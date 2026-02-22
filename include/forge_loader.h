#pragma once

// ============================================================================
// forge_loader.h — Load .forge model files via mmap
//
// .forge is a flat binary format:
//   - 512-byte header (magic, version, config, tensor count)
//   - Tensor table (name, offset, nbytes per tensor)
//   - 128-byte aligned FP16 tensor data
//
// The loader mmaps the entire file, then provides device pointers by
// copying each tensor to GPU memory. For models that fit in VRAM, this
// is a single bulk transfer.
//
// Usage:
//   ForgeModel model;
//   forge_load("model.forge", &model, stream);
//   // model.config has all hyperparameters
//   // model.layers[i].wq, .wk, etc. are device pointers
//   forge_free(&model, stream);
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#ifdef _WIN32
  #include <windows.h>
  #include <io.h>
#else
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  #include <unistd.h>
#endif

namespace transformer {

// ============================================================================
// CUDA_CHECK macro (if not already defined by tensor.h)
// ============================================================================
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)
#endif

// ============================================================================
// .forge file structures (must match convert.py exactly)
// ============================================================================

static constexpr uint32_t FORGE_MAGIC   = 0x45475246;  // "FRGE" little-endian
static constexpr uint32_t FORGE_VERSION = 1;
static constexpr uint32_t FORGE_HEADER_SIZE = 512;

// Packed model config — 96 bytes, matches convert.py's pack_model_config
#pragma pack(push, 1)
struct ForgeModelConfig {
    int32_t d_model;
    int32_t n_heads;
    int32_t n_kv_heads;
    int32_t d_head;
    int32_t d_ffn;
    int32_t n_layers;
    int32_t max_seq_len;
    int32_t vocab_size;
    int32_t bos_token_id;
    int32_t eos_token_id;
    float   rms_norm_eps;
    float   rope_theta;
    uint8_t use_rotary;
    uint8_t use_swiglu;
    uint8_t use_rmsnorm;
    uint8_t tie_word_embeddings;
    uint8_t _pad[44];  // Pad to 96 bytes
};
static_assert(sizeof(ForgeModelConfig) == 96, "ForgeModelConfig must be 96 bytes");

struct ForgeHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t header_size;
    uint32_t n_tensors;
    ForgeModelConfig config;
    uint8_t _pad[400];
};
static_assert(sizeof(ForgeHeader) == 512, "ForgeHeader must be 512 bytes");

struct ForgeTensorEntry {
    char     name[64];
    uint64_t offset;
    uint64_t nbytes;
};
static_assert(sizeof(ForgeTensorEntry) == 80, "ForgeTensorEntry must be 80 bytes");
#pragma pack(pop)

// ============================================================================
// Per-layer device weight pointers
// ============================================================================
struct ForgeLayerWeights {
    half* ln1_weight;    // [d_model]         RMSNorm weight
    half* wq;            // [d_model, d_model] transposed (col-major for NT)
    half* wk;            // [d_model, d_kv]    transposed
    half* wv;            // [d_model, d_kv]    transposed
    half* wo;            // [d_model, d_model] transposed
    half* ln2_weight;    // [d_model]         RMSNorm weight
    half* w_gate;        // [d_model, d_ffn]  transposed
    half* w_up;          // [d_model, d_ffn]  transposed
    half* w_down;        // [d_ffn, d_model]  transposed
};

// ============================================================================
// Full model — config + all weights on device
// ============================================================================
struct ForgeModel {
    ForgeModelConfig config;
    half*            embed_tokens;      // [vocab_size, d_model]
    ForgeLayerWeights* layers;          // [n_layers]
    half*            final_norm_weight; // [d_model]
    half*            lm_head;           // [d_model, vocab_size] transposed

    // Internal tracking
    half**  all_device_ptrs;   // Array of all allocated device pointers
    int     n_device_ptrs;
    void*   mmap_addr;         // For cleanup
    size_t  mmap_size;
#ifdef _WIN32
    HANDLE  file_handle;
    HANDLE  map_handle;
#else
    int     fd;
#endif
};

// ============================================================================
// Platform-specific mmap
// ============================================================================
static void* forge_mmap(const char* path, size_t* out_size, ForgeModel* model) {
#ifdef _WIN32
    HANDLE fh = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (fh == INVALID_HANDLE_VALUE) return nullptr;

    LARGE_INTEGER sz;
    GetFileSizeEx(fh, &sz);
    *out_size = (size_t)sz.QuadPart;

    HANDLE mh = CreateFileMappingA(fh, NULL, PAGE_READONLY, sz.HighPart, sz.LowPart, NULL);
    if (!mh) { CloseHandle(fh); return nullptr; }

    void* ptr = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    model->file_handle = fh;
    model->map_handle = mh;
    return ptr;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) return nullptr;

    struct stat st;
    fstat(fd, &st);
    *out_size = (size_t)st.st_size;

    void* ptr = mmap(nullptr, *out_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (ptr == MAP_FAILED) { close(fd); return nullptr; }

    // Advise sequential read
    madvise(ptr, *out_size, MADV_SEQUENTIAL);

    model->fd = fd;
    return ptr;
#endif
}

static void forge_munmap(ForgeModel* model) {
    if (!model->mmap_addr) return;
#ifdef _WIN32
    UnmapViewOfFile(model->mmap_addr);
    CloseHandle(model->map_handle);
    CloseHandle(model->file_handle);
#else
    munmap(model->mmap_addr, model->mmap_size);
    close(model->fd);
#endif
    model->mmap_addr = nullptr;
}

// ============================================================================
// Find tensor by name in the tensor table
// ============================================================================
static const ForgeTensorEntry* forge_find_tensor(
    const ForgeTensorEntry* table, uint32_t n_tensors, const char* name)
{
    for (uint32_t i = 0; i < n_tensors; i++) {
        if (strncmp(table[i].name, name, 63) == 0) {
            return &table[i];
        }
    }
    return nullptr;
}

// ============================================================================
// Copy a tensor from mmap'd host memory to GPU
// ============================================================================
static half* forge_upload_tensor(
    const void* file_base, const ForgeTensorEntry* entry,
    cudaStream_t stream)
{
    half* d_ptr = nullptr;
    size_t nbytes = entry->nbytes;

    // Allocate device memory (128-byte aligned by cudaMalloc)
    CUDA_CHECK(cudaMallocAsync(&d_ptr, nbytes, stream));

    // Copy from mmap'd file to device
    const void* src = static_cast<const uint8_t*>(file_base) + entry->offset;
    CUDA_CHECK(cudaMemcpyAsync(d_ptr, src, nbytes, cudaMemcpyHostToDevice, stream));

    return d_ptr;
}

// ============================================================================
// Load .forge file → ForgeModel with all weights on GPU
// ============================================================================
static bool forge_load(const char* path, ForgeModel* model, cudaStream_t stream = nullptr) {
    memset(model, 0, sizeof(ForgeModel));

    printf("Loading model: %s\n", path);

    // 1. mmap the file
    size_t file_size = 0;
    void* base = forge_mmap(path, &file_size, model);
    if (!base) {
        fprintf(stderr, "ERROR: Cannot open %s\n", path);
        return false;
    }
    model->mmap_addr = base;
    model->mmap_size = file_size;

    // 2. Validate header
    const ForgeHeader* hdr = static_cast<const ForgeHeader*>(base);
    if (hdr->magic != FORGE_MAGIC) {
        fprintf(stderr, "ERROR: Bad magic (expected FRGE)\n");
        forge_munmap(model);
        return false;
    }
    if (hdr->version != FORGE_VERSION) {
        fprintf(stderr, "ERROR: Unsupported version %u (expected %u)\n",
                hdr->version, FORGE_VERSION);
        forge_munmap(model);
        return false;
    }

    model->config = hdr->config;
    const ForgeModelConfig& cfg = model->config;

    printf("  d_model=%d, n_heads=%d/%d (Q/KV), n_layers=%d, vocab=%d\n",
           cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.n_layers, cfg.vocab_size);
    printf("  d_ffn=%d, d_head=%d, max_seq=%d\n",
           cfg.d_ffn, cfg.d_head, cfg.max_seq_len);

    // 3. Read tensor table
    const ForgeTensorEntry* table = reinterpret_cast<const ForgeTensorEntry*>(
        static_cast<const uint8_t*>(base) + FORGE_HEADER_SIZE);
    uint32_t n_tensors = hdr->n_tensors;

    printf("  %u tensors in file, %.1f MB\n",
           n_tensors, file_size / (1024.0 * 1024.0));

    // 4. Allocate tracking arrays
    int max_ptrs = 3 + cfg.n_layers * 9;  // embed + layers*9 + norm + lm_head
    model->all_device_ptrs = new half*[max_ptrs];
    model->n_device_ptrs = 0;
    model->layers = new ForgeLayerWeights[cfg.n_layers];
    memset(model->layers, 0, sizeof(ForgeLayerWeights) * cfg.n_layers);

    auto upload = [&](const char* name) -> half* {
        const ForgeTensorEntry* e = forge_find_tensor(table, n_tensors, name);
        if (!e) {
            fprintf(stderr, "  WARNING: tensor '%s' not found\n", name);
            return nullptr;
        }
        half* ptr = forge_upload_tensor(base, e, stream);
        model->all_device_ptrs[model->n_device_ptrs++] = ptr;
        return ptr;
    };

    // 5. Upload tensors to GPU
    printf("  Uploading to GPU...\n");

    model->embed_tokens = upload("embed_tokens");

    for (int l = 0; l < cfg.n_layers; l++) {
        char name[64];
        ForgeLayerWeights& lw = model->layers[l];

        snprintf(name, 64, "layers.%d.ln1_weight", l);   lw.ln1_weight = upload(name);
        snprintf(name, 64, "layers.%d.wq", l);           lw.wq = upload(name);
        snprintf(name, 64, "layers.%d.wk", l);           lw.wk = upload(name);
        snprintf(name, 64, "layers.%d.wv", l);           lw.wv = upload(name);
        snprintf(name, 64, "layers.%d.wo", l);           lw.wo = upload(name);
        snprintf(name, 64, "layers.%d.ln2_weight", l);   lw.ln2_weight = upload(name);
        snprintf(name, 64, "layers.%d.w_gate", l);       lw.w_gate = upload(name);
        snprintf(name, 64, "layers.%d.w_up", l);         lw.w_up = upload(name);
        snprintf(name, 64, "layers.%d.w_down", l);       lw.w_down = upload(name);
    }

    model->final_norm_weight = upload("final_norm_weight");
    model->lm_head = upload("lm_head");

    // Wait for all uploads
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 6. Unmap file — data is now on GPU
    forge_munmap(model);

    printf("  Loaded %d tensors to GPU\n", model->n_device_ptrs);
    return true;
}

// ============================================================================
// Free all GPU memory
// ============================================================================
static void forge_free(ForgeModel* model, cudaStream_t stream = nullptr) {
    for (int i = 0; i < model->n_device_ptrs; i++) {
        if (model->all_device_ptrs[i]) {
            cudaFreeAsync(model->all_device_ptrs[i], stream);
        }
    }
    delete[] model->all_device_ptrs;
    delete[] model->layers;
    model->all_device_ptrs = nullptr;
    model->layers = nullptr;
    model->n_device_ptrs = 0;
}

} // namespace transformer