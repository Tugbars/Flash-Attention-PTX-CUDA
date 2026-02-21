#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cassert>
#include <memory>
#include <cstdio>

namespace transformer {

// ============================================================================
// CUDA error checking
// ============================================================================
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                    \
            abort();                                                            \
        }                                                                       \
    } while (0)

// ============================================================================
// GPU Tensor — thin wrapper over device memory
// ============================================================================
template <typename T>
class Tensor {
public:
    Tensor() = default;

    // Allocate with 128-byte alignment
    Tensor(int32_t rows, int32_t cols, cudaStream_t stream = nullptr)
        : rows_(rows), cols_(cols), stream_(stream)
    {
        size_t bytes = static_cast<size_t>(rows) * cols * sizeof(T);
        // Pad to 128B alignment
        bytes = (bytes + 127) & ~127ULL;
        CUDA_CHECK(cudaMallocAsync(&data_, bytes, stream_));
        owns_data_ = true;
    }

    // View into existing memory (no ownership)
    Tensor(T* data, int32_t rows, int32_t cols)
        : data_(data), rows_(rows), cols_(cols), owns_data_(false) {}

    ~Tensor() {
        if (owns_data_ && data_) {
            cudaFreeAsync(data_, stream_);
        }
    }

    // Move only
    Tensor(Tensor&& other) noexcept
        : data_(other.data_), rows_(other.rows_), cols_(other.cols_),
          owns_data_(other.owns_data_), stream_(other.stream_)
    {
        other.data_ = nullptr;
        other.owns_data_ = false;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            if (owns_data_ && data_) cudaFreeAsync(data_, stream_);
            data_ = other.data_;
            rows_ = other.rows_;
            cols_ = other.cols_;
            owns_data_ = other.owns_data_;
            stream_ = other.stream_;
            other.data_ = nullptr;
            other.owns_data_ = false;
        }
        return *this;
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Accessors
    T*       data()       { return data_; }
    const T* data() const { return data_; }
    int32_t  rows() const { return rows_; }
    int32_t  cols() const { return cols_; }
    size_t   numel() const { return static_cast<size_t>(rows_) * cols_; }
    size_t   bytes() const { return numel() * sizeof(T); }

    // Slice a sub-view [row_start, row_start + n_rows) — zero-copy
    Tensor<T> slice_rows(int32_t row_start, int32_t n_rows) {
        assert(row_start + n_rows <= rows_);
        return Tensor<T>(data_ + static_cast<size_t>(row_start) * cols_,
                         n_rows, cols_);
    }

    // Zero-fill
    void zero(cudaStream_t s = nullptr) {
        CUDA_CHECK(cudaMemsetAsync(data_, 0, bytes(), s ? s : stream_));
    }

    // Copy from host
    void copy_from_host(const T* host_data, size_t count, cudaStream_t s = nullptr) {
        CUDA_CHECK(cudaMemcpyAsync(data_, host_data,
                                    count * sizeof(T),
                                    cudaMemcpyHostToDevice,
                                    s ? s : stream_));
    }

    // Copy to host
    void copy_to_host(T* host_data, size_t count, cudaStream_t s = nullptr) const {
        CUDA_CHECK(cudaMemcpyAsync(host_data, data_,
                                    count * sizeof(T),
                                    cudaMemcpyDeviceToHost,
                                    s ? s : stream_));
    }

private:
    T*           data_      = nullptr;
    int32_t      rows_      = 0;
    int32_t      cols_      = 0;
    bool         owns_data_ = false;
    cudaStream_t stream_    = nullptr;
};

// ============================================================================
// KV Cache — Pre-allocated ring buffer for autoregressive decoding
// ============================================================================
template <typename T>
struct KVCache {
    // Shape: [n_layers, 2(K+V), max_seq_len, n_heads, d_head]
    // Stored as contiguous block, indexed per-layer
    T*      data       = nullptr;
    int32_t max_seq    = 0;
    int32_t n_layers   = 0;
    int32_t n_heads    = 0;
    int32_t d_head     = 0;
    int32_t cur_len    = 0;  // Current sequence position

    size_t layer_bytes() const {
        return 2ULL * max_seq * n_heads * d_head * sizeof(T);
    }

    // Get K or V pointer for a specific layer
    // kv_idx: 0=K, 1=V
    T* get(int layer, int kv_idx) {
        size_t offset = static_cast<size_t>(layer) * 2 * max_seq * n_heads * d_head
                      + static_cast<size_t>(kv_idx) * max_seq * n_heads * d_head;
        return data + offset;
    }

    void allocate(int layers, int heads, int head_dim, int max_seq_len,
                  cudaStream_t stream = nullptr) {
        n_layers = layers;
        n_heads  = heads;
        d_head   = head_dim;
        max_seq  = max_seq_len;
        cur_len  = 0;
        size_t total = static_cast<size_t>(n_layers) * 2 * max_seq * n_heads * d_head * sizeof(T);
        CUDA_CHECK(cudaMallocAsync(&data, total, stream));
        CUDA_CHECK(cudaMemsetAsync(data, 0, total, stream));
    }

    void free(cudaStream_t stream = nullptr) {
        if (data) {
            CUDA_CHECK(cudaFreeAsync(data, stream));
            data = nullptr;
        }
    }
};

} // namespace transformer
