// ============================================================================
// Transformer Block — Full Forward Pass with CUDA Graph Support
//
// Ties together all kernel modules into a complete transformer layer:
//   Pre-LayerNorm → QKV Projection → RoPE → Flash Attention →
//   Attn Output Projection → Residual → Pre-LayerNorm →
//   SwiGLU FFN → Residual
//
// CUDA Graphs:
//   The forward pass is captured into a CUDA graph on first invocation for
//   each unique (batch_size, seq_len) pair. On subsequent calls, the graph
//   executable is updated in-place via cudaGraphExecUpdate (fast parameter
//   patch — no re-instantiation) and replayed. This eliminates ~5μs kernel
//   launch overhead per kernel × ~20 kernels/layer × N layers.
//
//   For a 12-layer model: ~240 kernel launches → 1 graph launch.
//   Expected savings: ~1.0–1.5ms per forward pass.
//
//   Why this works: all control flow (use_rotary, use_swiglu, n_layers) is
//   fixed at init time. The only per-call variable is start_pos, which
//   changes kernel parameters but not graph topology. cudaGraphExecUpdate
//   handles parameter-only changes as a fast path.
// ============================================================================

#include "../include/transformer_config.h"
#include "../include/tensor.h"
#include "../include/gemm_operations.h"
#include "../include/flash_attention.h"
#include "../include/layer_norm.h"
#include "../include/rotary_embedding.h"
#include "../include/activation_kernels.h"

#include <unordered_map>
#include <utility>

namespace transformer {

// ============================================================================
// Simple vector add kernel: output = a + b  (element-wise, FP16)
// ============================================================================
__global__ void vector_add_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half*       __restrict__ output,
    const int   n
) {
    int idx2 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx2 + 1 < n) {
        half2 va = *reinterpret_cast<const half2*>(&a[idx2]);
        half2 vb = *reinterpret_cast<const half2*>(&b[idx2]);
        float2 fa = __half22float2(va), fb = __half22float2(vb);
        float2 r = { fa.x + fb.x, fa.y + fb.y };
        *reinterpret_cast<half2*>(&output[idx2]) = __float22half2_rn(r);
    } else if (idx2 < n) {
        output[idx2] = __float2half(__half2float(a[idx2]) + __half2float(b[idx2]));
    }
}

static void launch_vector_add(const half* a, const half* b, half* output,
                               int n, cudaStream_t stream) {
    int block = 256;
    int grid  = (n / 2 + block - 1) / block;
    vector_add_kernel<<<grid, block, 0, stream>>>(a, b, output, n);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// KV Cache Update Kernel
//
// Replaces the host-side memcpy loop for graph compatibility.
// A host loop launching 2*batch_size memcpys creates variable node counts
// if batch_size changes. A single kernel is one fixed node, and start_pos
// is just a kernel parameter that gets patched by cudaGraphExecUpdate.
// ============================================================================
__global__ void kv_cache_update_kernel(
    half*       __restrict__ k_cache,   // [max_seq, d_model]
    half*       __restrict__ v_cache,   // [max_seq, d_model]
    const half* __restrict__ K_proj,    // [batch*seq, d_model]
    const half* __restrict__ V_proj,    // [batch*seq, d_model]
    const int   d_model,
    const int   seq_len,
    const int   start_pos,
    const int   batch_size
) {
    // blockIdx.y: 0 = K, 1 = V
    const int kv_select = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * seq_len * d_model;
    if (idx >= total) return;

    const int dim     = idx % d_model;
    const int seq_idx = (idx / d_model) % seq_len;
    const int b       = idx / (seq_len * d_model);

    const int src_offset = (b * seq_len + seq_idx) * d_model + dim;
    const int dst_offset = (start_pos + seq_idx) * d_model + dim;

    if (kv_select == 0)
        k_cache[dst_offset] = K_proj[src_offset];
    else
        v_cache[dst_offset] = V_proj[src_offset];
}

static void launch_kv_cache_update(
    half* k_cache, half* v_cache,
    const half* K_proj, const half* V_proj,
    int d_model, int seq_len, int start_pos, int batch_size,
    cudaStream_t stream)
{
    int total = batch_size * seq_len * d_model;
    int block = 256;
    int grid_x = (total + block - 1) / block;
    dim3 grid(grid_x, 2);  // y=0 K, y=1 V
    kv_cache_update_kernel<<<grid, block, 0, stream>>>(
        k_cache, v_cache, K_proj, V_proj,
        d_model, seq_len, start_pos, batch_size);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Per-Layer Weights (stored in column-major for CUTLASS NT layout)
// ============================================================================
struct TransformerLayerWeights {
    half* W_q       = nullptr;   // [d_model, d_model]
    half* W_k       = nullptr;
    half* W_v       = nullptr;
    half* W_o       = nullptr;
    half* b_q       = nullptr;   // [d_model] (optional)
    half* b_k       = nullptr;
    half* b_v       = nullptr;
    half* b_o       = nullptr;

    half* ln1_gamma  = nullptr;  // [d_model]
    half* ln1_beta   = nullptr;

    half* W_gate    = nullptr;   // [d_model, d_ffn] for SwiGLU
    half* W_up      = nullptr;   // [d_model, d_ffn]
    half* W_down    = nullptr;   // [d_ffn, d_model]

    half* ln2_gamma  = nullptr;  // [d_model]
    half* ln2_beta   = nullptr;
};

// ============================================================================
// Scratch Buffers (pre-allocated workspace)
// ============================================================================
struct ScratchBuffers {
    half* ln_out     = nullptr;
    half* qkv        = nullptr;
    half* attn_out   = nullptr;
    half* ffn_gate   = nullptr;
    half* ffn_up     = nullptr;
    half* ffn_inter  = nullptr;
    half* ffn_out    = nullptr;
    half* residual   = nullptr;
    float* attn_lse  = nullptr;

    void allocate(const ModelConfig& config, int max_batch, cudaStream_t stream) {
        size_t max_tokens = static_cast<size_t>(max_batch) * config.max_seq_len;
        int d = config.d_model;
        int d_ffn = config.d_ffn_gate();

        CUDA_CHECK(cudaMallocAsync(&ln_out,    max_tokens * d * sizeof(half), stream));
        CUDA_CHECK(cudaMallocAsync(&qkv,       max_tokens * 3 * d * sizeof(half), stream));
        CUDA_CHECK(cudaMallocAsync(&attn_out,  max_tokens * d * sizeof(half), stream));
        CUDA_CHECK(cudaMallocAsync(&ffn_gate,  max_tokens * d_ffn * sizeof(half), stream));
        CUDA_CHECK(cudaMallocAsync(&ffn_up,    max_tokens * d_ffn * sizeof(half), stream));
        CUDA_CHECK(cudaMallocAsync(&ffn_inter, max_tokens * d_ffn * sizeof(half), stream));
        CUDA_CHECK(cudaMallocAsync(&ffn_out,   max_tokens * d * sizeof(half), stream));
        CUDA_CHECK(cudaMallocAsync(&residual,  max_tokens * d * sizeof(half), stream));
        CUDA_CHECK(cudaMallocAsync(&attn_lse,
            static_cast<size_t>(max_batch) * config.n_heads * config.max_seq_len * sizeof(float),
            stream));
    }

    void free(cudaStream_t stream) {
        auto safe_free = [stream](half*& p) {
            if (p) { cudaFreeAsync(p, stream); p = nullptr; }
        };
        safe_free(ln_out); safe_free(qkv); safe_free(attn_out);
        safe_free(ffn_gate); safe_free(ffn_up); safe_free(ffn_inter);
        safe_free(ffn_out); safe_free(residual);
        if (attn_lse) { cudaFreeAsync(attn_lse, stream); attn_lse = nullptr; }
    }
};

// ============================================================================
// Transformer Layer Forward Pass
// ============================================================================
class TransformerLayer {
public:
    TransformerLayer(const ModelConfig& config, int layer_idx)
        : config_(config), layer_idx_(layer_idx) {}

    void forward(
        half* input, half* output,
        const TransformerLayerWeights& weights,
        ScratchBuffers& scratch,
        KVCache<half>& kv_cache,
        GemmManager& gemm,
        const RoPEConfig& rope,
        int batch_size, int seq_len, int start_pos,
        cudaStream_t stream
    ) {
        const int N = batch_size * seq_len;
        const int d = config_.d_model;
        const int n_heads = config_.n_heads;
        const int d_head = config_.d_head;
        const int d_ffn = config_.d_ffn_gate();

        gemm.set_stream(stream);

        // Step 1: Pre-LayerNorm (attention)
        {
            LayerNormParams ln = {};
            ln.input = input; ln.gamma = weights.ln1_gamma;
            ln.beta = weights.ln1_beta; ln.output = scratch.ln_out;
            ln.num_tokens = N; ln.d_model = d;
            ln.eps = 1e-5f; ln.use_rmsnorm = false; ln.stream = stream;
            launch_fused_layernorm(ln);
        }

        // Step 2: QKV projections
        half* Q      = scratch.qkv;
        half* K_proj = scratch.qkv + static_cast<size_t>(N) * d;
        half* V_proj = scratch.qkv + static_cast<size_t>(N) * d * 2;

        gemm.gemm_nt(scratch.ln_out, weights.W_q, Q,      N, d, d);
        gemm.gemm_nt(scratch.ln_out, weights.W_k, K_proj, N, d, d);
        gemm.gemm_nt(scratch.ln_out, weights.W_v, V_proj, N, d, d);

        // Step 3: RoPE
        if (config_.use_rotary) {
            launch_rope(Q, K_proj, rope, batch_size * n_heads,
                       seq_len, start_pos, stream);
        }

        // Step 4: KV cache update (kernel — graph-safe)
        {
            half* k_cache = kv_cache.get(layer_idx_, 0);
            half* v_cache = kv_cache.get(layer_idx_, 1);
            launch_kv_cache_update(k_cache, v_cache, K_proj, V_proj,
                                   d, seq_len, start_pos, batch_size, stream);
        }

        // Step 5: Flash Attention
        {
            int total_len = start_pos + seq_len;
            FlashAttentionParams fa = {};
            fa.Q = Q; fa.K = kv_cache.get(layer_idx_, 0);
            fa.V = kv_cache.get(layer_idx_, 1);
            fa.O = scratch.attn_out; fa.L = scratch.attn_lse;
            fa.batch_size = batch_size; fa.num_heads = n_heads;
            fa.seq_len = total_len; fa.d_head = d_head;
            fa.scale = 1.0f / sqrtf(static_cast<float>(d_head));
            fa.causal = true; fa.stream = stream;
            launch_flash_attention(fa);
        }

        // Step 6: Attention output projection
        gemm.gemm_nt(scratch.attn_out, weights.W_o, scratch.ffn_out, N, d, d);

        // Step 7: Pre-LayerNorm (FFN) with residual
        {
            LayerNormParams ln = {};
            ln.input = scratch.ffn_out; ln.residual = input;
            ln.bias = weights.b_o; ln.gamma = weights.ln2_gamma;
            ln.beta = weights.ln2_beta; ln.output = scratch.ln_out;
            ln.residual_out = scratch.residual;
            ln.num_tokens = N; ln.d_model = d;
            ln.eps = 1e-5f; ln.use_rmsnorm = false; ln.stream = stream;
            launch_fused_layernorm(ln);
        }

        // Step 8: FFN — SwiGLU or GELU
        if (config_.use_swiglu) {
            gemm.gemm_nt(scratch.ln_out, weights.W_gate, scratch.ffn_gate, N, d_ffn, d);
            gemm.gemm_nt(scratch.ln_out, weights.W_up,   scratch.ffn_up,   N, d_ffn, d);
            launch_fused_swiglu(scratch.ffn_gate, scratch.ffn_up,
                                scratch.ffn_inter, N, d_ffn, stream);
        } else {
            gemm.gemm_nt(scratch.ln_out, weights.W_up, scratch.ffn_up, N, d_ffn, d);
            launch_fused_gelu(scratch.ffn_up, scratch.ffn_inter, N, d_ffn, stream);
        }

        gemm.gemm_nt(scratch.ffn_inter, weights.W_down, scratch.ffn_out, N, d, d_ffn);

        // Step 9: Residual addition
        launch_vector_add(scratch.residual, scratch.ffn_out, output, N * d, stream);
    }

private:
    ModelConfig config_;
    int         layer_idx_;
};

// ============================================================================
// CUDA Graph Cache
//
// Keyed by (batch_size, seq_len). Topology is identical for all start_pos
// values within the same shape — same kernels, same grid dims, same deps.
// cudaGraphExecUpdate patches only the changed scalar parameters.
//
// For decode: seq_len=1 always, so ONE cached graph serves every step.
// For prefill: seq_len varies, so we cache per unique prefill length.
//   In practice there are 1-3 prefill shapes in a session.
// ============================================================================
struct GraphKey {
    int batch_size;
    int seq_len;
    bool operator==(const GraphKey& o) const {
        return batch_size == o.batch_size && seq_len == o.seq_len;
    }
};

struct GraphKeyHash {
    size_t operator()(const GraphKey& k) const {
        return std::hash<uint64_t>()(
            (static_cast<uint64_t>(k.batch_size) << 32) |
             static_cast<uint32_t>(k.seq_len));
    }
};

struct CachedGraph {
    cudaGraphExec_t exec = nullptr;
    int last_start_pos = -1;
};

// ============================================================================
// Full Transformer Model — with CUDA Graph acceleration
// ============================================================================
class Transformer {
public:
    Transformer(const ModelConfig& config, bool use_graphs = true)
        : config_(config), use_graphs_(use_graphs)
    {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    ~Transformer() {
        for (auto& [key, cached] : graph_cache_) {
            if (cached.exec) cudaGraphExecDestroy(cached.exec);
        }
        scratch_.free(stream_);
        kv_cache_.free(stream_);
        rope_.free(stream_);
        cudaStreamDestroy(stream_);
    }

    void init(int max_batch_size = 1) {
        kv_cache_.allocate(config_.n_layers, config_.n_heads,
                          config_.d_head, config_.max_seq_len, stream_);
        scratch_.allocate(config_, max_batch_size, stream_);
        if (config_.use_rotary) {
            rope_.init(config_.max_seq_len, config_.d_head, 10000.0f, stream_);
        }
        gemm_.set_stream(stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    void forward(half* input, half* output,
                 const TransformerLayerWeights* layer_weights,
                 int batch_size, int seq_len, int start_pos)
    {
        if (use_graphs_) {
            forward_graph(input, output, layer_weights,
                         batch_size, seq_len, start_pos);
        } else {
            forward_eager(input, output, layer_weights,
                         batch_size, seq_len, start_pos);
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        }
    }

private:
    ModelConfig     config_;
    KVCache<half>   kv_cache_;
    ScratchBuffers  scratch_;
    GemmManager     gemm_;
    RoPEConfig      rope_;
    cudaStream_t    stream_ = nullptr;
    bool            use_graphs_;

    std::unordered_map<GraphKey, CachedGraph, GraphKeyHash> graph_cache_;

    // -----------------------------------------------------------------------
    // Eager forward — no graph, used as capture body and fallback
    // -----------------------------------------------------------------------
    void forward_eager(half* input, half* output,
                       const TransformerLayerWeights* layer_weights,
                       int batch_size, int seq_len, int start_pos)
    {
        half* cur_in  = input;
        half* cur_out = output;

        for (int l = 0; l < config_.n_layers; l++) {
            TransformerLayer layer(config_, l);
            layer.forward(cur_in, cur_out, layer_weights[l],
                         scratch_, kv_cache_, gemm_, rope_,
                         batch_size, seq_len, start_pos, stream_);
            std::swap(cur_in, cur_out);
        }

        if (config_.n_layers % 2 == 1 && cur_in != output) {
            size_t bytes = static_cast<size_t>(batch_size) * seq_len
                         * config_.d_model * sizeof(half);
            CUDA_CHECK(cudaMemcpyAsync(output, cur_in, bytes,
                                        cudaMemcpyDeviceToDevice, stream_));
        }
    }

    // -----------------------------------------------------------------------
    // Graph-accelerated forward
    //
    // Strategy:
    //   1. Always capture a fresh graph (encodes current start_pos)
    //   2. If no cached exec for this shape → instantiate (one-time, ~100μs)
    //   3. If cached exec exists → cudaGraphExecUpdate (fast path, ~10μs)
    //      - Same topology, only scalar params differ → always succeeds
    //      - If it somehow fails → re-instantiate (defensive fallback)
    //   4. Launch the exec
    //
    // For decode (seq_len=1), step 3 runs every token. The update cost
    // (~10μs) replaces ~240 individual kernel launches (~5μs each = ~1.2ms).
    // Net savings: ~1.0–1.2ms per forward pass on a 12-layer model.
    //
    // Note on FA grid dims: Flash attention's grid is based on total_len
    // (start_pos + seq_len), which changes every decode step. BUT:
    // - For decode, seq_len=1, so Q-side tiles = 1 always
    // - Grid = batch_size * n_heads * ceil(1/BLOCK_M) = fixed
    // - K/V length changes but that's a loop bound inside the kernel,
    //   not a grid dimension. So topology is truly stable.
    // -----------------------------------------------------------------------
    void forward_graph(half* input, half* output,
                       const TransformerLayerWeights* layer_weights,
                       int batch_size, int seq_len, int start_pos)
    {
        GraphKey key{batch_size, seq_len};

        // Capture current forward into a graph
        cudaGraph_t graph = nullptr;
        CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
        forward_eager(input, output, layer_weights, batch_size, seq_len, start_pos);
        CUDA_CHECK(cudaStreamEndCapture(stream_, &graph));

        auto it = graph_cache_.find(key);

        if (it == graph_cache_.end()) {
            // First time — instantiate
            CachedGraph cached;
            CUDA_CHECK(cudaGraphInstantiate(&cached.exec, graph, nullptr, nullptr, 0));
            cached.last_start_pos = start_pos;
            graph_cache_[key] = cached;
            it = graph_cache_.find(key);
        } else {
            // Update existing executable with new parameters
            cudaGraphExecUpdateResult update_result;
            cudaGraphNode_t error_node;
            cudaError_t err = cudaGraphExecUpdate(
                it->second.exec, graph, &error_node, &update_result);

            if (err != cudaSuccess ||
                update_result != cudaGraphExecUpdateSuccess)
            {
                // Topology mismatch — shouldn't happen, but handle gracefully
                cudaGraphExecDestroy(it->second.exec);
                CUDA_CHECK(cudaGraphInstantiate(
                    &it->second.exec, graph, nullptr, nullptr, 0));
            }
            it->second.last_start_pos = start_pos;
        }

        cudaGraphDestroy(graph);

        // Launch
        CUDA_CHECK(cudaGraphLaunch(it->second.exec, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
};

} // namespace transformer
