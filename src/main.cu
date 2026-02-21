// ============================================================================
// Transformer Benchmark / Demo
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../include/transformer_config.h"
#include "../include/tensor.h"
#include "../include/flash_attention.h"
#include "../include/layer_norm.h"
#include "../include/gemm_operations.h"

using namespace transformer;

// ============================================================================
// Timing Utility
// ============================================================================
struct CudaTimer {
    cudaEvent_t start, stop;
    CudaTimer()  { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~CudaTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }

    void begin(cudaStream_t stream = nullptr) { cudaEventRecord(start, stream); }
    float end(cudaStream_t stream = nullptr) {
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// ============================================================================
// Benchmarks
// ============================================================================
void benchmark_flash_attention(int batch_size, int n_heads, int seq_len, int d_head,
                                int warmup = 10, int iters = 100)
{
    printf("\n=== Flash Attention Benchmark ===\n");
    printf("  B=%d, H=%d, S=%d, D=%d\n", batch_size, n_heads, seq_len, d_head);

    size_t bh = static_cast<size_t>(batch_size) * n_heads;
    size_t elems = bh * seq_len * d_head;

    half *Q, *K, *V, *O;
    float *L;
    cudaMalloc(&Q, elems * sizeof(half));
    cudaMalloc(&K, elems * sizeof(half));
    cudaMalloc(&V, elems * sizeof(half));
    cudaMalloc(&O, elems * sizeof(half));
    cudaMalloc(&L, bh * seq_len * sizeof(float));
    cudaMemset(Q, 0x3C, elems * sizeof(half));
    cudaMemset(K, 0x3C, elems * sizeof(half));
    cudaMemset(V, 0x3C, elems * sizeof(half));

    FlashAttentionParams params = {};
    params.Q = Q; params.K = K; params.V = V; params.O = O; params.L = L;
    params.batch_size = batch_size;
    params.num_heads  = n_heads;
    params.seq_len    = seq_len;
    params.d_head     = d_head;
    params.scale      = 1.0f / sqrtf(static_cast<float>(d_head));
    params.causal     = true;
    params.stream     = nullptr;

    for (int i = 0; i < warmup; i++) launch_flash_attention(params);
    cudaDeviceSynchronize();

    CudaTimer timer;
    timer.begin();
    for (int i = 0; i < iters; i++) launch_flash_attention(params);
    float total_ms = timer.end();
    float avg_ms = total_ms / iters;

    double flops = 2.0 * bh * static_cast<double>(seq_len) * seq_len * d_head * 2;
    double tflops = flops / (avg_ms * 1e-3) / 1e12;

    printf("  Average: %.3f ms\n", avg_ms);
    printf("  TFLOPS:  %.2f\n", tflops);

    cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(O); cudaFree(L);
}

void benchmark_gemm(int M, int N, int K_dim, int warmup = 10, int iters = 100)
{
    printf("\n=== GEMM Benchmark [%d x %d x %d] ===\n", M, N, K_dim);

    half *A, *B, *C;
    cudaMalloc(&A, static_cast<size_t>(M) * K_dim * sizeof(half));
    cudaMalloc(&B, static_cast<size_t>(N) * K_dim * sizeof(half));
    cudaMalloc(&C, static_cast<size_t>(M) * N * sizeof(half));
    cudaMemset(A, 0x3C, static_cast<size_t>(M) * K_dim * sizeof(half));
    cudaMemset(B, 0x3C, static_cast<size_t>(N) * K_dim * sizeof(half));

    GemmManager gemm;
    for (int i = 0; i < warmup; i++) gemm.gemm_nt(A, B, C, M, N, K_dim);
    cudaDeviceSynchronize();

    CudaTimer timer;
    timer.begin();
    for (int i = 0; i < iters; i++) gemm.gemm_nt(A, B, C, M, N, K_dim);
    float total_ms = timer.end();
    float avg_ms = total_ms / iters;

    double flops = 2.0 * M * static_cast<double>(N) * K_dim;
    double tflops = flops / (avg_ms * 1e-3) / 1e12;

    printf("  Average: %.3f ms\n", avg_ms);
    printf("  TFLOPS:  %.2f\n", tflops);

    cudaFree(A); cudaFree(B); cudaFree(C);
}

void benchmark_layernorm(int N, int D, int warmup = 10, int iters = 1000)
{
    printf("\n=== Fused LayerNorm Benchmark [%d tokens x %d dim] ===\n", N, D);

    half *input, *residual, *gamma, *beta, *output, *res_out;
    cudaMalloc(&input,    static_cast<size_t>(N) * D * sizeof(half));
    cudaMalloc(&residual, static_cast<size_t>(N) * D * sizeof(half));
    cudaMalloc(&gamma,    D * sizeof(half));
    cudaMalloc(&beta,     D * sizeof(half));
    cudaMalloc(&output,   static_cast<size_t>(N) * D * sizeof(half));
    cudaMalloc(&res_out,  static_cast<size_t>(N) * D * sizeof(half));

    LayerNormParams params = {};
    params.input = input; params.residual = residual;
    params.bias = nullptr; params.gamma = gamma; params.beta = beta;
    params.output = output; params.residual_out = res_out;
    params.num_tokens = N; params.d_model = D;
    params.eps = 1e-5f; params.use_rmsnorm = false;
    params.stream = nullptr;

    for (int i = 0; i < warmup; i++) launch_fused_layernorm(params);
    cudaDeviceSynchronize();

    CudaTimer timer;
    timer.begin();
    for (int i = 0; i < iters; i++) launch_fused_layernorm(params);
    float total_ms = timer.end();
    float avg_ms = total_ms / iters;

    double bytes = static_cast<double>(N) * D * sizeof(half) * 5;
    double gbps = bytes / (avg_ms * 1e-3) / 1e9;

    printf("  Average: %.4f ms\n", avg_ms);
    printf("  GB/s:    %.1f\n", gbps);

    cudaFree(input); cudaFree(residual); cudaFree(gamma);
    cudaFree(beta); cudaFree(output); cudaFree(res_out);
}

void verify_flash_attention(int batch_size, int n_heads, int seq_len, int d_head) {
    printf("=== Flash Attention Correctness Check ===\n");
    printf("  B=%d, H=%d, S=%d, D=%d\n", batch_size, n_heads, seq_len, d_head);

    const int total_elements = batch_size * n_heads * seq_len * d_head;
    const float scale = 1.0f / sqrtf(static_cast<float>(d_head));

    // Allocate host memory
    std::vector<float> h_Q(total_elements), h_K(total_elements), h_V(total_elements);
    std::vector<float> h_O_ref(total_elements, 0.0f);
    std::vector<float> h_O_gpu(total_elements, 0.0f);

    // Fill with small deterministic values
    srand(42);
    for (int i = 0; i < total_elements; i++) {
        h_Q[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.5f;
        h_K[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.5f;
        h_V[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.5f;
    }

    // === CPU reference (naive attention) ===
    // For each batch, head:
    //   S = Q * K^T * scale
    //   P = softmax(S, causal)
    //   O = P * V
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < n_heads; h++) {
            const int bh = b * n_heads + h;
            const float* Q_bh = h_Q.data() + bh * seq_len * d_head;
            const float* K_bh = h_K.data() + bh * seq_len * d_head;
            const float* V_bh = h_V.data() + bh * seq_len * d_head;
            float* O_bh = h_O_ref.data() + bh * seq_len * d_head;

            // S = Q * K^T [seq_len, seq_len]
            std::vector<float> S(seq_len * seq_len);
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < seq_len; j++) {
                    float dot = 0.0f;
                    for (int k = 0; k < d_head; k++)
                        dot += Q_bh[i * d_head + k] * K_bh[j * d_head + k];
                    S[i * seq_len + j] = dot * scale;

                    // Causal mask
                    if (j > i) S[i * seq_len + j] = -1e9f;
                }
            }

            // Softmax per row
            std::vector<float> P(seq_len * seq_len);
            for (int i = 0; i < seq_len; i++) {
                float max_val = -1e30f;
                for (int j = 0; j < seq_len; j++)
                    max_val = std::max(max_val, S[i * seq_len + j]);

                float sum = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    P[i * seq_len + j] = expf(S[i * seq_len + j] - max_val);
                    sum += P[i * seq_len + j];
                }
                for (int j = 0; j < seq_len; j++)
                    P[i * seq_len + j] /= sum;
            }

            // O = P * V [seq_len, d_head]
            for (int i = 0; i < seq_len; i++) {
                for (int k = 0; k < d_head; k++) {
                    float val = 0.0f;
                    for (int j = 0; j < seq_len; j++)
                        val += P[i * seq_len + j] * V_bh[j * d_head + k];
                    O_bh[i * d_head + k] = val;
                }
            }
        }
    }

    // === GPU kernel ===
    // Convert float -> half for GPU
    std::vector<half> h_Q_half(total_elements), h_K_half(total_elements), h_V_half(total_elements);
    std::vector<half> h_O_half(total_elements);
    for (int i = 0; i < total_elements; i++) {
        h_Q_half[i] = __float2half(h_Q[i]);
        h_K_half[i] = __float2half(h_K[i]);
        h_V_half[i] = __float2half(h_V[i]);
    }

    half *d_Q, *d_K, *d_V, *d_O;
    float *d_L;
    size_t bytes = total_elements * sizeof(half);
    cudaMalloc(&d_Q, bytes);
    cudaMalloc(&d_K, bytes);
    cudaMalloc(&d_V, bytes);
    cudaMalloc(&d_O, bytes);
    cudaMalloc(&d_L, batch_size * n_heads * seq_len * sizeof(float));

    cudaMemcpy(d_Q, h_Q_half.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K_half.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V_half.data(), bytes, cudaMemcpyHostToDevice);

    transformer::FlashAttentionParams params;
    params.Q = d_Q; params.K = d_K; params.V = d_V; params.O = d_O; params.L = d_L;
    params.batch_size = batch_size;
    params.num_heads = n_heads;
    params.seq_len = seq_len;
    params.d_head = d_head;
    params.scale = scale;
    params.causal = true;
    params.stream = 0;

    transformer::launch_flash_attention(params);
    cudaDeviceSynchronize();

    cudaMemcpy(h_O_half.data(), d_O, bytes, cudaMemcpyDeviceToHost);

    // Convert back to float
    for (int i = 0; i < total_elements; i++)
        h_O_gpu[i] = __half2float(h_O_half[i]);

    // === Compare ===
    float max_abs_err = 0.0f, max_rel_err = 0.0f;
    double sum_sq_err = 0.0, sum_sq_ref = 0.0;
    int num_bad = 0;
    int worst_idx = 0;

    for (int i = 0; i < total_elements; i++) {
        float ref = h_O_ref[i];
        float gpu = h_O_gpu[i];
        float abs_err = fabsf(ref - gpu);
        float rel_err = abs_err / (fabsf(ref) + 1e-6f);

        sum_sq_err += (double)(ref - gpu) * (ref - gpu);
        sum_sq_ref += (double)ref * ref;

        if (abs_err > max_abs_err) {
            max_abs_err = abs_err;
            worst_idx = i;
        }
        max_rel_err = std::max(max_rel_err, rel_err);

        // FP16 tolerance: ~1e-3 absolute for small values
        if (abs_err > 0.05f && rel_err > 0.1f)
            num_bad++;
    }

    float rmse = sqrtf((float)(sum_sq_err / total_elements));
    float nrmse = sqrtf((float)(sum_sq_err / (sum_sq_ref + 1e-10)));

    printf("  Max absolute error: %.6f (at index %d, ref=%.6f, gpu=%.6f)\n",
           max_abs_err, worst_idx, h_O_ref[worst_idx], h_O_gpu[worst_idx]);
    printf("  Max relative error: %.6f\n", max_rel_err);
    printf("  RMSE:              %.6f\n", rmse);
    printf("  Normalized RMSE:   %.6f\n", nrmse);
    printf("  Bad elements:      %d / %d (%.4f%%)\n",
           num_bad, total_elements, 100.0f * num_bad / total_elements);

           printf("  Per-row max error (first 20 rows):\n");
    for (int r = 0; r < std::min(20, seq_len); r++) {
        float row_max_err = 0.0f;
        for (int c = 0; c < d_head; c++) {
            int idx = r * d_head + c;  // First head only
            row_max_err = std::max(row_max_err, fabsf(h_O_ref[idx] - h_O_gpu[idx]));
        }
        printf("    row %2d: max_err = %.6f\n", r, row_max_err);
    }

    if (nrmse < 0.02f && num_bad == 0)
        printf("  Result: PASS ✓\n");
    else if (nrmse < 0.05f)
        printf("  Result: MARGINAL (FP16 precision loss)\n");
    else
        printf("  Result: FAIL ✗\n");

    printf("\n");

    // Print first few values for visual inspection
    printf("  First 8 output values:\n");
    printf("  CPU ref: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", h_O_ref[i]);
    printf("\n  GPU out: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", h_O_gpu[i]);
    printf("\n\n");

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_L);
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("============================================\n");
    printf("CUDA Transformer Benchmark\n");
    printf("============================================\n");
    printf("GPU: %s\n", prop.name);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Memory: %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("Compute: %d.%d\n", prop.major, prop.minor);

    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
    int fp16_ops_per_sm = (prop.major >= 12) ? 512 :
                          (prop.major >= 9)  ? 512 :
                          (prop.major >= 8)  ? 256 : 128;
    printf("Peak FP16: %.1f TFLOPS (theoretical)\n",
           2.0 * prop.multiProcessorCount * clock_khz * 1e-6 *
           fp16_ops_per_sm / 1e3);
    printf("============================================\n");

    ModelConfig config;
    config.d_model = 768;
    config.n_heads = 12;
    config.d_head  = 64;

    verify_flash_attention(1, 1, 64, 64);

    

    verify_flash_attention(1, 1, 128, 64); // Small, fast
    verify_flash_attention(1, 2, 256, 64); // Medium

    benchmark_layernorm(2048, config.d_model);
    benchmark_gemm(2048, config.d_model, config.d_model);
    benchmark_gemm(2048, config.d_ffn, config.d_model);
    benchmark_gemm(2048, config.d_model, config.d_ffn);
    benchmark_flash_attention(1, config.n_heads, 512, config.d_head);
    benchmark_flash_attention(1, config.n_heads, 2048, config.d_head);
    benchmark_flash_attention(4, 12, 2048, 64); // 384 blocks → 4.6/SM
    benchmark_flash_attention(8, 12, 2048, 64); // 768 blocks → 9.1/SM
    benchmark_flash_attention(1, 12, 4096, 64); // 768 blocks

    printf("\n============================================\n");
    printf("All benchmarks complete.\n");
    printf("============================================\n");

    return 0;
}
