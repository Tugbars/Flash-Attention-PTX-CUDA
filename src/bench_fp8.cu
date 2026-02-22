// ============================================================================
// FP8 Benchmark — FP16 vs FP8 E4M3 GEMM Comparison
//
// Tests:
//   1. Quantization kernel throughput (FP16 → FP8)
//   2. FP8 GEMM throughput vs FP16 baseline
//   3. Accuracy: max/mean error of FP8 vs FP16 reference
//   4. Full-layer projection (extrapolated speedup)
//
// Build:
//   cmake -DENABLE_FP8=ON --build build --target bench_fp8
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

#include "../include/transformer_config.h"
#include "../include/tensor.h"
#include "../include/gemm_operations.h"
#include "../include/fp8_quantize.h"

using namespace transformer;

// ============================================================================
// GPU Timer
// ============================================================================
struct GpuTimer {
    cudaEvent_t t0, t1;
    GpuTimer()  { cudaEventCreate(&t0); cudaEventCreate(&t1); }
    ~GpuTimer() { cudaEventDestroy(t0); cudaEventDestroy(t1); }
    void start(cudaStream_t s = 0) { cudaEventRecord(t0, s); }
    float stop(cudaStream_t s = 0) {
        cudaEventRecord(t1, s);
        cudaEventSynchronize(t1);
        float ms = 0;
        cudaEventElapsedTime(&ms, t0, t1);
        return ms;
    }
};

template<typename Fn>
static float bench_kernel(Fn&& fn, int warmup = 10, int iters = 50) {
    for (int i = 0; i < warmup; i++) fn();
    cudaDeviceSynchronize();
    GpuTimer t;
    t.start();
    for (int i = 0; i < iters; i++) fn();
    float ms = t.stop();
    return ms / iters;
}

// ============================================================================
// Helpers
// ============================================================================
static void fill_random_fp16(half* d_ptr, size_t n, float range = 0.05f) {
    std::vector<half> h(n);
    for (size_t i = 0; i < n; i++)
        h[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f * range - range);
    cudaMemcpy(d_ptr, h.data(), n * sizeof(half), cudaMemcpyHostToDevice);
}

// Quantize weights on host (offline quantization for static weights)
static void quantize_weights_host(
    const half* d_fp16, __nv_fp8_e4m3* d_fp8, float* h_scale,
    size_t n)
{
    std::vector<half> h_fp16(n);
    cudaMemcpy(h_fp16.data(), d_fp16, n * sizeof(half), cudaMemcpyDeviceToHost);

    // Find absmax
    float amax = 0.0f;
    for (size_t i = 0; i < n; i++)
        amax = std::max(amax, std::abs(__half2float(h_fp16[i])));

    if (amax < 1e-12f) amax = 1.0f;
    float scale = amax / 448.0f;
    float inv_scale = 448.0f / amax;
    *h_scale = scale;

    // Quantize
    std::vector<__nv_fp8_e4m3> h_fp8(n);
    for (size_t i = 0; i < n; i++) {
        float val = __half2float(h_fp16[i]) * inv_scale;
        val = std::min(std::max(val, -448.0f), 448.0f);
        h_fp8[i] = __nv_fp8_e4m3(val);
    }
    cudaMemcpy(d_fp8, h_fp8.data(), n * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
}

// Compute max and mean absolute error between two FP16 tensors
static void compute_error(const half* d_ref, const half* d_test, size_t n,
                          float* max_err, float* mean_err, float* rmse) {
    std::vector<half> h_ref(n), h_test(n);
    cudaMemcpy(h_ref.data(), d_ref, n * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_test.data(), d_test, n * sizeof(half), cudaMemcpyDeviceToHost);

    double sum_err = 0, sum_sq = 0;
    float maxe = 0;
    for (size_t i = 0; i < n; i++) {
        float r = __half2float(h_ref[i]);
        float t = __half2float(h_test[i]);
        float e = std::abs(r - t);
        maxe = std::max(maxe, e);
        sum_err += e;
        sum_sq += (double)e * e;
    }
    *max_err = maxe;
    *mean_err = (float)(sum_err / n);
    *rmse = (float)std::sqrt(sum_sq / n);
}

// ============================================================================
// Main
// ============================================================================
int main() {
    srand(42);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
    // FP16 tensor core peak (256 FP16 FMA per SM per cycle)
    float fp16_peak = 2.0f * prop.multiProcessorCount * (clock_khz * 1e-6f) * 256 / 1e3f;
    // FP8 tensor core peak (512 FP8 FMA per SM per cycle on sm_89+)
    float fp8_peak  = 2.0f * fp16_peak;

    printf("================================================================\n");
    printf("  FP8 E4M3 Quantization Benchmark\n");
    printf("  GPU: %s (%d SMs, CC %d.%d)\n",
           prop.name, prop.multiProcessorCount, prop.major, prop.minor);
    printf("  Theoretical peak FP16: %.1f TFLOPS\n", fp16_peak);
    printf("  Theoretical peak FP8:  %.1f TFLOPS (2x FP16)\n", fp8_peak);
    printf("================================================================\n\n");

    if (prop.major < 8 || (prop.major == 8 && prop.minor < 9)) {
        printf("ERROR: FP8 E4M3 requires sm_89+ (Ada Lovelace or newer)\n");
        return 1;
    }

    // --- LLaMA-7B scale ---
    const int d_model = 4096;
    const int d_ffn   = 11008;
    const int seq_len = 2048;
    const int B       = 1;
    const int N       = B * seq_len;

    printf("  Config: d=%d, d_ffn=%d, B=%d, S=%d, N=%d\n\n", d_model, d_ffn, B, seq_len, N);

    // ====================================================================
    // Allocate buffers
    // ====================================================================

    // FP16 buffers (baseline)
    half *d_A_fp16, *d_W_fp16, *d_C_ref;
    cudaMalloc(&d_A_fp16, (size_t)N * d_model * sizeof(half));
    cudaMalloc(&d_W_fp16, (size_t)d_model * d_model * sizeof(half));
    cudaMalloc(&d_C_ref,  (size_t)N * d_model * sizeof(half));

    // FP8 buffers
    __nv_fp8_e4m3 *d_A_fp8, *d_W_fp8;
    half *d_C_fp8;
    float *d_A_scale;  // 2 floats: [scale, inv_scale]
    cudaMalloc(&d_A_fp8,   (size_t)N * d_model * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_W_fp8,   (size_t)d_model * d_model * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_C_fp8,   (size_t)N * d_model * sizeof(half));
    cudaMalloc(&d_A_scale, 2 * sizeof(float));

    // Large FFN buffers
    half *d_A_ffn_fp16, *d_W_ffn_fp16, *d_C_ffn_ref, *d_C_ffn_fp8;
    __nv_fp8_e4m3 *d_A_ffn_fp8, *d_W_ffn_fp8;
    float *d_A_ffn_scale;
    cudaMalloc(&d_A_ffn_fp16, (size_t)N * d_model * sizeof(half));
    cudaMalloc(&d_W_ffn_fp16, (size_t)d_ffn * d_model * sizeof(half));
    cudaMalloc(&d_C_ffn_ref,  (size_t)N * d_ffn * sizeof(half));
    cudaMalloc(&d_C_ffn_fp8,  (size_t)N * d_ffn * sizeof(half));
    cudaMalloc(&d_A_ffn_fp8,  (size_t)N * d_model * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_W_ffn_fp8,  (size_t)d_ffn * d_model * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_A_ffn_scale, 2 * sizeof(float));

    // Fill with random data
    fill_random_fp16(d_A_fp16, (size_t)N * d_model, 0.05f);
    fill_random_fp16(d_W_fp16, (size_t)d_model * d_model, 0.02f);
    fill_random_fp16(d_A_ffn_fp16, (size_t)N * d_model, 0.05f);
    fill_random_fp16(d_W_ffn_fp16, (size_t)d_ffn * d_model, 0.02f);

    // Pre-quantize weights (offline — this would happen at model load time)
    float w_scale, w_ffn_scale;
    quantize_weights_host(d_W_fp16, d_W_fp8, &w_scale, (size_t)d_model * d_model);
    quantize_weights_host(d_W_ffn_fp16, d_W_ffn_fp8, &w_ffn_scale, (size_t)d_ffn * d_model);

    GemmManager gemm;
    cudaDeviceSynchronize();

    // ====================================================================
    // 1. Quantization kernel benchmark
    // ====================================================================
    printf("--- Quantization Kernel (FP16 → FP8 E4M3) ---\n\n");
    {
        size_t n = (size_t)N * d_model;
        float ms = bench_kernel([&]{
            launch_quantize_fp16_to_fp8(d_A_fp16, d_A_fp8, d_A_scale, n, 0);
        });
        double bytes = n * (sizeof(half) + sizeof(__nv_fp8_e4m3));  // read fp16, write fp8
        double gbps = bytes / (ms * 1e-3) / 1e9;
        printf("  Quantize [%d x %d] (%zu elems):  %.3f ms  (%.0f GB/s)\n",
               N, d_model, n, ms, gbps);

        // Also time just the quantize (no absmax) with prescaled
        float ms_pre = bench_kernel([&]{
            launch_quantize_fp16_to_fp8_prescaled(d_A_fp16, d_A_fp8, 448.0f / 0.05f, n, 0);
        });
        printf("  Prescaled quantize:              %.3f ms  (%.0f GB/s)\n",
               ms_pre, bytes / (ms_pre * 1e-3) / 1e9);
    }

    // ====================================================================
    // 2. GEMM throughput comparison: FP16 vs FP8
    // ====================================================================
    printf("\n--- GEMM Throughput: FP16 vs FP8 ---\n\n");

    // --- Attention projection: [N, d] x [d, d] ---
    printf("  Attention projection [%d x %d x %d]:\n", N, d_model, d_model);
    {
        double flops = 2.0 * N * d_model * d_model;

        // FP16 baseline
        float ms_fp16 = bench_kernel([&]{
            gemm.gemm_nt(d_A_fp16, d_W_fp16, d_C_ref, N, d_model, d_model);
        });
        double tflops_fp16 = flops / (ms_fp16 * 1e-3) / 1e12;

        // FP8: quantize A + GEMM (using dynamic quantize convenience method)
        // First, get activation scale (run quantize once)
        launch_quantize_fp16_to_fp8(d_A_fp16, d_A_fp8, d_A_scale, (size_t)N * d_model, 0);
        cudaDeviceSynchronize();
        float a_scale;
        cudaMemcpy(&a_scale, d_A_scale, sizeof(float), cudaMemcpyDeviceToHost);

        // Bench FP8 GEMM only (pre-quantized A and W)
        float ms_fp8_gemm = bench_kernel([&]{
            gemm.gemm_nt_fp8(d_A_fp8, d_W_fp8, d_C_fp8, N, d_model, d_model,
                             a_scale, w_scale);
        });
        double tflops_fp8 = flops / (ms_fp8_gemm * 1e-3) / 1e12;

        // Bench FP8 end-to-end (quantize A + GEMM)
        float ms_fp8_e2e = bench_kernel([&]{
            gemm.gemm_nt_fp8_dynamic(d_A_fp16, d_W_fp8, d_C_fp8, N, d_model, d_model,
                                      w_scale, d_A_fp8, d_A_scale);
        });
        double tflops_fp8_e2e = flops / (ms_fp8_e2e * 1e-3) / 1e12;

        printf("    FP16:         %.3f ms  %.1f TFLOPS (%.0f%% of FP16 peak)\n",
               ms_fp16, tflops_fp16, 100 * tflops_fp16 / fp16_peak);
        printf("    FP8 GEMM:     %.3f ms  %.1f TFLOPS (%.0f%% of FP8 peak)\n",
               ms_fp8_gemm, tflops_fp8, 100 * tflops_fp8 / fp8_peak);
        printf("    FP8 E2E:      %.3f ms  %.1f TFLOPS (quant+sync+GEMM)\n",
               ms_fp8_e2e, tflops_fp8_e2e);
        printf("    Speedup GEMM: %.2fx    Speedup E2E: %.2fx\n",
               ms_fp16 / ms_fp8_gemm, ms_fp16 / ms_fp8_e2e);

        // Accuracy
        float max_err, mean_err, rmse;
        compute_error(d_C_ref, d_C_fp8, (size_t)N * d_model, &max_err, &mean_err, &rmse);
        printf("    Error: max=%.6f  mean=%.6f  RMSE=%.6f\n", max_err, mean_err, rmse);
    }

    printf("\n");

    // --- FFN up-projection: [N, d] x [d_ffn, d] (larger K) ---
    printf("  FFN gate [%d x %d x %d]:\n", N, d_ffn, d_model);
    {
        double flops = 2.0 * N * d_ffn * d_model;

        float ms_fp16 = bench_kernel([&]{
            gemm.gemm_nt(d_A_ffn_fp16, d_W_ffn_fp16, d_C_ffn_ref, N, d_ffn, d_model);
        });
        double tflops_fp16 = flops / (ms_fp16 * 1e-3) / 1e12;

        // Pre-quantize activations
        launch_quantize_fp16_to_fp8(d_A_ffn_fp16, d_A_ffn_fp8, d_A_ffn_scale,
                                     (size_t)N * d_model, 0);
        cudaDeviceSynchronize();
        float a_ffn_scale;
        cudaMemcpy(&a_ffn_scale, d_A_ffn_scale, sizeof(float), cudaMemcpyDeviceToHost);

        float ms_fp8_gemm = bench_kernel([&]{
            gemm.gemm_nt_fp8(d_A_ffn_fp8, d_W_ffn_fp8, d_C_ffn_fp8, N, d_ffn, d_model,
                             a_ffn_scale, w_ffn_scale);
        });
        double tflops_fp8 = flops / (ms_fp8_gemm * 1e-3) / 1e12;

        float ms_fp8_e2e = bench_kernel([&]{
            gemm.gemm_nt_fp8_dynamic(d_A_ffn_fp16, d_W_ffn_fp8, d_C_ffn_fp8, N, d_ffn, d_model,
                                      w_ffn_scale, d_A_ffn_fp8, d_A_ffn_scale);
        });
        double tflops_fp8_e2e = flops / (ms_fp8_e2e * 1e-3) / 1e12;

        printf("    FP16:         %.3f ms  %.1f TFLOPS (%.0f%% of FP16 peak)\n",
               ms_fp16, tflops_fp16, 100 * tflops_fp16 / fp16_peak);
        printf("    FP8 GEMM:     %.3f ms  %.1f TFLOPS (%.0f%% of FP8 peak)\n",
               ms_fp8_gemm, tflops_fp8, 100 * tflops_fp8 / fp8_peak);
        printf("    FP8 E2E:      %.3f ms  %.1f TFLOPS (quant+sync+GEMM)\n",
               ms_fp8_e2e, tflops_fp8_e2e);
        printf("    Speedup GEMM: %.2fx    Speedup E2E: %.2fx\n",
               ms_fp16 / ms_fp8_gemm, ms_fp16 / ms_fp8_e2e);

        float max_err, mean_err, rmse;
        compute_error(d_C_ffn_ref, d_C_ffn_fp8, (size_t)N * d_ffn, &max_err, &mean_err, &rmse);
        printf("    Error: max=%.6f  mean=%.6f  RMSE=%.6f\n", max_err, mean_err, rmse);
    }

    // ====================================================================
    // 3. Full-layer projection (all GEMMs)
    // ====================================================================
    printf("\n--- Full Layer GEMM Projection ---\n\n");
    printf("  LLaMA-7B: Q,K,V,O (4x [%d,%d,%d]) + gate,up,down (3x [%d,%d,%d])\n",
           N, d_model, d_model, N, d_ffn, d_model);
    {
        // FP16 total: 4 * attn_gemm + 2 * ffn_up + 1 * ffn_down
        float ms_attn_fp16 = bench_kernel([&]{
            gemm.gemm_nt(d_A_fp16, d_W_fp16, d_C_ref, N, d_model, d_model);
        });
        float ms_ffn_fp16 = bench_kernel([&]{
            gemm.gemm_nt(d_A_ffn_fp16, d_W_ffn_fp16, d_C_ffn_ref, N, d_ffn, d_model);
        });
        // Note: FFN down is [N, d_ffn] x [d, d_ffn]^T — similar shape to gate/up
        float total_fp16 = 4 * ms_attn_fp16 + 3 * ms_ffn_fp16;

        // Get pre-quantized scales
        launch_quantize_fp16_to_fp8(d_A_fp16, d_A_fp8, d_A_scale, (size_t)N * d_model, 0);
        cudaDeviceSynchronize();
        float a_scale;
        cudaMemcpy(&a_scale, d_A_scale, sizeof(float), cudaMemcpyDeviceToHost);
        launch_quantize_fp16_to_fp8(d_A_ffn_fp16, d_A_ffn_fp8, d_A_ffn_scale,
                                     (size_t)N * d_model, 0);
        cudaDeviceSynchronize();
        float a_ffn_scale;
        cudaMemcpy(&a_ffn_scale, d_A_ffn_scale, sizeof(float), cudaMemcpyDeviceToHost);

        float ms_attn_fp8 = bench_kernel([&]{
            gemm.gemm_nt_fp8(d_A_fp8, d_W_fp8, d_C_fp8, N, d_model, d_model,
                             a_scale, w_scale);
        });
        float ms_ffn_fp8 = bench_kernel([&]{
            gemm.gemm_nt_fp8(d_A_ffn_fp8, d_W_ffn_fp8, d_C_ffn_fp8, N, d_ffn, d_model,
                             a_ffn_scale, w_ffn_scale);
        });
        float total_fp8 = 4 * ms_attn_fp8 + 3 * ms_ffn_fp8;

        // Quantize overhead (4 quant ops for attn, 3 for FFN activations)
        float ms_quant_attn = bench_kernel([&]{
            launch_quantize_fp16_to_fp8(d_A_fp16, d_A_fp8, d_A_scale,
                                         (size_t)N * d_model, 0);
        });
        float ms_quant_ffn = bench_kernel([&]{
            launch_quantize_fp16_to_fp8(d_A_ffn_fp16, d_A_ffn_fp8, d_A_ffn_scale,
                                         (size_t)N * d_model, 0);
        });
        float total_quant = 4 * ms_quant_attn + 3 * ms_quant_ffn;

        printf("  FP16 GEMM total:     %.3f ms  (4×%.3f + 3×%.3f)\n",
               total_fp16, ms_attn_fp16, ms_ffn_fp16);
        printf("  FP8 GEMM total:      %.3f ms  (4×%.3f + 3×%.3f)\n",
               total_fp8, ms_attn_fp8, ms_ffn_fp8);
        printf("  Quant overhead:      %.3f ms  (4×%.3f + 3×%.3f)\n",
               total_quant, ms_quant_attn, ms_quant_ffn);
        printf("  FP8 total (w/quant): %.3f ms\n", total_fp8 + total_quant);
        printf("\n");
        printf("  GEMM-only speedup:   %.2fx\n", total_fp16 / total_fp8);
        printf("  E2E speedup (w/qnt): %.2fx\n", total_fp16 / (total_fp8 + total_quant));

        // Full layer estimate (add non-GEMM from baseline)
        float non_gemm_ms = 0.53f + 0.28f + 0.05f;  // attn + bw-bound + launch overhead
        float layer_fp16 = total_fp16 + non_gemm_ms;
        float layer_fp8  = total_fp8 + total_quant + non_gemm_ms;
        printf("\n  Full layer estimate (GEMM + attn + bw-kernels):\n");
        printf("    FP16: %.3f ms (%.3f GEMM + %.2f other)\n",
               layer_fp16, total_fp16, non_gemm_ms);
        printf("    FP8:  %.3f ms (%.3f GEMM + %.3f quant + %.2f other)\n",
               layer_fp8, total_fp8, total_quant, non_gemm_ms);
        printf("    Layer speedup: %.2fx\n", layer_fp16 / layer_fp8);

        // FLOP accounting
        double total_flops = 4 * 2.0 * N * d_model * d_model +
                             3 * 2.0 * N * d_ffn * d_model;
        printf("\n  Total GFLOP:   %.1f\n", total_flops / 1e9);
        printf("  FP16 eff:      %.1f TFLOPS\n", total_flops / (total_fp16 * 1e-3) / 1e12);
        printf("  FP8 eff:       %.1f TFLOPS\n", total_flops / (total_fp8 * 1e-3) / 1e12);
    }

    // ====================================================================
    // 4. Memory savings
    // ====================================================================
    printf("\n--- Memory ---\n\n");
    {
        size_t fp16_weights = (4ULL * d_model * d_model + 3ULL * d_ffn * d_model) * 2;
        size_t fp8_weights  = (4ULL * d_model * d_model + 3ULL * d_ffn * d_model) * 1;
        size_t fp8_scales   = 7 * sizeof(float);  // 7 weight matrices, 1 scale each
        printf("  Weight memory per layer:\n");
        printf("    FP16: %.1f MB\n", fp16_weights / 1e6);
        printf("    FP8:  %.1f MB (%.1fx smaller)\n",
               (fp8_weights + fp8_scales) / 1e6,
               (float)fp16_weights / (fp8_weights + fp8_scales));
        printf("  Full model (32 layers):\n");
        printf("    FP16: %.0f MB\n", fp16_weights * 32.0 / 1e6);
        printf("    FP8:  %.0f MB\n", (fp8_weights + fp8_scales) * 32.0 / 1e6);
    }

    printf("\n================================================================\n");
    printf("  Done.\n");
    printf("================================================================\n");

    // Cleanup
    cudaFree(d_A_fp16); cudaFree(d_W_fp16); cudaFree(d_C_ref);
    cudaFree(d_A_fp8); cudaFree(d_W_fp8); cudaFree(d_C_fp8); cudaFree(d_A_scale);
    cudaFree(d_A_ffn_fp16); cudaFree(d_W_ffn_fp16);
    cudaFree(d_C_ffn_ref); cudaFree(d_C_ffn_fp8);
    cudaFree(d_A_ffn_fp8); cudaFree(d_W_ffn_fp8); cudaFree(d_A_ffn_scale);

    return 0;
}
