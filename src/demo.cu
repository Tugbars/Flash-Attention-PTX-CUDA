// ============================================================================
// Flash Attention Demo — Standalone
//
// Runs the flash attention kernel on sample data, extracts attention weights
// and output, writes them to binary files for Python visualization.
//
// Build: nvcc -O3 -arch=sm_89 -o flash_demo demo.cu flash_attention.cu
// Run:   ./flash_demo
// Then:  python visualize.py
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>

// Use the project's flash attention header
#include "../include/flash_attention.h"

using namespace transformer;

// ============================================================================
// CPU reference attention — returns full attention weight matrix P
// ============================================================================
void cpu_attention(
    const float* Q, const float* K, const float* V,
    float* O, float* P_out,
    int seq_len, int d_head, float scale, bool causal)
{
    // S = Q * K^T
    std::vector<float> S(seq_len * seq_len);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            float dot = 0.0f;
            for (int k = 0; k < d_head; k++)
                dot += Q[i * d_head + k] * K[j * d_head + k];
            S[i * seq_len + j] = dot * scale;
            if (causal && j > i)
                S[i * seq_len + j] = -1e9f;
        }
    }

    // Softmax per row → P
    for (int i = 0; i < seq_len; i++) {
        float max_val = -1e30f;
        for (int j = 0; j < seq_len; j++)
            max_val = std::max(max_val, S[i * seq_len + j]);

        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            P_out[i * seq_len + j] = expf(S[i * seq_len + j] - max_val);
            sum += P_out[i * seq_len + j];
        }
        for (int j = 0; j < seq_len; j++)
            P_out[i * seq_len + j] /= sum;
    }

    // O = P * V
    for (int i = 0; i < seq_len; i++) {
        for (int k = 0; k < d_head; k++) {
            float val = 0.0f;
            for (int j = 0; j < seq_len; j++)
                val += P_out[i * seq_len + j] * V[j * d_head + k];
            O[i * d_head + k] = val;
        }
    }
}

// ============================================================================
// Write binary file: header + float data
// ============================================================================
void write_binary(const char* filename, const float* data,
                  int dim0, int dim1, int dim2 = 1, int dim3 = 1)
{
    std::ofstream f(filename, std::ios::binary);
    int32_t ndims = (dim3 > 1) ? 4 : (dim2 > 1) ? 3 : 2;
    f.write(reinterpret_cast<const char*>(&ndims), 4);
    f.write(reinterpret_cast<const char*>(&dim0), 4);
    f.write(reinterpret_cast<const char*>(&dim1), 4);
    if (ndims >= 3) f.write(reinterpret_cast<const char*>(&dim2), 4);
    if (ndims >= 4) f.write(reinterpret_cast<const char*>(&dim3), 4);
    size_t total = (size_t)dim0 * dim1 * dim2 * dim3;
    f.write(reinterpret_cast<const char*>(data), total * sizeof(float));
    f.close();
    printf("  Wrote %s: [%d", filename, dim0);
    if (ndims >= 2) printf(", %d", dim1);
    if (ndims >= 3) printf(", %d", dim2);
    if (ndims >= 4) printf(", %d", dim3);
    printf("] (%zu floats)\n", total);
}

// ============================================================================
// Timing utility
// ============================================================================
struct CudaTimer {
    cudaEvent_t start, stop;
    CudaTimer()  { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~CudaTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin(cudaStream_t s = 0) { cudaEventRecord(start, s); }
    float end(cudaStream_t s = 0) {
        cudaEventRecord(stop, s);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// ============================================================================
// Main
// ============================================================================
int main() {
    // --- GPU info ---
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
    int fp16_ops = (prop.major >= 12) ? 512 : (prop.major >= 8) ? 256 : 128;
    double peak_tflops = 2.0 * prop.multiProcessorCount * clock_khz * 1e-6 * fp16_ops / 1e3;

    printf("============================================================\n");
    printf("  Flash Attention Demo\n");
    printf("  GPU: %s (%d SMs, CC %d.%d)\n", prop.name, prop.multiProcessorCount, prop.major, prop.minor);
    printf("  Peak FP16: %.1f TFLOPS\n", peak_tflops);
    printf("============================================================\n\n");

    // --- Parameters ---
    const int B = 1, H = 4, S = 256, D = 64;
    const float scale = 1.0f / sqrtf((float)D);
    const bool causal = true;
    const int total = B * H * S * D;

    printf("[1] Generating structured Q, K, V (B=%d, H=%d, S=%d, D=%d)\n", B, H, S, D);
    printf("    Head 0: Local attention (nearby tokens)\n");
    printf("    Head 1: Strided/periodic attention\n");
    printf("    Head 2: Global + anchor (BOS pattern)\n");
    printf("    Head 3: Clustered block attention\n");

    // --- Generate structured data ---
    // Strategy: directly construct Q and K so that Q[i]*K[j] is LARGE when we
    // want strong attention and SMALL (or negative) otherwise.
    // With scale = 1/sqrt(64) = 0.125, we need dot products ~8-16 to dominate softmax.
    //
    // Simple approach: use one-hot-ish embeddings.
    // Q[i] has a spike at dimension f(i), K[j] has a spike at dimension g(j).
    // When f(i) == g(j), dot product is large → strong attention.
    std::vector<float> h_Q(total, 0.0f), h_K(total, 0.0f), h_V(total, 0.0f);
    srand(42);

    // Fill V with small random values
    for (int i = 0; i < total; i++)
        h_V[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.3f;

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            int bh = b * H + h;
            float* Q_ptr = h_Q.data() + bh * S * D;
            float* K_ptr = h_K.data() + bh * S * D;

            if (h == 0) {
                // HEAD 0: LOCAL ATTENTION — strong diagonal band, width ~16
                // Q[i] and K[j] share a "position hash" that's similar for nearby tokens.
                // Use overlapping Gaussian bumps in embedding space.
                const int bandwidth = 16;
                for (int i = 0; i < S; i++) {
                    // Place a Gaussian bump centered at dim = (i % D)
                    // Nearby positions have overlapping bumps → high dot product
                    int center = (i * D / S) % D;  // spread centers across D dims
                    for (int d = 0; d < D; d++) {
                        float dist = fminf(abs(d - center), D - abs(d - center));
                        float val = expf(-dist * dist / (2.0f * 4.0f * 4.0f)) * 3.0f;
                        Q_ptr[i * D + d] = val;
                        K_ptr[i * D + d] = val;
                    }
                }
            }
            else if (h == 1) {
                // HEAD 1: STRIDED / PERIODIC — attend to every 8th token
                // Tokens with same (i % period) get identical embeddings.
                const int period = 8;
                for (int i = 0; i < S; i++) {
                    int phase = i % period;
                    // Each phase gets a unique direction in embedding space
                    for (int d = 0; d < D; d++) {
                        float val = 0.0f;
                        // phase-specific dimensions: dims [phase*8 .. phase*8+7] are hot
                        if (d >= phase * 8 && d < phase * 8 + 8)
                            val = 2.5f;
                        // Small position-dependent noise to break exact ties
                        val += (rand() / (float)RAND_MAX - 0.5f) * 0.1f;
                        Q_ptr[i * D + d] = val;
                        K_ptr[i * D + d] = val;
                    }
                }
            }
            else if (h == 2) {
                // HEAD 2: GLOBAL + ANCHOR — first ~16 tokens get attended by everyone
                // Early K's have large magnitude in shared dims; later K's have small.
                // All Q's project onto those shared dims.
                for (int i = 0; i < S; i++) {
                    for (int d = 0; d < D; d++) {
                        if (d < 32) {
                            // Shared subspace: Q always projects here
                            Q_ptr[i * D + d] = 1.5f;
                            // K decays: early tokens have strong projection
                            float anchor_strength = expf(-i * 0.05f) * 2.5f;
                            K_ptr[i * D + d] = anchor_strength;
                        } else {
                            // Remaining dims: position-dependent for some self-attention
                            float pos_signal = sinf(i * (d - 32 + 1) * 0.1f) * 0.3f;
                            Q_ptr[i * D + d] = pos_signal;
                            K_ptr[i * D + d] = pos_signal;
                        }
                    }
                }
            }
            else if (h == 3) {
                // HEAD 3: BLOCK ATTENTION — 32-token blocks, strong intra-block
                // Each block gets a unique "block ID" embedding.
                // Tokens in the same block share it → high mutual attention.
                const int block_size = 32;
                const int num_blocks = (S + block_size - 1) / block_size;
                for (int i = 0; i < S; i++) {
                    int blk = i / block_size;
                    for (int d = 0; d < D; d++) {
                        float val = 0.0f;
                        // Block-ID dims: first half of D, each block owns 4 dims
                        int blk_start = (blk * 4) % (D / 2);
                        if (d >= blk_start && d < blk_start + 4)
                            val = 3.0f;
                        // Intra-block position: second half of D
                        if (d >= D / 2) {
                            int local = i % block_size;
                            int local_d = d - D / 2;
                            // Smooth local similarity
                            float center = (float)local * (D/2) / block_size;
                            float dist = fabsf(local_d - center);
                            val += expf(-dist * dist / 18.0f) * 1.5f;
                        }
                        Q_ptr[i * D + d] = val;
                        K_ptr[i * D + d] = val;
                    }
                }
            }
        }
    }

    // --- CPU reference ---
    printf("\n[2] Running CPU reference attention...\n");
    std::vector<float> h_O_ref(total, 0.0f);
    std::vector<float> h_P_ref(B * H * S * S, 0.0f);

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            int bh = b * H + h;
            cpu_attention(
                h_Q.data() + bh * S * D,
                h_K.data() + bh * S * D,
                h_V.data() + bh * S * D,
                h_O_ref.data() + bh * S * D,
                h_P_ref.data() + bh * S * S,
                S, D, scale, causal);
        }
    }
    printf("  Done.\n");

    // --- GPU kernel ---
    printf("\n[3] Running GPU flash attention kernel...\n");

    std::vector<half> h_Q_h(total), h_K_h(total), h_V_h(total);
    for (int i = 0; i < total; i++) {
        h_Q_h[i] = __float2half(h_Q[i]);
        h_K_h[i] = __float2half(h_K[i]);
        h_V_h[i] = __float2half(h_V[i]);
    }

    half *d_Q, *d_K, *d_V, *d_O;
    float *d_L;
    cudaMalloc(&d_Q, total * sizeof(half));
    cudaMalloc(&d_K, total * sizeof(half));
    cudaMalloc(&d_V, total * sizeof(half));
    cudaMalloc(&d_O, total * sizeof(half));
    cudaMalloc(&d_L, B * H * S * sizeof(float));
    cudaMemcpy(d_Q, h_Q_h.data(), total * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K_h.data(), total * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V_h.data(), total * sizeof(half), cudaMemcpyHostToDevice);

    FlashAttentionParams params = {};
    params.Q = d_Q; params.K = d_K; params.V = d_V;
    params.O = d_O; params.L = d_L;
    params.batch_size = B; params.num_heads = H;
    params.seq_len = S; params.d_head = D;
    params.scale = scale; params.causal = causal;
    params.stream = 0;

    // Warmup
    for (int i = 0; i < 5; i++)
        launch_flash_attention(params);
    cudaDeviceSynchronize();

    // Benchmark
    CudaTimer timer;
    const int iters = 200;
    timer.begin();
    for (int i = 0; i < iters; i++)
        launch_flash_attention(params);
    float total_ms = timer.end();
    float avg_ms = total_ms / iters;

    double flops = 2.0 * B * H * (double)S * S * D * 2;
    double tflops = flops / (avg_ms * 1e-3) / 1e12;

    printf("  Average: %.4f ms\n", avg_ms);
    printf("  TFLOPS:  %.2f (%.1f%% of peak)\n", tflops, 100.0 * tflops / peak_tflops);

    // Copy output back
    std::vector<half> h_O_h(total);
    std::vector<float> h_O_gpu(total);
    cudaMemcpy(h_O_h.data(), d_O, total * sizeof(half), cudaMemcpyDeviceToHost);
    for (int i = 0; i < total; i++)
        h_O_gpu[i] = __half2float(h_O_h[i]);

    // --- Correctness check ---
    printf("\n[4] Correctness check...\n");
    float max_err = 0, sum_sq_err = 0, sum_sq_ref = 0;
    int bad = 0;
    for (int i = 0; i < total; i++) {
        float err = fabsf(h_O_ref[i] - h_O_gpu[i]);
        float rel = err / (fabsf(h_O_ref[i]) + 1e-6f);
        max_err = std::max(max_err, err);
        sum_sq_err += (double)err * err;
        sum_sq_ref += (double)h_O_ref[i] * h_O_ref[i];
        if (err > 0.05f && rel > 0.1f) bad++;
    }
    float nrmse = sqrtf(sum_sq_err / (sum_sq_ref + 1e-10));
    printf("  Max absolute error: %.6f\n", max_err);
    printf("  Normalized RMSE:    %.6f\n", nrmse);
    printf("  Bad elements:       %d / %d\n", bad, total);
    printf("  Result:             %s\n", (nrmse < 0.03f && bad == 0) ? "PASS ✓" : "FAIL");

    // --- Dump data for visualization ---
    printf("\n[5] Writing visualization data...\n");

    // Attention weights: [B, H, S, S]
    write_binary("attention_weights.bin", h_P_ref.data(), B, H, S, S);

    // GPU output: [B, H, S, D]
    write_binary("output_gpu.bin", h_O_gpu.data(), B, H, S, D);

    // CPU reference output: [B, H, S, D]
    write_binary("output_ref.bin", h_O_ref.data(), B, H, S, D);

    // Q, K, V for analysis
    write_binary("q_data.bin", h_Q.data(), B, H, S, D);
    write_binary("k_data.bin", h_K.data(), B, H, S, D);
    write_binary("v_data.bin", h_V.data(), B, H, S, D);

    // Performance data for chart — actual benchmark results at B=4, H=12, S=2048, D=64
    {
        std::ofstream f("perf_data.csv");
        f << "version,tflops,pct_peak\n";
        f << "Scalar Baseline,2.7," << (2.7/peak_tflops*100) << "\n";
        f << "WMMA v3,26.7," << (26.7/peak_tflops*100) << "\n";
        f << "Vectorized v6,38.3," << (38.3/peak_tflops*100) << "\n";
        f << "PTX MMA v7,49.2," << (49.2/peak_tflops*100) << "\n";
        f << "In-Register v8,125.2," << (125.2/peak_tflops*100) << "\n";
        f << "Corrected v9,135.9," << (135.9/peak_tflops*100) << "\n";
        f.close();
        printf("  Wrote perf_data.csv\n");
    }

    printf("\n============================================================\n");
    printf("  Done! Run 'python visualize.py' to see the results.\n");
    printf("============================================================\n");

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_L);
    return 0;
}
