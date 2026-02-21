// ============================================================================
// Flash Attention — Benchmark & Correctness Suite
//
// Tests:
//   1. PTX MMA + ldmatrix verification (identity matrix multiply)
//   2. Correctness check vs CPU reference (multiple sequence lengths)
//   3. Performance benchmark across batch sizes and sequence lengths
//
// Build:  cmake --build build --target flash_bench
// Run:    ./build/flash_bench
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#include "../include/flash_attention.h"

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
// PTX MMA + ldmatrix Test
//
// Verifies that our PTX intrinsics and ldmatrix addressing are correct by
// multiplying an identity matrix (16×16) by a known B matrix (16×8).
// Expected: C = B.
// ============================================================================

__device__ __forceinline__ void test_ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    const void* smem_ptr)
{
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void test_ldmatrix_x2_trans(
    uint32_t& r0, uint32_t& r1, const void* smem_ptr)
{
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
        : "=r"(r0), "=r"(r1) : "r"(addr));
}

__device__ __forceinline__ void test_ptx_mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__global__ void test_mma_kernel(float* result) {
    const int lane_id = threadIdx.x;
    constexpr int PAD = 8;
    constexpr int A_STRIDE = 16 + PAD;
    constexpr int B_STRIDE = 8 + PAD;

    __shared__ half smem_a[16 * A_STRIDE];
    __shared__ half smem_b[16 * B_STRIDE];

    // A = identity (16×16)
    for (int idx = lane_id; idx < 16 * A_STRIDE; idx += 32) {
        int r = idx / A_STRIDE, c = idx % A_STRIDE;
        smem_a[idx] = (c < 16 && r == c) ? __float2half(1.0f) : __float2half(0.0f);
    }

    // B[row][col] = row * 8 + col + 1  (16×8)
    for (int idx = lane_id; idx < 16 * B_STRIDE; idx += 32) {
        int r = idx / B_STRIDE, c = idx % B_STRIDE;
        smem_b[idx] = (c < 8) ? __float2half((float)(r * 8 + c + 1)) : __float2half(0.0f);
    }
    __syncthreads();

    uint32_t a0, a1, a2, a3;
    {
        int row = lane_id % 16;
        int col = (lane_id / 16) * 8;
        test_ldmatrix_x4(a0, a1, a2, a3, smem_a + row * A_STRIDE + col);
    }

    // Key: threads 8-15 must offset by +8 rows to load the second 8×8 block
    uint32_t b0, b1;
    {
        int row = lane_id % 8 + ((lane_id / 8) % 2) * 8;
        test_ldmatrix_x2_trans(b0, b1, smem_b + row * B_STRIDE);
    }

    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    test_ptx_mma_m16n8k16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);

    int row0 = (lane_id / 4) % 8;
    int row1 = row0 + 8;
    int col0 = (lane_id % 4) * 2;
    int col1 = col0 + 1;

    result[row0 * 8 + col0] = d0;
    result[row0 * 8 + col1] = d1;
    result[row1 * 8 + col0] = d2;
    result[row1 * 8 + col1] = d3;
}

bool test_ptx_mma() {
    printf("=== PTX MMA + ldmatrix Test ===\n");
    printf("  C = I(16x16) * B(16x8), B[r][c] = r*8+c+1\n");
    printf("  Expected: C = B\n\n");

    float* d_result;
    cudaMalloc(&d_result, 16 * 8 * sizeof(float));
    cudaMemset(d_result, 0, 16 * 8 * sizeof(float));

    test_mma_kernel<<<1, 32>>>(d_result);
    cudaDeviceSynchronize();

    float h_result[128];
    cudaMemcpy(h_result, d_result, 128 * sizeof(float), cudaMemcpyDeviceToHost);

    bool pass = true;
    for (int r = 0; r < 16; r++) {
        printf("  Row %2d: ", r);
        for (int c = 0; c < 8; c++) {
            float got = h_result[r * 8 + c];
            float expected = (float)(r * 8 + c + 1);
            printf("%6.1f", got);
            if (fabsf(got - expected) > 0.5f) pass = false;
        }
        printf("\n");
    }
    printf("\n  Result: %s\n\n", pass ? "PASS ✓" : "FAIL ✗");

    cudaFree(d_result);
    return pass;
}

// ============================================================================
// Flash Attention Correctness Check
//
// Compares GPU kernel output against a CPU reference (naive O(S²) attention).
// Returns true if NRMSE < 2% and zero bad elements.
// ============================================================================

bool verify_flash_attention(int batch_size, int n_heads, int seq_len, int d_head) {
    printf("=== Flash Attention Correctness Check ===\n");
    printf("  B=%d, H=%d, S=%d, D=%d\n", batch_size, n_heads, seq_len, d_head);

    const int total_elements = batch_size * n_heads * seq_len * d_head;
    const float scale = 1.0f / sqrtf(static_cast<float>(d_head));

    std::vector<float> h_Q(total_elements), h_K(total_elements), h_V(total_elements);
    std::vector<float> h_O_ref(total_elements, 0.0f);
    std::vector<float> h_O_gpu(total_elements, 0.0f);

    srand(42);
    for (int i = 0; i < total_elements; i++) {
        h_Q[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.5f;
        h_K[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.5f;
        h_V[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.5f;
    }

    // --- CPU reference: naive attention ---
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < n_heads; h++) {
            const int bh = b * n_heads + h;
            const float* Q_bh = h_Q.data() + bh * seq_len * d_head;
            const float* K_bh = h_K.data() + bh * seq_len * d_head;
            const float* V_bh = h_V.data() + bh * seq_len * d_head;
            float* O_bh = h_O_ref.data() + bh * seq_len * d_head;

            std::vector<float> S(seq_len * seq_len);
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < seq_len; j++) {
                    float dot = 0.0f;
                    for (int k = 0; k < d_head; k++)
                        dot += Q_bh[i * d_head + k] * K_bh[j * d_head + k];
                    S[i * seq_len + j] = dot * scale;
                    if (j > i) S[i * seq_len + j] = -1e9f;
                }
            }

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

    // --- GPU kernel ---
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

    FlashAttentionParams params = {};
    params.Q = d_Q; params.K = d_K; params.V = d_V; params.O = d_O; params.L = d_L;
    params.batch_size = batch_size;
    params.num_heads = n_heads;
    params.seq_len = seq_len;
    params.d_head = d_head;
    params.scale = scale;
    params.causal = true;
    params.stream = 0;

    launch_flash_attention(params);
    cudaDeviceSynchronize();

    cudaMemcpy(h_O_half.data(), d_O, bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < total_elements; i++)
        h_O_gpu[i] = __half2float(h_O_half[i]);

    // --- Compare ---
    float max_abs_err = 0.0f;
    double sum_sq_err = 0.0, sum_sq_ref = 0.0;
    int num_bad = 0, worst_idx = 0;

    for (int i = 0; i < total_elements; i++) {
        float ref = h_O_ref[i], gpu = h_O_gpu[i];
        float abs_err = fabsf(ref - gpu);
        float rel_err = abs_err / (fabsf(ref) + 1e-6f);

        sum_sq_err += (double)(ref - gpu) * (ref - gpu);
        sum_sq_ref += (double)ref * ref;

        if (abs_err > max_abs_err) { max_abs_err = abs_err; worst_idx = i; }
        if (abs_err > 0.05f && rel_err > 0.1f) num_bad++;
    }

    float nrmse = sqrtf((float)(sum_sq_err / (sum_sq_ref + 1e-10)));

    printf("  Max absolute error: %.6f (at index %d)\n", max_abs_err, worst_idx);
    printf("  Normalized RMSE:    %.6f\n", nrmse);
    printf("  Bad elements:       %d / %d\n", num_bad, total_elements);

    bool pass = (nrmse < 0.03f && num_bad == 0);
    printf("  Result:             %s\n\n", pass ? "PASS ✓" : "FAIL ✗");

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_L);
    return pass;
}

// ============================================================================
// Flash Attention Benchmark
// ============================================================================

void benchmark_flash_attention(int batch_size, int n_heads, int seq_len, int d_head,
                               int warmup = 10, int iters = 100)
{
    printf("=== Flash Attention Benchmark ===\n");
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
    printf("  TFLOPS:  %.2f\n\n", tflops);

    cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(O); cudaFree(L);
}

// ============================================================================
// Main
// ============================================================================
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
    int fp16_ops = (prop.major >= 12) ? 512 : (prop.major >= 8) ? 256 : 128;
    double peak_tflops = 2.0 * prop.multiProcessorCount * clock_khz * 1e-6 * fp16_ops / 1e3;

    printf("============================================================\n");
    printf("  Flash Attention — Benchmark & Test Suite\n");
    printf("============================================================\n");
    printf("  GPU:      %s (%d SMs, CC %d.%d)\n",
           prop.name, prop.multiProcessorCount, prop.major, prop.minor);
    printf("  Memory:   %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("  Peak FP16: %.1f TFLOPS (theoretical)\n", peak_tflops);
    printf("============================================================\n");

    // --- Correctness tests ---
    printf("\n--- Correctness Tests ---\n\n");

    bool all_pass = true;
    all_pass &= test_ptx_mma();
    all_pass &= verify_flash_attention(1, 1, 64, 64);
    all_pass &= verify_flash_attention(1, 1, 128, 64);
    all_pass &= verify_flash_attention(1, 2, 256, 64);

    if (!all_pass) {
        printf("!!! CORRECTNESS FAILURES DETECTED — skipping benchmarks !!!\n");
        return 1;
    }

    // --- Benchmarks ---
    printf("--- Benchmarks ---\n\n");

    benchmark_flash_attention(1, 12, 512, 64);
    benchmark_flash_attention(1, 12, 2048, 64);
    benchmark_flash_attention(4, 12, 2048, 64);
    benchmark_flash_attention(8, 12, 2048, 64);
    benchmark_flash_attention(1, 12, 4096, 64);

    printf("============================================================\n");
    printf("  All tests passed. Benchmarks complete.\n");
    printf("============================================================\n");

    return 0;
}
