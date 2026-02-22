// ============================================================================
// Transformer Benchmark + Accuracy Tests
//
// Modules tested for correctness (CPU reference vs GPU):
//   ✓ PTX MMA (identity multiply)
//   ✓ Flash Attention (causal, multi-head)
//   ✓ GEMM NT (via cuBLAS/CUTLASS)
//   ✓ LayerNorm (standard + RMSNorm, with/without residual)
//   ✓ RoPE (rotary position embedding)
//   ✓ SwiGLU activation
//   ✓ GELU activation
//   ✓ SiLU activation (in-place)
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

#include "../include/transformer_config.h"
#include "../include/tensor.h"
#include "../include/flash_attention.h"
#include "../include/layer_norm.h"
#include "../include/rotary_embedding.h"
#include "../include/activation_kernels.h"
#include "../include/gemm_operations.h"

using namespace transformer;

// ============================================================================
// Utilities
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

// Deterministic RNG for reproducible tests
static float rand_float(unsigned& seed, float lo = -0.5f, float hi = 0.5f) {
    seed = seed * 1103515245u + 12345u;
    float t = ((seed >> 16) & 0x7FFF) / 32767.0f;
    return lo + t * (hi - lo);
}

// Upload float array to device as FP16
static half* upload_fp16(const std::vector<float>& host, size_t n) {
    std::vector<half> tmp(n);
    for (size_t i = 0; i < n; i++) tmp[i] = __float2half(host[i]);
    half* d;
    cudaMalloc(&d, n * sizeof(half));
    cudaMemcpy(d, tmp.data(), n * sizeof(half), cudaMemcpyHostToDevice);
    return d;
}

// Download device FP16 to host float
static void download_fp16(half* d, std::vector<float>& host, size_t n) {
    std::vector<half> tmp(n);
    cudaMemcpy(tmp.data(), d, n * sizeof(half), cudaMemcpyDeviceToHost);
    host.resize(n);
    for (size_t i = 0; i < n; i++) host[i] = __half2float(tmp[i]);
}

// Compare two float arrays, return pass/fail and print stats
struct CompareResult {
    float max_abs_err;
    float max_rel_err;
    float rmse;
    float nrmse;
    int   num_bad;
    int   total;
    bool  pass;
};

static CompareResult compare_arrays(const float* ref, const float* gpu, int n,
                                     float abs_tol = 0.05f, float rel_tol = 0.1f,
                                     float nrmse_pass = 0.02f) {
    CompareResult r = {};
    r.total = n;
    double sum_sq_err = 0, sum_sq_ref = 0;

    for (int i = 0; i < n; i++) {
        float ae = fabsf(ref[i] - gpu[i]);
        float re = ae / (fabsf(ref[i]) + 1e-6f);
        sum_sq_err += (double)(ref[i] - gpu[i]) * (ref[i] - gpu[i]);
        sum_sq_ref += (double)ref[i] * ref[i];

        if (ae > r.max_abs_err) r.max_abs_err = ae;
        if (re > r.max_rel_err) r.max_rel_err = re;
        if (ae > abs_tol && re > rel_tol) r.num_bad++;
    }

    r.rmse  = sqrtf((float)(sum_sq_err / n));
    r.nrmse = sqrtf((float)(sum_sq_err / (sum_sq_ref + 1e-10)));
    r.pass  = (r.nrmse < nrmse_pass && r.num_bad == 0);
    return r;
}

static void print_result(const char* name, const CompareResult& r) {
    printf("  Max abs err:  %.6f\n", r.max_abs_err);
    printf("  Max rel err:  %.6f\n", r.max_rel_err);
    printf("  RMSE:         %.6f\n", r.rmse);
    printf("  NRMSE:        %.6f\n", r.nrmse);
    printf("  Bad elements: %d / %d\n", r.num_bad, r.total);
    printf("  Result: %s %s\n\n", name, r.pass ? "PASS" : "FAIL");
}

// ============================================================================
// 1. PTX MMA Test (existing)
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

    for (int idx = lane_id; idx < 16 * A_STRIDE; idx += 32) {
        int r = idx / A_STRIDE, c = idx % A_STRIDE;
        smem_a[idx] = (c < 16 && r == c) ? __float2half(1.0f) : __float2half(0.0f);
    }
    for (int idx = lane_id; idx < 16 * B_STRIDE; idx += 32) {
        int r = idx / B_STRIDE, c = idx % B_STRIDE;
        smem_b[idx] = (c < 8) ? __float2half((float)(r * 8 + c + 1)) : __float2half(0.0f);
    }
    __syncthreads();

    uint32_t a0, a1, a2, a3;
    { int row = lane_id % 16; int col = (lane_id / 16) * 8;
      test_ldmatrix_x4(a0, a1, a2, a3, smem_a + row * A_STRIDE + col); }

    uint32_t b0, b1;
    { int row = lane_id % 8 + ((lane_id / 8) % 2) * 8;
      test_ldmatrix_x2_trans(b0, b1, smem_b + row * B_STRIDE); }

    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    test_ptx_mma_m16n8k16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);

    int row0 = (lane_id / 4) % 8, row1 = row0 + 8;
    int col0 = (lane_id % 4) * 2, col1 = col0 + 1;
    result[row0 * 8 + col0] = d0;
    result[row0 * 8 + col1] = d1;
    result[row1 * 8 + col0] = d2;
    result[row1 * 8 + col1] = d3;
}

bool test_ptx_mma() {
    printf("=== PTX MMA + ldmatrix Test ===\n");
    printf("  C = I(16x16) * B(16x8), B[r][c] = r*8+c+1\n");

    float* d_result;
    cudaMalloc(&d_result, 16 * 8 * sizeof(float));
    cudaMemset(d_result, 0, 16 * 8 * sizeof(float));
    test_mma_kernel<<<1, 32>>>(d_result);
    cudaDeviceSynchronize();

    float h_result[128];
    cudaMemcpy(h_result, d_result, 128 * sizeof(float), cudaMemcpyDeviceToHost);

    bool pass = true;
    for (int r = 0; r < 16; r++) {
        for (int c = 0; c < 8; c++) {
            float expected = (float)(r * 8 + c + 1);
            if (fabsf(h_result[r * 8 + c] - expected) > 0.5f) pass = false;
        }
    }
    printf("  Result: %s\n\n", pass ? "PASS" : "FAIL");
    cudaFree(d_result);
    return pass;
}

// ============================================================================
// 2. GEMM NT Accuracy Test
//    CPU ref: C[i][j] = sum_k A[i][k] * B[j][k]  (B stored as [N,K])
// ============================================================================
bool verify_gemm_nt(int M, int N, int K_dim) {
    printf("=== GEMM NT Accuracy [M=%d, N=%d, K=%d] ===\n", M, N, K_dim);

    unsigned seed = 12345;
    size_t sA = (size_t)M * K_dim, sB = (size_t)N * K_dim, sC = (size_t)M * N;

    std::vector<float> hA(sA), hB(sB), hC_ref(sC, 0.0f), hC_gpu(sC);

    // Small values to stay in FP16 range
    for (size_t i = 0; i < sA; i++) hA[i] = rand_float(seed, -0.3f, 0.3f);
    for (size_t i = 0; i < sB; i++) hB[i] = rand_float(seed, -0.3f, 0.3f);

    // CPU reference: C = A * B^T
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float dot = 0.0f;
            for (int k = 0; k < K_dim; k++)
                dot += hA[i * K_dim + k] * hB[j * K_dim + k];
            hC_ref[i * N + j] = dot;
        }
    }

    half* dA = upload_fp16(hA, sA);
    half* dB = upload_fp16(hB, sB);
    half* dC;
    cudaMalloc(&dC, sC * sizeof(half));

    GemmManager gemm;
    gemm.gemm_nt(dA, dB, dC, M, N, K_dim);
    cudaDeviceSynchronize();

    download_fp16(dC, hC_gpu, sC);

    // FP16 GEMM accumulates in FP32 but inputs are quantized — relax tolerance
    // For K=768, each dot product sums 768 terms: expect ~sqrt(768)*eps ≈ 0.03
    auto r = compare_arrays(hC_ref.data(), hC_gpu.data(), (int)sC,
                            0.1f, 0.15f, 0.05f);
    print_result("GEMM NT", r);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return r.pass;
}

// ============================================================================
// 3. LayerNorm Accuracy Test
//    CPU ref: y = gamma * (x - mean) / sqrt(var + eps) + beta
//    With optional residual: x_combined = input + residual + bias
// ============================================================================
bool verify_layernorm(int N, int D, bool with_residual, bool rmsnorm) {
    const char* mode = rmsnorm ? "RMSNorm" : "LayerNorm";
    const char* res  = with_residual ? "+residual" : "";
    printf("=== %s%s Accuracy [N=%d, D=%d] ===\n", mode, res, N, D);

    unsigned seed = 54321;
    size_t total = (size_t)N * D;

    std::vector<float> h_input(total), h_gamma(D), h_beta(D);
    std::vector<float> h_residual(with_residual ? total : 0);
    std::vector<float> h_ref(total), h_gpu(total);

    for (size_t i = 0; i < total; i++) h_input[i] = rand_float(seed, -1.0f, 1.0f);
    for (int i = 0; i < D; i++) h_gamma[i] = rand_float(seed, 0.5f, 1.5f);
    for (int i = 0; i < D; i++) h_beta[i]  = rand_float(seed, -0.5f, 0.5f);
    if (with_residual)
        for (size_t i = 0; i < total; i++) h_residual[i] = rand_float(seed, -0.5f, 0.5f);

    // CPU reference
    float eps = 1e-5f;
    for (int row = 0; row < N; row++) {
        // Combine input + residual
        std::vector<float> x(D);
        for (int j = 0; j < D; j++) {
            x[j] = h_input[row * D + j];
            if (with_residual) x[j] += h_residual[row * D + j];
        }

        if (rmsnorm) {
            // RMSNorm: y = gamma * x / sqrt(mean(x^2) + eps)
            float sumsq = 0.0f;
            for (int j = 0; j < D; j++) sumsq += x[j] * x[j];
            float rms_scale = 1.0f / sqrtf(sumsq / D + eps);
            for (int j = 0; j < D; j++)
                h_ref[row * D + j] = h_gamma[j] * x[j] * rms_scale;
        } else {
            // LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
            float mean = 0.0f;
            for (int j = 0; j < D; j++) mean += x[j];
            mean /= D;

            float var = 0.0f;
            for (int j = 0; j < D; j++) var += (x[j] - mean) * (x[j] - mean);
            var /= D;

            float inv_std = 1.0f / sqrtf(var + eps);
            for (int j = 0; j < D; j++)
                h_ref[row * D + j] = h_gamma[j] * (x[j] - mean) * inv_std + h_beta[j];
        }
    }

    // GPU
    half* d_input = upload_fp16(h_input, total);
    half* d_residual = with_residual ? upload_fp16(h_residual, total) : nullptr;
    half* d_gamma = upload_fp16(h_gamma, D);
    half* d_beta  = upload_fp16(h_beta, D);
    half* d_output;
    half* d_res_out;
    cudaMalloc(&d_output, total * sizeof(half));
    cudaMalloc(&d_res_out, total * sizeof(half));

    LayerNormParams params = {};
    params.input       = d_input;
    params.residual    = d_residual;
    params.bias        = nullptr;
    params.gamma       = d_gamma;
    params.beta        = rmsnorm ? nullptr : d_beta;
    params.output      = d_output;
    params.residual_out = with_residual ? d_res_out : nullptr;
    params.num_tokens  = N;
    params.d_model     = D;
    params.eps         = eps;
    params.use_rmsnorm = rmsnorm;
    params.stream      = nullptr;

    launch_fused_layernorm(params);
    cudaDeviceSynchronize();

    download_fp16(d_output, h_gpu, total);

    auto r = compare_arrays(h_ref.data(), h_gpu.data(), (int)total,
                            0.02f, 0.05f, 0.01f);
    print_result(mode, r);

    // Also verify residual_out if applicable
    if (with_residual) {
        std::vector<float> h_resout_gpu;
        download_fp16(d_res_out, h_resout_gpu, total);

        std::vector<float> h_resout_ref(total);
        for (size_t i = 0; i < total; i++)
            h_resout_ref[i] = h_input[i] + h_residual[i];

        auto r2 = compare_arrays(h_resout_ref.data(), h_resout_gpu.data(), (int)total,
                                 0.01f, 0.05f, 0.005f);
        printf("  --- Residual output check ---\n");
        print_result("Residual Add", r2);
        if (!r2.pass) r.pass = false;
    }

    cudaFree(d_input); cudaFree(d_output); cudaFree(d_gamma);
    cudaFree(d_beta); cudaFree(d_res_out);
    if (d_residual) cudaFree(d_residual);
    return r.pass;
}

// ============================================================================
// 4. RoPE Accuracy Test
//    CPU ref: standard rotary embedding pair rotation
// ============================================================================
bool verify_rope(int batch_heads, int seq_len, int d_head, int start_pos) {
    printf("=== RoPE Accuracy [BH=%d, S=%d, D=%d, start=%d] ===\n",
           batch_heads, seq_len, d_head, start_pos);

    unsigned seed = 99999;
    size_t total = (size_t)batch_heads * seq_len * d_head;
    int half_d = d_head / 2;
    float base = 10000.0f;

    std::vector<float> h_Q(total), h_K(total);
    std::vector<float> h_Q_ref(total), h_K_ref(total);
    std::vector<float> h_Q_gpu(total), h_K_gpu(total);

    for (size_t i = 0; i < total; i++) {
        h_Q[i] = rand_float(seed, -1.0f, 1.0f);
        h_K[i] = rand_float(seed, -1.0f, 1.0f);
    }
    h_Q_ref = h_Q;
    h_K_ref = h_K;

    // CPU reference
    for (int bh = 0; bh < batch_heads; bh++) {
        for (int s = 0; s < seq_len; s++) {
            int abs_pos = start_pos + s;
            for (int p = 0; p < half_d; p++) {
                float freq = expf(-2.0f * p / (float)d_head * logf(base));
                float angle = abs_pos * freq;
                float cos_val = cosf(angle);
                float sin_val = sinf(angle);

                size_t off = (size_t)bh * seq_len * d_head + s * d_head;
                int d0 = p * 2, d1 = p * 2 + 1;

                float q0 = h_Q[off + d0], q1 = h_Q[off + d1];
                h_Q_ref[off + d0] = q0 * cos_val - q1 * sin_val;
                h_Q_ref[off + d1] = q0 * sin_val + q1 * cos_val;

                float k0 = h_K[off + d0], k1 = h_K[off + d1];
                h_K_ref[off + d0] = k0 * cos_val - k1 * sin_val;
                h_K_ref[off + d1] = k0 * sin_val + k1 * cos_val;
            }
        }
    }

    // GPU
    half* d_Q = upload_fp16(h_Q, total);
    half* d_K = upload_fp16(h_K, total);

    RoPEConfig rope;
    rope.init(start_pos + seq_len + 64, d_head, base, nullptr);
    cudaDeviceSynchronize();

    launch_rope(d_Q, d_K, rope, batch_heads, seq_len, start_pos, nullptr);
    cudaDeviceSynchronize();

    download_fp16(d_Q, h_Q_gpu, total);
    download_fp16(d_K, h_K_gpu, total);

    auto rq = compare_arrays(h_Q_ref.data(), h_Q_gpu.data(), (int)total,
                             0.02f, 0.05f, 0.01f);
    printf("  --- Q rotation ---\n");
    print_result("RoPE Q", rq);

    auto rk = compare_arrays(h_K_ref.data(), h_K_gpu.data(), (int)total,
                             0.02f, 0.05f, 0.01f);
    printf("  --- K rotation ---\n");
    print_result("RoPE K", rk);

    rope.free(nullptr);
    cudaFree(d_Q); cudaFree(d_K);
    return rq.pass && rk.pass;
}

// ============================================================================
// 5. SwiGLU Accuracy Test
//    CPU ref: output = SiLU(gate) * up = (gate * sigmoid(gate)) * up
// ============================================================================
bool verify_swiglu(int N, int D) {
    printf("=== SwiGLU Accuracy [N=%d, D=%d] ===\n", N, D);

    unsigned seed = 77777;
    size_t total = (size_t)N * D;

    std::vector<float> h_gate(total), h_up(total), h_ref(total), h_gpu(total);
    for (size_t i = 0; i < total; i++) {
        h_gate[i] = rand_float(seed, -3.0f, 3.0f);
        h_up[i]   = rand_float(seed, -2.0f, 2.0f);
    }

    // CPU reference: SiLU(gate) * up
    for (size_t i = 0; i < total; i++) {
        float g = h_gate[i];
        float silu_g = g / (1.0f + expf(-g));
        h_ref[i] = silu_g * h_up[i];
    }

    half* d_gate = upload_fp16(h_gate, total);
    half* d_up   = upload_fp16(h_up, total);
    half* d_out;
    cudaMalloc(&d_out, total * sizeof(half));

    launch_fused_swiglu(d_gate, d_up, d_out, N, D, nullptr);
    cudaDeviceSynchronize();

    download_fp16(d_out, h_gpu, total);

    auto r = compare_arrays(h_ref.data(), h_gpu.data(), (int)total,
                            0.05f, 0.1f, 0.02f);
    print_result("SwiGLU", r);

    cudaFree(d_gate); cudaFree(d_up); cudaFree(d_out);
    return r.pass;
}

// ============================================================================
// 6. GELU Accuracy Test
//    CPU ref: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// ============================================================================
bool verify_gelu(int N, int D) {
    printf("=== GELU Accuracy [N=%d, D=%d] ===\n", N, D);

    unsigned seed = 33333;
    size_t total = (size_t)N * D;

    std::vector<float> h_input(total), h_ref(total), h_gpu(total);
    for (size_t i = 0; i < total; i++)
        h_input[i] = rand_float(seed, -4.0f, 4.0f);

    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float GELU_COEFF     = 0.044715f;
    for (size_t i = 0; i < total; i++) {
        float x = h_input[i];
        float x3 = x * x * x;
        h_ref[i] = 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * (x + GELU_COEFF * x3)));
    }

    half* d_input = upload_fp16(h_input, total);
    half* d_output;
    cudaMalloc(&d_output, total * sizeof(half));

    launch_fused_gelu(d_input, d_output, N, D, nullptr);
    cudaDeviceSynchronize();

    download_fp16(d_output, h_gpu, total);

    auto r = compare_arrays(h_ref.data(), h_gpu.data(), (int)total,
                            0.05f, 0.1f, 0.02f);
    print_result("GELU", r);

    cudaFree(d_input); cudaFree(d_output);
    return r.pass;
}

// ============================================================================
// 7. SiLU In-Place Accuracy Test
//    CPU ref: x * sigmoid(x)
// ============================================================================
bool verify_silu(int N) {
    printf("=== SiLU In-Place Accuracy [N=%d] ===\n", N);

    unsigned seed = 11111;
    std::vector<float> h_input(N), h_ref(N), h_gpu(N);

    for (int i = 0; i < N; i++)
        h_input[i] = rand_float(seed, -5.0f, 5.0f);

    for (int i = 0; i < N; i++) {
        float x = h_input[i];
        h_ref[i] = x / (1.0f + expf(-x));
    }

    half* d_data = upload_fp16(h_input, N);

    launch_silu_inplace(d_data, N, nullptr);
    cudaDeviceSynchronize();

    download_fp16(d_data, h_gpu, N);

    auto r = compare_arrays(h_ref.data(), h_gpu.data(), N,
                            0.05f, 0.1f, 0.02f);
    print_result("SiLU", r);

    cudaFree(d_data);
    return r.pass;
}

// ============================================================================
// 8. Flash Attention Accuracy Test (existing, refactored)
// ============================================================================
bool verify_flash_attention(int batch_size, int n_heads, int seq_len, int d_head) {
    printf("=== Flash Attention Accuracy [B=%d, H=%d, S=%d, D=%d] ===\n",
           batch_size, n_heads, seq_len, d_head);

    const int total_elements = batch_size * n_heads * seq_len * d_head;
    const float scale = 1.0f / sqrtf(static_cast<float>(d_head));

    std::vector<float> h_Q(total_elements), h_K(total_elements), h_V(total_elements);
    std::vector<float> h_O_ref(total_elements, 0.0f), h_O_gpu(total_elements);

    srand(42);
    for (int i = 0; i < total_elements; i++) {
        h_Q[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.5f;
        h_K[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.5f;
        h_V[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.5f;
    }

    // CPU reference
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < n_heads; h++) {
            const int bh = b * n_heads + h;
            const float* Q_bh = h_Q.data() + bh * seq_len * d_head;
            const float* K_bh = h_K.data() + bh * seq_len * d_head;
            const float* V_bh = h_V.data() + bh * seq_len * d_head;
            float* O_bh = h_O_ref.data() + bh * seq_len * d_head;

            std::vector<float> S(seq_len * seq_len);
            for (int i = 0; i < seq_len; i++)
                for (int j = 0; j < seq_len; j++) {
                    float dot = 0.0f;
                    for (int k = 0; k < d_head; k++)
                        dot += Q_bh[i * d_head + k] * K_bh[j * d_head + k];
                    S[i * seq_len + j] = (j > i) ? -1e9f : dot * scale;
                }

            std::vector<float> P(seq_len * seq_len);
            for (int i = 0; i < seq_len; i++) {
                float mx = *std::max_element(&S[i * seq_len], &S[(i+1) * seq_len]);
                float sum = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    P[i * seq_len + j] = expf(S[i * seq_len + j] - mx);
                    sum += P[i * seq_len + j];
                }
                for (int j = 0; j < seq_len; j++) P[i * seq_len + j] /= sum;
            }

            for (int i = 0; i < seq_len; i++)
                for (int k = 0; k < d_head; k++) {
                    float val = 0.0f;
                    for (int j = 0; j < seq_len; j++)
                        val += P[i * seq_len + j] * V_bh[j * d_head + k];
                    O_bh[i * d_head + k] = val;
                }
        }
    }

    // GPU
    half* dQ = upload_fp16(h_Q, total_elements);
    half* dK = upload_fp16(h_K, total_elements);
    half* dV = upload_fp16(h_V, total_elements);
    half* dO; cudaMalloc(&dO, total_elements * sizeof(half));
    float* dL; cudaMalloc(&dL, batch_size * n_heads * seq_len * sizeof(float));

    FlashAttentionParams params = {};
    params.Q = dQ; params.K = dK; params.V = dV; params.O = dO; params.L = dL;
    params.batch_size = batch_size; params.num_heads = n_heads;
    params.seq_len = seq_len; params.d_head = d_head;
    params.scale = scale; params.causal = true; params.stream = 0;

    launch_flash_attention(params);
    cudaDeviceSynchronize();

    download_fp16(dO, h_O_gpu, total_elements);

    // FA chains 3 matmuls + softmax in FP16 — NRMSE ~0.024 is expected
    auto r = compare_arrays(h_O_ref.data(), h_O_gpu.data(), total_elements,
                            0.05f, 0.1f, 0.03f);
    print_result("Flash Attention", r);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dL);
    return r.pass;
}

// ============================================================================
// Benchmarks (unchanged)
// ============================================================================
void benchmark_flash_attention(int batch_size, int n_heads, int seq_len, int d_head,
                                int warmup = 10, int iters = 100) {
    printf("\n--- FA Bench [B=%d,H=%d,S=%d,D=%d] ", batch_size, n_heads, seq_len, d_head);
    size_t bh = (size_t)batch_size * n_heads;
    size_t elems = bh * seq_len * d_head;
    half *Q, *K, *V, *O; float *L;
    cudaMalloc(&Q, elems*2); cudaMalloc(&K, elems*2);
    cudaMalloc(&V, elems*2); cudaMalloc(&O, elems*2);
    cudaMalloc(&L, bh * seq_len * 4);
    cudaMemset(Q, 0x3C, elems*2); cudaMemset(K, 0x3C, elems*2); cudaMemset(V, 0x3C, elems*2);

    FlashAttentionParams p = {};
    p.Q=Q; p.K=K; p.V=V; p.O=O; p.L=L;
    p.batch_size=batch_size; p.num_heads=n_heads; p.seq_len=seq_len; p.d_head=d_head;
    p.scale=1.0f/sqrtf((float)d_head); p.causal=true; p.stream=nullptr;
    for(int i=0;i<warmup;i++) launch_flash_attention(p);
    cudaDeviceSynchronize();
    CudaTimer t; t.begin();
    for(int i=0;i<iters;i++) launch_flash_attention(p);
    float ms = t.end()/iters;
    double flops = 2.0*bh*(double)seq_len*seq_len*d_head*2;
    printf("%.3f ms  %.2f TFLOPS\n", ms, flops/(ms*1e-3)/1e12);
    cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(O); cudaFree(L);
}

void benchmark_gemm(int M, int N, int K_dim, int warmup = 10, int iters = 100) {
    printf("--- GEMM Bench [%dx%dx%d] ", M, N, K_dim);
    half *A,*B,*C;
    cudaMalloc(&A,(size_t)M*K_dim*2); cudaMalloc(&B,(size_t)N*K_dim*2); cudaMalloc(&C,(size_t)M*N*2);
    cudaMemset(A,0x3C,(size_t)M*K_dim*2); cudaMemset(B,0x3C,(size_t)N*K_dim*2);
    GemmManager g;
    for(int i=0;i<warmup;i++) g.gemm_nt(A,B,C,M,N,K_dim);
    cudaDeviceSynchronize();
    CudaTimer t; t.begin();
    for(int i=0;i<iters;i++) g.gemm_nt(A,B,C,M,N,K_dim);
    float ms = t.end()/iters;
    double flops = 2.0*M*(double)N*K_dim;
    printf("%.3f ms  %.2f TFLOPS\n", ms, flops/(ms*1e-3)/1e12);
    cudaFree(A); cudaFree(B); cudaFree(C);
}

void benchmark_layernorm(int N, int D, int warmup = 10, int iters = 1000) {
    printf("--- LN Bench [%d x %d] ", N, D);
    half *in,*res,*gm,*bt,*out,*ro;
    cudaMalloc(&in,(size_t)N*D*2); cudaMalloc(&res,(size_t)N*D*2);
    cudaMalloc(&gm,D*2); cudaMalloc(&bt,D*2);
    cudaMalloc(&out,(size_t)N*D*2); cudaMalloc(&ro,(size_t)N*D*2);
    LayerNormParams p={};
    p.input=in; p.residual=res; p.gamma=gm; p.beta=bt;
    p.output=out; p.residual_out=ro; p.num_tokens=N; p.d_model=D;
    p.eps=1e-5f; p.use_rmsnorm=false; p.stream=nullptr;
    for(int i=0;i<warmup;i++) launch_fused_layernorm(p);
    cudaDeviceSynchronize();
    CudaTimer t; t.begin();
    for(int i=0;i<iters;i++) launch_fused_layernorm(p);
    float ms = t.end()/iters;
    double bytes = (double)N*D*2*5;
    printf("%.4f ms  %.1f GB/s\n", ms, bytes/(ms*1e-3)/1e9);
    cudaFree(in); cudaFree(res); cudaFree(gm); cudaFree(bt); cudaFree(out); cudaFree(ro);
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("============================================\n");
    printf("  Transformer Engine — Tests & Benchmarks\n");
    printf("============================================\n");
    printf("GPU: %s\n", prop.name);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Memory: %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("Compute: sm_%d%d\n", prop.major, prop.minor);

    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
    int fp16_ops = (prop.major >= 12) ? 512 : (prop.major >= 8) ? 256 : 128;
    printf("Peak FP16: %.1f TFLOPS\n",
           2.0 * prop.multiProcessorCount * clock_khz * 1e-6 * fp16_ops / 1e3);
    printf("============================================\n\n");

    int pass = 0, fail = 0;
    auto check = [&](bool ok) { ok ? pass++ : fail++; };

    // ── Accuracy Tests ──────────────────────────────────────────────────
    printf("╔══════════════════════════════════════════╗\n");
    printf("║          ACCURACY TESTS                  ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    // PTX MMA
    check(test_ptx_mma());

    // Flash Attention — multiple configs
    check(verify_flash_attention(1, 1, 64, 64));
    check(verify_flash_attention(1, 1, 128, 64));
    check(verify_flash_attention(1, 2, 256, 64));

    // GEMM NT — small, medium, transformer-sized
    check(verify_gemm_nt(16, 16, 16));
    check(verify_gemm_nt(128, 128, 128));
    check(verify_gemm_nt(512, 768, 768));     // QKV projection size
    check(verify_gemm_nt(512, 3072, 768));    // FFN up-projection size

    // LayerNorm variants
    check(verify_layernorm(64,  768, false, false));  // Standard LN
    check(verify_layernorm(64,  768, true,  false));  // LN + residual
    check(verify_layernorm(64,  768, false, true));   // RMSNorm
    check(verify_layernorm(64,  768, true,  true));   // RMSNorm + residual
    check(verify_layernorm(256, 768, true,  false));  // Larger batch

    // RoPE
    check(verify_rope(12, 64, 64, 0));        // 12 heads, start_pos=0
    check(verify_rope(12, 128, 64, 0));       // Longer sequence
    check(verify_rope(12, 32, 64, 100));      // Nonzero start (decoding)

    // Activations
    check(verify_swiglu(256, 2048));
    check(verify_swiglu(1024, 768));
    check(verify_gelu(256, 3072));
    check(verify_gelu(1024, 768));
    check(verify_silu(65536));
    check(verify_silu(1000000));

    // ── Summary ─────────────────────────────────────────────────────────
    printf("============================================\n");
    printf("  ACCURACY SUMMARY: %d passed, %d failed\n", pass, fail);
    printf("============================================\n");

    if (fail > 0) {
        printf("  *** %d TEST(S) FAILED ***\n\n", fail);
    } else {
        printf("  All tests passed.\n\n");
    }

    // ── Benchmarks ──────────────────────────────────────────────────────
    printf("╔══════════════════════════════════════════╗\n");
    printf("║          BENCHMARKS                      ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    ModelConfig config;
    config.d_model = 768; config.n_heads = 12; config.d_head = 64;

    benchmark_layernorm(2048, config.d_model);
    benchmark_gemm(2048, config.d_model, config.d_model);
    benchmark_gemm(2048, config.d_ffn, config.d_model);
    benchmark_gemm(2048, config.d_model, config.d_ffn);
    benchmark_flash_attention(1, 12, 512, 64);
    benchmark_flash_attention(1, 12, 2048, 64);
    benchmark_flash_attention(4, 12, 2048, 64);
    benchmark_flash_attention(8, 12, 2048, 64);
    benchmark_flash_attention(1, 12, 4096, 64);

    printf("\n============================================\n");
    printf("  Complete. %d/%d accuracy tests passed.\n", pass, pass + fail);
    printf("============================================\n");

    return fail > 0 ? 1 : 0;
}
