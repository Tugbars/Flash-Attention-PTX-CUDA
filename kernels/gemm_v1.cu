// ============================================================================
// GEMM v1 — Basic Tensor Core GEMM with PTX MMA
//
// C[M,N] = A[M,K] × B[K,N]   (A row-major, B column-major, C row-major)
//
// This is the baseline: one output tile per block, single-stage K-loop,
// no pipelining, no double buffering. The goal is correctness and a
// baseline TFLOPS number to optimize from.
//
// Tile: BLOCK_M=128, BLOCK_N=128, BLOCK_K=32
// Warps: 8 (4×2 warp arrangement over M×N)
// MMA: m16n8k16 (same as flash attention)
// Precision: FP16 compute, FP32 accumulation, FP16 output
//
// B is stored column-major (equivalently, B^T is row-major) so that
// the B operand can be loaded with ldmatrix.x2.trans — matching the
// MMA's col-major B requirement naturally.
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>
#include <algorithm>

// ============================================================================
// Config
// ============================================================================
static constexpr int BLOCK_M  = 128;
static constexpr int BLOCK_N  = 128;
static constexpr int BLOCK_K  = 32;
static constexpr int NUM_WARPS = 8;
static constexpr int WARP_SIZE = 32;
static constexpr int THREADS   = WARP_SIZE * NUM_WARPS;

// Warp arrangement: 4 warps along M, 2 along N
// Each warp computes a 32×64 output sub-tile via 2×8 = 16 m16n8k16 MMAs
static constexpr int WARPS_M = 4;
static constexpr int WARPS_N = 2;

// Each warp handles: (BLOCK_M/WARPS_M) = 32 rows, (BLOCK_N/WARPS_N) = 64 cols
static constexpr int WARP_M = BLOCK_M / WARPS_M;  // 32
static constexpr int WARP_N = BLOCK_N / WARPS_N;   // 64

// MMA tiles per warp
static constexpr int MMA_M = 16;
static constexpr int MMA_N = 8;
static constexpr int MMA_K = 16;
static constexpr int WARP_MMA_M = WARP_M / MMA_M;  // 2 m-tiles per warp
static constexpr int WARP_MMA_N = WARP_N / MMA_N;   // 8 n-tiles per warp

// Shared memory padding to avoid bank conflicts
static constexpr int SMEM_PAD = 8;
static constexpr int A_STRIDE = BLOCK_K + SMEM_PAD;  // row-major A tile: [BLOCK_M, BLOCK_K+pad]
static constexpr int B_STRIDE = BLOCK_K + SMEM_PAD;  // col-major B tile: [BLOCK_N, BLOCK_K+pad]

// ============================================================================
// PTX Intrinsics (same as flash attention)
// ============================================================================
__device__ __forceinline__ void ptx_mma_m16n8k16(
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

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    const void* smem_ptr)
{
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_trans(
    uint32_t& r0, uint32_t& r1, const void* smem_ptr)
{
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
        : "=r"(r0), "=r"(r1) : "r"(addr));
}

// ============================================================================
// GEMM Kernel
// ============================================================================
__global__ void gemm_v1_kernel(
    const half* __restrict__ A,   // [M, K] row-major
    const half* __restrict__ B,   // [N, K] row-major (column-major B means B^T is row-major)
    half*       __restrict__ C,   // [M, N] row-major
    int M, int N, int K)
{
    // Block computes C[bm*128 .. bm*128+127, bn*128 .. bn*128+127]
    const int bm = blockIdx.y;
    const int bn = blockIdx.x;
    const int tid = threadIdx.x + threadIdx.y * WARP_SIZE;
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;

    // Warp position in the 4×2 grid
    const int warp_m = warp_id / WARPS_N;  // 0-3
    const int warp_n = warp_id % WARPS_N;  // 0-1

    // Global row/col start for this block
    const int block_row = bm * BLOCK_M;
    const int block_col = bn * BLOCK_N;

    // -- Shared memory --
    // A tile: [BLOCK_M, BLOCK_K + pad] in row-major
    // B tile: [BLOCK_N, BLOCK_K + pad] in row-major (B is stored as [N,K])
    extern __shared__ char smem[];
    half* smem_a = reinterpret_cast<half*>(smem);
    half* smem_b = smem_a + BLOCK_M * A_STRIDE;

    // -- Output accumulator in registers --
    // Each warp computes WARP_MMA_M × WARP_MMA_N = 2×8 = 16 MMA tiles
    // Each MMA produces 4 floats per thread
    float acc[WARP_MMA_M][WARP_MMA_N][4];
    #pragma unroll
    for (int mi = 0; mi < WARP_MMA_M; mi++)
        #pragma unroll
        for (int ni = 0; ni < WARP_MMA_N; ni++) {
            acc[mi][ni][0] = 0.0f;
            acc[mi][ni][1] = 0.0f;
            acc[mi][ni][2] = 0.0f;
            acc[mi][ni][3] = 0.0f;
        }

    // -- K-loop: iterate over K in BLOCK_K chunks --
    const int num_k_tiles = (K + BLOCK_K - 1) / BLOCK_K;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k_start = kt * BLOCK_K;

        // === Load A tile: [BLOCK_M, BLOCK_K] from global → smem ===
        // 128-bit vectorized loads (uint4 = 8 halfs)
        {
            constexpr int VEC_COLS = BLOCK_K / 8;  // 32/8 = 4 vectors per row
            for (int idx = tid; idx < BLOCK_M * VEC_COLS; idx += THREADS) {
                int row = idx / VEC_COLS;
                int col = idx % VEC_COLS;
                int g_row = block_row + row;
                int g_col = k_start + col * 8;
                uint4 val = make_uint4(0, 0, 0, 0);
                if (g_row < M && g_col + 7 < K) {
                    val = reinterpret_cast<const uint4*>(A + g_row * K + g_col)[0];
                } else if (g_row < M) {
                    // Partial vector at K boundary — scalar fallback
                    half tmp[8] = {};
                    for (int i = 0; i < 8 && g_col + i < K; i++)
                        tmp[i] = A[g_row * K + g_col + i];
                    val = *reinterpret_cast<uint4*>(tmp);
                }
                reinterpret_cast<uint4*>(smem_a + row * A_STRIDE)[col] = val;
            }
        }

        // === Load B tile: [BLOCK_N, BLOCK_K] from global → smem ===
        // B is [N, K] row-major. We load B[bn*128+row, k_start:k_start+32]
        {
            constexpr int VEC_COLS = BLOCK_K / 8;
            for (int idx = tid; idx < BLOCK_N * VEC_COLS; idx += THREADS) {
                int row = idx / VEC_COLS;
                int col = idx % VEC_COLS;
                int g_row = block_col + row;
                int g_col = k_start + col * 8;
                uint4 val = make_uint4(0, 0, 0, 0);
                if (g_row < N && g_col + 7 < K) {
                    val = reinterpret_cast<const uint4*>(B + g_row * K + g_col)[0];
                } else if (g_row < N) {
                    half tmp[8] = {};
                    for (int i = 0; i < 8 && g_col + i < K; i++)
                        tmp[i] = B[g_row * K + g_col + i];
                    val = *reinterpret_cast<uint4*>(tmp);
                }
                reinterpret_cast<uint4*>(smem_b + row * B_STRIDE)[col] = val;
            }
        }

        __syncthreads();

        // === MMA: accumulate A_tile × B_tile^T into registers ===
        // K-dimension of the tile is BLOCK_K=32, MMA_K=16 → 2 k-steps
        #pragma unroll
        for (int ki = 0; ki < BLOCK_K / MMA_K; ki++) {

            // For each m-tile this warp owns
            #pragma unroll
            for (int mi = 0; mi < WARP_MMA_M; mi++) {

                // Load A sub-tile: 16 rows × 16 cols from smem (once per mi)
                uint32_t a0, a1, a2, a3;
                {
                    int a_row = warp_m * WARP_M + mi * MMA_M + (lane_id % 16);
                    int a_col = ki * MMA_K + (lane_id / 16) * 8;
                    ldmatrix_x4(a0, a1, a2, a3,
                        smem_a + a_row * A_STRIDE + a_col);
                }

                // For each n-tile this warp owns
                #pragma unroll
                for (int ni = 0; ni < WARP_MMA_N; ni++) {

                    // Load B sub-tile via ldmatrix.x2.trans
                    // B in smem: [BLOCK_N, BLOCK_K+pad] row-major
                    // For m16n8k16: B operand is 16×8 col-major
                    // ldmatrix.x2.trans loads from [N-rows, K-cols] and transposes
                    // threads 0-7 provide addresses for first 8×8, threads 8-15 for second 8×8
                    // The two 8×8 blocks are stacked along K: k=[0..7] and k=[8..15]
                    uint32_t b0, b1;
                    {
                        int b_n_row = warp_n * WARP_N + ni * MMA_N + (lane_id % 8);
                        int b_k_col = ki * MMA_K + ((lane_id / 8) % 2) * 8;
                        ldmatrix_x2_trans(b0, b1,
                            smem_b + b_n_row * B_STRIDE + b_k_col);
                    }

                    ptx_mma_m16n8k16(
                        acc[mi][ni][0], acc[mi][ni][1],
                        acc[mi][ni][2], acc[mi][ni][3],
                        a0, a1, a2, a3, b0, b1,
                        acc[mi][ni][0], acc[mi][ni][1],
                        acc[mi][ni][2], acc[mi][ni][3]);
                }
            }
        }

        __syncthreads();
    }

    // -- Write output: convert FP32 accumulators to FP16 and store to C --
    // Each MMA tile: thread owns C[row0,col0], C[row0,col1], C[row1,col0], C[row1,col1]
    // row0 = (lane_id/4)%8, row1 = row0+8, col0 = (lane_id%4)*2, col1 = col0+1
    {
        const int local_row0 = (lane_id / 4) % 8;
        const int local_col0 = (lane_id % 4) * 2;

        #pragma unroll
        for (int mi = 0; mi < WARP_MMA_M; mi++) {
            #pragma unroll
            for (int ni = 0; ni < WARP_MMA_N; ni++) {
                int g_row0 = block_row + warp_m * WARP_M + mi * MMA_M + local_row0;
                int g_row1 = g_row0 + 8;
                int g_col0 = block_col + warp_n * WARP_N + ni * MMA_N + local_col0;
                int g_col1 = g_col0 + 1;

                if (g_row0 < M && g_col0 < N)
                    C[g_row0 * N + g_col0] = __float2half(acc[mi][ni][0]);
                if (g_row0 < M && g_col1 < N)
                    C[g_row0 * N + g_col1] = __float2half(acc[mi][ni][1]);
                if (g_row1 < M && g_col0 < N)
                    C[g_row1 * N + g_col0] = __float2half(acc[mi][ni][2]);
                if (g_row1 < M && g_col1 < N)
                    C[g_row1 * N + g_col1] = __float2half(acc[mi][ni][3]);
            }
        }
    }
}

// ============================================================================
// CPU Reference GEMM
// ============================================================================
void cpu_gemm(const float* A, const float* B_col, float* C,
              int M, int N, int K)
{
    // B_col is [N, K] (i.e., column-major B transposed to row-major)
    // C = A × B_col^T → C[i,j] = sum_k A[i,k] * B_col[j,k]
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i * K + k] * B_col[j * K + k];
            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// Timing
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
// Main — Correctness + Benchmark
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
    printf("  GEMM v1 — Basic PTX Tensor Core GEMM  [build 003]\n");
    printf("  GPU: %s (%d SMs, CC %d.%d)\n",
           prop.name, prop.multiProcessorCount, prop.major, prop.minor);
    printf("  Peak FP16: %.1f TFLOPS\n", peak_tflops);
    printf("============================================================\n\n");

    // --- Test configs ---
    struct TestConfig { int M, N, K; const char* name; };
    TestConfig configs[] = {
        {128,  128,  128,  "Small square"},
        {256,  256,  256,  "Medium square"},
        {768,  768,  768,  "Transformer d_model"},
        {1024, 3072, 768,  "FFN up (B*S=1024)"},
        {1024, 768,  3072, "FFN down (B*S=1024)"},
        {2048, 2304, 768,  "QKV projection (B*S=2048)"},
    };
    int num_configs = sizeof(configs) / sizeof(configs[0]);

    bool all_pass = true;

    for (int ci = 0; ci < num_configs; ci++) {
        int M = configs[ci].M, N = configs[ci].N, K = configs[ci].K;
        printf("--- %s: M=%d, N=%d, K=%d ---\n", configs[ci].name, M, N, K);

        // Generate data
        std::vector<float> h_A(M * K), h_B(N * K), h_C_ref(M * N, 0.0f);
        srand(42 + ci);
        for (int i = 0; i < M * K; i++) h_A[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.5f;
        for (int i = 0; i < N * K; i++) h_B[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.5f;

        // CPU reference
        cpu_gemm(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);

        // Convert to FP16
        std::vector<half> h_A_h(M * K), h_B_h(N * K);
        for (int i = 0; i < M * K; i++) h_A_h[i] = __float2half(h_A[i]);
        for (int i = 0; i < N * K; i++) h_B_h[i] = __float2half(h_B[i]);

        // GPU
        half *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, M * K * sizeof(half));
        cudaMalloc(&d_B, N * K * sizeof(half));
        cudaMalloc(&d_C, M * N * sizeof(half));
        cudaMemcpy(d_A, h_A_h.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B_h.data(), N * K * sizeof(half), cudaMemcpyHostToDevice);

        dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
        dim3 block(WARP_SIZE, NUM_WARPS);
        size_t smem_bytes = (BLOCK_M * A_STRIDE + BLOCK_N * B_STRIDE) * sizeof(half);

        if (smem_bytes > 48 * 1024) {
            cudaFuncSetAttribute(gemm_v1_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem_bytes));
        }

        // Warmup
        for (int i = 0; i < 5; i++)
            gemm_v1_kernel<<<grid, block, smem_bytes>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();

        // Benchmark
        CudaTimer timer;
        const int iters = 100;
        timer.begin();
        for (int i = 0; i < iters; i++)
            gemm_v1_kernel<<<grid, block, smem_bytes>>>(d_A, d_B, d_C, M, N, K);
        float total_ms = timer.end();
        float avg_ms = total_ms / iters;

        double flops = 2.0 * M * (double)N * K;
        double tflops = flops / (avg_ms * 1e-3) / 1e12;

        // Copy back and check
        std::vector<half> h_C_h(M * N);
        std::vector<float> h_C_gpu(M * N);
        cudaMemcpy(h_C_h.data(), d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);
        for (int i = 0; i < M * N; i++) h_C_gpu[i] = __half2float(h_C_h[i]);

        float max_err = 0;
        double sum_sq_err = 0, sum_sq_ref = 0;
        int bad = 0;
        for (int i = 0; i < M * N; i++) {
            float err = fabsf(h_C_ref[i] - h_C_gpu[i]);
            float rel = err / (fabsf(h_C_ref[i]) + 1e-6f);
            max_err = std::max(max_err, err);
            sum_sq_err += (double)err * err;
            sum_sq_ref += (double)h_C_ref[i] * h_C_ref[i];
            if (err > 0.1f && rel > 0.15f) bad++;
        }
        float nrmse = sqrtf((float)(sum_sq_err / (sum_sq_ref + 1e-10)));
        bool pass = (nrmse < 0.05f && bad == 0);
        all_pass &= pass;

        printf("  Time:  %.3f ms\n", avg_ms);
        printf("  TFLOPS: %.2f (%.1f%% peak)\n", tflops, 100.0 * tflops / peak_tflops);
        printf("  NRMSE: %.6f | Max err: %.6f | Bad: %d/%d\n",
               nrmse, max_err, bad, M * N);

        // Debug values for first 2 tests
        if (ci < 2) {
            printf("  Row0 CPU:");
            for (int i = 0; i < 8; i++) printf(" %.4f", h_C_ref[i]);
            printf("\n  Row0 GPU:");
            for (int i = 0; i < 8; i++) printf(" %.4f", h_C_gpu[i]);
            printf("\n  Row1 CPU:");
            for (int i = 0; i < 8; i++) printf(" %.4f", h_C_ref[N+i]);
            printf("\n  Row1 GPU:");
            for (int i = 0; i < 8; i++) printf(" %.4f", h_C_gpu[N+i]);
            printf("\n");
            fflush(stdout);
        }

        printf("  %s\n\n", pass ? "PASS" : "FAIL");

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    printf("============================================================\n");
    printf("  %s\n", all_pass ? "All tests passed." : "FAILURES DETECTED.");
    printf("============================================================\n");

    return all_pass ? 0 : 1;
}
