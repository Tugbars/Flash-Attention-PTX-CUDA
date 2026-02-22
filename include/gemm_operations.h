#pragma once

// ============================================================================
// CUTLASS GEMM Operations for Transformer
// 
// Wraps CUTLASS 3.x GEMMs with optimal tile shapes for transformer workloads.
// Falls back to cuBLAS if CUTLASS is not available.
// ============================================================================

#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "transformer_config.h"
#include "tensor.h"
#include "activation_kernels.h"

// CUTLASS headers (3.x)
#ifdef USE_CUTLASS
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/epilogue/thread/linear_combination_silu.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>
#endif

// FP8 support
#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#include "fp8_quantize.h"
#endif

namespace transformer {

// ============================================================================
// GEMM Configuration Types
// ============================================================================

enum class GemmType {
    QKV_PROJ,       // [B*S, D] x [D, 3D] — fused Q/K/V projection
    ATTN_OUT,       // [B*S, D] x [D, D]  — attention output projection
    FFN_UP,         // [B*S, D] x [D, 4D] — FFN up-projection
    FFN_GATE,       // [B*S, D] x [D, 4D] — SwiGLU gate projection
    FFN_DOWN,       // [B*S, 4D] x [4D, D] — FFN down-projection
    BMM_QK,         // Batched [S, D_h] x [D_h, S] — Q*K^T
    BMM_AV,         // Batched [S, S] x [S, D_h]   — Attn*V
};

// ============================================================================
// CUTLASS GEMM Kernel Type Definitions
// ============================================================================

#ifdef USE_CUTLASS

// -- FP16 Tensor Core GEMM (Ampere/Ada/Hopper) -----------------------------
// C = alpha * A * B + beta * C
// A: RowMajor, B: ColumnMajor (for NT layout — optimal for weight matrices)
using GemmFP16_NT = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t,                                    // ElementA
    cutlass::layout::RowMajor,                          // LayoutA
    cutlass::half_t,                                    // ElementB
    cutlass::layout::ColumnMajor,                       // LayoutB (transposed weights)
    cutlass::half_t,                                    // ElementC
    cutlass::layout::RowMajor,                          // LayoutC
    float,                                              // Accumulator
    cutlass::arch::OpClassTensorOp,                     // Use Tensor Cores
    cutlass::arch::Sm80,                                // Ampere+
    cutlass::gemm::GemmShape<
        tuning::GEMM_TILE_M,
        tuning::GEMM_TILE_N,
        tuning::GEMM_TILE_K>,                           // ThreadBlock shape
    cutlass::gemm::GemmShape<
        tuning::GEMM_WARP_M,
        tuning::GEMM_WARP_N,
        tuning::GEMM_WARP_K>,                           // Warp shape
    cutlass::gemm::GemmShape<16, 8, 16>,                // MMA shape (Tensor Core)
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t, 8, float, float>,              // Epilogue
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    tuning::GEMM_STAGES                                 // Pipeline stages
>;

// -- Batched GEMM for Q*K^T -------------------------------------------------
using BatchedGemmFP16_NT = cutlass::gemm::device::GemmBatched<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 32>,               // Smaller tiles for BMM
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2   // stages
>;

// -- GEMM with fused SiLU activation (for SwiGLU gate) ----------------------
using GemmFP16_SiLU = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationSilu<
        cutlass::half_t, 8, float, float>,              // Fused SiLU in epilogue!
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4
>;

#ifdef ENABLE_FP8
// -- FP8 E4M3 Tensor Core GEMM (Ada sm_89 / Blackwell sm_120) ----------------
// A,B: FP8 E4M3, C/D: FP16, Accumulator: FP32
// Dequantization via alpha = scale_a * scale_b in epilogue
using GemmFP8_NT = cutlass::gemm::device::GemmUniversal<
    cutlass::float_e4m3_t,                              // ElementA
    cutlass::layout::RowMajor,                          // LayoutA
    cutlass::float_e4m3_t,                              // ElementB
    cutlass::layout::ColumnMajor,                       // LayoutB (transposed weights)
    cutlass::half_t,                                    // ElementC (output FP16)
    cutlass::layout::RowMajor,                          // LayoutC
    float,                                              // Accumulator FP32
    cutlass::arch::OpClassTensorOp,                     // Tensor Cores
    cutlass::arch::Sm89,                                // Ada+ (works on sm_120)
    cutlass::gemm::GemmShape<
        tuning::FP8_GEMM_TILE_M,
        tuning::FP8_GEMM_TILE_N,
        tuning::FP8_GEMM_TILE_K>,                       // ThreadBlock (128,64,128)
    cutlass::gemm::GemmShape<
        tuning::FP8_GEMM_WARP_M,
        tuning::FP8_GEMM_WARP_N,
        tuning::FP8_GEMM_WARP_K>,                       // Warp (64,32,128)
    cutlass::gemm::GemmShape<16, 8, 32>,                // MMA shape for FP8
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t, 8, float, float>,              // alpha*acc → FP16
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    tuning::FP8_GEMM_STAGES
>;
#endif // ENABLE_FP8

#endif // USE_CUTLASS

// ============================================================================
// GEMM Manager — Dispatches to CUTLASS or cuBLAS
// ============================================================================

class GemmManager {
public:
    GemmManager() {
        CUBLAS_CHECK(cublasCreate(&cublas_handle_));
        CUBLAS_CHECK(cublasSetMathMode(cublas_handle_, CUBLAS_TENSOR_OP_MATH));
    }

    ~GemmManager() {
        if (cublas_handle_) cublasDestroy(cublas_handle_);
    }

    void set_stream(cudaStream_t stream) {
        CUBLAS_CHECK(cublasSetStream(cublas_handle_, stream));
        stream_ = stream;
    }

    // -----------------------------------------------------------------------
    // Standard GEMM: C = alpha * A * B^T + beta * C
    // A: [M, K], B: [N, K] (stored transposed), C: [M, N]
    // -----------------------------------------------------------------------
    void gemm_nt(const half* A, const half* B, half* C,
                 int M, int N, int K,
                 float alpha = 1.0f, float beta = 0.0f)
    {
#ifdef USE_CUTLASS
        launch_cutlass_gemm_nt(A, B, C, M, N, K, alpha, beta);
#else
        launch_cublas_gemm_nt(A, B, C, M, N, K, alpha, beta);
#endif
    }

    // -----------------------------------------------------------------------
    // Batched GEMM: C_i = alpha * A_i * B_i^T + beta * C_i
    // For multi-head attention Q*K^T and Attn*V
    // -----------------------------------------------------------------------
    void batched_gemm_nt(const half* A, const half* B, half* C,
                         int M, int N, int K, int batch_count,
                         int64_t stride_A, int64_t stride_B, int64_t stride_C,
                         float alpha = 1.0f, float beta = 0.0f)
    {
#ifdef USE_CUTLASS
        launch_cutlass_batched_gemm(A, B, C, M, N, K, batch_count,
                                     stride_A, stride_B, stride_C,
                                     alpha, beta);
#else
        launch_cublas_batched_gemm(A, B, C, M, N, K, batch_count,
                                    stride_A, stride_B, stride_C,
                                    alpha, beta);
#endif
    }

    // -----------------------------------------------------------------------
    // GEMM with fused SiLU — for SwiGLU gate projection
    // C = SiLU(A * B^T)
    // -----------------------------------------------------------------------
    void gemm_nt_silu(const half* A, const half* B, half* C,
                      int M, int N, int K)
    {
#ifdef USE_CUTLASS
        launch_cutlass_gemm_silu(A, B, C, M, N, K);
#else
        // Fallback: separate GEMM + SiLU kernel
        launch_cublas_gemm_nt(A, B, C, M, N, K, 1.0f, 0.0f);
        silu_fallback(C, static_cast<size_t>(M) * N);
#endif
    }

#ifdef ENABLE_FP8
    // -----------------------------------------------------------------------
    // FP8 GEMM: D_fp16 = (scale_a * scale_b) * (A_fp8 * B_fp8^T)
    //
    // A: [M, K] FP8, B: [N, K] FP8 (stored transposed)
    // scale_a, scale_b: per-tensor scale factors (absmax / 448)
    // D: [M, N] FP16 output
    // -----------------------------------------------------------------------
    void gemm_nt_fp8(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B,
                     half* D,
                     int M, int N, int K,
                     float scale_a, float scale_b)
    {
#ifdef USE_CUTLASS
        launch_cutlass_gemm_nt_fp8(A, B, D, M, N, K, scale_a, scale_b);
#else
        // No cuBLAS FP8 fallback — require CUTLASS
        assert(false && "FP8 GEMM requires CUTLASS");
#endif
    }

    // -----------------------------------------------------------------------
    // FP8 GEMM convenience: quantize FP16 activations on the fly, then GEMM
    //
    // Takes FP16 A and pre-quantized FP8 weights B.
    // Quantizes A dynamically, runs FP8 GEMM, outputs FP16.
    //
    // NOTE: This path contains a cudaStreamSynchronize to read the activation
    // scale back to host (CUTLASS requires host-side alpha at launch time).
    // This causes a ~5μs pipeline bubble per GEMM.
    //
    // Production alternatives to eliminate the sync:
    //   1. Static calibration: pre-compute activation scales offline
    //   2. CUTLASS EVT (Epilogue Visitor Tree): device-side alpha pointer
    //   3. Post-GEMM scale: GEMM with alpha=1, then fused scale+dequant kernel
    //   4. Scale from previous layer: reuse previous layer's activation stats
    // -----------------------------------------------------------------------
    void gemm_nt_fp8_dynamic(const half* A_fp16,
                              const __nv_fp8_e4m3* B_fp8,
                              half* D,
                              int M, int N, int K,
                              float scale_b,
                              __nv_fp8_e4m3* a_fp8_buf,
                              float* a_scale_buf)  // 2 device floats
    {
        // Step 1: Quantize A to FP8 (3 kernels: absmax → scale → quantize)
        launch_quantize_fp16_to_fp8(A_fp16, a_fp8_buf, a_scale_buf, M * K, stream_);

        // Step 2: Read scale_a from device — requires sync
        // TODO(perf): replace with EVT epilogue or post-GEMM scale kernel
        cudaStreamSynchronize(stream_);
        float scale_a;
        cudaMemcpy(&scale_a, a_scale_buf, sizeof(float), cudaMemcpyDeviceToHost);

        // Step 3: FP8 GEMM with dequant in epilogue
        gemm_nt_fp8(a_fp8_buf, B_fp8, D, M, N, K, scale_a, scale_b);
    }
#endif // ENABLE_FP8

private:
    cublasHandle_t cublas_handle_ = nullptr;
    cudaStream_t   stream_        = nullptr;

    // -- cuBLAS fallback implementations ------------------------------------

    void launch_cublas_gemm_nt(const half* A, const half* B, half* C,
                                int M, int N, int K,
                                float alpha, float beta)
    {
        // cuBLAS uses column-major, so we compute C^T = B * A^T
        // which gives us row-major C = A * B^T
        const half alpha_h = __float2half(alpha);
        const half beta_h  = __float2half(beta);

        cublasStatus_t status = cublasHgemm(
            cublas_handle_,
            CUBLAS_OP_T,    // B transposed
            CUBLAS_OP_N,    // A not transposed
            N, M, K,        // Swapped M,N for col-major
            &alpha_h,
            B, K,            // ldb = K (B is [N, K] row-major)
            A, K,            // lda = K (A is [M, K] row-major)
            &beta_h,
            C, N             // ldc = N
        );
        assert(status == CUBLAS_STATUS_SUCCESS);
    }

    void launch_cublas_batched_gemm(const half* A, const half* B, half* C,
                                     int M, int N, int K, int batch_count,
                                     int64_t stride_A, int64_t stride_B, int64_t stride_C,
                                     float alpha, float beta)
    {
        const half alpha_h = __float2half(alpha);
        const half beta_h  = __float2half(beta);

        cublasStatus_t status = cublasHgemmStridedBatched(
            cublas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha_h,
            B, K, stride_B,
            A, K, stride_A,
            &beta_h,
            C, N, stride_C,
            batch_count
        );
        assert(status == CUBLAS_STATUS_SUCCESS);
    }

#ifdef USE_CUTLASS
    void launch_cutlass_gemm_nt(const half* A, const half* B, half* C,
                                 int M, int N, int K,
                                 float alpha, float beta)
    {
        using Gemm = GemmFP16_NT;
        typename Gemm::Arguments args(
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K},          // Problem size
            1,                   // Batch count
            {alpha, beta},       // Epilogue params
            reinterpret_cast<const cutlass::half_t*>(A), // A
            reinterpret_cast<const cutlass::half_t*>(B), // B
            reinterpret_cast<cutlass::half_t*>(C),       // C (source)
            reinterpret_cast<cutlass::half_t*>(C),       // D (destination)
            M * K, N * K, M * N, M * N,  // Batch strides
            K, K, N, N                    // Leading dims
        );

        Gemm gemm_op;
        cutlass::Status status = gemm_op(args, nullptr, stream_);
        assert(status == cutlass::Status::kSuccess);
    }

    void launch_cutlass_batched_gemm(const half* A, const half* B, half* C,
                                      int M, int N, int K, int batch_count,
                                      int64_t stride_A, int64_t stride_B, int64_t stride_C,
                                      float alpha, float beta)
    {
        using Gemm = BatchedGemmFP16_NT;
        typename Gemm::Arguments args(
            {M, N, K},
            {reinterpret_cast<const cutlass::half_t*>(A), K},
            stride_A,
            {reinterpret_cast<const cutlass::half_t*>(B), K},
            stride_B,
            {reinterpret_cast<cutlass::half_t*>(C), N},
            stride_C,
            {reinterpret_cast<cutlass::half_t*>(C), N},
            stride_C,
            {alpha, beta},
            batch_count
        );

        Gemm gemm_op;
        cutlass::Status status = gemm_op(args, nullptr, stream_);
        assert(status == cutlass::Status::kSuccess);
    }

    void launch_cutlass_gemm_silu(const half* A, const half* B, half* C,
                                   int M, int N, int K)
    {
        using Gemm = GemmFP16_SiLU;
        typename Gemm::Arguments args(
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K}, 1,
            {1.0f, 0.0f},
            reinterpret_cast<const cutlass::half_t*>(A),
            reinterpret_cast<const cutlass::half_t*>(B),
            reinterpret_cast<cutlass::half_t*>(C),
            reinterpret_cast<cutlass::half_t*>(C),
            M * K, N * K, M * N, M * N,
            K, K, N, N
        );

        Gemm gemm_op;
        cutlass::Status status = gemm_op(args, nullptr, stream_);
        assert(status == cutlass::Status::kSuccess);
    }

#ifdef ENABLE_FP8
    // -- FP8 CUTLASS GEMM launcher -------------------------------------------
    // D_fp16 = (scale_a * scale_b) * (A_fp8 * B_fp8^T)
    // Alpha encodes the dequantization: alpha = scale_a * scale_b
    // This converts INT32-range accumulator back to proper FP16 magnitude.
    void launch_cutlass_gemm_nt_fp8(const __nv_fp8_e4m3* A,
                                     const __nv_fp8_e4m3* B,
                                     half* D,
                                     int M, int N, int K,
                                     float scale_a, float scale_b)
    {
        using Gemm = GemmFP8_NT;
        float alpha = scale_a * scale_b;
        float beta  = 0.0f;

        typename Gemm::Arguments args(
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K},
            1,                          // batch count
            {alpha, beta},              // Epilogue: D = alpha * acc + beta * C
            reinterpret_cast<const cutlass::float_e4m3_t*>(A),
            reinterpret_cast<const cutlass::float_e4m3_t*>(B),
            reinterpret_cast<cutlass::half_t*>(D),       // C (source, unused with beta=0)
            reinterpret_cast<cutlass::half_t*>(D),       // D (destination)
            M * K, N * K, M * N, M * N, // Batch strides
            K, K, N, N                   // Leading dims
        );

        Gemm gemm_op;
        cutlass::Status status = gemm_op(args, nullptr, stream_);
        assert(status == cutlass::Status::kSuccess);
    }
#endif // ENABLE_FP8
#endif // USE_CUTLASS

    // SiLU fallback — delegates to launch_silu_inplace from activation_kernels
    void silu_fallback(half* data, size_t n) {
        launch_silu_inplace(data, n, stream_);
    }
};

} // namespace transformer