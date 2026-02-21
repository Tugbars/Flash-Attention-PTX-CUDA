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
        launch_silu_inplace(C, static_cast<size_t>(M) * N, stream_);
#endif
    }

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
#endif

    // SiLU fallback kernel (used when CUTLASS fused epilogue not available)
    static void launch_silu_inplace(half* data, size_t n, cudaStream_t stream);
};

} // namespace transformer