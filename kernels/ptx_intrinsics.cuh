#pragma once

/**
 * @file ptx_intrinsics.cuh
 * @brief PTX wrappers shared by the attention kernels.
 *
 * Raw PTX is used rather than WMMA because mma.sync has a documented, fixed
 * fragment layout. That is what allows the softmax to run in registers and
 * the QK accumulator to be repacked directly as the A operand of the PV MMA.
 */

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <type_traits>

namespace transformer {
namespace ptx {

/// @name Element conversion
/// ldmatrix / cp.async / uint4 are width-agnostic, so the element type only
/// appears here and in the MMA opcode.
/// @{

/// Convert fp32 to the 16-bit element type @p T (round to nearest).
template <class T> __device__ __forceinline__ T from_float(float x);
template <> __device__ __forceinline__ half from_float<half>(float x) {
  return __float2half(x);
}
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
  return __float2bfloat16(x);
}

/// Pack two fp32 values into one 32-bit register holding two @p T elements,
/// @p a in the low half.
template <class T> __device__ __forceinline__ uint32_t pack2(float a, float b);
template <> __device__ __forceinline__ uint32_t pack2<half>(float a, float b) {
  __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<uint32_t *>(&h);
}
template <>
__device__ __forceinline__ uint32_t pack2<__nv_bfloat16>(float a, float b) {
  __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t *>(&h);
}

/// Two e4m3 (fp8) values, in the low 16 bits of @p two_bytes with the first in
/// the low byte, to two fp16 values in one register. The conversion is exact.
/// One `cvt` instruction from sm_89 on; earlier targets use the cuda_fp8.h
/// emulation. (Calling __nv_cvt_fp8x2_to_halfraw2 directly on sm_89+ emits the
/// same instruction but returns a two-field struct, which costs two extracts
/// and a re-pack per conversion.)
__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint32_t two_bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
  uint32_t r;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;"
      : "=r"(r)
      : "h"(static_cast<unsigned short>(two_bytes)));
  return r;
#else
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(
      static_cast<__nv_fp8x2_storage_t>(two_bytes), __NV_E4M3);
  return (uint32_t)h.x | ((uint32_t)h.y << 16);
#endif
}

/// @}

/**
 * @brief In-place `mma.sync.aligned.m16n8k16.row.col`, fp32 accumulate:
 *        `d += a * b`.
 *
 * Accumulator fragment per lane, with `g = lane / 4` and `c = (lane % 4) * 2`:
 * @code
 *   d[0] = D[g][c]    d[1] = D[g][c+1]
 *   d[2] = D[g+8][c]  d[3] = D[g+8][c+1]
 * @endcode
 * The A fragment has the same (row, column-pair) ownership per register, so an
 * accumulator produced by one MMA can be packed (pack2()) into the A operand of
 * the next without a shared-memory round trip.
 *
 * @tparam T  half or __nv_bfloat16; selects the opcode only.
 * @param d   Accumulator fragment (read and written).
 * @param a   A fragment (16x16, row-major), four registers from ldmatrix_x4().
 * @param b0  B fragment (16x8, col-major), first register from ldmatrix_x2().
 * @param b1  B fragment, second register.
 */
template <class T>
__device__ __forceinline__ void mma_m16n8k16(float (&d)[4],
                                             const uint32_t (&a)[4],
                                             uint32_t b0, uint32_t b1) {
  if constexpr (::std::is_same<T, __nv_bfloat16>::value) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
}

/// Generic pointer to a shared-memory address usable by PTX.
__device__ __forceinline__ uint32_t smem_addr(const void *p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

/// @name ldmatrix
/// Every lane supplies a row address; lanes 0-7 address matrix 0, 8-15
/// matrix 1, and so on. The caller offsets the later lane groups so each 8x8
/// sub-matrix covers a distinct part of the k dimension.
///
/// B-operand orientation for mma.row.col: lane i receives `B[k=(i%4)*2][n=i/4]`.
///   - A row-major K tile (rows = kv position = n) needs the plain x2 load.
///   - A row-major V tile (rows = kv position = k of the PV GEMM) needs .trans.
/// Using .trans on K feeds the MMA a feature-shuffled tile whose error is only
/// visible at realistic score magnitudes.
///
/// An x4 load is either one m16k16 A operand or the B operands of TWO adjacent
/// n8 tiles: registers 0-1 feed the first MMA, 2-3 the second. Both prefill
/// kernels load K and V that way (one load per two MMAs). The x2 forms are the
/// single-tile equivalents; nothing uses them today, they are kept for tile
/// shapes with an odd number of n8 tiles.
/// @{

/// Load four 8x8 b16 matrices without transpose: an m16k16 A operand, or the
/// K B-operands of two adjacent n8 tiles (lanes 0-7 / 8-15 address k 0-7 /
/// 8-15 of the first tile, lanes 16-23 / 24-31 those of the second).
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void *p) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(smem_addr(p)));
}

/// Load two 8x8 b16 matrices without transpose (K as B operand).
__device__ __forceinline__ void ldmatrix_x2(uint32_t &r0, uint32_t &r1,
                                            const void *p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
               : "=r"(r0), "=r"(r1)
               : "r"(smem_addr(p)));
}

/// Load two 8x8 b16 matrices with transpose (V as B operand).
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t &r0, uint32_t &r1,
                                                  const void *p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
               : "=r"(r0), "=r"(r1)
               : "r"(smem_addr(p)));
}

/// Load four 8x8 b16 matrices with transpose: the V B-operands of two adjacent
/// n8 tiles in one instruction. Lanes 0-7 / 8-15 address the k 0-7 / 8-15 rows
/// of the first n8 tile, lanes 16-23 / 24-31 those of the second.
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4],
                                                  const void *p) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(smem_addr(p)));
}

/// @}

/// @name cp.async
/// @{

/**
 * @brief 16-byte global -> shared copy that bypasses registers.
 *
 * With @p pred == false the source is not read and the destination is
 * zero-filled; @p gmem_src must still be a valid address.
 */
__device__ __forceinline__ void cp_async_16(void *smem_dst,
                                            const void *gmem_src, bool pred) {
  const int src_size = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(
                   smem_addr(smem_dst)),
               "l"(gmem_src), "r"(src_size));
}

/// Unpredicated 16-byte global -> shared copy to a precomputed smem address.
__device__ __forceinline__ void cp_async_16_full(uint32_t smem_dst,
                                                 const void *gmem_src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_dst),
               "l"(gmem_src));
}

/// Close the current cp.async group.
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

/// Block until at most @p N committed groups are still in flight.
template <int N> __device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

/// @}

} // namespace ptx
} // namespace transformer
