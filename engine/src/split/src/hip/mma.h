// HIP stand-in for CUDA's <mma.h>: nvcuda::wmma's 16x16x16 half / float fragments on gfx12, gfx11, and gfx906 (MI50).
#pragma once
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#if defined(__GFX11__)
#define FNX_WMMA_GFX11 1
#else
#define FNX_WMMA_GFX11 0
#endif

#if defined(__gfx906__) || defined(STRATA_HIP_GFX906)
#define FNX_WMMA_GFX906 1
#else
#define FNX_WMMA_GFX906 0
#endif

#define FNX_WMMA_FULL_K (FNX_WMMA_GFX11 || FNX_WMMA_GFX906)

namespace nvcuda {
namespace wmma {

struct row_major {};
struct col_major {};
struct matrix_a {};
struct matrix_b {};
struct accumulator {};
enum layout_t { mem_row_major, mem_col_major };

typedef _Float16 fnx_h2 __attribute__((ext_vector_type(2)));
typedef _Float16 fnx_h8 __attribute__((ext_vector_type(8)));
typedef _Float16 fnx_h16 __attribute__((ext_vector_type(16)));
typedef float fnx_f8 __attribute__((ext_vector_type(8)));
#if FNX_WMMA_FULL_K
typedef fnx_h16 fnx_hab;
constexpr int FNX_KPL = 16;
#else
typedef fnx_h8 fnx_hab;
constexpr int FNX_KPL = 8;
#endif

__device__ __forceinline__ fnx_hab fnx_ld_ab(const __half* p) {
#if FNX_WMMA_FULL_K
    const fnx_h8 lo = *reinterpret_cast<const fnx_h8*>(p), hi = *reinterpret_cast<const fnx_h8*>(p + 8);
    return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
#else
    return *reinterpret_cast<const fnx_h8*>(p);
#endif
}

template <typename Use, int M, int N, int K, typename T, typename Layout = void> struct fragment;
template <typename L> struct fragment<matrix_a, 16, 16, 16, __half, L> { fnx_hab v; };
template <typename L> struct fragment<matrix_b, 16, 16, 16, __half, L> { fnx_hab v; };
template <> struct fragment<accumulator, 16, 16, 16, float, void> { fnx_f8 v; };

__device__ __forceinline__ int fnx_lane() { return threadIdx.x & 31; }

__device__ __forceinline__ void load_matrix_sync(fragment<matrix_a, 16, 16, 16, __half, row_major>& f, const __half* p, unsigned ldm) {
    const int l = fnx_lane(), m = l & 15, k0 = FNX_WMMA_FULL_K ? 0 : 8 * (l >> 4);
    f.v = fnx_ld_ab(p + (size_t) m * ldm + k0);
}
__device__ __forceinline__ void load_matrix_sync(fragment<matrix_a, 16, 16, 16, __half, col_major>& f, const __half* p, unsigned ldm) {
    const int l = fnx_lane(), m = l & 15, k0 = FNX_WMMA_FULL_K ? 0 : 8 * (l >> 4);
#pragma unroll
    for (int i = 0; i < FNX_KPL; ++i) f.v[i] = (_Float16) __half2float(p[(size_t) (k0 + i) * ldm + m]);
}
__device__ __forceinline__ void load_matrix_sync(fragment<matrix_b, 16, 16, 16, __half, col_major>& f, const __half* p, unsigned ldm) {
    const int l = fnx_lane(), n = l & 15, k0 = FNX_WMMA_FULL_K ? 0 : 8 * (l >> 4);
    f.v = fnx_ld_ab(p + (size_t) n * ldm + k0);
}
__device__ __forceinline__ void load_matrix_sync(fragment<matrix_b, 16, 16, 16, __half, row_major>& f, const __half* p, unsigned ldm) {
    const int l = fnx_lane(), n = l & 15, k0 = FNX_WMMA_FULL_K ? 0 : 8 * (l >> 4);
#pragma unroll
    for (int i = 0; i < FNX_KPL; ++i) f.v[i] = (_Float16) __half2float(p[(size_t) (k0 + i) * ldm + n]);
}
__device__ __forceinline__ void fill_fragment(fragment<accumulator, 16, 16, 16, float>& f, float x) {
#pragma unroll
    for (int i = 0; i < 8; ++i) f.v[i] = x;
}
__device__ __forceinline__ fnx_f8 fnx_mma_raw(const fnx_hab& a, const fnx_hab& b, const fnx_f8& c) {
#if FNX_WMMA_GFX11
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#elif defined(__GFX12__) || defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#elif FNX_WMMA_GFX906
    const int l = fnx_lane();
    const int odd = l >> 4;
    unsigned a_u[8];
    fnx_h2 b_h[8];
#pragma unroll
    for (int p = 0; p < 8; ++p) {
        const fnx_h2 ap = {a[2 * p], a[2 * p + 1]};
        __builtin_memcpy(&a_u[p], &ap, 4);
        b_h[p] = {b[2 * p], b[2 * p + 1]};
    }
    fnx_f8 d;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r = 2 * i + odd;
        float acc = c[i];
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            const unsigned u = __shfl(a_u[p], r, 32);
            fnx_h2 ap;
            __builtin_memcpy(&ap, &u, 4);
#if defined(__gfx906__)
            acc = __builtin_amdgcn_fdot2(ap, b_h[p], acc, false);
#else
            acc += (float) ap[0] * (float) b_h[p][0] + (float) ap[1] * (float) b_h[p][1];
#endif
        }
        d[i] = acc;
    }
    return d;
#else
    (void) a; (void) b;
    __builtin_trap();
    return c;
#endif
}
template <typename LA, typename LB>
__device__ __forceinline__ void mma_sync(fragment<accumulator, 16, 16, 16, float>& d, const fragment<matrix_a, 16, 16, 16, __half, LA>& a,
                                         const fragment<matrix_b, 16, 16, 16, __half, LB>& b,
                                         const fragment<accumulator, 16, 16, 16, float>& c) {
    d.v = fnx_mma_raw(a.v, b.v, c.v);
}
__device__ __forceinline__ void store_matrix_sync(float* p, const fragment<accumulator, 16, 16, 16, float>& f, unsigned ldm, layout_t layout) {
    const int l = fnx_lane(), n = l & 15;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r = FNX_WMMA_FULL_K ? 2 * i + (l >> 4) : 8 * (l >> 4) + i;
        if (layout == mem_row_major) p[(size_t) r * ldm + n] = f.v[i];
        else p[(size_t) n * ldm + r] = f.v[i];
    }
}

}  // namespace wmma
}  // namespace nvcuda
