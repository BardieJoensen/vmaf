/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#include "stdio.h"
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include "feature/integer_vif.h"
#include "feature/common/macros.h"
#include "mem.h"

/*
 * This file is also built with -mavxvnni as vif_avxvnni.c, which defines
 * VIF_AVXVNNI: the kernels are the same, but every vpmaddwd + vpaddd
 * accumulation is a single vpdpwssd there, and the exported functions get
 * an _avxvnni suffix instead of _avx2.
 */
#ifdef VIF_AVXVNNI
#define VIF_X86_FN(name) name##_avxvnni
#else
#define VIF_X86_FN(name) name##_avx2
#endif

// acc + the sum of the products of each pair of int16 in a and b
static FORCE_INLINE __m256i vif_madd_acc(__m256i acc, __m256i a, __m256i b)
{
#ifdef VIF_AVXVNNI
    return _mm256_dpwssd_avx_epi32(acc, a, b);
#else
    return _mm256_add_epi32(acc, _mm256_madd_epi16(a, b));
#endif
}

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

#if defined __GNUC__
#define ALIGNED(x) __attribute__ ((aligned (x)))
#elif defined (_MSC_VER)  && (!defined UNDER_CE)
#define ALIGNED(x) __declspec (align(x))
#else
#define ALIGNED(x)
#endif


static FORCE_INLINE void
pad_top_and_bottom(VifBuffer buf, unsigned h, int fwidth)
{
    const unsigned fwidth_half = fwidth / 2;
    unsigned char *ref = buf.ref;
    unsigned char *dis = buf.dis;
    for (unsigned i = 1; i <= fwidth_half; ++i) {
        size_t offset = buf.stride * i;
        memcpy(ref - offset, ref + offset, buf.stride);
        memcpy(dis - offset, dis + offset, buf.stride);
        memcpy(ref + buf.stride * (h - 1) + buf.stride * i,
            ref + buf.stride * (h - 1) - buf.stride * i,
            buf.stride);
        memcpy(dis + buf.stride * (h - 1) + buf.stride * i,
            dis + buf.stride * (h - 1) - buf.stride * i,
            buf.stride);
    }
}

static FORCE_INLINE void
copy_and_pad(VifBuffer buf, unsigned w, unsigned h, int scale)
{
    uint16_t *ref = buf.ref;
    uint16_t *dis = buf.dis;
    const ptrdiff_t stride = buf.stride / sizeof(uint16_t);
    const ptrdiff_t mu_stride = buf.stride_16 / sizeof(uint16_t);

    for (unsigned i = 0; i < h / 2; ++i) {
        for (unsigned j = 0; j < w / 2; ++j) {
            ref[i * stride + j] = buf.mu1[i * mu_stride + j];
            dis[i * stride + j] = buf.mu2[i * mu_stride + j];
        }
    }
    pad_top_and_bottom(buf, h / 2, vif_filter1d_width[scale]);
}

/*
 * Row scratch for vif_statistic_8/16.
 *
 * The vertical pass leaves, per column, the filtered mu1, mu2 (at most 16
 * bits after the vertical shift) and T = filtered ref^2, dis^2, ref*dis
 * (after the vertical shift, all 32 bits). The C code filters T
 * horizontally in 64 bits. Here every value is kept as int16 planes
 * offset by -32768, T split into its upper and lower 16 bits, so that the
 * horizontal pass is plain vpmaddwd in 32-bit lanes: all VIF filters sum
 * to exactly 65536, so a plane filtered with them is its exact filtered
 * value minus 2^31 and every partial sum fits in int32. Recombining,
 *
 *   (sum f * T + 2^15) >> 16 = (Fhi + 2^31) + ((Flo + 2^31 + 2^15) >> 16)
 *
 * in uint32 arithmetic, with Fhi and Flo the filtered offset planes, which
 * is bit-exact with the C code.
 *
 * The planes live in the tmp row buffers (each has room for w + 64 uint32,
 * i.e. two int16 planes of w + 64 with 32 samples of margin on each side).
 */
typedef struct VifPlanes {
    int16_t *mu1, *mu2;
    int16_t *ref_hi, *ref_lo;
    int16_t *dis_hi, *dis_lo;
    int16_t *ref_dis_hi, *ref_dis_lo;
} VifPlanes;

static VifPlanes vif_planes(VifBuffer buf, unsigned w)
{
    const ptrdiff_t second = w + 2 * MAX_ALIGN;
    VifPlanes p;
    p.mu1 = (int16_t *) buf.tmp.mu1 + MAX_ALIGN;
    p.mu2 = (int16_t *) buf.tmp.mu2 + MAX_ALIGN;
    p.ref_hi = (int16_t *) buf.tmp.ref + MAX_ALIGN;
    p.ref_lo = p.ref_hi + second;
    p.dis_hi = (int16_t *) buf.tmp.dis + MAX_ALIGN;
    p.dis_lo = p.dis_hi + second;
    p.ref_dis_hi = (int16_t *) buf.tmp.ref_dis + MAX_ALIGN;
    p.ref_dis_lo = p.ref_dis_hi + second;
    return p;
}

static FORCE_INLINE void vif_mirror(int16_t *p, int w, int fwidth_half)
{
    for (int f = 1; f <= fwidth_half; ++f) {
        p[-f] = p[f];
        p[w - 1 + f] = p[w - 1 - f];
    }
}

static FORCE_INLINE void vif_mirror_planes(const VifPlanes *p, int w,
                                           int fwidth_half)
{
    vif_mirror(p->mu1, w, fwidth_half);
    vif_mirror(p->mu2, w, fwidth_half);
    vif_mirror(p->ref_hi, w, fwidth_half);
    vif_mirror(p->ref_lo, w, fwidth_half);
    vif_mirror(p->dis_hi, w, fwidth_half);
    vif_mirror(p->dis_lo, w, fwidth_half);
    vif_mirror(p->ref_dis_hi, w, fwidth_half);
    vif_mirror(p->ref_dis_lo, w, fwidth_half);
}

static FORCE_INLINE void vif_store_scalar(const VifPlanes *p, unsigned j,
                                          uint32_t mu1, uint32_t mu2,
                                          uint32_t ref, uint32_t dis,
                                          uint32_t ref_dis)
{
    p->mu1[j] = (int16_t) ((int32_t) mu1 - 32768);
    p->mu2[j] = (int16_t) ((int32_t) mu2 - 32768);
    p->ref_hi[j] = (int16_t) ((int32_t) (ref >> 16) - 32768);
    p->ref_lo[j] = (int16_t) ((int32_t) (ref & 0xffff) - 32768);
    p->dis_hi[j] = (int16_t) ((int32_t) (dis >> 16) - 32768);
    p->dis_lo[j] = (int16_t) ((int32_t) (dis & 0xffff) - 32768);
    p->ref_dis_hi[j] = (int16_t) ((int32_t) (ref_dis >> 16) - 32768);
    p->ref_dis_lo[j] = (int16_t) ((int32_t) (ref_dis & 0xffff) - 32768);
}

// Store uint32 T (unpacklo/hi_epi16 lane order) as offset hi/lo planes.
static FORCE_INLINE void vif_store_split(int16_t *hi, int16_t *lo,
                                         __m256i t_lo, __m256i t_hi)
{
    const __m256i low16 = _mm256_set1_epi32(0xffff);
    const __m256i offset16 = _mm256_set1_epi16((int16_t) 0x8000);
    _mm256_storeu_si256((__m256i *) hi,
        _mm256_xor_si256(_mm256_packus_epi32(_mm256_srli_epi32(t_lo, 16),
                                             _mm256_srli_epi32(t_hi, 16)),
                         offset16));
    _mm256_storeu_si256((__m256i *) lo,
        _mm256_xor_si256(_mm256_packus_epi32(_mm256_and_si256(t_lo, low16),
                                             _mm256_and_si256(t_hi, low16)),
                         offset16));
}

// As vif_store_split(), from T - 2^31 (int32) instead of T.
static FORCE_INLINE void vif_store_split_signed(int16_t *hi, int16_t *lo,
                                                __m256i t_lo, __m256i t_hi)
{
    const __m256i low16 = _mm256_set1_epi32(0xffff);
    _mm256_storeu_si256((__m256i *) hi,
        _mm256_packs_epi32(_mm256_srai_epi32(t_lo, 16),
                           _mm256_srai_epi32(t_hi, 16)));
    _mm256_storeu_si256((__m256i *) lo,
        _mm256_xor_si256(_mm256_packus_epi32(_mm256_and_si256(t_lo, low16),
                                             _mm256_and_si256(t_hi, low16)),
                         _mm256_set1_epi16((int16_t) 0x8000)));
}

/*
 * Horizontal filter of an offset int16 plane at columns j..j+15, as npairs
 * vpmaddwd over adjacent sample pairs starting at p + first + step * m:
 * even gets columns j, j+2, ..., j+14, odd gets j+1, ..., j+15.
 */
static FORCE_INLINE void vif_hfilt(const int16_t *p, const __m256i *coef,
                                   const int first, const int step,
                                   const int npairs,
                                   __m256i *even, __m256i *odd)
{
    __m256i e = _mm256_madd_epi16(
        _mm256_loadu_si256((const __m256i *) (p + first)), coef[0]);
    __m256i o = _mm256_madd_epi16(
        _mm256_loadu_si256((const __m256i *) (p + first + 1)), coef[0]);
    for (int m = 1; m < npairs; m++) {
        e = vif_madd_acc(e, 
                _mm256_loadu_si256((const __m256i *) (p + first + step * m)),
                coef[m]);
        o = vif_madd_acc(o, 
                _mm256_loadu_si256((const __m256i *) (p + first + step * m + 1)),
                coef[m]);
    }
    *even = e;
    *odd = o;
}

/*
 * Tap pairs for vif_hfilt(). Adjacent taps (f_2m, f_2m+1) starting at
 * -fwidth / 2, except for the 3-tap filter, whose centre tap does not fit
 * in int16 and is split over two overlapping pairs.
 */
static FORCE_INLINE void vif_hcoef(const uint16_t *filt, unsigned fwidth,
                                   __m256i *coef)
{
    if (fwidth == 3) {
        coef[0] = _mm256_set1_epi32(filt[0] | ((uint32_t) (filt[1] / 2) << 16));
        coef[1] = _mm256_set1_epi32((filt[1] - filt[1] / 2) |
                                    ((uint32_t) filt[2] << 16));
        return;
    }
    // filt[fwidth] is 0 in vif_filter1d_table
    for (unsigned m = 0; m <= fwidth / 2; m++)
        coef[m] = _mm256_set1_epi32(filt[2 * m] |
                                    ((uint32_t) filt[2 * m + 1] << 16));
}

// (a * b + 2^31) >> 32 per uint32 lane
static FORCE_INLINE __m256i vif_mulhi_round(__m256i a, __m256i b)
{
    const __m256i round = _mm256_set1_epi64x(0x80000000);
    __m256i ev = _mm256_add_epi64(_mm256_mul_epu32(a, b), round);
    __m256i od = _mm256_add_epi64(_mm256_mul_epu32(_mm256_srli_epi64(a, 32),
                                                   _mm256_srli_epi64(b, 32)),
                                  round);
    return _mm256_blend_epi32(_mm256_srli_epi64(ev, 32), od, 0xAA);
}

// (sum f * T + 2^15) >> 16 from the filtered offset hi/lo planes
static FORCE_INLINE __m256i vif_combine(__m256i fhi, __m256i flo)
{
    return _mm256_add_epi32(
        _mm256_xor_si256(fhi, _mm256_set1_epi32((int32_t) 0x80000000)),
        _mm256_srli_epi32(_mm256_add_epi32(flo,
                              _mm256_set1_epi32((int32_t) 0x80008000)), 16));
}

typedef struct VifAccum {
    int64_t num_log;
    int64_t den_log;
    int64_t num_non_log;
    int64_t den_non_log;
} VifAccum;

static FORCE_INLINE void vif_accumulate(int32_t sigma1_sq, int32_t sigma2_sq,
                                        int32_t sigma12,
                                        const uint16_t *log2_table,
                                        double vif_enhn_gain_limit,
                                        VifAccum *acc)
{
    //float equivalent of 2. (2 * 65536)
    static const int32_t sigma_nsq = 65536 << 1;

    if (sigma1_sq >= sigma_nsq) {
        /**
        * log values are taken from the look-up table generated by
        * log_generate() function which is called in integer_combo_threadfunc
        * den_val in float is log2(1 + sigma1_sq/2)
        * here it is converted to equivalent of log2(2+sigma1_sq) - log2(2) i.e log2(2*65536+sigma1_sq) - 17
        * multiplied by 2048 as log_value = log2(i)*2048 i=16384 to 65535 generated using log_value
        * x because best 16 bits are taken
        */
        acc->den_log += log2_32(log2_table, sigma_nsq + sigma1_sq) - 2048 * 17;

        if (sigma12 > 0 && sigma2_sq > 0)
        {
            // num_val = log2f(1.0f + (g * g * sigma1_sq) / (sv_sq + sigma_nsq));
            /**
            * In floating-point numerator = log2((1.0f + (g * g * sigma1_sq)/(sv_sq + sigma_nsq))
            *
            * In Fixed-point the above is converted to
            * numerator = log2((sv_sq + sigma_nsq)+(g * g * sigma1_sq))- log2(sv_sq + sigma_nsq)
            */

            const double eps = 65536 * 1.0e-10;
            double g = sigma12 / (sigma1_sq + eps); // this epsilon can go away
            int32_t sv_sq = sigma2_sq - g * sigma12;

            sv_sq = (uint32_t)(MAX(sv_sq, 0));

            g = MIN(g, vif_enhn_gain_limit);

            uint32_t numer1 = (sv_sq + sigma_nsq);
            int64_t numer1_tmp = (int64_t)((g * g * sigma1_sq)) + numer1; //numerator
            acc->num_log += log2_64(log2_table, numer1_tmp) - log2_64(log2_table, numer1);
        }
    }
    else {
        acc->num_non_log += sigma2_sq;
        acc->den_non_log += 1;
    }
}

/*
 * Horizontal pass and per-sample accumulation of one row, from the planes
 * the vertical pass left (mirrored at both ends). hfirst, hstep and hpairs
 * describe the tap pairs set up by vif_hcoef(); filt and fwidth are the
 * same filter for the scalar column tail.
 */
static FORCE_INLINE void vif_horizontal_row(const VifPlanes *p, unsigned w,
                                            const __m256i *hcoef,
                                            const int hfirst, const int hstep,
                                            const int hpairs,
                                            const uint16_t *filt,
                                            unsigned fwidth,
                                            const uint16_t *log2_table,
                                            double vif_enhn_gain_limit,
                                            VifAccum *accum)
{
    const __m256i zero = _mm256_setzero_si256();
    const __m256i sign = _mm256_set1_epi32((int32_t) 0x80000000);
    VifAccum acc_row = *accum, *acc = &acc_row;

    // variables used for 16 sample block vif computation
    ALIGNED(32) uint32_t xx[16];
    ALIGNED(32) uint32_t yy[16];
    ALIGNED(32) uint32_t xy[16];

    const unsigned n = w >> 4;
    for (unsigned j = 0; j < n << 4; j += 16) {
        // filter and consume one quantity at a time to keep few vectors live;
        // even columns go to [0, 8), odd columns to [8, 16), and the
        // accumulation below does not depend on the order
        __m256i m1[2], m2[2], mu1_sq[2], mu2_sq[2], mu1_mu2[2], hi[2], lo[2];
        vif_hfilt(p->mu1 + j, hcoef, hfirst, hstep, hpairs, &m1[0], &m1[1]);
        vif_hfilt(p->mu2 + j, hcoef, hfirst, hstep, hpairs, &m2[0], &m2[1]);
        for (int e = 0; e < 2; e++) {
            const __m256i mu1_val = _mm256_xor_si256(m1[e], sign);
            const __m256i mu2_val = _mm256_xor_si256(m2[e], sign);
            mu1_sq[e] = vif_mulhi_round(mu1_val, mu1_val);
            mu2_sq[e] = vif_mulhi_round(mu2_val, mu2_val);
            mu1_mu2[e] = vif_mulhi_round(mu1_val, mu2_val);
        }

        vif_hfilt(p->ref_hi + j, hcoef, hfirst, hstep, hpairs, &hi[0], &hi[1]);
        vif_hfilt(p->ref_lo + j, hcoef, hfirst, hstep, hpairs, &lo[0], &lo[1]);
        for (int e = 0; e < 2; e++)
            _mm256_store_si256((__m256i *) &xx[8 * e],
                _mm256_sub_epi32(vif_combine(hi[e], lo[e]), mu1_sq[e]));

        vif_hfilt(p->dis_hi + j, hcoef, hfirst, hstep, hpairs, &hi[0], &hi[1]);
        vif_hfilt(p->dis_lo + j, hcoef, hfirst, hstep, hpairs, &lo[0], &lo[1]);
        for (int e = 0; e < 2; e++)
            _mm256_store_si256((__m256i *) &yy[8 * e], _mm256_max_epi32(
                _mm256_sub_epi32(vif_combine(hi[e], lo[e]), mu2_sq[e]), zero));

        vif_hfilt(p->ref_dis_hi + j, hcoef, hfirst, hstep, hpairs, &hi[0], &hi[1]);
        vif_hfilt(p->ref_dis_lo + j, hcoef, hfirst, hstep, hpairs, &lo[0], &lo[1]);
        for (int e = 0; e < 2; e++)
            _mm256_store_si256((__m256i *) &xy[8 * e],
                _mm256_sub_epi32(vif_combine(hi[e], lo[e]), mu1_mu2[e]));

        for (unsigned int b = 0; b < 16; b++)
            vif_accumulate(xx[b], yy[b], xy[b], log2_table, vif_enhn_gain_limit, acc);
    }

    // remaining columns, scalar, from the same planes
    for (unsigned j = n << 4; j < w; ++j) {
        uint32_t accum_mu1 = 0;
        uint32_t accum_mu2 = 0;
        uint64_t accum_ref = 0;
        uint64_t accum_dis = 0;
        uint64_t accum_ref_dis = 0;
        for (unsigned fj = 0; fj < fwidth; ++fj) {
            const int jj = j - fwidth / 2 + fj;
            const uint32_t f = filt[fj];
            accum_mu1 += f * (uint32_t) (p->mu1[jj] + 32768);
            accum_mu2 += f * (uint32_t) (p->mu2[jj] + 32768);
            accum_ref += f * (uint64_t) (((uint32_t) (p->ref_hi[jj] + 32768) << 16) |
                                         (uint32_t) (p->ref_lo[jj] + 32768));
            accum_dis += f * (uint64_t) (((uint32_t) (p->dis_hi[jj] + 32768) << 16) |
                                         (uint32_t) (p->dis_lo[jj] + 32768));
            accum_ref_dis += f * (uint64_t) (((uint32_t) (p->ref_dis_hi[jj] + 32768) << 16) |
                                             (uint32_t) (p->ref_dis_lo[jj] + 32768));
        }
        const uint32_t mu1_sq_val = (uint32_t) ((((uint64_t) accum_mu1 * accum_mu1) + 2147483648) >> 32);
        const uint32_t mu2_sq_val = (uint32_t) ((((uint64_t) accum_mu2 * accum_mu2) + 2147483648) >> 32);
        const uint32_t mu1_mu2_val = (uint32_t) ((((uint64_t) accum_mu1 * accum_mu2) + 2147483648) >> 32);
        const uint32_t xx_filt_val = (uint32_t) ((accum_ref + 32768) >> 16);
        const uint32_t yy_filt_val = (uint32_t) ((accum_dis + 32768) >> 16);
        const uint32_t xy_filt_val = (uint32_t) ((accum_ref_dis + 32768) >> 16);

        const int32_t sigma1_sq = (int32_t) (xx_filt_val - mu1_sq_val);
        const int32_t sigma2_sq = MAX((int32_t) (yy_filt_val - mu2_sq_val), 0);
        const int32_t sigma12 = (int32_t) (xy_filt_val - mu1_mu2_val);

        vif_accumulate(sigma1_sq, sigma2_sq, sigma12, log2_table,
                       vif_enhn_gain_limit, acc);
    }
    *accum = acc_row;
}

void VIF_X86_FN(vif_statistic_8)(struct VifPublicState *s, float *num, float *den, unsigned w, unsigned h) {
    assert(vif_filter1d_width[0] == 17);
    static const unsigned fwidth = 17;
    const uint16_t *vif_filt_s0 = vif_filter1d_table[0];
    VifBuffer buf = s->buf;
    const VifPlanes p = vif_planes(buf, w);
    VifAccum acc = { 0 };

    // vertical: symmetric rows k and 16 - k share a tap, (f_k, f_k) pairs;
    // the centre row is paired with zeros, (f_8, 0)
    __m256i vcoef[9];
    for (unsigned k = 0; k < 8; k++)
        vcoef[k] = _mm256_set1_epi32(vif_filt_s0[k] * 0x10001);
    vcoef[8] = _mm256_set1_epi32(vif_filt_s0[8]);
    __m256i hcoef[9];
    vif_hcoef(vif_filt_s0, fwidth, hcoef);

    const __m256i zero = _mm256_setzero_si256();
    const __m256i offset16 = _mm256_set1_epi16((int16_t) 0x8000);
    // mu = (sum + 128) >> 8, stored minus 32768
    const __m256i mu_round = _mm256_set1_epi32(128 - (32768 << 8));

    // loop on row, each iteration produces one line of output
    for (unsigned i = 0; i < h; ++i) {
        // Filter vertically
        // First consider all blocks of 16 elements until it's not possible anymore
        unsigned n = w >> 4;
        for (unsigned jj = 0; jj < n << 4; jj += 16) {
            const uint8_t *ref = (uint8_t *) buf.ref + jj;
            const uint8_t *dis = (uint8_t *) buf.dis + jj;

            __m256i r = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *) (ref + buf.stride * i)));
            __m256i d = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *) (dis + buf.stride * i)));
            __m256i rr = _mm256_xor_si256(_mm256_mullo_epi16(r, r), offset16);
            __m256i dd = _mm256_xor_si256(_mm256_mullo_epi16(d, d), offset16);
            __m256i rd = _mm256_xor_si256(_mm256_mullo_epi16(r, d), offset16);

            __m256i mu1_lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(r, zero), vcoef[8]);
            __m256i mu1_hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(r, zero), vcoef[8]);
            __m256i mu2_lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(d, zero), vcoef[8]);
            __m256i mu2_hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(d, zero), vcoef[8]);
            __m256i rr_lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(rr, zero), vcoef[8]);
            __m256i rr_hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(rr, zero), vcoef[8]);
            __m256i dd_lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(dd, zero), vcoef[8]);
            __m256i dd_hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(dd, zero), vcoef[8]);
            __m256i rd_lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(rd, zero), vcoef[8]);
            __m256i rd_hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(rd, zero), vcoef[8]);

            for (unsigned k = 0; k < fwidth / 2; k++) {
                const ptrdiff_t ra = buf.stride * ((int) i - (int) (fwidth / 2) + (int) k);
                const ptrdiff_t rb = buf.stride * ((int) i + (int) (fwidth / 2) - (int) k);
                const __m256i f = vcoef[k];

                __m256i r0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *) (ref + ra)));
                __m256i r1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *) (ref + rb)));
                __m256i d0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *) (dis + ra)));
                __m256i d1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *) (dis + rb)));

                mu1_lo = vif_madd_acc(mu1_lo, _mm256_unpacklo_epi16(r0, r1), f);
                mu1_hi = vif_madd_acc(mu1_hi, _mm256_unpackhi_epi16(r0, r1), f);
                mu2_lo = vif_madd_acc(mu2_lo, _mm256_unpacklo_epi16(d0, d1), f);
                mu2_hi = vif_madd_acc(mu2_hi, _mm256_unpackhi_epi16(d0, d1), f);

                __m256i rr0 = _mm256_xor_si256(_mm256_mullo_epi16(r0, r0), offset16);
                __m256i rr1 = _mm256_xor_si256(_mm256_mullo_epi16(r1, r1), offset16);
                __m256i dd0 = _mm256_xor_si256(_mm256_mullo_epi16(d0, d0), offset16);
                __m256i dd1 = _mm256_xor_si256(_mm256_mullo_epi16(d1, d1), offset16);
                __m256i rd0 = _mm256_xor_si256(_mm256_mullo_epi16(r0, d0), offset16);
                __m256i rd1 = _mm256_xor_si256(_mm256_mullo_epi16(r1, d1), offset16);

                rr_lo = vif_madd_acc(rr_lo, _mm256_unpacklo_epi16(rr0, rr1), f);
                rr_hi = vif_madd_acc(rr_hi, _mm256_unpackhi_epi16(rr0, rr1), f);
                dd_lo = vif_madd_acc(dd_lo, _mm256_unpacklo_epi16(dd0, dd1), f);
                dd_hi = vif_madd_acc(dd_hi, _mm256_unpackhi_epi16(dd0, dd1), f);
                rd_lo = vif_madd_acc(rd_lo, _mm256_unpacklo_epi16(rd0, rd1), f);
                rd_hi = vif_madd_acc(rd_hi, _mm256_unpackhi_epi16(rd0, rd1), f);
            }

            mu1_lo = _mm256_srai_epi32(_mm256_add_epi32(mu1_lo, mu_round), 8);
            mu1_hi = _mm256_srai_epi32(_mm256_add_epi32(mu1_hi, mu_round), 8);
            mu2_lo = _mm256_srai_epi32(_mm256_add_epi32(mu2_lo, mu_round), 8);
            mu2_hi = _mm256_srai_epi32(_mm256_add_epi32(mu2_hi, mu_round), 8);
            _mm256_storeu_si256((__m256i *) (p.mu1 + jj), _mm256_packs_epi32(mu1_lo, mu1_hi));
            _mm256_storeu_si256((__m256i *) (p.mu2 + jj), _mm256_packs_epi32(mu2_lo, mu2_hi));

            vif_store_split_signed(p.ref_hi + jj, p.ref_lo + jj, rr_lo, rr_hi);
            vif_store_split_signed(p.dis_hi + jj, p.dis_lo + jj, dd_lo, dd_hi);
            vif_store_split_signed(p.ref_dis_hi + jj, p.ref_dis_lo + jj, rd_lo, rd_hi);
        }

        // Then consider the remaining elements individually
        for (unsigned j = n << 4; j < w; ++j) {
            uint32_t accum_mu1 = 0;
            uint32_t accum_mu2 = 0;
            uint32_t accum_ref = 0;
            uint32_t accum_dis = 0;
            uint32_t accum_ref_dis = 0;

            for (unsigned fi = 0; fi < fwidth; ++fi) {
                int ii = i - fwidth / 2;
                int ii_check = ii + fi;
                const uint16_t fcoeff = vif_filt_s0[fi];
                const uint8_t *ref = (uint8_t*)buf.ref;
                const uint8_t *dis = (uint8_t*)buf.dis;
                uint16_t imgcoeff_ref = ref[ii_check * buf.stride + j];
                uint16_t imgcoeff_dis = dis[ii_check * buf.stride + j];
                uint32_t img_coeff_ref = fcoeff * (uint32_t)imgcoeff_ref;
                uint32_t img_coeff_dis = fcoeff * (uint32_t)imgcoeff_dis;
                accum_mu1 += img_coeff_ref;
                accum_mu2 += img_coeff_dis;
                accum_ref += img_coeff_ref * (uint32_t)imgcoeff_ref;
                accum_dis += img_coeff_dis * (uint32_t)imgcoeff_dis;
                accum_ref_dis += img_coeff_ref * (uint32_t)imgcoeff_dis;
            }

            vif_store_scalar(&p, j, (accum_mu1 + 128) >> 8, (accum_mu2 + 128) >> 8,
                             accum_ref, accum_dis, accum_ref_dis);
        }

        vif_mirror_planes(&p, w, fwidth / 2);

        //HORIZONTAL
        vif_horizontal_row(&p, w, hcoef, -8, 2, 9, vif_filt_s0, fwidth,
                           s->log2_table, s->vif_enhn_gain_limit, &acc);
    }

    //log has to be divided by 2048 as log_value = log2(i*2048)  i=16384 to 65535
    //num[0] = accum_num_log / 2048.0 + (accum_den_non_log - (accum_num_non_log / 65536.0) / (255.0*255.0));
    //den[0] = accum_den_log / 2048.0 + accum_den_non_log;

    //changed calculation to increase performance
    num[0] = acc.num_log / 2048.0 + (acc.den_non_log - ((acc.num_non_log) / 16384.0) / (65025.0));
    den[0] = acc.den_log / 2048.0 + acc.den_non_log;

}

/*
 * Vertical pass of vif_statistic_16 for 16 columns. Samples and products
 * can use all 16 and 32 bits, so every product is split into its low and
 * high 16 bits (_mm256_mullo_epi16 / _mm256_mulhi_epu16) and, like the
 * samples, offset by -32768; symmetric rows share a tap and are filtered
 * as (f_k, f_k) vpmaddwd pairs, the centre row as (x, x) with its tap split
 * in two. The ref^2 and mu passes and the dis^2 / ref*dis pass are
 * separate so that each keeps its eight accumulators in registers.
 */
static FORCE_INLINE void vif16_vertical_block(const uint16_t *ref,
                                              const uint16_t *dis,
                                              ptrdiff_t stride, const int fh,
                                              const __m256i *vcoef,
                                              int shift_vp, int round_vp,
                                              int shift_vp_sq, int round_vp_sq,
                                              const VifPlanes *p, unsigned j)
{
    const __m256i offset16 = _mm256_set1_epi16((int16_t) 0x8000);
    const __m256i sign = _mm256_set1_epi32((int32_t) 0x80000000);
    const __m128i s_vp = _mm_cvtsi32_si128(shift_vp);
    const __m128i s_sq = _mm_cvtsi32_si128(shift_vp_sq);
    const __m128i s_sq_hi = _mm_cvtsi32_si128(16 - shift_vp_sq);
    const __m256i rnd_vp = _mm256_set1_epi32(round_vp);
    const __m256i rnd_sq = _mm256_set1_epi32(round_vp_sq);

// accumulate row pair (a, b) or the centre row (a, a) with tap pair f
#define VIF16_PAIR(acc_lo, acc_hi, a, b, f)                                   \
    do {                                                                      \
        acc_lo = vif_madd_acc(acc_lo, _mm256_unpacklo_epi16(a, b), f);        \
        acc_hi = vif_madd_acc(acc_hi, _mm256_unpackhi_epi16(a, b), f);        \
    } while (0)

// T = (sum + round) >> shift from the offset hi/lo sums, as uint32
#define VIF16_SQ(h, l)                                                        \
    _mm256_add_epi32(_mm256_sll_epi32(_mm256_xor_si256(h, sign), s_sq_hi),    \
        _mm256_srl_epi32(_mm256_add_epi32(_mm256_xor_si256(l, sign), rnd_sq), \
                         s_sq))

// mu = (sum + round) >> shift, minus 32768
#define VIF16_MU(acc)                                                         \
    _mm256_sub_epi32(_mm256_srl_epi32(_mm256_add_epi32(                       \
        _mm256_xor_si256(acc, sign), rnd_vp), s_vp), _mm256_set1_epi32(32768))

    {
        __m256i mu1_lo = _mm256_setzero_si256(), mu1_hi = mu1_lo;
        __m256i mu2_lo = mu1_lo, mu2_hi = mu1_lo;
        __m256i rrl_lo = mu1_lo, rrl_hi = mu1_lo, rrh_lo = mu1_lo, rrh_hi = mu1_lo;
        for (int k = 0; k <= fh; k++) {
            const __m256i f = vcoef[k];
            const __m256i r0 = _mm256_loadu_si256((const __m256i *) (ref + stride * (k - fh)));
            const __m256i r1 = _mm256_loadu_si256((const __m256i *) (ref + stride * (fh - k)));
            const __m256i d0 = _mm256_loadu_si256((const __m256i *) (dis + stride * (k - fh)));
            const __m256i d1 = _mm256_loadu_si256((const __m256i *) (dis + stride * (fh - k)));
            VIF16_PAIR(mu1_lo, mu1_hi, _mm256_xor_si256(r0, offset16),
                       _mm256_xor_si256(r1, offset16), f);
            VIF16_PAIR(mu2_lo, mu2_hi, _mm256_xor_si256(d0, offset16),
                       _mm256_xor_si256(d1, offset16), f);
            VIF16_PAIR(rrl_lo, rrl_hi,
                       _mm256_xor_si256(_mm256_mullo_epi16(r0, r0), offset16),
                       _mm256_xor_si256(_mm256_mullo_epi16(r1, r1), offset16), f);
            VIF16_PAIR(rrh_lo, rrh_hi,
                       _mm256_xor_si256(_mm256_mulhi_epu16(r0, r0), offset16),
                       _mm256_xor_si256(_mm256_mulhi_epu16(r1, r1), offset16), f);
        }
        _mm256_storeu_si256((__m256i *) (p->mu1 + j),
                            _mm256_packs_epi32(VIF16_MU(mu1_lo), VIF16_MU(mu1_hi)));
        _mm256_storeu_si256((__m256i *) (p->mu2 + j),
                            _mm256_packs_epi32(VIF16_MU(mu2_lo), VIF16_MU(mu2_hi)));
        vif_store_split(p->ref_hi + j, p->ref_lo + j,
                        VIF16_SQ(rrh_lo, rrl_lo), VIF16_SQ(rrh_hi, rrl_hi));
    }
    {
        __m256i ddl_lo = _mm256_setzero_si256(), ddl_hi = ddl_lo;
        __m256i ddh_lo = ddl_lo, ddh_hi = ddl_lo;
        __m256i rdl_lo = ddl_lo, rdl_hi = ddl_lo, rdh_lo = ddl_lo, rdh_hi = ddl_lo;
        for (int k = 0; k <= fh; k++) {
            const __m256i f = vcoef[k];
            const __m256i r0 = _mm256_loadu_si256((const __m256i *) (ref + stride * (k - fh)));
            const __m256i r1 = _mm256_loadu_si256((const __m256i *) (ref + stride * (fh - k)));
            const __m256i d0 = _mm256_loadu_si256((const __m256i *) (dis + stride * (k - fh)));
            const __m256i d1 = _mm256_loadu_si256((const __m256i *) (dis + stride * (fh - k)));
            VIF16_PAIR(ddl_lo, ddl_hi,
                       _mm256_xor_si256(_mm256_mullo_epi16(d0, d0), offset16),
                       _mm256_xor_si256(_mm256_mullo_epi16(d1, d1), offset16), f);
            VIF16_PAIR(ddh_lo, ddh_hi,
                       _mm256_xor_si256(_mm256_mulhi_epu16(d0, d0), offset16),
                       _mm256_xor_si256(_mm256_mulhi_epu16(d1, d1), offset16), f);
            VIF16_PAIR(rdl_lo, rdl_hi,
                       _mm256_xor_si256(_mm256_mullo_epi16(r0, d0), offset16),
                       _mm256_xor_si256(_mm256_mullo_epi16(r1, d1), offset16), f);
            VIF16_PAIR(rdh_lo, rdh_hi,
                       _mm256_xor_si256(_mm256_mulhi_epu16(r0, d0), offset16),
                       _mm256_xor_si256(_mm256_mulhi_epu16(r1, d1), offset16), f);
        }
        vif_store_split(p->dis_hi + j, p->dis_lo + j,
                        VIF16_SQ(ddh_lo, ddl_lo), VIF16_SQ(ddh_hi, ddl_hi));
        vif_store_split(p->ref_dis_hi + j, p->ref_dis_lo + j,
                        VIF16_SQ(rdh_lo, rdl_lo), VIF16_SQ(rdh_hi, rdl_hi));
    }

#undef VIF16_PAIR
#undef VIF16_SQ
#undef VIF16_MU
}

static FORCE_INLINE void vif_statistic_16_scale(struct VifPublicState *s,
                                                float *num, float *den,
                                                unsigned w, unsigned h,
                                                int bpc, const int scale,
                                                const int hfirst,
                                                const int hstep,
                                                const int hpairs)
{
    const unsigned fwidth = vif_filter1d_width[scale];
    const uint16_t *vif_filt = vif_filter1d_table[scale];
    const int fh = fwidth / 2;
    VifBuffer buf = s->buf;
    const ptrdiff_t stride = buf.stride / sizeof(uint16_t);
    const VifPlanes p = vif_planes(buf, w);
    VifAccum acc = { 0 };

    int32_t add_shift_round_VP, shift_VP;
    int32_t add_shift_round_VP_sq, shift_VP_sq;
    if (scale == 0) {
        shift_VP = bpc;
        add_shift_round_VP = 1 << (bpc - 1);
        shift_VP_sq = (bpc - 8) * 2;
        add_shift_round_VP_sq = (bpc == 8) ? 0 : 1 << (shift_VP_sq - 1);
    } else {
        shift_VP = 16;
        add_shift_round_VP = 32768;
        shift_VP_sq = 16;
        add_shift_round_VP_sq = 32768;
    }

    // vertical: rows k and fwidth - 1 - k share a tap, (f_k, f_k); the
    // centre row is taken twice, (x, x), with its tap split in two
    __m256i vcoef[9];
    for (int k = 0; k < fh; k++)
        vcoef[k] = _mm256_set1_epi32(vif_filt[k] * 0x10001);
    vcoef[fh] = _mm256_set1_epi32((vif_filt[fh] / 2) |
                                  ((uint32_t) (vif_filt[fh] - vif_filt[fh] / 2) << 16));
    __m256i hcoef[9];
    vif_hcoef(vif_filt, fwidth, hcoef);

    for (unsigned i = 0; i < h; ++i) {
        // VERTICAL
        const uint16_t *ref = (uint16_t *) buf.ref + stride * i;
        const uint16_t *dis = (uint16_t *) buf.dis + stride * i;
        unsigned n = w >> 4;
        for (unsigned j = 0; j < n << 4; j += 16) {
            vif16_vertical_block(ref + j, dis + j, stride, fh, vcoef,
                                 shift_VP, add_shift_round_VP,
                                 shift_VP_sq, add_shift_round_VP_sq, &p, j);
        }

        for (unsigned j = n << 4; j < w; ++j) {
            uint32_t accum_mu1 = 0;
            uint32_t accum_mu2 = 0;
            uint64_t accum_ref = 0;
            uint64_t accum_dis = 0;
            uint64_t accum_ref_dis = 0;
            for (unsigned fi = 0; fi < fwidth; ++fi) {
                const ptrdiff_t row = stride * ((int) fi - fh);
                const uint16_t fcoeff = vif_filt[fi];
                uint16_t imgcoeff_ref = ref[row + j];
                uint16_t imgcoeff_dis = dis[row + j];
                uint32_t img_coeff_ref = fcoeff * (uint32_t)imgcoeff_ref;
                uint32_t img_coeff_dis = fcoeff * (uint32_t)imgcoeff_dis;
                accum_mu1 += img_coeff_ref;
                accum_mu2 += img_coeff_dis;
                accum_ref += img_coeff_ref * (uint64_t)imgcoeff_ref;
                accum_dis += img_coeff_dis * (uint64_t)imgcoeff_dis;
                accum_ref_dis += img_coeff_ref * (uint64_t)imgcoeff_dis;
            }
            vif_store_scalar(&p, j,
                (uint16_t)((accum_mu1 + add_shift_round_VP) >> shift_VP),
                (uint16_t)((accum_mu2 + add_shift_round_VP) >> shift_VP),
                (uint32_t)((accum_ref + add_shift_round_VP_sq) >> shift_VP_sq),
                (uint32_t)((accum_dis + add_shift_round_VP_sq) >> shift_VP_sq),
                (uint32_t)((accum_ref_dis + add_shift_round_VP_sq) >> shift_VP_sq));
        }

        vif_mirror_planes(&p, w, fh);

        //HORIZONTAL
        vif_horizontal_row(&p, w, hcoef, hfirst, hstep, hpairs, vif_filt,
                           fwidth, s->log2_table, s->vif_enhn_gain_limit, &acc);
    }

    num[0] = acc.num_log / 2048.0 + (acc.den_non_log - ((acc.num_non_log) / 16384.0) / (65025.0));
    den[0] = acc.den_log / 2048.0 + acc.den_non_log;
}

void VIF_X86_FN(vif_statistic_16)(struct VifPublicState *s, float *num, float *den, unsigned w, unsigned h, int bpc, int scale) {
    // hfirst/hstep/hpairs as set up by vif_hcoef() for the scale's filter
    switch (scale) {
    case 0: vif_statistic_16_scale(s, num, den, w, h, bpc, 0, -8, 2, 9); break;
    case 1: vif_statistic_16_scale(s, num, den, w, h, bpc, 1, -4, 2, 5); break;
    case 2: vif_statistic_16_scale(s, num, den, w, h, bpc, 2, -2, 2, 3); break;
    case 3: vif_statistic_16_scale(s, num, den, w, h, bpc, 3, -1, 1, 2); break;
    }
}

// Even columns j, j+2, ..., j+14 of vif_hfilt(), for decimating filters.
static FORCE_INLINE __m256i vif_hfilt_even(const int16_t *p, const __m256i *coef,
                                           const int first, const int step,
                                           const int npairs)
{
    __m256i e = _mm256_madd_epi16(
        _mm256_loadu_si256((const __m256i *) (p + first)), coef[0]);
    for (int m = 1; m < npairs; m++)
        e = vif_madd_acc(e, 
                _mm256_loadu_si256((const __m256i *) (p + first + step * m)),
                coef[m]);
    return e;
}

/*
 * Horizontal half of the subsampling filters: output m of row i is column
 * 2m of the filtered offset plane p, (sum f * mu + 2^15) >> 16, written to
 * dst[m]. Only the columns that survive the decimation are filtered.
 */
static FORCE_INLINE void vif_subsample_horizontal(const int16_t *p, uint16_t *dst,
                                                  unsigned w,
                                                  const __m256i *hcoef,
                                                  const int hfirst,
                                                  const int hstep,
                                                  const int hpairs,
                                                  const uint16_t *filt,
                                                  unsigned fwidth)
{
    const __m256i sign = _mm256_set1_epi32((int32_t) 0x80000000);
    const __m256i round = _mm256_set1_epi32(32768);
    const unsigned nout = w / 2;
    unsigned m = 0;
    for (; m + 8 <= nout; m += 8) {
        __m256i v = vif_hfilt_even(p + 2 * m, hcoef, hfirst, hstep, hpairs);
        v = _mm256_srli_epi32(_mm256_add_epi32(_mm256_xor_si256(v, sign), round), 16);
        v = _mm256_permute4x64_epi64(_mm256_packus_epi32(v, v), 0x08);
        _mm_storeu_si128((__m128i *) (dst + m), _mm256_castsi256_si128(v));
    }
    for (; m < nout; ++m) {
        uint32_t accum = 0;
        for (unsigned fj = 0; fj < fwidth; ++fj) {
            const int jj = (int) (2 * m + fj) - (int) (fwidth / 2);
            accum += filt[fj] * (uint32_t) (p[jj] + 32768);
        }
        dst[m] = (uint16_t) ((accum + 32768) >> 16);
    }
}

/*
 * The subsampling filters in the same offset int16 plane layout as the
 * statistics: the vertical pass of every second row goes into an offset
 * plane (tmp.ref_convol / tmp.dis_convol), and the horizontal pass only
 * computes the even columns that the decimation keeps. The results are
 * written decimated into mu1/mu2 and copied into ref/dis as before.
 */
void VIF_X86_FN(vif_subsample_rd_8)(VifBuffer buf, unsigned w, unsigned h) {
    const unsigned fwidth = vif_filter1d_width[1];
    const uint16_t *vif_filt_s1 = vif_filter1d_table[1];
    const int fh = fwidth / 2;
    const ptrdiff_t mu_stride = buf.stride_16 / sizeof(uint16_t);
    int16_t *pr = (int16_t *) buf.tmp.ref_convol + MAX_ALIGN;
    int16_t *pd = (int16_t *) buf.tmp.dis_convol + MAX_ALIGN;

    // symmetric rows share a tap, (f_k, f_k); the centre row is (x, 0)
    __m256i vcoef[5];
    for (int k = 0; k < fh; k++)
        vcoef[k] = _mm256_set1_epi32(vif_filt_s1[k] * 0x10001);
    vcoef[fh] = _mm256_set1_epi32(vif_filt_s1[fh]);
    __m256i hcoef[5];
    vif_hcoef(vif_filt_s1, fwidth, hcoef);

    const __m256i zero = _mm256_setzero_si256();
    // mu = (sum + 128) >> 8, stored minus 32768
    const __m256i mu_round = _mm256_set1_epi32(128 - (32768 << 8));

    for (unsigned i = 0; i < h / 2; i++) {
        const uint8_t *ref = (uint8_t *) buf.ref + buf.stride * (2 * i);
        const uint8_t *dis = (uint8_t *) buf.dis + buf.stride * (2 * i);

        // VERTICAL
        unsigned j = 0;
        for (; j + 16 <= w; j += 16) {
            const __m256i r = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *) (ref + j)));
            const __m256i d = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *) (dis + j)));
            __m256i r_lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(r, zero), vcoef[fh]);
            __m256i r_hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(r, zero), vcoef[fh]);
            __m256i d_lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(d, zero), vcoef[fh]);
            __m256i d_hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(d, zero), vcoef[fh]);
            for (int k = 0; k < fh; k++) {
                const ptrdiff_t ra = buf.stride * (k - fh);
                const ptrdiff_t rb = buf.stride * (fh - k);
                const __m256i r0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *) (ref + ra + j)));
                const __m256i r1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *) (ref + rb + j)));
                const __m256i d0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *) (dis + ra + j)));
                const __m256i d1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *) (dis + rb + j)));
                r_lo = vif_madd_acc(r_lo, _mm256_unpacklo_epi16(r0, r1), vcoef[k]);
                r_hi = vif_madd_acc(r_hi, _mm256_unpackhi_epi16(r0, r1), vcoef[k]);
                d_lo = vif_madd_acc(d_lo, _mm256_unpacklo_epi16(d0, d1), vcoef[k]);
                d_hi = vif_madd_acc(d_hi, _mm256_unpackhi_epi16(d0, d1), vcoef[k]);
            }
            _mm256_storeu_si256((__m256i *) (pr + j), _mm256_packs_epi32(
                _mm256_srai_epi32(_mm256_add_epi32(r_lo, mu_round), 8),
                _mm256_srai_epi32(_mm256_add_epi32(r_hi, mu_round), 8)));
            _mm256_storeu_si256((__m256i *) (pd + j), _mm256_packs_epi32(
                _mm256_srai_epi32(_mm256_add_epi32(d_lo, mu_round), 8),
                _mm256_srai_epi32(_mm256_add_epi32(d_hi, mu_round), 8)));
        }
        for (; j < w; ++j) {
            uint32_t accum_ref = 0;
            uint32_t accum_dis = 0;
            for (unsigned fi = 0; fi < fwidth; ++fi) {
                const ptrdiff_t row = buf.stride * ((int) fi - fh);
                accum_ref += vif_filt_s1[fi] * (uint32_t) ref[row + j];
                accum_dis += vif_filt_s1[fi] * (uint32_t) dis[row + j];
            }
            pr[j] = (int16_t) ((int32_t) ((accum_ref + 128) >> 8) - 32768);
            pd[j] = (int16_t) ((int32_t) ((accum_dis + 128) >> 8) - 32768);
        }

        vif_mirror(pr, w, fh);
        vif_mirror(pd, w, fh);

        // HORIZONTAL
        vif_subsample_horizontal(pr, buf.mu1 + i * mu_stride, w, hcoef, -4, 2, 5,
                                 vif_filt_s1, fwidth);
        vif_subsample_horizontal(pd, buf.mu2 + i * mu_stride, w, hcoef, -4, 2, 5,
                                 vif_filt_s1, fwidth);
    }
    copy_and_pad(buf, w, h, 0);
}

static FORCE_INLINE void vif_subsample_rd_16_scale(VifBuffer buf, unsigned w,
                                                   unsigned h, int scale,
                                                   int bpc, const int hfirst,
                                                   const int hstep,
                                                   const int hpairs)
{
    const unsigned fwidth = vif_filter1d_width[scale + 1];
    const uint16_t *vif_filt = vif_filter1d_table[scale + 1];
    const int fh = fwidth / 2;
    const ptrdiff_t stride = buf.stride / sizeof(uint16_t);
    const ptrdiff_t mu_stride = buf.stride_16 / sizeof(uint16_t);
    int16_t *pr = (int16_t *) buf.tmp.ref_convol + MAX_ALIGN;
    int16_t *pd = (int16_t *) buf.tmp.dis_convol + MAX_ALIGN;

    int32_t add_shift_round_VP, shift_VP;
    if (scale == 0) {
        add_shift_round_VP = 1 << (bpc - 1);
        shift_VP = bpc;
    } else {
        add_shift_round_VP = 32768;
        shift_VP = 16;
    }

    // symmetric rows share a tap, (f_k, f_k); the centre row is taken twice,
    // (x, x), with its tap split in two (the 3-tap centre does not fit int16)
    __m256i vcoef[5];
    for (int k = 0; k < fh; k++)
        vcoef[k] = _mm256_set1_epi32(vif_filt[k] * 0x10001);
    vcoef[fh] = _mm256_set1_epi32((vif_filt[fh] / 2) |
                                  ((uint32_t) (vif_filt[fh] - vif_filt[fh] / 2) << 16));
    __m256i hcoef[5];
    vif_hcoef(vif_filt, fwidth, hcoef);

    const __m256i offset16 = _mm256_set1_epi16((int16_t) 0x8000);
    const __m256i sign = _mm256_set1_epi32((int32_t) 0x80000000);
    const __m256i low16 = _mm256_set1_epi32(0xffff);
    const __m256i rnd_vp = _mm256_set1_epi32(add_shift_round_VP);
    const __m256i off_mu = _mm256_set1_epi32(32768);
    const __m128i s_vp = _mm_cvtsi32_si128(shift_VP);

    for (unsigned i = 0; i < h / 2; i++) {
        const uint16_t *ref = (uint16_t *) buf.ref + stride * (2 * i);
        const uint16_t *dis = (uint16_t *) buf.dis + stride * (2 * i);

        // VERTICAL
        unsigned j = 0;
        for (; j + 16 <= w; j += 16) {
            __m256i r_lo = _mm256_setzero_si256(), r_hi = r_lo, d_lo = r_lo, d_hi = r_lo;
            for (int k = 0; k <= fh; k++) {
                const __m256i r0 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *) (ref + stride * (k - fh) + j)), offset16);
                const __m256i r1 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *) (ref + stride * (fh - k) + j)), offset16);
                const __m256i d0 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *) (dis + stride * (k - fh) + j)), offset16);
                const __m256i d1 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *) (dis + stride * (fh - k) + j)), offset16);
                r_lo = vif_madd_acc(r_lo, _mm256_unpacklo_epi16(r0, r1), vcoef[k]);
                r_hi = vif_madd_acc(r_hi, _mm256_unpackhi_epi16(r0, r1), vcoef[k]);
                d_lo = vif_madd_acc(d_lo, _mm256_unpacklo_epi16(d0, d1), vcoef[k]);
                d_hi = vif_madd_acc(d_hi, _mm256_unpackhi_epi16(d0, d1), vcoef[k]);
            }
            // mu = (uint16_t)((sum + round) >> shift), stored minus 32768
#define VIF_SUB16_MU(acc)                                                     \
            _mm256_sub_epi32(_mm256_and_si256(_mm256_srl_epi32(               \
                _mm256_add_epi32(_mm256_xor_si256(acc, sign), rnd_vp), s_vp), \
                low16), off_mu)
            _mm256_storeu_si256((__m256i *) (pr + j),
                _mm256_packs_epi32(VIF_SUB16_MU(r_lo), VIF_SUB16_MU(r_hi)));
            _mm256_storeu_si256((__m256i *) (pd + j),
                _mm256_packs_epi32(VIF_SUB16_MU(d_lo), VIF_SUB16_MU(d_hi)));
#undef VIF_SUB16_MU
        }
        for (; j < w; ++j) {
            uint32_t accum_ref = 0;
            uint32_t accum_dis = 0;
            for (unsigned fi = 0; fi < fwidth; ++fi) {
                const ptrdiff_t row = stride * ((int) fi - fh);
                accum_ref += vif_filt[fi] * (uint32_t) ref[row + j];
                accum_dis += vif_filt[fi] * (uint32_t) dis[row + j];
            }
            pr[j] = (int16_t) ((int32_t) (uint16_t) ((accum_ref + add_shift_round_VP) >> shift_VP) - 32768);
            pd[j] = (int16_t) ((int32_t) (uint16_t) ((accum_dis + add_shift_round_VP) >> shift_VP) - 32768);
        }

        vif_mirror(pr, w, fh);
        vif_mirror(pd, w, fh);

        // HORIZONTAL
        vif_subsample_horizontal(pr, buf.mu1 + i * mu_stride, w, hcoef,
                                 hfirst, hstep, hpairs, vif_filt, fwidth);
        vif_subsample_horizontal(pd, buf.mu2 + i * mu_stride, w, hcoef,
                                 hfirst, hstep, hpairs, vif_filt, fwidth);
    }
    copy_and_pad(buf, w, h, scale);
}

void VIF_X86_FN(vif_subsample_rd_16)(VifBuffer buf, unsigned w, unsigned h, int scale,
                              int bpc) {
    // filter scale + 1; hfirst/hstep/hpairs as set up by vif_hcoef()
    switch (scale) {
    case 0: vif_subsample_rd_16_scale(buf, w, h, 0, bpc, -4, 2, 5); break;
    case 1: vif_subsample_rd_16_scale(buf, w, h, 1, bpc, -2, 2, 3); break;
    case 2: vif_subsample_rd_16_scale(buf, w, h, 2, bpc, -1, 1, 2); break;
    }
}
