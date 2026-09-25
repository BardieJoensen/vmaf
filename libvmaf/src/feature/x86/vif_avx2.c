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

// multiply r0 * f and store in 32-bit accumulators (shuffled 0 1 2 3 8 9 10 11 / 4 5 6 7 12 13 14 15)
#define multiply2(acc_left, acc_right, r0, f) \
{ \
__m256i zero = _mm256_setzero_si256(); \
acc_left = _mm256_madd_epi16(_mm256_unpacklo_epi16(r0, zero), f); \
acc_right = _mm256_madd_epi16(_mm256_unpackhi_epi16(r0, zero), f); \
}

// multiply r0 * f and r1 * f and store in 32-bit accumulators (shuffled 0 1 2 3 8 9 10 11 / 4 5 6 7 12 13 14 15)
#define multiply2_and_accumulate(acc_left, acc_right, r0, r1, f) \
  acc_left = _mm256_add_epi32(acc_left, _mm256_madd_epi16(_mm256_unpacklo_epi16(r0, r1), f)); \
  acc_right = _mm256_add_epi32(acc_right, _mm256_madd_epi16(_mm256_unpackhi_epi16(r0, r1), f));


// compute r0 * r1 * f and set 32-bit accumulators (shuffled 0 1 2 3 8 9 10 11 / 4 5 6 7 12 13 14 15)
#define multiply3(accum_ref_left, accum_ref_right, r0, r1, f) \
{ \
    __m256i mul = _mm256_mullo_epi16(r0, r1); \
    __m256i lo = _mm256_mullo_epi16(mul, f); \
    __m256i hi = _mm256_mulhi_epu16(mul, f); \
    accum_ref_left = _mm256_unpacklo_epi16(lo, hi); \
    accum_ref_right = _mm256_unpackhi_epi16(lo, hi); \
}

// compute r0 * r1 * f and add to 32-bit accumulators (shuffled 0 1 2 3 8 9 10 11 / 4 5 6 7 12 13 14 15)
#define multiply3_and_accumulate(accum_ref_left, accum_ref_right, r0, r1, f) \
{ \
    __m256i mul = _mm256_mullo_epi16(r0, r1); \
    __m256i lo = _mm256_mullo_epi16(mul, f); \
    __m256i hi = _mm256_mulhi_epu16(mul, f); \
    __m256i left = _mm256_unpacklo_epi16(lo, hi); \
    __m256i right = _mm256_unpackhi_epi16(lo, hi); \
    accum_ref_left = _mm256_add_epi32(accum_ref_left, left); \
    accum_ref_right = _mm256_add_epi32(accum_ref_right, right); \
}

#define shuffle_and_save(addr, x, y) \
{ \
   __m256i left = _mm256_permute2x128_si256(x, y, 0x20); \
   __m256i right = _mm256_permute2x128_si256(x, y, 0x31); \
   _mm256_storeu_si256((__m256i*)(addr), left); \
   _mm256_storeu_si256(((__m256i*)(addr)) + 1, right); \
}


/*
 * Row scratch for the scale 0 statistic.
 *
 * The vertical pass leaves, per column, the 17-tap sums mu1, mu2 (at most
 * 16 bits after the >> 8) and T = sum f * p for p = ref^2, dis^2, ref*dis,
 * which fill all 32 bits. The C code filters T horizontally in 64 bits.
 * Here every value is kept as int16 planes offset by -32768, T split into
 * its upper and lower 16 bits, so that the horizontal pass is plain
 * vpmaddwd in 32-bit lanes: the scale 0 taps sum to exactly 65536, so a
 * filtered plane is the exact filtered value minus 2^31 and every partial
 * sum fits in int32. Recombining,
 *
 *   (sum f * T + 2^15) >> 16 = (Fhi + 2^31) + ((Flo + 2^31 + 2^15) >> 16)
 *
 * in uint32 arithmetic, with Fhi and Flo the filtered offset planes, which
 * is bit-exact with the C code.
 *
 * The planes live in the tmp row buffers (each has room for w + 64 uint32,
 * i.e. two int16 planes of w + 64 with 32 samples of margin on each side).
 */
typedef struct Vif8Planes {
    int16_t *mu1, *mu2;
    int16_t *ref_hi, *ref_lo;
    int16_t *dis_hi, *dis_lo;
    int16_t *ref_dis_hi, *ref_dis_lo;
} Vif8Planes;

static Vif8Planes vif8_planes(VifBuffer buf, unsigned w)
{
    const ptrdiff_t second = w + 2 * MAX_ALIGN;
    Vif8Planes p;
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

static FORCE_INLINE void vif8_mirror(int16_t *p, int w, int fwidth_half)
{
    for (int f = 1; f <= fwidth_half; ++f) {
        p[-f] = p[f];
        p[w - 1 + f] = p[w - 1 - f];
    }
}

static FORCE_INLINE void vif8_store_scalar(const Vif8Planes *p, unsigned j,
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

// Store T - 2^31 (int32, unpacklo/hi_epi16 lane order) as offset hi/lo planes.
static FORCE_INLINE void vif8_store_split(int16_t *hi, int16_t *lo,
                                          __m256i t_lo, __m256i t_hi)
{
    const __m256i low16 = _mm256_set1_epi32(0xffff);
    _mm256_storeu_si256((__m256i *) hi,
        _mm256_packs_epi32(_mm256_srai_epi32(t_lo, 16),
                           _mm256_srai_epi32(t_hi, 16)));
    _mm256_storeu_si256((__m256i *) lo,
        _mm256_xor_si256(
            _mm256_packus_epi32(_mm256_and_si256(t_lo, low16),
                                _mm256_and_si256(t_hi, low16)),
            _mm256_set1_epi16((int16_t) 0x8000)));
}

// 17-tap horizontal filter of an offset int16 plane at columns j..j+15:
// even gets columns j, j+2, ..., j+14, odd gets j+1, ..., j+15.
static FORCE_INLINE void vif8_hfilt(const int16_t *p, const __m256i coef[9],
                                    __m256i *even, __m256i *odd)
{
    __m256i e = _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *) (p - 8)),
                                  coef[0]);
    __m256i o = _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *) (p - 7)),
                                  coef[0]);
    for (int m = 1; m < 9; m++) {
        e = _mm256_add_epi32(e, _mm256_madd_epi16(
                _mm256_loadu_si256((const __m256i *) (p - 8 + 2 * m)), coef[m]));
        o = _mm256_add_epi32(o, _mm256_madd_epi16(
                _mm256_loadu_si256((const __m256i *) (p - 7 + 2 * m)), coef[m]));
    }
    *even = e;
    *odd = o;
}

// (a * b + 2^31) >> 32 per uint32 lane
static FORCE_INLINE __m256i vif8_mulhi_round(__m256i a, __m256i b)
{
    const __m256i round = _mm256_set1_epi64x(0x80000000);
    __m256i ev = _mm256_add_epi64(_mm256_mul_epu32(a, b), round);
    __m256i od = _mm256_add_epi64(_mm256_mul_epu32(_mm256_srli_epi64(a, 32),
                                                   _mm256_srli_epi64(b, 32)),
                                  round);
    return _mm256_blend_epi32(_mm256_srli_epi64(ev, 32), od, 0xAA);
}

// (sum f * T + 2^15) >> 16 from the filtered offset hi/lo planes
static FORCE_INLINE __m256i vif8_combine(__m256i fhi, __m256i flo)
{
    return _mm256_add_epi32(
        _mm256_xor_si256(fhi, _mm256_set1_epi32((int32_t) 0x80000000)),
        _mm256_srli_epi32(_mm256_add_epi32(flo,
                              _mm256_set1_epi32((int32_t) 0x80008000)), 16));
}

static FORCE_INLINE void vif8_accumulate(int32_t sigma1_sq, int32_t sigma2_sq,
                                         int32_t sigma12,
                                         const uint16_t *log2_table,
                                         double vif_enhn_gain_limit,
                                         int64_t *accum_num_log,
                                         int64_t *accum_den_log,
                                         int64_t *accum_num_non_log,
                                         int64_t *accum_den_non_log)
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
        *accum_den_log += log2_32(log2_table, sigma_nsq + sigma1_sq) - 2048 * 17;

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
            *accum_num_log += log2_64(log2_table, numer1_tmp) - log2_64(log2_table, numer1);
        }
    }
    else {
        *accum_num_non_log += sigma2_sq;
        *accum_den_non_log += 1;
    }
}

void vif_statistic_8_avx2(struct VifPublicState *s, float *num, float *den, unsigned w, unsigned h) {
    assert(vif_filter1d_width[0] == 17);
    static const unsigned fwidth = 17;
    const uint16_t *vif_filt_s0 = vif_filter1d_table[0];
    VifBuffer buf = s->buf;
    const Vif8Planes p = vif8_planes(buf, w);
    const uint16_t *log2_table = s->log2_table;
    const double vif_enhn_gain_limit = s->vif_enhn_gain_limit;

    int64_t accum_num_log = 0;
    int64_t accum_den_log = 0;
    int64_t accum_num_non_log = 0;
    int64_t accum_den_non_log = 0;

    // vertical: symmetric rows k and 16 - k share a tap, (f_k, f_k) pairs;
    // the centre row is paired with zeros, (f_8, 0)
    __m256i vcoef[9];
    for (unsigned k = 0; k < 8; k++)
        vcoef[k] = _mm256_set1_epi32(vif_filt_s0[k] * 0x10001);
    vcoef[8] = _mm256_set1_epi32(vif_filt_s0[8]);
    // horizontal: adjacent taps (f_2m, f_2m+1), f_17 = 0
    __m256i hcoef[9];
    for (unsigned m = 0; m < 9; m++)
        hcoef[m] = _mm256_set1_epi32(vif_filt_s0[2 * m] |
                                     ((uint32_t) vif_filt_s0[2 * m + 1] << 16));

    const __m256i zero = _mm256_setzero_si256();
    const __m256i offset16 = _mm256_set1_epi16((int16_t) 0x8000);

    // variables used for 16 sample block vif computation
    ALIGNED(32) uint32_t xx[16];
    ALIGNED(32) uint32_t yy[16];
    ALIGNED(32) uint32_t xy[16];

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

                mu1_lo = _mm256_add_epi32(mu1_lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(r0, r1), f));
                mu1_hi = _mm256_add_epi32(mu1_hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(r0, r1), f));
                mu2_lo = _mm256_add_epi32(mu2_lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(d0, d1), f));
                mu2_hi = _mm256_add_epi32(mu2_hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(d0, d1), f));

                __m256i rr0 = _mm256_xor_si256(_mm256_mullo_epi16(r0, r0), offset16);
                __m256i rr1 = _mm256_xor_si256(_mm256_mullo_epi16(r1, r1), offset16);
                __m256i dd0 = _mm256_xor_si256(_mm256_mullo_epi16(d0, d0), offset16);
                __m256i dd1 = _mm256_xor_si256(_mm256_mullo_epi16(d1, d1), offset16);
                __m256i rd0 = _mm256_xor_si256(_mm256_mullo_epi16(r0, d0), offset16);
                __m256i rd1 = _mm256_xor_si256(_mm256_mullo_epi16(r1, d1), offset16);

                rr_lo = _mm256_add_epi32(rr_lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(rr0, rr1), f));
                rr_hi = _mm256_add_epi32(rr_hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(rr0, rr1), f));
                dd_lo = _mm256_add_epi32(dd_lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(dd0, dd1), f));
                dd_hi = _mm256_add_epi32(dd_hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(dd0, dd1), f));
                rd_lo = _mm256_add_epi32(rd_lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(rd0, rd1), f));
                rd_hi = _mm256_add_epi32(rd_hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(rd0, rd1), f));
            }

            // mu = (sum + 128) >> 8, stored minus 32768
            const __m256i mu_round = _mm256_set1_epi32(128 - (32768 << 8));
            mu1_lo = _mm256_srai_epi32(_mm256_add_epi32(mu1_lo, mu_round), 8);
            mu1_hi = _mm256_srai_epi32(_mm256_add_epi32(mu1_hi, mu_round), 8);
            mu2_lo = _mm256_srai_epi32(_mm256_add_epi32(mu2_lo, mu_round), 8);
            mu2_hi = _mm256_srai_epi32(_mm256_add_epi32(mu2_hi, mu_round), 8);
            _mm256_storeu_si256((__m256i *) (p.mu1 + jj), _mm256_packs_epi32(mu1_lo, mu1_hi));
            _mm256_storeu_si256((__m256i *) (p.mu2 + jj), _mm256_packs_epi32(mu2_lo, mu2_hi));

            vif8_store_split(p.ref_hi + jj, p.ref_lo + jj, rr_lo, rr_hi);
            vif8_store_split(p.dis_hi + jj, p.dis_lo + jj, dd_lo, dd_hi);
            vif8_store_split(p.ref_dis_hi + jj, p.ref_dis_lo + jj, rd_lo, rd_hi);
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

            vif8_store_scalar(&p, j, (accum_mu1 + 128) >> 8, (accum_mu2 + 128) >> 8,
                              accum_ref, accum_dis, accum_ref_dis);
        }

        vif8_mirror(p.mu1, w, fwidth / 2);
        vif8_mirror(p.mu2, w, fwidth / 2);
        vif8_mirror(p.ref_hi, w, fwidth / 2);
        vif8_mirror(p.ref_lo, w, fwidth / 2);
        vif8_mirror(p.dis_hi, w, fwidth / 2);
        vif8_mirror(p.dis_lo, w, fwidth / 2);
        vif8_mirror(p.ref_dis_hi, w, fwidth / 2);
        vif8_mirror(p.ref_dis_lo, w, fwidth / 2);

        //HORIZONTAL
        for (unsigned j = 0; j < n << 4; j += 16) {
            __m256i mu1[2], mu2[2], rh[2], rl[2], dh[2], dl[2], xh[2], xl[2];
            vif8_hfilt(p.mu1 + j, hcoef, &mu1[0], &mu1[1]);
            vif8_hfilt(p.mu2 + j, hcoef, &mu2[0], &mu2[1]);
            vif8_hfilt(p.ref_hi + j, hcoef, &rh[0], &rh[1]);
            vif8_hfilt(p.ref_lo + j, hcoef, &rl[0], &rl[1]);
            vif8_hfilt(p.dis_hi + j, hcoef, &dh[0], &dh[1]);
            vif8_hfilt(p.dis_lo + j, hcoef, &dl[0], &dl[1]);
            vif8_hfilt(p.ref_dis_hi + j, hcoef, &xh[0], &xh[1]);
            vif8_hfilt(p.ref_dis_lo + j, hcoef, &xl[0], &xl[1]);

            // even columns go to [0, 8), odd columns to [8, 16); the
            // accumulation below does not depend on the order
            for (int e = 0; e < 2; e++) {
                const __m256i sign = _mm256_set1_epi32((int32_t) 0x80000000);
                const __m256i mu1_val = _mm256_xor_si256(mu1[e], sign);
                const __m256i mu2_val = _mm256_xor_si256(mu2[e], sign);
                const __m256i mu1_sq = vif8_mulhi_round(mu1_val, mu1_val);
                const __m256i mu2_sq = vif8_mulhi_round(mu2_val, mu2_val);
                const __m256i mu1_mu2 = vif8_mulhi_round(mu1_val, mu2_val);

                const __m256i sigma1_sq = _mm256_sub_epi32(vif8_combine(rh[e], rl[e]), mu1_sq);
                const __m256i sigma2_sq = _mm256_sub_epi32(vif8_combine(dh[e], dl[e]), mu2_sq);
                const __m256i sigma12 = _mm256_sub_epi32(vif8_combine(xh[e], xl[e]), mu1_mu2);

                _mm256_store_si256((__m256i *) &xx[8 * e], sigma1_sq);
                _mm256_store_si256((__m256i *) &yy[8 * e], _mm256_max_epi32(sigma2_sq, zero));
                _mm256_store_si256((__m256i *) &xy[8 * e], sigma12);
            }

            for (unsigned int b = 0; b < 16; b++) {
                vif8_accumulate(xx[b], yy[b], xy[b], log2_table, vif_enhn_gain_limit,
                                &accum_num_log, &accum_den_log,
                                &accum_num_non_log, &accum_den_non_log);
            }
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
                const uint32_t f = vif_filt_s0[fj];
                accum_mu1 += f * (uint32_t) (p.mu1[jj] + 32768);
                accum_mu2 += f * (uint32_t) (p.mu2[jj] + 32768);
                accum_ref += f * (uint64_t) (((uint32_t) (p.ref_hi[jj] + 32768) << 16) |
                                             (uint32_t) (p.ref_lo[jj] + 32768));
                accum_dis += f * (uint64_t) (((uint32_t) (p.dis_hi[jj] + 32768) << 16) |
                                             (uint32_t) (p.dis_lo[jj] + 32768));
                accum_ref_dis += f * (uint64_t) (((uint32_t) (p.ref_dis_hi[jj] + 32768) << 16) |
                                                 (uint32_t) (p.ref_dis_lo[jj] + 32768));
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

            vif8_accumulate(sigma1_sq, sigma2_sq, sigma12, log2_table, vif_enhn_gain_limit,
                            &accum_num_log, &accum_den_log,
                            &accum_num_non_log, &accum_den_non_log);
        }
    }

    //log has to be divided by 2048 as log_value = log2(i*2048)  i=16384 to 65535
    //num[0] = accum_num_log / 2048.0 + (accum_den_non_log - (accum_num_non_log / 65536.0) / (255.0*255.0));
    //den[0] = accum_den_log / 2048.0 + accum_den_non_log;

    //changed calculation to increase performance
    num[0] = accum_num_log / 2048.0 + (accum_den_non_log - ((accum_num_non_log) / 16384.0) / (65025.0));
    den[0] = accum_den_log / 2048.0 + accum_den_non_log;

}

void vif_statistic_16_avx2(struct VifPublicState *s, float *num, float *den, unsigned w, unsigned h, int bpc, int scale) {
    const unsigned fwidth = vif_filter1d_width[scale];
    const uint16_t *vif_filt = vif_filter1d_table[scale];
    VifBuffer buf = s->buf;
    const ptrdiff_t stride = buf.stride / sizeof(uint16_t);
    int fwidth_half = fwidth >> 1;

    int32_t add_shift_round_VP, shift_VP;
    int32_t add_shift_round_VP_sq, shift_VP_sq;

    //float equivalent of 2. (2 * 65536)
    static const int32_t sigma_nsq = 65536 << 1;

    int64_t accum_num_log = 0;
    int64_t accum_den_log = 0;
    int64_t accum_num_non_log = 0;
    int64_t accum_den_non_log = 0;
    const uint16_t *log2_table = s->log2_table;
    double vif_enhn_gain_limit = s->vif_enhn_gain_limit;

    // variables used for 16 sample block vif computation
    ALIGNED(32) uint32_t xx[16];
    ALIGNED(32) uint32_t yy[16];
    ALIGNED(32) uint32_t xy[16];

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

    for (unsigned i = 0; i < h; ++i) {
        // VERTICAL
        int ii = i - fwidth_half;
        unsigned n = w >> 4;
        for (unsigned j = 0; j < n << 4; j = j + 16) {
            __m256i mask2 = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
            int ii_check = ii;

            uint16_t *ref = buf.ref;
            uint16_t *dis = buf.dis;
            __m256i accumr_lo, accumr_hi, accumd_lo, accumd_hi, rmul1, rmul2,
                dmul1, dmul2, accumref1, accumref2, accumref3, accumref4,
                accumrefdis1, accumrefdis2, accumrefdis3, accumrefdis4,
                accumdis1, accumdis2, accumdis3, accumdis4;
            accumr_lo = accumr_hi = accumd_lo = accumd_hi = rmul1 = rmul2 =
                dmul1 = dmul2 = accumref1 = accumref2 = accumref3 = accumref4 =
                    accumrefdis1 = accumrefdis2 = accumrefdis3 = accumrefdis4 =
                        accumdis1 = accumdis2 = accumdis3 = accumdis4 =
                            _mm256_setzero_si256();
            __m256i addnum = _mm256_set1_epi32(add_shift_round_VP);
            for (unsigned fi = 0; fi < fwidth; ++fi, ii_check = ii + fi) {
                __m256i f1 = _mm256_set1_epi16(vif_filt[fi]);
                __m256i ref1 = _mm256_loadu_si256(
                    (__m256i *)(ref + (ii_check * stride) + j));
                __m256i dis1 = _mm256_loadu_si256(
                    (__m256i *)(dis + (ii_check * stride) + j));
                __m256i result2 = _mm256_mulhi_epu16(ref1, f1);
                __m256i result2lo = _mm256_mullo_epi16(ref1, f1);
                rmul1 = _mm256_unpacklo_epi16(result2lo, result2);
                rmul2 = _mm256_unpackhi_epi16(result2lo, result2);
                accumr_lo = _mm256_add_epi32(accumr_lo, rmul1);
                accumr_hi = _mm256_add_epi32(accumr_hi, rmul2);
                __m256i d0 = _mm256_mulhi_epu16(dis1, f1);
                __m256i d0lo = _mm256_mullo_epi16(dis1, f1);
                dmul1 = _mm256_unpacklo_epi16(d0lo, d0);
                dmul2 = _mm256_unpackhi_epi16(d0lo, d0);
                accumd_lo = _mm256_add_epi32(accumd_lo, dmul1);
                accumd_hi = _mm256_add_epi32(accumd_hi, dmul2);

                __m256i sg0 =
                    _mm256_cvtepu32_epi64(_mm256_castsi256_si128(rmul1));
                __m256i sg1 =
                    _mm256_cvtepu32_epi64(_mm256_extracti128_si256(rmul1, 1));
                __m256i sg2 =
                    _mm256_cvtepu32_epi64(_mm256_castsi256_si128(rmul2));
                __m256i sg3 =
                    _mm256_cvtepu32_epi64(_mm256_extracti128_si256(rmul2, 1));
                __m128i l0 = _mm256_castsi256_si128(ref1);
                __m128i l1 = _mm256_extracti128_si256(ref1, 1);
                accumref1 = _mm256_add_epi64(
                    accumref1,
                    _mm256_mul_epu32(sg0, _mm256_cvtepu16_epi64(l0)));
                accumref2 = _mm256_add_epi64(
                    accumref2,
                    _mm256_mul_epu32(
                        sg2, _mm256_cvtepu16_epi64(_mm_bsrli_si128(l0, 8))));
                accumref3 = _mm256_add_epi64(
                    accumref3,
                    _mm256_mul_epu32(sg1, _mm256_cvtepu16_epi64(l1)));
                accumref4 = _mm256_add_epi64(
                    accumref4,
                    _mm256_mul_epu32(
                        sg3, _mm256_cvtepu16_epi64(_mm_bsrli_si128(l1, 8))));
                l0 = _mm256_castsi256_si128(dis1);
                l1 = _mm256_extracti128_si256(dis1, 1);
                accumrefdis1 = _mm256_add_epi64(
                    accumrefdis1,
                    _mm256_mul_epu32(sg0, _mm256_cvtepu16_epi64(l0)));
                accumrefdis2 = _mm256_add_epi64(
                    accumrefdis2,
                    _mm256_mul_epu32(
                        sg2, _mm256_cvtepu16_epi64(_mm_bsrli_si128(l0, 8))));
                accumrefdis3 = _mm256_add_epi64(
                    accumrefdis3,
                    _mm256_mul_epu32(sg1, _mm256_cvtepu16_epi64(l1)));
                accumrefdis4 = _mm256_add_epi64(
                    accumrefdis4,
                    _mm256_mul_epu32(
                        sg3, _mm256_cvtepu16_epi64(_mm_bsrli_si128(l1, 8))));
                __m256i sd0 =
                    _mm256_cvtepu32_epi64(_mm256_castsi256_si128(dmul1));
                __m256i sd1 =
                    _mm256_cvtepu32_epi64(_mm256_extracti128_si256(dmul1, 1));
                __m256i sd2 =
                    _mm256_cvtepu32_epi64(_mm256_castsi256_si128(dmul2));
                __m256i sd3 =
                    _mm256_cvtepu32_epi64(_mm256_extracti128_si256(dmul2, 1));
                accumdis1 = _mm256_add_epi64(
                    accumdis1,
                    _mm256_mul_epu32(sd0, _mm256_cvtepu16_epi64(l0)));
                accumdis2 = _mm256_add_epi64(
                    accumdis2,
                    _mm256_mul_epu32(
                        sd2, _mm256_cvtepu16_epi64(_mm_bsrli_si128(l0, 8))));
                accumdis3 = _mm256_add_epi64(
                    accumdis3,
                    _mm256_mul_epu32(sd1, _mm256_cvtepu16_epi64(l1)));
                accumdis4 = _mm256_add_epi64(
                    accumdis4,
                    _mm256_mul_epu32(
                        sd3, _mm256_cvtepu16_epi64(_mm_bsrli_si128(l1, 8))));
            }
            accumr_lo = _mm256_add_epi32(accumr_lo, addnum);
            accumr_hi = _mm256_add_epi32(accumr_hi, addnum);
            accumr_lo = _mm256_srli_epi32(accumr_lo, shift_VP);
            accumr_hi = _mm256_srli_epi32(accumr_hi, shift_VP);
            __m256i accumu2_lo =
                _mm256_permute2x128_si256(accumr_lo, accumr_hi, 0x20);
            __m256i accumu2_hi =
                _mm256_permute2x128_si256(accumr_lo, accumr_hi, 0x31);
            _mm256_storeu_si256((__m256i *)(buf.tmp.mu1 + j), accumu2_lo);
            _mm256_storeu_si256((__m256i *)(buf.tmp.mu1 + j + 8), accumu2_hi);

            accumd_lo = _mm256_add_epi32(accumd_lo, addnum);
            accumd_hi = _mm256_add_epi32(accumd_hi, addnum);
            accumd_lo = _mm256_srli_epi32(accumd_lo, shift_VP);
            accumd_hi = _mm256_srli_epi32(accumd_hi, shift_VP);
            __m256i accumu3_lo =
                _mm256_permute2x128_si256(accumd_lo, accumd_hi, 0x20);
            __m256i accumu3_hi =
                _mm256_permute2x128_si256(accumd_lo, accumd_hi, 0x31);
            _mm256_storeu_si256((__m256i *)(buf.tmp.mu2 + j), accumu3_lo);
            _mm256_storeu_si256((__m256i *)(buf.tmp.mu2 + j + 8), accumu3_hi);
            addnum = _mm256_set1_epi64x(add_shift_round_VP_sq);
            accumref1 = _mm256_add_epi64(accumref1, addnum);
            accumref2 = _mm256_add_epi64(accumref2, addnum);
            accumref3 = _mm256_add_epi64(accumref3, addnum);
            accumref4 = _mm256_add_epi64(accumref4, addnum);
            accumref1 = _mm256_srli_epi64(accumref1, shift_VP_sq);
            accumref2 = _mm256_srli_epi64(accumref2, shift_VP_sq);
            accumref3 = _mm256_srli_epi64(accumref3, shift_VP_sq);
            accumref4 = _mm256_srli_epi64(accumref4, shift_VP_sq);

            accumref2 = _mm256_slli_si256(accumref2, 4);
            accumref1 = _mm256_blend_epi32(accumref1, accumref2, 0xAA);
            accumref1 = _mm256_permutevar8x32_epi32(accumref1, mask2);

            _mm256_storeu_si256((__m256i *)(buf.tmp.ref + j), accumref1);
            accumref4 = _mm256_slli_si256(accumref4, 4);
            accumref3 = _mm256_blend_epi32(accumref3, accumref4, 0xAA);
            accumref3 = _mm256_permutevar8x32_epi32(accumref3, mask2);

            _mm256_storeu_si256((__m256i *)(buf.tmp.ref + j + 8), accumref3);

            accumrefdis1 = _mm256_add_epi64(accumrefdis1, addnum);
            accumrefdis2 = _mm256_add_epi64(accumrefdis2, addnum);
            accumrefdis3 = _mm256_add_epi64(accumrefdis3, addnum);
            accumrefdis4 = _mm256_add_epi64(accumrefdis4, addnum);
            accumrefdis1 = _mm256_srli_epi64(accumrefdis1, shift_VP_sq);
            accumrefdis2 = _mm256_srli_epi64(accumrefdis2, shift_VP_sq);
            accumrefdis3 = _mm256_srli_epi64(accumrefdis3, shift_VP_sq);
            accumrefdis4 = _mm256_srli_epi64(accumrefdis4, shift_VP_sq);

            accumrefdis2 = _mm256_slli_si256(accumrefdis2, 4);
            accumrefdis1 = _mm256_blend_epi32(accumrefdis1, accumrefdis2, 0xAA);
            accumrefdis1 = _mm256_permutevar8x32_epi32(accumrefdis1, mask2);

            _mm256_storeu_si256((__m256i *)(buf.tmp.ref_dis + j), accumrefdis1);
            accumrefdis4 = _mm256_slli_si256(accumrefdis4, 4);
            accumrefdis3 = _mm256_blend_epi32(accumrefdis3, accumrefdis4, 0xAA);
            accumrefdis3 = _mm256_permutevar8x32_epi32(accumrefdis3, mask2);

            _mm256_storeu_si256((__m256i *)(buf.tmp.ref_dis + j + 8),
                                accumrefdis3);

            accumdis1 = _mm256_add_epi64(accumdis1, addnum);
            accumdis2 = _mm256_add_epi64(accumdis2, addnum);
            accumdis3 = _mm256_add_epi64(accumdis3, addnum);
            accumdis4 = _mm256_add_epi64(accumdis4, addnum);
            accumdis1 = _mm256_srli_epi64(accumdis1, shift_VP_sq);
            accumdis2 = _mm256_srli_epi64(accumdis2, shift_VP_sq);
            accumdis3 = _mm256_srli_epi64(accumdis3, shift_VP_sq);
            accumdis4 = _mm256_srli_epi64(accumdis4, shift_VP_sq);

            accumdis2 = _mm256_slli_si256(accumdis2, 4);
            accumdis1 = _mm256_blend_epi32(accumdis1, accumdis2, 0xAA);
            accumdis1 = _mm256_permutevar8x32_epi32(accumdis1, mask2);

            _mm256_storeu_si256((__m256i *)(buf.tmp.dis + j), accumdis1);
            accumdis4 = _mm256_slli_si256(accumdis4, 4);
            accumdis3 = _mm256_blend_epi32(accumdis3, accumdis4, 0xAA);
            accumdis3 = _mm256_permutevar8x32_epi32(accumdis3, mask2);

            _mm256_storeu_si256((__m256i *)(buf.tmp.dis + j + 8), accumdis3);
        }

        for (unsigned j = n << 4; j < w; ++j) {
            uint32_t accum_mu1 = 0;
            uint32_t accum_mu2 = 0;
            uint64_t accum_ref = 0;
            uint64_t accum_dis = 0;
            uint64_t accum_ref_dis = 0;

            int ii_check = ii;
            for (unsigned fi = 0; fi < fwidth; ++fi, ii_check = ii + fi) {
                const uint16_t fcoeff = vif_filt[fi];
                uint16_t *ref = buf.ref;
                uint16_t *dis = buf.dis;
                uint16_t imgcoeff_ref = ref[ii_check * stride + j];
                uint16_t imgcoeff_dis = dis[ii_check * stride + j];
                uint32_t img_coeff_ref = fcoeff * (uint32_t)imgcoeff_ref;
                uint32_t img_coeff_dis = fcoeff * (uint32_t)imgcoeff_dis;
                accum_mu1 += img_coeff_ref;
                accum_mu2 += img_coeff_dis;
                accum_ref += img_coeff_ref * (uint64_t)imgcoeff_ref;
                accum_dis += img_coeff_dis * (uint64_t)imgcoeff_dis;
                accum_ref_dis += img_coeff_ref * (uint64_t)imgcoeff_dis;
            }
            buf.tmp.mu1[j] =
                (uint16_t)((accum_mu1 + add_shift_round_VP) >> shift_VP);
            buf.tmp.mu2[j] =
                (uint16_t)((accum_mu2 + add_shift_round_VP) >> shift_VP);
            buf.tmp.ref[j] =
                (uint32_t)((accum_ref + add_shift_round_VP_sq) >> shift_VP_sq);
            buf.tmp.ref_dis[j] = (uint32_t)(
                (accum_ref_dis + add_shift_round_VP_sq) >> shift_VP_sq);
            buf.tmp.dis[j] =
                (uint32_t)((accum_dis + add_shift_round_VP_sq) >> shift_VP_sq);
        }

        PADDING_SQ_DATA(buf, w, fwidth_half);

        //HORIZONTAL
        for (unsigned jj = 0; jj < n << 4; jj += 16) {
            __m256i mu1_lo;
            __m256i mu1_hi;
            __m256i mu1sq_lo;
            __m256i mu1sq_hi;
            __m256i mu2sq_lo;
            __m256i mu2sq_hi;
            __m256i mu1mu2_lo;
            __m256i mu1mu2_hi;
            {
                __m256i fq = _mm256_set1_epi32(vif_filt[fwidth / 2]);
                __m256i acc0 = _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)(buf.tmp.mu1 + jj + 0)), fq);
                __m256i acc1 = _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)(buf.tmp.mu1 + jj + 8)), fq);
                for (unsigned fj = 0; fj < fwidth / 2; ++fj) {
                    __m256i fq = _mm256_set1_epi32(vif_filt[fj]);
                    acc0 = _mm256_add_epi64(acc0, _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)(buf.tmp.mu1 + jj - fwidth / 2 + fj + 0)), fq));
                    acc1 = _mm256_add_epi64(acc1, _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)(buf.tmp.mu1 + jj - fwidth / 2 + fj + 8)), fq));
                    acc0 = _mm256_add_epi64(acc0, _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)(buf.tmp.mu1 + jj + fwidth / 2 - fj + 0)), fq));
                    acc1 = _mm256_add_epi64(acc1, _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)(buf.tmp.mu1 + jj + fwidth / 2 - fj + 8)), fq));
                }

                mu1_lo = acc0;
                mu1_hi = acc1;

                __m256i acc0_lo = _mm256_unpacklo_epi32(acc0, _mm256_setzero_si256());
                __m256i acc0_hi = _mm256_unpackhi_epi32(acc0, _mm256_setzero_si256());
                acc0_lo = _mm256_mul_epu32(acc0_lo, acc0_lo);
                acc0_hi = _mm256_mul_epu32(acc0_hi, acc0_hi);
                acc0_lo = _mm256_srli_epi64(_mm256_add_epi64(acc0_lo, _mm256_set1_epi64x(0x80000000)), 32);
                acc0_hi = _mm256_srli_epi64(_mm256_add_epi64(acc0_hi, _mm256_set1_epi64x(0x80000000)), 32);

                __m256i acc1_lo = _mm256_unpacklo_epi32(acc1, _mm256_setzero_si256());
                __m256i acc1_hi = _mm256_unpackhi_epi32(acc1, _mm256_setzero_si256());
                acc1_lo = _mm256_mul_epu32(acc1_lo, acc1_lo);
                acc1_hi = _mm256_mul_epu32(acc1_hi, acc1_hi);
                acc1_lo = _mm256_srli_epi64(_mm256_add_epi64(acc1_lo, _mm256_set1_epi64x(0x80000000)), 32);
                acc1_hi = _mm256_srli_epi64(_mm256_add_epi64(acc1_hi, _mm256_set1_epi64x(0x80000000)), 32);

                __m256i acc0_sq = _mm256_blend_epi32(acc0_lo, _mm256_slli_si256(acc0_hi, 4), 0xAA);
                acc0_sq = _mm256_shuffle_epi32(acc0_sq, 0xD8);
                __m256i acc1_sq = _mm256_blend_epi32(acc1_lo, _mm256_slli_si256(acc1_hi, 4), 0xAA);
                acc1_sq = _mm256_shuffle_epi32(acc1_sq, 0xD8);
                mu1sq_lo = acc0_sq;
                mu1sq_hi = acc1_sq;
            }

            {
                __m256i fq = _mm256_set1_epi32(vif_filt[fwidth / 2]);
                __m256i acc0 = _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)(buf.tmp.mu2 + jj + 0)), fq);
                __m256i acc1 = _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)(buf.tmp.mu2 + jj + 8)), fq);
                for (unsigned fj = 0; fj < fwidth / 2; ++fj) {
                    __m256i fq = _mm256_set1_epi32(vif_filt[fj]);
                    acc0 = _mm256_add_epi64(acc0, _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)(buf.tmp.mu2 + jj - fwidth / 2 + fj + 0)), fq));
                    acc1 = _mm256_add_epi64(acc1, _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)(buf.tmp.mu2 + jj - fwidth / 2 + fj + 8)), fq));
                    acc0 = _mm256_add_epi64(acc0, _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)(buf.tmp.mu2 + jj + fwidth / 2 - fj + 0)), fq));
                    acc1 = _mm256_add_epi64(acc1, _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)(buf.tmp.mu2 + jj + fwidth / 2 - fj + 8)), fq));
                }

                __m256i acc0_lo = _mm256_unpacklo_epi32(acc0, _mm256_setzero_si256());
                __m256i acc0_hi = _mm256_unpackhi_epi32(acc0, _mm256_setzero_si256());

                __m256i mu1lo_lo = _mm256_unpacklo_epi32(mu1_lo, _mm256_setzero_si256());
                __m256i mu1lo_hi = _mm256_unpackhi_epi32(mu1_lo, _mm256_setzero_si256());
                __m256i mu1hi_lo = _mm256_unpacklo_epi32(mu1_hi, _mm256_setzero_si256());
                __m256i mu1hi_hi = _mm256_unpackhi_epi32(mu1_hi, _mm256_setzero_si256());


                mu1lo_lo = _mm256_mul_epu32(mu1lo_lo, acc0_lo);
                mu1lo_hi = _mm256_mul_epu32(mu1lo_hi, acc0_hi);
                mu1lo_lo = _mm256_srli_epi64(_mm256_add_epi64(mu1lo_lo, _mm256_set1_epi64x(0x80000000)), 32);
                mu1lo_hi = _mm256_srli_epi64(_mm256_add_epi64(mu1lo_hi, _mm256_set1_epi64x(0x80000000)), 32);

                acc0_lo = _mm256_mul_epu32(acc0_lo, acc0_lo);
                acc0_hi = _mm256_mul_epu32(acc0_hi, acc0_hi);
                acc0_lo = _mm256_srli_epi64(_mm256_add_epi64(acc0_lo, _mm256_set1_epi64x(0x80000000)), 32);
                acc0_hi = _mm256_srli_epi64(_mm256_add_epi64(acc0_hi, _mm256_set1_epi64x(0x80000000)), 32);


                __m256i acc1_lo = _mm256_unpacklo_epi32(acc1, _mm256_setzero_si256());
                __m256i acc1_hi = _mm256_unpackhi_epi32(acc1, _mm256_setzero_si256());

                mu1hi_lo = _mm256_mul_epu32(mu1hi_lo, acc1_lo);
                mu1hi_hi = _mm256_mul_epu32(mu1hi_hi, acc1_hi);
                mu1hi_lo = _mm256_srli_epi64(_mm256_add_epi64(mu1hi_lo, _mm256_set1_epi64x(0x80000000)), 32);
                mu1hi_hi = _mm256_srli_epi64(_mm256_add_epi64(mu1hi_hi, _mm256_set1_epi64x(0x80000000)), 32);

                acc1_lo = _mm256_mul_epu32(acc1_lo, acc1_lo);
                acc1_hi = _mm256_mul_epu32(acc1_hi, acc1_hi);
                acc1_lo = _mm256_srli_epi64(_mm256_add_epi64(acc1_lo, _mm256_set1_epi64x(0x80000000)), 32);
                acc1_hi = _mm256_srli_epi64(_mm256_add_epi64(acc1_hi, _mm256_set1_epi64x(0x80000000)), 32);


                __m256i acc0_sq = _mm256_blend_epi32(acc0_lo, _mm256_slli_si256(acc0_hi, 4), 0xAA);
                acc0_sq = _mm256_shuffle_epi32(acc0_sq, 0xD8);
                __m256i acc1_sq = _mm256_blend_epi32(acc1_lo, _mm256_slli_si256(acc1_hi, 4), 0xAA);
                acc1_sq = _mm256_shuffle_epi32(acc1_sq, 0xD8);
                mu2sq_lo = acc0_sq;
                mu2sq_hi = acc1_sq;

                mu1mu2_lo = _mm256_blend_epi32(mu1lo_lo, _mm256_slli_si256(mu1lo_hi, 4), 0xAA);
                mu1mu2_lo = _mm256_shuffle_epi32(mu1mu2_lo, 0xD8);
                mu1mu2_hi = _mm256_blend_epi32(mu1hi_lo, _mm256_slli_si256(mu1hi_hi, 4), 0xAA);
                mu1mu2_hi = _mm256_shuffle_epi32(mu1mu2_hi, 0xD8);


            }

            // filter horizontally ref
            {
                __m256i rounder = _mm256_set1_epi64x(0x8000);
                __m256i fq = _mm256_set1_epi64x(vif_filt[fwidth / 2]);
                __m256i acc0 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref + jj + 0))), fq));
                __m256i acc1 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref + jj + 4))), fq));
                __m256i acc2 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref + jj + 8))), fq));
                __m256i acc3 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref + jj + 12))), fq));
                for (unsigned fj = 0; fj < fwidth / 2; ++fj) {
                    __m256i fq = _mm256_set1_epi64x(vif_filt[fj]);
                    acc0 = _mm256_add_epi64(acc0, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref + jj - fwidth / 2 + fj + 0))), fq));
                    acc1 = _mm256_add_epi64(acc1, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref + jj - fwidth / 2 + fj + 4))), fq));
                    acc2 = _mm256_add_epi64(acc2, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref + jj - fwidth / 2 + fj + 8))), fq));
                    acc3 = _mm256_add_epi64(acc3, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref + jj - fwidth / 2 + fj + 12))), fq));
                    acc0 = _mm256_add_epi64(acc0, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref + jj + fwidth / 2 - fj + 0))), fq));
                    acc1 = _mm256_add_epi64(acc1, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref + jj + fwidth / 2 - fj + 4))), fq));
                    acc2 = _mm256_add_epi64(acc2, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref + jj + fwidth / 2 - fj + 8))), fq));
                    acc3 = _mm256_add_epi64(acc3, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref + jj + fwidth / 2 - fj + 12))), fq));
                }
                acc0 = _mm256_srli_epi64(acc0, 16);
                acc1 = _mm256_srli_epi64(acc1, 16);
                acc2 = _mm256_srli_epi64(acc2, 16);
                acc3 = _mm256_srli_epi64(acc3, 16);

                __m256i mask1 = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

                // pack acc0,acc1,acc2,acc3 to acc0, acc1
                acc1 = _mm256_slli_si256(acc1, 4);
                acc1 = _mm256_blend_epi32(acc0, acc1, 0xAA);
                acc0 = _mm256_permutevar8x32_epi32(acc1, mask1);
                acc3 = _mm256_slli_si256(acc3, 4);
                acc3 = _mm256_blend_epi32(acc2, acc3, 0xAA);

                acc1 = _mm256_permutevar8x32_epi32(acc3, mask1);

                acc0 = _mm256_sub_epi32(acc0, mu1sq_lo);
                acc1 = _mm256_sub_epi32(acc1, mu1sq_hi);
                _mm256_storeu_si256((__m256i*) & xx[0], acc0);
                _mm256_storeu_si256((__m256i*) & xx[8], acc1);
            }

            // filter horizontally dis
            {
                __m256i rounder = _mm256_set1_epi64x(0x8000);
                __m256i fq = _mm256_set1_epi64x(vif_filt[fwidth / 2]);
                __m256i acc0 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.dis + jj + 0))), fq));
                __m256i acc1 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.dis + jj + 4))), fq));
                __m256i acc2 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.dis + jj + 8))), fq));
                __m256i acc3 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.dis + jj + 12))), fq));
                for (unsigned fj = 0; fj < fwidth / 2; ++fj) {
                    __m256i fq = _mm256_set1_epi64x(vif_filt[fj]);
                    acc0 = _mm256_add_epi64(acc0, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.dis + jj - fwidth / 2 + fj + 0))), fq));
                    acc1 = _mm256_add_epi64(acc1, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.dis + jj - fwidth / 2 + fj + 4))), fq));
                    acc2 = _mm256_add_epi64(acc2, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.dis + jj - fwidth / 2 + fj + 8))), fq));
                    acc3 = _mm256_add_epi64(acc3, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.dis + jj - fwidth / 2 + fj + 12))), fq));
                    acc0 = _mm256_add_epi64(acc0, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.dis + jj + fwidth / 2 - fj + 0))), fq));
                    acc1 = _mm256_add_epi64(acc1, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.dis + jj + fwidth / 2 - fj + 4))), fq));
                    acc2 = _mm256_add_epi64(acc2, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.dis + jj + fwidth / 2 - fj + 8))), fq));
                    acc3 = _mm256_add_epi64(acc3, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.dis + jj + fwidth / 2 - fj + 12))), fq));
                }
                acc0 = _mm256_srli_epi64(acc0, 16);
                acc1 = _mm256_srli_epi64(acc1, 16);
                acc2 = _mm256_srli_epi64(acc2, 16);
                acc3 = _mm256_srli_epi64(acc3, 16);

                __m256i mask1 = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

                // pack acc0,acc1,acc2,acc3 to acc0, acc1
                acc1 = _mm256_slli_si256(acc1, 4);
                acc1 = _mm256_blend_epi32(acc0, acc1, 0xAA);
                acc0 = _mm256_permutevar8x32_epi32(acc1, mask1);
                acc3 = _mm256_slli_si256(acc3, 4);
                acc3 = _mm256_blend_epi32(acc2, acc3, 0xAA);
                acc1 = _mm256_permutevar8x32_epi32(acc3, mask1);

                acc0 = _mm256_sub_epi32(acc0, mu2sq_lo);
                acc1 = _mm256_sub_epi32(acc1, mu2sq_hi);
                _mm256_storeu_si256((__m256i*) & yy[0], _mm256_max_epi32(acc0, _mm256_setzero_si256()));
                _mm256_storeu_si256((__m256i*) & yy[8], _mm256_max_epi32(acc1, _mm256_setzero_si256()));
            }

            // filter horizontally ref_dis, producing 16 samples
            {
                __m256i rounder = _mm256_set1_epi64x(0x8000);
                __m256i fq = _mm256_set1_epi64x(vif_filt[fwidth / 2]);
                __m256i acc0 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref_dis + jj + 0))), fq));
                __m256i acc1 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref_dis + jj + 4))), fq));
                __m256i acc2 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref_dis + jj + 8))), fq));
                __m256i acc3 = _mm256_add_epi64(rounder, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref_dis + jj + 12))), fq));

                for (unsigned fj = 0; fj < fwidth / 2; ++fj) {
                    __m256i fq = _mm256_set1_epi64x(vif_filt[fj]);
                    acc0 = _mm256_add_epi64(acc0, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref_dis + jj - fwidth / 2 + fj + 0))), fq));
                    acc1 = _mm256_add_epi64(acc1, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref_dis + jj - fwidth / 2 + fj + 4))), fq));
                    acc2 = _mm256_add_epi64(acc2, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref_dis + jj - fwidth / 2 + fj + 8))), fq));
                    acc3 = _mm256_add_epi64(acc3, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref_dis + jj - fwidth / 2 + fj + 12))), fq));
                    acc0 = _mm256_add_epi64(acc0, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref_dis + jj + fwidth / 2 - fj + 0))), fq));
                    acc1 = _mm256_add_epi64(acc1, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref_dis + jj + fwidth / 2 - fj + 4))), fq));
                    acc2 = _mm256_add_epi64(acc2, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref_dis + jj + fwidth / 2 - fj + 8))), fq));
                    acc3 = _mm256_add_epi64(acc3, _mm256_mul_epu32(_mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.tmp.ref_dis + jj + fwidth / 2 - fj + 12))), fq));
                }

                acc0 = _mm256_srli_epi64(acc0, 16);
                acc1 = _mm256_srli_epi64(acc1, 16);
                acc2 = _mm256_srli_epi64(acc2, 16);
                acc3 = _mm256_srli_epi64(acc3, 16);

                __m256i mask1 = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

                // pack acc0,acc1,acc2,acc3 to acc0, acc1
                acc1 = _mm256_slli_si256(acc1, 4);
                acc1 = _mm256_blend_epi32(acc0, acc1, 0xAA);
                acc0 = _mm256_permutevar8x32_epi32(acc1, mask1);
                acc3 = _mm256_slli_si256(acc3, 4);
                acc3 = _mm256_blend_epi32(acc2, acc3, 0xAA);
                acc1 = _mm256_permutevar8x32_epi32(acc3, mask1);

                acc0 = _mm256_sub_epi32(acc0, mu1mu2_lo);
                acc1 = _mm256_sub_epi32(acc1, mu1mu2_hi);
                _mm256_storeu_si256((__m256i*) & xy[0], acc0);
                _mm256_storeu_si256((__m256i*) & xy[8], acc1);
            }

            for (unsigned int j = 0; j < 16; j++) {
                int32_t sigma1_sq = xx[j];
                int32_t sigma2_sq = yy[j];
                int32_t sigma12 = xy[j];

                if (sigma1_sq >= sigma_nsq) {
                    /**
                    * log values are taken from the look-up table generated by
                    * log_generate() function which is called in integer_combo_threadfunc
                    * den_val in float is log2(1 + sigma1_sq/2)
                    * here it is converted to equivalent of log2(2+sigma1_sq) - log2(2) i.e log2(2*65536+sigma1_sq) - 17
                    * multiplied by 2048 as log_value = log2(i)*2048 i=16384 to 65535 generated using log_value
                    * x because best 16 bits are taken
                    */
                    accum_den_log += log2_32(log2_table, sigma_nsq + sigma1_sq) - 2048 * 17;

                    if (sigma12 > 0 && sigma2_sq > 0) {
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
                        accum_num_log += log2_64(log2_table, numer1_tmp) - log2_64(log2_table, numer1);
                    }
                }
                else {
                    accum_num_non_log += sigma2_sq;
                    accum_den_non_log++;
                }
            }
        }


        if ((n << 4) != w) {
            VifResiduals residuals =
                vif_compute_line_residuals(s, n << 4, w, scale);
            accum_num_log += residuals.accum_num_log;
            accum_den_log += residuals.accum_den_log;
            accum_num_non_log += residuals.accum_num_non_log;
            accum_den_non_log += residuals.accum_den_non_log;
        }
    }

    num[0] = accum_num_log / 2048.0 + (accum_den_non_log - ((accum_num_non_log) / 16384.0) / (65025.0));
    den[0] = accum_den_log / 2048.0 + accum_den_non_log;
}

void vif_subsample_rd_8_avx2(VifBuffer buf, unsigned w, unsigned h) {
    const unsigned fwidth = vif_filter1d_width[1];
    const uint16_t *vif_filt_s1 = vif_filter1d_table[1];
    const uint8_t *ref = (uint8_t *)buf.ref;
    const uint8_t *dis = (uint8_t *)buf.dis;
    const ptrdiff_t stride = buf.stride_16 / sizeof(uint16_t);
    __m256i addnum = _mm256_set1_epi32(32768);
    __m256i mask1 = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
    __m256i x = _mm256_set1_epi32(128);
    int fwidth_half = fwidth >> 1;

    __m256i fcoeff0 = _mm256_set1_epi16(vif_filt_s1[0]);
    __m256i fcoeff1 = _mm256_set1_epi16(vif_filt_s1[1]);
    __m256i fcoeff2 = _mm256_set1_epi16(vif_filt_s1[2]);
    __m256i fcoeff3 = _mm256_set1_epi16(vif_filt_s1[3]);
    __m256i fcoeff4 = _mm256_set1_epi16(vif_filt_s1[4]);

    for (unsigned i = 0; i < h / 2; i ++) {
        // VERTICAL
        unsigned n = w >> 4;
        for (unsigned j = 0; j < n << 4; j = j + 16) {
            int ii = i * 2 - fwidth_half;
            int ii_check = ii;
            __m256i accum_mu1_lo, accum_mu1_hi;
            __m256i accum_mu2_lo, accum_mu2_hi;
            __m256i g0, g1, g2, g3, g4, g5, g6, g7, g8;
            __m256i s0, s1, s2, s3, s4, s5, s6, s7, s8;

            g0 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(ref + (buf.stride * ii_check) + j)));
            g1 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(ref + buf.stride * (ii_check + 1) + j)));
            g2 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(ref + buf.stride * (ii_check + 2) + j)));
            g3 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(ref + buf.stride * (ii_check + 3) + j)));
            g4 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(ref + buf.stride * (ii_check + 4) + j)));
            g5 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(ref + buf.stride * (ii_check + 5) + j)));
            g6 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(ref + buf.stride * (ii_check + 6) + j)));
            g7 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(ref + buf.stride * (ii_check + 7) + j)));
            g8 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(ref + buf.stride * (ii_check + 8) + j)));

            s0 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(dis + (buf.stride * ii_check) + j)));
            s1 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(dis + buf.stride * (ii_check + 1) + j)));
            s2 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(dis + buf.stride * (ii_check + 2) + j)));
            s3 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(dis + buf.stride * (ii_check + 3) + j)));
            s4 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(dis + buf.stride * (ii_check + 4) + j)));
            s5 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(dis + buf.stride * (ii_check + 5) + j)));
            s6 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(dis + buf.stride * (ii_check + 6) + j)));
            s7 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(dis + buf.stride * (ii_check + 7) + j)));
            s8 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(dis + buf.stride * (ii_check + 8) + j)));

            multiply2(accum_mu2_lo, accum_mu2_hi, s4, fcoeff4);
            multiply2_and_accumulate(accum_mu2_lo, accum_mu2_hi, s0, s8, fcoeff0);
            multiply2_and_accumulate(accum_mu2_lo, accum_mu2_hi, s1, s7, fcoeff1);
            multiply2_and_accumulate(accum_mu2_lo, accum_mu2_hi, s2, s6, fcoeff2);
            multiply2_and_accumulate(accum_mu2_lo, accum_mu2_hi, s3, s5, fcoeff3);

            multiply2(accum_mu1_lo, accum_mu1_hi, g4, fcoeff4);
            multiply2_and_accumulate(accum_mu1_lo, accum_mu1_hi, g0, g8, fcoeff0);
            multiply2_and_accumulate(accum_mu1_lo, accum_mu1_hi, g1, g7, fcoeff1);
            multiply2_and_accumulate(accum_mu1_lo, accum_mu1_hi, g2, g6, fcoeff2);
            multiply2_and_accumulate(accum_mu1_lo, accum_mu1_hi, g3, g5, fcoeff3);

            __m256i accumu1_lo = _mm256_add_epi32(
                x, _mm256_permute2x128_si256(accum_mu1_lo, accum_mu1_hi, 0x20));
            __m256i accumu1_hi = _mm256_add_epi32(
                x, _mm256_permute2x128_si256(accum_mu1_lo, accum_mu1_hi, 0x31));
            __m256i accumu2_lo = _mm256_add_epi32(
                x, _mm256_permute2x128_si256(accum_mu2_lo, accum_mu2_hi, 0x20));
            __m256i accumu2_hi = _mm256_add_epi32(
                x, _mm256_permute2x128_si256(accum_mu2_lo, accum_mu2_hi, 0x31));
            accumu1_lo = _mm256_srli_epi32(accumu1_lo, 0x08);
            accumu1_hi = _mm256_srli_epi32(accumu1_hi, 0x08);
            accumu2_lo = _mm256_srli_epi32(accumu2_lo, 0x08);
            accumu2_hi = _mm256_srli_epi32(accumu2_hi, 0x08);
            _mm256_storeu_si256((__m256i *)(buf.tmp.ref_convol + j),
                                accumu1_lo);
            _mm256_storeu_si256((__m256i *)(buf.tmp.ref_convol + j + 8),
                                accumu1_hi);
            _mm256_storeu_si256((__m256i *)(buf.tmp.dis_convol + j),
                                accumu2_lo);
            _mm256_storeu_si256((__m256i *)(buf.tmp.dis_convol + j + 8),
                                accumu2_hi);
        }
        for (unsigned j = n << 4; j < w; ++j) {
            uint32_t accum_ref = 0;
            uint32_t accum_dis = 0;
            for (unsigned fi = 0; fi < fwidth; ++fi) {
                int ii = i * 2 - fwidth_half;
                int ii_check = ii + fi;
                const uint16_t fcoeff = vif_filt_s1[fi];
                const uint8_t *ref = (uint8_t *)buf.ref;
                const uint8_t *dis = (uint8_t *)buf.dis;
                accum_ref += fcoeff * (uint32_t)ref[ii_check * buf.stride + j];
                accum_dis += fcoeff * (uint32_t)dis[ii_check * buf.stride + j];
            }
            buf.tmp.ref_convol[j] = (accum_ref + 128) >> 8;
            buf.tmp.dis_convol[j] = (accum_dis + 128) >> 8;
        }

        PADDING_SQ_DATA_2(buf, w, fwidth_half);

        // HORIZONTAL
        n = w >> 3;
        for (unsigned j = 0; j < n << 3; j = j + 8) {
            int jj = j - fwidth_half;
            int jj_check = jj;
            __m256i accumrlo, accumdlo, accumrhi, accumdhi;
            accumrlo = accumdlo = accumrhi = accumdhi = _mm256_setzero_si256();
            __m256i refconvol0 = _mm256_loadu_si256((__m256i *)(buf.tmp.ref_convol + jj_check));
            __m256i refconvol4 = _mm256_loadu_si256((__m256i*)(buf.tmp.ref_convol + jj_check + 4));
            __m256i refconvol8 = _mm256_loadu_si256((__m256i*)(buf.tmp.ref_convol + jj_check + 8));
            __m256i refconvol1 = _mm256_alignr_epi8(refconvol4, refconvol0, 4);
            __m256i refconvol2 = _mm256_alignr_epi8(refconvol4, refconvol0, 8);
            __m256i refconvol3 = _mm256_alignr_epi8(refconvol4, refconvol0, 12);
            __m256i refconvol5 = _mm256_alignr_epi8(refconvol8, refconvol4, 4);
            __m256i refconvol6 = _mm256_alignr_epi8(refconvol8, refconvol4, 8);
            __m256i refconvol7 = _mm256_alignr_epi8(refconvol8, refconvol4, 12);

            __m256i result2 = _mm256_mulhi_epu16(refconvol0, fcoeff0);
            __m256i result2lo = _mm256_mullo_epi16(refconvol0, fcoeff0);
            accumrlo = _mm256_add_epi32(
                accumrlo, _mm256_unpacklo_epi16(result2lo, result2));
            accumrhi = _mm256_add_epi32(
                accumrhi, _mm256_unpackhi_epi16(result2lo, result2));
            __m256i result3 = _mm256_mulhi_epu16(refconvol1, fcoeff1);
            __m256i result3lo = _mm256_mullo_epi16(refconvol1, fcoeff1);
            accumrlo = _mm256_add_epi32(
                accumrlo, _mm256_unpacklo_epi16(result3lo, result3));
            accumrhi = _mm256_add_epi32(
                accumrhi, _mm256_unpackhi_epi16(result3lo, result3));
            __m256i result4 = _mm256_mulhi_epu16(refconvol2, fcoeff2);
            __m256i result4lo = _mm256_mullo_epi16(refconvol2, fcoeff2);
            accumrlo = _mm256_add_epi32(
                accumrlo, _mm256_unpacklo_epi16(result4lo, result4));
            accumrhi = _mm256_add_epi32(
                accumrhi, _mm256_unpackhi_epi16(result4lo, result4));
            __m256i result5 = _mm256_mulhi_epu16(refconvol3, fcoeff3);
            __m256i result5lo = _mm256_mullo_epi16(refconvol3, fcoeff3);
            accumrlo = _mm256_add_epi32(
                accumrlo, _mm256_unpacklo_epi16(result5lo, result5));
            accumrhi = _mm256_add_epi32(
                accumrhi, _mm256_unpackhi_epi16(result5lo, result5));
            __m256i result6 = _mm256_mulhi_epu16(refconvol4, fcoeff4);
            __m256i result6lo = _mm256_mullo_epi16(refconvol4, fcoeff4);
            accumrlo = _mm256_add_epi32(
                accumrlo, _mm256_unpacklo_epi16(result6lo, result6));
            accumrhi = _mm256_add_epi32(
                accumrhi, _mm256_unpackhi_epi16(result6lo, result6));
            __m256i result7 = _mm256_mulhi_epu16(refconvol5, fcoeff3);
            __m256i result7lo = _mm256_mullo_epi16(refconvol5, fcoeff3);
            accumrlo = _mm256_add_epi32(
                accumrlo, _mm256_unpacklo_epi16(result7lo, result7));
            accumrhi = _mm256_add_epi32(
                accumrhi, _mm256_unpackhi_epi16(result7lo, result7));
            __m256i result8 = _mm256_mulhi_epu16(refconvol6, fcoeff2);
            __m256i result8lo = _mm256_mullo_epi16(refconvol6, fcoeff2);
            accumrlo = _mm256_add_epi32(
                accumrlo, _mm256_unpacklo_epi16(result8lo, result8));
            accumrhi = _mm256_add_epi32(
                accumrhi, _mm256_unpackhi_epi16(result8lo, result8));
            __m256i result9 = _mm256_mulhi_epu16(refconvol7, fcoeff1);
            __m256i result9lo = _mm256_mullo_epi16(refconvol7, fcoeff1);
            accumrlo = _mm256_add_epi32(
                accumrlo, _mm256_unpacklo_epi16(result9lo, result9));
            accumrhi = _mm256_add_epi32(
                accumrhi, _mm256_unpackhi_epi16(result9lo, result9));
            __m256i result10 = _mm256_mulhi_epu16(refconvol8, fcoeff0);
            __m256i result10lo = _mm256_mullo_epi16(refconvol8, fcoeff0);
            accumrlo = _mm256_add_epi32(
                accumrlo, _mm256_unpacklo_epi16(result10lo, result10));
            accumrhi = _mm256_add_epi32(
                accumrhi, _mm256_unpackhi_epi16(result10lo, result10));

            __m256i disconvol0 =_mm256_loadu_si256((__m256i *)(buf.tmp.dis_convol + jj_check));
            __m256i disconvol4 = _mm256_loadu_si256((__m256i*)(buf.tmp.dis_convol + jj_check + 4));
            __m256i disconvol8 = _mm256_loadu_si256((__m256i*)(buf.tmp.dis_convol + jj_check + 8));
            __m256i disconvol1 = _mm256_alignr_epi8(disconvol4, disconvol0, 4);
            __m256i disconvol2 = _mm256_alignr_epi8(disconvol4, disconvol0, 8);
            __m256i disconvol3 = _mm256_alignr_epi8(disconvol4, disconvol0, 12);
            __m256i disconvol5 = _mm256_alignr_epi8(disconvol8, disconvol4, 4);
            __m256i disconvol6 = _mm256_alignr_epi8(disconvol8, disconvol4, 8);
            __m256i disconvol7 = _mm256_alignr_epi8(disconvol8, disconvol4, 12);
            result2 = _mm256_mulhi_epu16(disconvol0, fcoeff0);
            result2lo = _mm256_mullo_epi16(disconvol0, fcoeff0);
            accumdlo = _mm256_add_epi32(
                accumdlo, _mm256_unpacklo_epi16(result2lo, result2));
            accumdhi = _mm256_add_epi32(
                accumdhi, _mm256_unpackhi_epi16(result2lo, result2));
            result3 = _mm256_mulhi_epu16(disconvol1, fcoeff1);
            result3lo = _mm256_mullo_epi16(disconvol1, fcoeff1);
            accumdlo = _mm256_add_epi32(
                accumdlo, _mm256_unpacklo_epi16(result3lo, result3));
            accumdhi = _mm256_add_epi32(
                accumdhi, _mm256_unpackhi_epi16(result3lo, result3));
            result4 = _mm256_mulhi_epu16(disconvol2, fcoeff2);
            result4lo = _mm256_mullo_epi16(disconvol2, fcoeff2);
            accumdlo = _mm256_add_epi32(
                accumdlo, _mm256_unpacklo_epi16(result4lo, result4));
            accumdhi = _mm256_add_epi32(
                accumdhi, _mm256_unpackhi_epi16(result4lo, result4));
            result5 = _mm256_mulhi_epu16(disconvol3, fcoeff3);
            result5lo = _mm256_mullo_epi16(disconvol3, fcoeff3);
            accumdlo = _mm256_add_epi32(
                accumdlo, _mm256_unpacklo_epi16(result5lo, result5));
            accumdhi = _mm256_add_epi32(
                accumdhi, _mm256_unpackhi_epi16(result5lo, result5));
            result6 = _mm256_mulhi_epu16(disconvol4, fcoeff4);
            result6lo = _mm256_mullo_epi16(disconvol4, fcoeff4);
            accumdlo = _mm256_add_epi32(
                accumdlo, _mm256_unpacklo_epi16(result6lo, result6));
            accumdhi = _mm256_add_epi32(
                accumdhi, _mm256_unpackhi_epi16(result6lo, result6));
            result7 = _mm256_mulhi_epu16(disconvol5, fcoeff3);
            result7lo = _mm256_mullo_epi16(disconvol5, fcoeff3);
            accumdlo = _mm256_add_epi32(
                accumdlo, _mm256_unpacklo_epi16(result7lo, result7));
            accumdhi = _mm256_add_epi32(
                accumdhi, _mm256_unpackhi_epi16(result7lo, result7));
            result8 = _mm256_mulhi_epu16(disconvol6, fcoeff2);
            result8lo = _mm256_mullo_epi16(disconvol6, fcoeff2);
            accumdlo = _mm256_add_epi32(
                accumdlo, _mm256_unpacklo_epi16(result8lo, result8));
            accumdhi = _mm256_add_epi32(
                accumdhi, _mm256_unpackhi_epi16(result8lo, result8));
            result9 = _mm256_mulhi_epu16(disconvol7, fcoeff1);
            result9lo = _mm256_mullo_epi16(disconvol7, fcoeff1);
            accumdlo = _mm256_add_epi32(
                accumdlo, _mm256_unpacklo_epi16(result9lo, result9));
            accumdhi = _mm256_add_epi32(
                accumdhi, _mm256_unpackhi_epi16(result9lo, result9));
            result10 = _mm256_mulhi_epu16(disconvol8, fcoeff0);
            result10lo = _mm256_mullo_epi16(disconvol8, fcoeff0);
            accumdlo = _mm256_add_epi32(
                accumdlo, _mm256_unpacklo_epi16(result10lo, result10));
            accumdhi = _mm256_add_epi32(
                accumdhi, _mm256_unpackhi_epi16(result10lo, result10));

            accumdlo = _mm256_add_epi32(accumdlo, addnum);
            accumdhi = _mm256_add_epi32(accumdhi, addnum);
            accumrlo = _mm256_add_epi32(accumrlo, addnum);
            accumrhi = _mm256_add_epi32(accumrhi, addnum);
            accumdlo = _mm256_srli_epi32(accumdlo, 0x10);
            accumdhi = _mm256_srli_epi32(accumdhi, 0x10);
            accumrlo = _mm256_srli_epi32(accumrlo, 0x10);
            accumrhi = _mm256_srli_epi32(accumrhi, 0x10);

            __m256i result = _mm256_packus_epi32(accumdlo, accumdhi);
            __m256i resultd = _mm256_packus_epi32(accumrlo, accumrhi);
            resultd = _mm256_permutevar8x32_epi32(resultd, mask1);
            result = _mm256_permutevar8x32_epi32(result, mask1);
            resultd = _mm256_packus_epi32(resultd, resultd);
            result = _mm256_packus_epi32(result, result);
            _mm_storel_epi64((__m128i *)(buf.mu1 + i  * stride + (j >> 1)), _mm256_castsi256_si128(resultd));
            _mm_storel_epi64((__m128i *)(buf.mu2 + i  * stride + (j >> 1)), _mm256_castsi256_si128(result));
        }
        for (unsigned j = n << 3; j < w; j += 2) {
            uint32_t accum_ref = 0;
            uint32_t accum_dis = 0;
            int jj = j - fwidth_half;
            int jj_check = jj;
            for (unsigned fj = 0; fj < fwidth; ++fj, jj_check = jj + fj) {
                const uint16_t fcoeff = vif_filt_s1[fj];
                accum_ref += fcoeff * buf.tmp.ref_convol[jj_check];
                accum_dis += fcoeff * buf.tmp.dis_convol[jj_check];
            }
            buf.mu1[i * stride + (j >> 1)] = (uint16_t)((accum_ref + 32768) >> 16);
            buf.mu2[i * stride + (j >> 1)] = (uint16_t)((accum_dis + 32768) >> 16);
        }
    }
    copy_and_pad(buf, w, h, 0);
}

void vif_subsample_rd_16_avx2(VifBuffer buf, unsigned w, unsigned h, int scale,
                             int bpc) {
    const unsigned fwidth = vif_filter1d_width[scale + 1];
    const uint16_t *vif_filt = vif_filter1d_table[scale + 1];
    int32_t add_shift_round_VP, shift_VP;
    int fwidth_half = fwidth >> 1;
    const ptrdiff_t stride = buf.stride / sizeof(uint16_t);
    const ptrdiff_t stride16 = buf.stride_16 / sizeof(uint16_t);
    uint16_t *ref = buf.ref;
    uint16_t *dis = buf.dis;
    __m256i mask1 = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);

    if (scale == 0) {
        add_shift_round_VP = 1 << (bpc - 1);
        shift_VP = bpc;
    } else {
        add_shift_round_VP = 32768;
        shift_VP = 16;
    }

    for (unsigned i = 0; i < h / 2; i++) {
        // VERTICAL

        unsigned n = w >> 4;
        int ii = i * 2 - fwidth_half;
        for (unsigned j = 0; j < n << 4; j = j + 16) {
            int ii_check = ii;
            __m256i accumr_lo, accumr_hi, accumd_lo, accumd_hi, rmul1, rmul2,
                dmul1, dmul2;
            accumr_lo = accumr_hi = accumd_lo = accumd_hi = rmul1 = rmul2 =
                dmul1 = dmul2 = _mm256_setzero_si256();
            for (unsigned fi = 0; fi < fwidth; ++fi, ii_check = ii + fi) {
                __m256i f1 = _mm256_set1_epi16(vif_filt[fi]);
                __m256i ref1 = _mm256_loadu_si256(
                    (__m256i *)(ref + (ii_check * stride) + j));
                __m256i dis1 = _mm256_loadu_si256(
                    (__m256i *)(dis + (ii_check * stride) + j));
                __m256i result2 = _mm256_mulhi_epu16(ref1, f1);
                __m256i result2lo = _mm256_mullo_epi16(ref1, f1);
                rmul1 = _mm256_unpacklo_epi16(result2lo, result2);
                rmul2 = _mm256_unpackhi_epi16(result2lo, result2);
                accumr_lo = _mm256_add_epi32(accumr_lo, rmul1);
                accumr_hi = _mm256_add_epi32(accumr_hi, rmul2);

                __m256i d0 = _mm256_mulhi_epu16(dis1, f1);
                __m256i d0lo = _mm256_mullo_epi16(dis1, f1);
                dmul1 = _mm256_unpacklo_epi16(d0lo, d0);
                dmul2 = _mm256_unpackhi_epi16(d0lo, d0);
                accumd_lo = _mm256_add_epi32(accumd_lo, dmul1);
                accumd_hi = _mm256_add_epi32(accumd_hi, dmul2);
            }
            __m256i addnum = _mm256_set1_epi32(add_shift_round_VP);
            accumr_lo = _mm256_add_epi32(accumr_lo, addnum);
            accumr_hi = _mm256_add_epi32(accumr_hi, addnum);
            accumr_lo = _mm256_srli_epi32(accumr_lo, shift_VP);
            accumr_hi = _mm256_srli_epi32(accumr_hi, shift_VP);

            __m256i accumu2_lo =
                _mm256_permute2x128_si256(accumr_lo, accumr_hi, 0x20);
            __m256i accumu2_hi =
                _mm256_permute2x128_si256(accumr_lo, accumr_hi, 0x31);
            _mm256_storeu_si256((__m256i *)(buf.tmp.ref_convol + j),
                                accumu2_lo);
            _mm256_storeu_si256((__m256i *)(buf.tmp.ref_convol + j + 8),
                                accumu2_hi);

            accumd_lo = _mm256_add_epi32(accumd_lo, addnum);
            accumd_hi = _mm256_add_epi32(accumd_hi, addnum);
            accumd_lo = _mm256_srli_epi32(accumd_lo, shift_VP);
            accumd_hi = _mm256_srli_epi32(accumd_hi, shift_VP);
            accumu2_lo = _mm256_permute2x128_si256(accumd_lo, accumd_hi, 0x20);
            accumu2_hi = _mm256_permute2x128_si256(accumd_lo, accumd_hi, 0x31);
            _mm256_storeu_si256((__m256i *)(buf.tmp.dis_convol + j),
                                accumu2_lo);
            _mm256_storeu_si256((__m256i *)(buf.tmp.dis_convol + j + 8),
                                accumu2_hi);
        }

        // VERTICAL
        for (unsigned j = n << 4; j < w; ++j) {
            uint32_t accum_ref = 0;
            uint32_t accum_dis = 0;
            int ii_check = ii;
            for (unsigned fi = 0; fi < fwidth; ++fi, ii_check = ii + fi) {
                const uint16_t fcoeff = vif_filt[fi];
                accum_ref += fcoeff * ((uint32_t)ref[ii_check * stride + j]);
                accum_dis += fcoeff * ((uint32_t)dis[ii_check * stride + j]);
            }
            buf.tmp.ref_convol[j] =
                (uint16_t)((accum_ref + add_shift_round_VP) >> shift_VP);
            buf.tmp.dis_convol[j] =
                (uint16_t)((accum_dis + add_shift_round_VP) >> shift_VP);
        }

        PADDING_SQ_DATA_2(buf, w, fwidth_half);

        // HORIZONTAL
        n = w >> 3;
        for (unsigned j = 0; j < n << 3; j = j + 8) {
            int jj = j - fwidth_half;
            int jj_check = jj;
            __m256i accumrlo, accumdlo, accumrhi, accumdhi;
            accumrlo = accumdlo = accumrhi = accumdhi = _mm256_setzero_si256();
            for (unsigned fj = 0; fj < fwidth; ++fj, jj_check = jj + fj) {
                __m256i refconvol = _mm256_loadu_si256(
                    (__m256i *)(buf.tmp.ref_convol + jj_check));
                __m256i fcoeff = _mm256_set1_epi16(vif_filt[fj]);
                __m256i result2 = _mm256_mulhi_epu16(refconvol, fcoeff);
                __m256i result2lo = _mm256_mullo_epi16(refconvol, fcoeff);
                accumrlo = _mm256_add_epi32(
                    accumrlo, _mm256_unpacklo_epi16(result2lo, result2));
                accumrhi = _mm256_add_epi32(
                    accumrhi, _mm256_unpackhi_epi16(result2lo, result2));
                __m256i disconvol = _mm256_loadu_si256(
                    (__m256i *)(buf.tmp.dis_convol + jj_check));
                result2 = _mm256_mulhi_epu16(disconvol, fcoeff);
                result2lo = _mm256_mullo_epi16(disconvol, fcoeff);
                accumdlo = _mm256_add_epi32(
                    accumdlo, _mm256_unpacklo_epi16(result2lo, result2));
                accumdhi = _mm256_add_epi32(
                    accumdhi, _mm256_unpackhi_epi16(result2lo, result2));
            }

            __m256i addnum = _mm256_set1_epi32(32768);
            accumdlo = _mm256_add_epi32(accumdlo, addnum);
            accumdhi = _mm256_add_epi32(accumdhi, addnum);
            accumrlo = _mm256_add_epi32(accumrlo, addnum);
            accumrhi = _mm256_add_epi32(accumrhi, addnum);
            accumdlo = _mm256_srli_epi32(accumdlo, 0x10);
            accumdhi = _mm256_srli_epi32(accumdhi, 0x10);
            accumrlo = _mm256_srli_epi32(accumrlo, 0x10);
            accumrhi = _mm256_srli_epi32(accumrhi, 0x10);

            __m256i result = _mm256_packus_epi32(accumdlo, accumdhi);
            __m256i resultd = _mm256_packus_epi32(accumrlo, accumrhi);
            __m256i resulttmp = _mm256_srli_si256(resultd, 2);
            resultd = _mm256_blend_epi16(resultd, resulttmp, 0xAA);
            resultd = _mm256_permutevar8x32_epi32(resultd, mask1);
            _mm_storeu_si128((__m128i *)(buf.mu1 + i * stride16 + j),
                             _mm256_castsi256_si128(resultd));

            resulttmp = _mm256_srli_si256(result, 2);
            result = _mm256_blend_epi16(result, resulttmp, 0xAA);
            result = _mm256_permutevar8x32_epi32(result, mask1);
            _mm_storeu_si128((__m128i *)(buf.mu2 + i * stride16 + j),
                             _mm256_castsi256_si128(result));
        }

        for (unsigned j = n << 3; j < w; ++j) {
            uint32_t accum_ref = 0;
            uint32_t accum_dis = 0;
            int jj = j - fwidth_half;
            int jj_check = jj;
            for (unsigned fj = 0; fj < fwidth; ++fj, jj_check = jj + fj) {
                const uint16_t fcoeff = vif_filt[fj];
                accum_ref += fcoeff * ((uint32_t)buf.tmp.ref_convol[jj_check]);
                accum_dis += fcoeff * ((uint32_t)buf.tmp.dis_convol[jj_check]);
            }
            buf.mu1[i * stride16 + j] = (uint16_t)((accum_ref + 32768) >> 16);
            buf.mu2[i * stride16 + j] = (uint16_t)((accum_dis + 32768) >> 16);
        }
    }

    ref = buf.ref;
    dis = buf.dis;

    for (unsigned i = 0; i < h / 2; ++i) {
        for (unsigned j = 0; j < w / 2; ++j) {
            ref[i * stride + j] = buf.mu1[i * stride16 + (j * 2)];
            dis[i * stride + j] = buf.mu2[i * stride16 + (j * 2)];
        }
    }
    pad_top_and_bottom(buf, h / 2, vif_filter1d_width[scale]);
}
