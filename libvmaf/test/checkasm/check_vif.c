/**
 *
 *  Copyright 2016-2026 Netflix, Inc.
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

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <checkasm/checkasm.h>
#include <checkasm/test.h>
#include <checkasm/utils.h>

#include "config.h"
#include "cpu.h"
#include "feature/integer_vif.h"

#if ARCH_X86
#include "feature/x86/vif_avx2.h"
#if HAVE_AVX512
#include "feature/x86/vif_avx512.h"
#endif
#elif ARCH_AARCH64
#include "feature/arm64/vif_neon.h"
#endif

void pad_top_and_bottom(VifBuffer buf, unsigned h, int fwidth);

typedef void (*vif_statistic_8_fn)(struct VifPublicState *s, float *num,
                                    float *den, unsigned w, unsigned h);
typedef void (*vif_statistic_16_fn)(struct VifPublicState *s, float *num,
                                     float *den, unsigned w, unsigned h,
                                     int bpc, int scale);
typedef void (*subsample_rd_8_fn)(VifBuffer buf, unsigned w, unsigned h);
typedef void (*subsample_rd_16_fn)(VifBuffer buf, unsigned w, unsigned h,
                                    int scale, int bpc);

static vif_statistic_8_fn get_vif_statistic_8(unsigned cpu_flags)
{
    vif_statistic_8_fn fn = vif_statistic_8;
#if ARCH_X86
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX2)
        fn = vif_statistic_8_avx2;
#if HAVE_AVX512
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX512)
        fn = vif_statistic_8_avx512;
#endif
#elif ARCH_AARCH64
    if (cpu_flags & VMAF_ARM_CPU_FLAG_NEON)
        fn = vif_statistic_8_neon;
#endif
    return fn;
}

static vif_statistic_16_fn get_vif_statistic_16(unsigned cpu_flags)
{
    vif_statistic_16_fn fn = vif_statistic_16;
#if ARCH_X86
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX2)
        fn = vif_statistic_16_avx2;
#if HAVE_AVX512
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX512)
        fn = vif_statistic_16_avx512;
#endif
#elif ARCH_AARCH64
    if (cpu_flags & VMAF_ARM_CPU_FLAG_NEON)
        fn = vif_statistic_16_neon;
#endif
    return fn;
}

static subsample_rd_8_fn get_subsample_rd_8(unsigned cpu_flags)
{
    subsample_rd_8_fn fn = subsample_rd_8;
#if ARCH_X86
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX2)
        fn = vif_subsample_rd_8_avx2;
#if HAVE_AVX512
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX512)
        fn = vif_subsample_rd_8_avx512;
#endif
#elif ARCH_AARCH64
    if (cpu_flags & VMAF_ARM_CPU_FLAG_NEON)
        fn = vif_subsample_rd_8_neon;
#endif
    return fn;
}

static subsample_rd_16_fn get_subsample_rd_16(unsigned cpu_flags)
{
    subsample_rd_16_fn fn = subsample_rd_16;
#if ARCH_X86
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX2)
        fn = vif_subsample_rd_16_avx2;
#if HAVE_AVX512
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX512)
        fn = vif_subsample_rd_16_avx512;
#endif
#elif ARCH_AARCH64
    if (cpu_flags & VMAF_ARM_CPU_FLAG_NEON)
        fn = vif_subsample_rd_16_neon;
#endif
    return fn;
}

static const struct { unsigned w, h; } sizes[] = {
    { 32,  32 },
    { 64,  36 },
    { 173, 41 },
    { 480, 64 },
};

enum vif_pattern {
    VIF_NOISE,    // independent noise: large sigma1_sq, sigma12 ~ 0
    VIF_FLAT,     // flat blocks, sparse +-1: sigma1_sq < 2 * 65536 branch
    VIF_TRACKING, // dis = ref + small noise: sigma12 > 0, g near 1
    VIF_EXTREME,  // full-range checkerboard: the largest variances
    VIF_N_PATTERNS,
};

static const char *const vif_pattern_names[VIF_N_PATTERNS] = {
    "noise", "flat", "tracking", "extreme",
};

// vif_enhn_gain_limit values: the default, and the 1.0 of the NEG models,
// which clamps g on most of the tracking content.
static const double vif_gain_limits[] = { DEFAULT_VIF_ENHN_GAIN_LIMIT, 1.0 };

static unsigned clip_sample(int v, unsigned max)
{
    return v < 0 ? 0 : v > (int) max ? max : (unsigned) v;
}

static void fill_vif_planes(void *ref, void *dis, ptrdiff_t stride,
                            unsigned w, unsigned h, int bpc,
                            enum vif_pattern pattern)
{
    const unsigned max = (1u << bpc) - 1;
    for (unsigned r = 0; r < h; r++)
        for (unsigned c = 0; c < w; c++) {
            const uint32_t x = checkasm_rand_uint32();
            unsigned rv, dv;
            switch (pattern) {
            case VIF_FLAT:
                rv = (((r / 32) * 7 + (c / 32) * 13) % 11) * max / 10;
                if (x % 29 == 0) rv = rv ? rv - 1 : 1;
                dv = rv;
                if ((x >> 8) % 31 == 0) dv = dv < max ? dv + 1 : dv - 1;
                break;
            case VIF_TRACKING:
                rv = x & max;
                dv = clip_sample((int) rv + (int) ((x >> 20) % 9) - 4, max);
                break;
            case VIF_EXTREME:
                rv = ((r ^ c) & 1) ? max : 0;
                dv = ((x >> 3) % 5) ? rv : max - rv;
                break;
            default:
                rv = x & max;
                dv = checkasm_rand_uint32() & max;
                break;
            }
            if (bpc == 8) {
                ((uint8_t *) ref)[r * stride + c] = (uint8_t) rv;
                ((uint8_t *) dis)[r * stride + c] = (uint8_t) dv;
            } else {
                ((uint16_t *) ref)[r * stride + c] = (uint16_t) rv;
                ((uint16_t *) dis)[r * stride + c] = (uint16_t) dv;
            }
        }
}

static void check_vif_statistic_8(void)
{
    checkasm_declare(void, struct VifPublicState *, float *, float *,
                      unsigned, unsigned);

    if (!checkasm_check_func(get_vif_statistic_8(checkasm_get_cpu_flags()),
                              "vif_statistic_8"))
        return;

    for (int p = 0; p < VIF_N_PATTERNS; p++) {
        for (size_t i = 0; i < sizeof(sizes) / sizeof(*sizes); i++) {
            const unsigned w = sizes[i].w, h = sizes[i].h;

            VifPublicState s;
            if (vif_buffer_alloc(&s.buf, w, h, 8)) continue;
            log_generate(s.log2_table);

            fill_vif_planes(s.buf.ref, s.buf.dis, s.buf.stride, w, h, 8, p);
            pad_top_and_bottom(s.buf, h, vif_filter1d_width[0]);

            float num_c = 0, den_c = 0, num_a = 0, den_a = 0;
            for (size_t g = 0;
                 g < sizeof(vif_gain_limits) / sizeof(*vif_gain_limits); g++)
            {
                s.vif_enhn_gain_limit = vif_gain_limits[g];
                checkasm_call_ref(&s, &num_c, &den_c, w, h);
                checkasm_call_new(&s, &num_a, &den_a, w, h);

                if (num_c != num_a || den_c != den_a) {
                    if (checkasm_fail())
                        fprintf(stderr,
                                "%s %ux%u gain_limit=%g: expected {%f,%f}, "
                                "got {%f,%f}\n",
                                vif_pattern_names[p], w, h,
                                vif_gain_limits[g], num_c, den_c, num_a,
                                den_a);
                }
            }

            if (p == VIF_TRACKING && i == sizeof(sizes) / sizeof(*sizes) - 1) {
                s.vif_enhn_gain_limit = DEFAULT_VIF_ENHN_GAIN_LIMIT;
                checkasm_bench_new(&s, &num_a, &den_a, w, h);
            }

            vif_buffer_free(&s.buf);
        }
    }
}

static void check_vif_statistic_16(void)
{
    checkasm_declare(void, struct VifPublicState *, float *, float *,
                      unsigned, unsigned, int, int);

    static const struct { int bpc, scale; } configs[] = {
        { 10, 0 }, { 12, 0 }, { 16, 0 },
        { 10, 1 }, { 10, 2 }, { 10, 3 },
    };

    for (size_t j = 0; j < sizeof(configs) / sizeof(*configs); j++) {
        const int bpc = configs[j].bpc, scale = configs[j].scale;

        if (!checkasm_check_func(get_vif_statistic_16(checkasm_get_cpu_flags()),
                                  "vif_statistic_16_%dbpc_scale%d", bpc,
                                  scale))
            continue;

        for (int p = 0; p < VIF_N_PATTERNS; p++) {
            for (size_t i = 0; i < sizeof(sizes) / sizeof(*sizes); i++) {
                const unsigned w = sizes[i].w, h = sizes[i].h;

                VifPublicState s;
                if (vif_buffer_alloc(&s.buf, w, h, bpc)) continue;
                log_generate(s.log2_table);

                fill_vif_planes(s.buf.ref, s.buf.dis,
                                s.buf.stride / sizeof(uint16_t), w, h, bpc, p);
                pad_top_and_bottom(s.buf, h, vif_filter1d_width[scale]);

                float num_c = 0, den_c = 0, num_a = 0, den_a = 0;
                for (size_t g = 0;
                     g < sizeof(vif_gain_limits) / sizeof(*vif_gain_limits);
                     g++)
                {
                    s.vif_enhn_gain_limit = vif_gain_limits[g];
                    checkasm_call_ref(&s, &num_c, &den_c, w, h, bpc, scale);
                    checkasm_call_new(&s, &num_a, &den_a, w, h, bpc, scale);

                    if (num_c != num_a || den_c != den_a) {
                        if (checkasm_fail())
                            fprintf(stderr,
                                    "%s %ux%u bpc=%d scale=%d gain_limit=%g: "
                                    "expected {%f,%f}, got {%f,%f}\n",
                                    vif_pattern_names[p], w, h, bpc, scale,
                                    vif_gain_limits[g], num_c, den_c, num_a,
                                    den_a);
                    }
                }

                if (p == VIF_TRACKING &&
                    i == sizeof(sizes) / sizeof(*sizes) - 1)
                {
                    s.vif_enhn_gain_limit = DEFAULT_VIF_ENHN_GAIN_LIMIT;
                    checkasm_bench_new(&s, &num_a, &den_a, w, h, bpc, scale);
                }

                vif_buffer_free(&s.buf);
            }
        }
    }
}

static void check_subsample_rd_8(void)
{
    checkasm_declare(void, VifBuffer, unsigned, unsigned);

    if (!checkasm_check_func(get_subsample_rd_8(checkasm_get_cpu_flags()),
                              "subsample_rd_8"))
        return;

    for (size_t i = 0; i < sizeof(sizes) / sizeof(*sizes); i++) {
        const unsigned w = sizes[i].w, h = sizes[i].h;

        VifBuffer buf_c, buf_a;
        if (vif_buffer_alloc(&buf_c, w, h, 8)) continue;
        if (vif_buffer_alloc(&buf_a, w, h, 8)) {
            vif_buffer_free(&buf_c);
            continue;
        }

        uint8_t *ref_c = buf_c.ref, *dis_c = buf_c.dis;
        uint8_t *ref_a = buf_a.ref, *dis_a = buf_a.dis;
        for (unsigned r = 0; r < h; r++)
            for (unsigned c = 0; c < w; c++) {
                const uint8_t rv = (uint8_t) checkasm_rand_uint32();
                const uint8_t dv = (uint8_t) checkasm_rand_uint32();
                ref_c[r * buf_c.stride + c] = ref_a[r * buf_a.stride + c] = rv;
                dis_c[r * buf_c.stride + c] = dis_a[r * buf_a.stride + c] = dv;
            }
        pad_top_and_bottom(buf_c, h, vif_filter1d_width[1]);
        pad_top_and_bottom(buf_a, h, vif_filter1d_width[1]);

        checkasm_call_ref(buf_c, w, h);
        checkasm_call(checkasm_func_new, buf_a, w, h);

        const ptrdiff_t dec_stride = buf_c.stride / sizeof(uint16_t);
        checkasm_check2d(uint16_t, (uint16_t *) buf_c.ref, dec_stride,
                          (uint16_t *) buf_a.ref, dec_stride, w / 2, h / 2,
                          "decimated_ref");
        checkasm_check2d(uint16_t, (uint16_t *) buf_c.dis, dec_stride,
                          (uint16_t *) buf_a.dis, dec_stride, w / 2, h / 2,
                          "decimated_dis");

        vif_buffer_free(&buf_c);
        vif_buffer_free(&buf_a);
    }
}

static void check_subsample_rd_16(void)
{
    checkasm_declare(void, VifBuffer, unsigned, unsigned, int, int);

    if (!checkasm_check_func(get_subsample_rd_16(checkasm_get_cpu_flags()),
                              "subsample_rd_16"))
        return;

    static const struct { int bpc, scale; } configs[] = {
        { 10, 0 },
        { 10, 1 },
        { 10, 2 },
    };

    for (size_t i = 0; i < sizeof(sizes) / sizeof(*sizes); i++) {
        for (size_t j = 0; j < sizeof(configs) / sizeof(*configs); j++) {
            const unsigned w = sizes[i].w, h = sizes[i].h;
            const int bpc = configs[j].bpc, scale = configs[j].scale;
            const uint16_t mask = (uint16_t) ((1 << bpc) - 1);

            VifBuffer buf_c, buf_a;
            if (vif_buffer_alloc(&buf_c, w, h, bpc)) continue;
            if (vif_buffer_alloc(&buf_a, w, h, bpc)) {
                vif_buffer_free(&buf_c);
                continue;
            }

            uint16_t *ref_c = buf_c.ref, *dis_c = buf_c.dis;
            uint16_t *ref_a = buf_a.ref, *dis_a = buf_a.dis;
            const ptrdiff_t stride = buf_c.stride / sizeof(uint16_t);
            for (unsigned r = 0; r < h; r++)
                for (unsigned c = 0; c < w; c++) {
                    const uint16_t rv =
                        (uint16_t) checkasm_rand_uint32() & mask;
                    const uint16_t dv =
                        (uint16_t) checkasm_rand_uint32() & mask;
                    ref_c[r * stride + c] = ref_a[r * stride + c] = rv;
                    dis_c[r * stride + c] = dis_a[r * stride + c] = dv;
                }
            pad_top_and_bottom(buf_c, h, vif_filter1d_width[scale + 1]);
            pad_top_and_bottom(buf_a, h, vif_filter1d_width[scale + 1]);

            checkasm_call_ref(buf_c, w, h, scale, bpc);
            checkasm_call(checkasm_func_new, buf_a, w, h, scale, bpc);

            char name[64];
            const ptrdiff_t dec_stride = buf_c.stride / sizeof(uint16_t);
            snprintf(name, sizeof(name), "decimated_ref_bpc%d_scale%d", bpc,
                      scale);
            checkasm_check2d(uint16_t, (uint16_t *) buf_c.ref, dec_stride,
                              (uint16_t *) buf_a.ref, dec_stride, w / 2,
                              h / 2, name);
            snprintf(name, sizeof(name), "decimated_dis_bpc%d_scale%d", bpc,
                      scale);
            checkasm_check2d(uint16_t, (uint16_t *) buf_c.dis, dec_stride,
                              (uint16_t *) buf_a.dis, dec_stride, w / 2,
                              h / 2, name);

            vif_buffer_free(&buf_c);
            vif_buffer_free(&buf_a);
        }
    }
}

void checkasm_check_vif(void)
{
    check_vif_statistic_8();
    checkasm_report("vif_statistic_8");

    check_vif_statistic_16();
    checkasm_report("vif_statistic_16");

    check_subsample_rd_8();
    checkasm_report("subsample_rd_8");

    check_subsample_rd_16();
    checkasm_report("subsample_rd_16");
}
