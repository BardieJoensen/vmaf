/**
 *
 *  Copyright 2026 Bardie Høgh Joensen
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

#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#include "test.h"
#include "libvmaf/libvmaf.h"
#include "libvmaf/libvmaf_cuda.h"
#include "cuda/common.h"

static CudaFunctions driver;
static unsigned delayed_resets;

static void CUDAAPI delay(void *unused)
{
    (void)unused;
    const struct timespec interval = { .tv_nsec = 5000000 };
    nanosleep(&interval, NULL);
}

static CUresult CUDAAPI delayed_memset(CUdeviceptr dst, unsigned char value,
                                       size_t size, CUstream stream)
{
    /* Add backpressure to the SAD reset's own stream. The motion kernel
     * must wait for this reset even when another stream can run sooner. */
    if (size == sizeof(uint64_t)) {
        CUresult err = driver.cuLaunchHostFunc(stream, delay, NULL);
        if (err) return err;
        delayed_resets++;
    }
    return driver.cuMemsetD8Async(dst, value, size, stream);
}

static char *run_motion(unsigned bpc)
{
    VmafContext *ctx;
    int err = vmaf_init(&ctx, (VmafConfiguration){ .log_level = VMAF_LOG_LEVEL_ERROR });
    mu_assert("context init failed", !err);
    VmafCudaState *state = NULL;
    err = vmaf_cuda_state_init(&state, (VmafCudaConfiguration){0});
    if (err) { vmaf_close(ctx); exit(77); }
    driver = *state->f;
    state->f->cuMemsetD8Async = delayed_memset;
    delayed_resets = 0;
    err = vmaf_cuda_import_state(ctx, state);
    mu_assert("CUDA import failed", !err);
    err = vmaf_use_feature(ctx, "motion_cuda", NULL);
    mu_assert("motion registration failed", !err);
    for (unsigned i = 0; i < 6; i++) {
        VmafPicture ref, dist;
        err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, bpc, 64, 48);
        err |= vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, bpc, 64, 48);
        mu_assert("picture allocation failed", !err);
        for (unsigned p = 0; p < 3; p++)
            for (unsigned y = 0; y < ref.h[p]; y++) {
                uint8_t *r = (uint8_t *)ref.data[p] + y * ref.stride[p];
                uint8_t *d = (uint8_t *)dist.data[p] + y * dist.stride[p];
                const unsigned value = (32 + i) << (bpc - 8);
                for (unsigned x = 0; x < ref.w[p]; x++) {
                    if (bpc == 8) r[x] = d[x] = value;
                    else ((uint16_t *)r)[x] = ((uint16_t *)d)[x] = value;
                }
            }
        err = vmaf_read_pictures(ctx, &ref, &dist, i);
        mu_assert("motion extraction failed", !err);
    }
    err = vmaf_read_pictures(ctx, NULL, NULL, 0);
    mu_assert("motion flush failed", !err);
    int mismatch = 0;
    for (unsigned i = 0; i < 6; i++) {
        double score;
        err = vmaf_feature_score_at_index(ctx, "VMAF_integer_feature_motion2_score", &score, i);
        mu_assert("missing motion score", !err);
        if (score != (i ? 1.0 : 0.0)) {
            fprintf(stderr, "bpc=%u frame=%u motion2=%.12f expected=%u\n", bpc, i, score, i ? 1 : 0);
            mismatch = 1;
        }
    }
    err = vmaf_close(ctx);
    free(state);
    mu_assert("motion close failed", !err);
    mu_assert("SAD reset delay was not exercised", delayed_resets == 6);
    mu_assert("motion kernel did not wait for its accumulator reset", !mismatch);
    return NULL;
}

static char *test_motion_order(void)
{
    char *fail = run_motion(8);
    return fail ? fail : run_motion(10);
}

char *run_tests(void)
{
    mu_run_test(test_motion_order);
    return NULL;
}
