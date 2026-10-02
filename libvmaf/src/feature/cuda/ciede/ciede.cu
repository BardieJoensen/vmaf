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

// CUDA port of feature/ciede.c, which is in large part a port of the
// ciede2000 implementation from av-metrics
// (https://github.com/rust-av/av-metrics) with the following license:

/*
The MIT License (MIT)
Copyright (c) 2019 Joshua Holmer

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/*
 * This implementation uses float32 color conversion and CIEDE math, with
 * double partial sums and deterministic host pooling. The CPU implementation
 * uses double color conversion and float LAB/CIEDE intermediates, together
 * with a different math library; bit-exact agreement is not expected.
 *
 * A 0.001 score tolerance is checked for the noisy-frame parity fixtures.
 * It is not a universal bound: near-identical colors can have larger score
 * differences because 45 - 20*log10(mean Delta E) magnifies errors near zero.
 * Accuracy should also be evaluated in the underlying mean Delta E domain.
 */

#include "cuda_helper.cuh"

#include "common.h"

#define M_PI_F 3.14159265358979323846f

typedef struct LABColorf {
    float l;
    float a;
    float b;
} LABColorf;

__device__ __forceinline__ float get_h_prime(const float x, const float y)
{
    if ((x == 0.0f) && (y == 0.0f))
        return 0.0f;
    float hue_angle = atan2f(x, y);
    if (hue_angle < 0.0f)
        hue_angle += 2.f * M_PI_F;
    return hue_angle;
}

__device__ __forceinline__ float get_delta_h_prime(const float c1,
        const float c2, const float h_prime_1, const float h_prime_2)
{
    if ((c1 == 0.0f) || (c2 == 0.0f))
        return 0.0f;
    if (fabsf(h_prime_1 - h_prime_2) <= M_PI_F)
        return h_prime_2 - h_prime_1;
    if (h_prime_2 <= h_prime_1)
        return h_prime_2 - h_prime_1 + 2.f * M_PI_F;
    else
        return h_prime_2 - h_prime_1 - 2.f * M_PI_F;
}

__device__ __forceinline__ float get_upcase_h_bar_prime(const float h_prime_1,
        const float h_prime_2)
{
    return fabsf(h_prime_1 - h_prime_2) > M_PI_F ?
        (h_prime_1 + h_prime_2 + 2.0f * M_PI_F) / 2.0f :
        (h_prime_1 + h_prime_2) / 2.0f;
}

__device__ __forceinline__ float get_upcase_t(const float upcase_h_bar_prime)
{
    return 1.0f -
           0.17f * cosf(upcase_h_bar_prime - M_PI_F / 6.0f) +
           0.24f * cosf(2.0f * upcase_h_bar_prime) +
           0.32f * cosf(3.0f * upcase_h_bar_prime + M_PI_F / 30.0f) -
           0.20f * cosf(4.0f * upcase_h_bar_prime - 7.0f * M_PI_F / 20.0f);
}

__device__ __forceinline__ float get_r_sub_t(const float c_bar_prime,
        const float upcase_h_bar_prime)
{
    const float degrees =
        (upcase_h_bar_prime * (180.0f / M_PI_F) - 275.0f) * (1.0f / 25.0f);
    const float c7 = powf(c_bar_prime, 7);

    return -2.0f *
          sqrtf(c7 / (c7 + powf(25.f, 7))) *
          sinf((60.0f * expf(-(degrees * degrees))) * (M_PI_F / 180.0f));
}

__device__ float ciede2000(LABColorf color_1, LABColorf color_2)
{
    // default ksub from feature/ciede.c: l = 0.65, c = 1.0, h = 4.0
    const float ksub_l = 0.65f, ksub_c = 1.0f, ksub_h = 4.0f;

    const float delta_l_prime = color_2.l - color_1.l;
    const float l_bar = (color_1.l + color_2.l) / 2;
    const float c1 = sqrtf(color_1.a * color_1.a + color_1.b * color_1.b);
    const float c2 = sqrtf(color_2.a * color_2.a + color_2.b * color_2.b);
    const float c_bar = (c1 + c2) / 2;
    const float c_bar7 = powf(c_bar, 7);
    const float chroma_shift = 1 - sqrtf(c_bar7 / (c_bar7 + powf(25.f, 7)));
    const float a_prime_1 = color_1.a + (color_1.a / 2) * chroma_shift;
    const float a_prime_2 = color_2.a + (color_2.a / 2) * chroma_shift;
    const float c_prime_1 =
        sqrtf(a_prime_1 * a_prime_1 + color_1.b * color_1.b);
    const float c_prime_2 =
        sqrtf(a_prime_2 * a_prime_2 + color_2.b * color_2.b);
    const float c_bar_prime = (c_prime_1 + c_prime_2) / 2;
    const float delta_c_prime = c_prime_2 - c_prime_1;
    const float s_sub_l = 1.f + ((0.015f * ((l_bar - 50) * (l_bar - 50))) /
                          sqrtf(20 + ((l_bar - 50) * (l_bar - 50))));
    const float s_sub_c = 1.f + 0.045f * c_bar_prime;
    const float h_prime_1 = get_h_prime(color_1.b, a_prime_1);
    const float h_prime_2 = get_h_prime(color_2.b, a_prime_2);
    const float delta_h_prime = get_delta_h_prime(c1, c2, h_prime_1, h_prime_2);
    const float delta_upcase_h_prime =
            2.0f * sqrtf(c_prime_1 * c_prime_2) * sinf(delta_h_prime / 2.0f);
    const float upcase_h_bar_prime =
        get_upcase_h_bar_prime(h_prime_1, h_prime_2);
    const float upcase_t = get_upcase_t(upcase_h_bar_prime);
    const float s_sub_upcase_h = 1.0f + 0.015f * c_bar_prime * upcase_t;
    const float r_sub_t = get_r_sub_t(c_bar_prime, upcase_h_bar_prime);
    const float lightness = delta_l_prime / (ksub_l * s_sub_l);
    const float chroma  = delta_c_prime / (ksub_c * s_sub_c);
    const float hue = delta_upcase_h_prime / (ksub_h * s_sub_upcase_h);

    return sqrtf(lightness * lightness + chroma * chroma +
                 hue * hue + r_sub_t * chroma * hue);
}

__device__ double pow_2_4(double x)
{
    return pow(x, 2.4);
}

__device__ double rgb_to_xyz_map(double c)
{
    if (c > 10. / 255.) {
        const double A = 0.055;
        const double D = 1.0 / 1.055;
        return pow_2_4((c + A) * D);
    } else {
        const double D = 1.0 / 12.92;
        return (c * D);
    }
}

__device__ double cbrt_approx(double c)
{
    return pow(c, 1.0 / 3.0);
}

__device__ float xyz_to_lab_map(double c)
{
    const double KAPPA = 24389.0 / 27.0;
    const double EPSILON = 216.0 / 24389.0;

    if (c > EPSILON) {
        return cbrt_approx(c);
    } else {
        return (KAPPA * c + 16.0) * (1.0 / 116.0);
    }
}

__device__ LABColorf get_lab_color(double y, double u, double v, unsigned bpc)
{
    const double scale = 1 << (bpc - 8);

    y = (y - 16.  * scale) * (1. / (219. * scale));
    u = (u - 128. * scale) * (1. / (224. * scale));
    v = (v - 128. * scale) * (1. / (224. * scale));

    // Assumes BT.709
    double r = y + 1.28033 * v;
    double g = y - 0.21482 * u - 0.38059 * v;
    double b = y + 2.12798 * u;

    r = rgb_to_xyz_map(r);
    g = rgb_to_xyz_map(g);
    b = rgb_to_xyz_map(b);

    double x = r * 0.4124564390896921 + g * 0.357576077643909 +
              b * 0.18043748326639894;
          y = r * 0.21267285140562248 + g * 0.715152155287818 +
              b * 0.07217499330655958;
    double z = r * 0.019333895582329317 + g * 0.119192025881303 +
              b * 0.9503040785363677;

    x = xyz_to_lab_map(x * (1.0 / 0.95047));
    y = xyz_to_lab_map(y);
    z = xyz_to_lab_map(z * (1.0 / 1.08883));

    LABColorf lab_color = {
        (116.0 * y) - 16.0,
        500.0 * (x - y),
        200.0 * (y - z),
    };

    return lab_color;
}

// Block-reduce per-thread ΔE (double partials) and write one partial per
// block; the host sums the partials sequentially so pooling is deterministic
__device__ __forceinline__ void block_reduce_de(double de,
        VmafCudaBuffer partials)
{
    __shared__ double sh[256];

    const int t = threadIdx.y * blockDim.x + threadIdx.x;
    sh[t] = de;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (t < stride)
            sh[t] += sh[t + stride];
        __syncthreads();
    }
    if (t == 0)
        reinterpret_cast<double*>(partials.data)[
            blockIdx.y * gridDim.x + blockIdx.x] = sh[0];
}

// scale_chroma_planes in feature/ciede.c halves the column index for
// horizontal subsampling and advances the source row every second output row
// for vertical subsampling.
__device__ __forceinline__ int chroma_col(int j, int ss_hor)
{
    return ss_hor ? j / 2 : j;
}

__device__ __forceinline__ int chroma_row(int i, int ss_ver)
{
    return ss_ver ? i / 2 : i;
}

extern "C" {

__global__ void ciede_kernel_8bpc(const VmafPicture ref, const VmafPicture dis,
        VmafCudaBuffer partials, unsigned width, unsigned height,
        int ss_hor, int ss_ver)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;

    double de00 = 0.0;
    if (j < width && i < height) {
        const int cj = chroma_col(j, ss_hor);
        const int ci = chroma_row(i, ss_ver);

        const float r_y = (reinterpret_cast<const uint8_t*>(ref.data[0]) +
                (size_t)i * ref.stride[0])[j];
        const float r_u = (reinterpret_cast<const uint8_t*>(ref.data[1]) +
                (size_t)ci * ref.stride[1])[cj];
        const float r_v = (reinterpret_cast<const uint8_t*>(ref.data[2]) +
                (size_t)ci * ref.stride[2])[cj];
        const float d_y = (reinterpret_cast<const uint8_t*>(dis.data[0]) +
                (size_t)i * dis.stride[0])[j];
        const float d_u = (reinterpret_cast<const uint8_t*>(dis.data[1]) +
                (size_t)ci * dis.stride[1])[cj];
        const float d_v = (reinterpret_cast<const uint8_t*>(dis.data[2]) +
                (size_t)ci * dis.stride[2])[cj];

        const LABColorf color_1 = get_lab_color(r_y, r_u, r_v, 8);
        const LABColorf color_2 = get_lab_color(d_y, d_u, d_v, 8);
        de00 = ciede2000(color_1, color_2);
    }

    block_reduce_de(de00, partials);
}

__global__ void ciede_kernel_16bpc(const VmafPicture ref, const VmafPicture dis,
        VmafCudaBuffer partials, unsigned width, unsigned height,
        int ss_hor, int ss_ver)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;

    double de00 = 0.0;
    if (j < width && i < height) {
        const int cj = chroma_col(j, ss_hor);
        const int ci = chroma_row(i, ss_ver);

        const float r_y = reinterpret_cast<const uint16_t*>(
                reinterpret_cast<const uint8_t*>(ref.data[0]) +
                (size_t)i * ref.stride[0])[j];
        const float r_u = reinterpret_cast<const uint16_t*>(
                reinterpret_cast<const uint8_t*>(ref.data[1]) +
                (size_t)ci * ref.stride[1])[cj];
        const float r_v = reinterpret_cast<const uint16_t*>(
                reinterpret_cast<const uint8_t*>(ref.data[2]) +
                (size_t)ci * ref.stride[2])[cj];
        const float d_y = reinterpret_cast<const uint16_t*>(
                reinterpret_cast<const uint8_t*>(dis.data[0]) +
                (size_t)i * dis.stride[0])[j];
        const float d_u = reinterpret_cast<const uint16_t*>(
                reinterpret_cast<const uint8_t*>(dis.data[1]) +
                (size_t)ci * dis.stride[1])[cj];
        const float d_v = reinterpret_cast<const uint16_t*>(
                reinterpret_cast<const uint8_t*>(dis.data[2]) +
                (size_t)ci * dis.stride[2])[cj];

        const LABColorf color_1 = get_lab_color(r_y, r_u, r_v, ref.bpc);
        const LABColorf color_2 = get_lab_color(d_y, d_u, d_v, dis.bpc);
        de00 = ciede2000(color_1, color_2);
    }

    block_reduce_de(de00, partials);
}

}
