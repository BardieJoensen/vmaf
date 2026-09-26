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

/*
 * The AVX2 VIF kernels, built with -mavxvnni: every vpmaddwd + vpaddd
 * accumulation becomes a single vpdpwssd (see vif_madd_acc()).
 */
#define VIF_AVXVNNI 1
#include "vif_avx2.c"
