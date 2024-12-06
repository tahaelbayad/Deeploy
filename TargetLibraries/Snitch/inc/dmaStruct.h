/* ----------------------------------------------------------------------
#
# File: dmaStruct.h
#
# Last edited: 03.06.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author:
# - Victor Jung, jungvi@iis.ee.ethz.ch, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
*/

#include "snrt.h"

typedef struct {
  void *dst;
  void *src;
  size_t size;
  size_t dst_stride;
  size_t src_stride;
  size_t repeat;
  snrt_dma_txid_t tid;
} DMA_copy;