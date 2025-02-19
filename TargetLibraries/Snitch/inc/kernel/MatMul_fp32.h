#ifndef __DEEPLOY_MATH_ADD_KERNEL_HEADER_
#define __DEEPLOY_MATH_ADD_KERNEL_HEADER_
#include "DeeploySnitchMath.h"


void MatMul_fp32(const float32_t *__restrict__ pSrcA,
    const float32_t *__restrict__ pSrcB,
    float32_t *__restrict__ pDstY,
    uint32_t M,
    uint32_t N, 
    uint32_t O);

#endif