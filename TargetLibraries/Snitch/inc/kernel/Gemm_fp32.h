#ifndef __DEEPLOY_MATH_GEMM_KERNEL_HEADER_
#define __DEEPLOY_MATH_GEMM_KERNEL_HEADER_

#include "DeeploySnitchMath.h"

/*
 * 
 * FP32 GEMM with the following format:
 * A is an M x K matrix, B is a K x N matrix, and C is a M x N matrix
 * 
 * A' = transpose(A) if transA else A
 * B' = transpose(B) if transB else B
 * 
 * Y = ALPHA * A' * B' + C * BETA
 * 
 */


/*
 * 
 * transposed A    = no
 * transposed B    = yes
 * multi-core      = yes
 * unrolling       = yes
 * simd            = yes
 * parallelization = row-wise
 */

void gemm_fp32_transB_opt(uint32_t M, uint32_t N, uint32_t K, float32_t *A,
                   uint32_t ldA, float32_t *B, uint32_t ldB, float32_t *C,
                   uint32_t ldC, float32_t *Y, uint32_t BETA,
                   uint32_t setup_SSR);


/*
 * 
 * transposed A    = no
 * transposed B    = no
 * multi-core      = yes
 * unrolling       = yes
 * simd            = yes
 * parallelization = row-wise
 */


void gemm_fp32_opt(uint32_t M, uint32_t N, uint32_t K, float32_t *A, uint32_t ldA,
               float32_t *B, uint32_t ldB, float32_t *C, uint32_t ldC,
               float32_t *Y, uint32_t BETA, uint32_t setup_SSR);



#endif //__DEEPLOY_MATH_GEMM_KERNEL_HEADER_