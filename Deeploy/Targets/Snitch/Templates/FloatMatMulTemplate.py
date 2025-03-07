# ----------------------------------------------------------------------
#
# File: MatMul.py.py
#
# Last edited: 27.01.2025
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author: Run Wang, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the Licens
from Deeploy.DeeployTypes import NodeTemplate


referenceTemplate = NodeTemplate("""
// GEMM (Name: ${nodeName}, Op: ${nodeOp})
uint32_t core_id = snrt_global_core_idx();
uint32_t compute_num = snrt_cluster_compute_core_num();

${A_type.typeName} ref_${data_out}_${A} = ${A};
${B_type.typeName} ref_${data_out}_${B} = ${B};
${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

for(uint32_t i=0; i<${batch}; i++){
    gemm_fp32_naive(${M} / compute_num, ${O}, ${N}, ref_${data_out}_${A}, ${N} * compute_num, ref_${data_out}_${B}, ${O}, 0, ${O} * compute_num, ref_${data_out}_${data_out}, 0, 1 );
    //MatMul_fp32( ref_${data_out}_${A}, ref_${data_out}_${B}, ref_${data_out}_${data_out}, ${M}, ${N}, ${O} );

    ref_${data_out}_${A} += ${M} * ${N};
    ref_${data_out}_${B} += ${N} * ${O};
    ref_${data_out}_${data_out} += ${M} * ${O};
    }
                
""")



# referenceTemplate = NodeTemplate("""
# // GEMM (Name: ${nodeName}, Op: ${nodeOp})
# uint32_t compute_num = snrt_cluster_compute_core_num();

# ${A_type.typeName} ref_${data_out}_${A} = ${A};
# ${B_type.typeName} ref_${data_out}_${B} = ${B};
# ${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};
# for(uint32_t i=0; i<${batch}; i++){                             
# % if transB:
#     gemm_fp32_transB_opt(${M} / compute_num, ${O}, ${N}, ref_${data_out}_${A}, ${N} * compute_num, ref_${data_out}_${B}, ${N}, 0, ${O} * compute_num, ref_${data_out}_${data_out}, 0, 1 );
# % else:                                 
#     gemm_fp32_opt(${M} / compute_num, ${O}, ${N}, ref_${data_out}_${A}, ${N} * compute_num, ref_${data_out}_${B}, ${O}, 0, ${O} * compute_num, ref_${data_out}_${data_out}, 0, 1 );
# %endif

#     ref_${data_out}_${A} += ${M} * ${N};
#     ref_${data_out}_${B} += ${N} * ${O};
#     ref_${data_out}_${data_out} += ${M} * ${O};
# }

# """)