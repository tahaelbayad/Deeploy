from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class SnitchFloatGemmTemplate(NodeTemplate):
    
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        
        operatorRepresentation['kernelName'] = 'gemm_fp32_opt'
        return ctxt, operatorRepresentation, []
    
SnitchFloatGemmTemplateStr = r"""
    uint32_t ldA = ${N};
    uint32_t ldC = ${O};
    uint32_t ldB = ${N};    
    uint32_t beta = ${beta};
    
    ${kernelName}( ${M}, ${O}, ${N}, ${A}, ldA, ${B}, ldB, ${C}, ldC, ${data_out}, &beta, 1);
"""
SnitchFloatGemm_Template = SnitchFloatGemmTemplate(SnitchFloatGemmTemplateStr)