# ----------------------------------------------------------------------
#
# File: SnitchPlatform.py
#
# Last edited: 23.04.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Authors:
# - Philip Wiese (wiesep@iis.ee.ethz.ch), ETH Zurich
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
# limitations under the License.

from typing import List

import numpy as np

from Deeploy.DeeployTypes import ConstantBuffer, DeploymentEngine, DeploymentPlatform, NodeMapper, NodeTemplate, \
    StructBuffer, TopologyOptimizer, TransientBuffer, VariableBuffer
from Deeploy.Targets.Generic.Bindings import BasicGatherBindings, BasicMatMulBinding, BasicPad1DBindings, \
    BasicPad2DBindings, BasicReshapeBindings, BasicRQIntegerDivBinding
from Deeploy.Targets.Generic.Layers import GatherLayer, MatMulLayer, PadLayer, ReshapeLayer, RQIntegerDivLayer
from Deeploy.Targets.Generic.Parsers import GatherParser, MatMulParser, Pad1DParser, Pad2DParser, RQIntegerDivParser, \
    UnsqueezeParser
from Deeploy.Targets.Generic.Templates import AllocateTemplate as BasicAllocateTemplate
from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import IntegerDivRequantMergePass, \
    MergeConstAddAndRequantPass, MergeTrueIntegerDivRequantShiftPass, RQSSplitPass, SkipEmptyConcatPass, \
    SkipUnityRequantPass, iGELURequantMergePass, iHardswishRequantMergePass
from Deeploy.Targets.Snitch.Templates import AllocateTemplate, FreeTemplate

GatherMapper = NodeMapper(GatherParser(), BasicGatherBindings)
Pad1DMapper = NodeMapper(Pad1DParser(), BasicPad1DBindings)
Pad2DMapper = NodeMapper(Pad2DParser(), BasicPad2DBindings)
UnsqueezeMapper = NodeMapper(UnsqueezeParser(), BasicReshapeBindings)

RQIntegerDivMapper = NodeMapper(RQIntegerDivParser(), [BasicRQIntegerDivBinding])

MatMulMapper = NodeMapper(MatMulParser(), [BasicMatMulBinding])

SnitchMapping = {
    'RQIntegerDiv': RQIntegerDivLayer([RQIntegerDivMapper]),
    'Gather': GatherLayer([GatherMapper]),
    'Pad': PadLayer([Pad1DMapper, Pad2DMapper]),
    'Unsqueeze': ReshapeLayer([UnsqueezeMapper]),
    'MatMul': MatMulLayer([MatMulMapper]),
}


class SnitchVariableBuffer(VariableBuffer):

    initTemplate = AllocateTemplate.snitchL2InitTemplate
    allocTemplate = AllocateTemplate.snitchGenericAllocate
    deallocTemplate = FreeTemplate.snitchGenericFree

    def _bufferRepresentation(self):

        if hasattr(self, "_memoryLevel"):
            memoryLevel = self._memoryLevel
        else:
            memoryLevel = None

        return {
            "type": self._instance,
            "name": self.name,
            "size": int(np.prod(self.shape)),
            "_memoryLevel": memoryLevel
        }


class SnitchTransientBuffer(TransientBuffer):

    initTemplate = AllocateTemplate.snitchL2InitTemplate
    allocTemplate = AllocateTemplate.snitchGenericAllocate
    deallocTemplate = FreeTemplate.snitchGenericFree

    # allocTemplate = AllocateTemplate.snitchL2AllocateTemplate
    # deallocTemplate = FreeTemplate.snitchL2GlobalTemplate

    def _bufferRepresentation(self):

        if hasattr(self, "_memoryLevel"):
            memoryLevel = self._memoryLevel
        else:
            memoryLevel = None

        return {"type": self._type, "name": self.name, "size": self.size, "_memoryLevel": memoryLevel}


class SnitchConstantBuffer(ConstantBuffer):

    initTemplate = AllocateTemplate.snitchGenericGlobalInitTemplate
    allocTemplate = AllocateTemplate.snitchL2GlobalAllocateTemplate
    deallocTemplate = FreeTemplate.snitchL2GlobalTemplate

    def _bufferRepresentation(self):
        nodeRep = super()._bufferRepresentation()

        if hasattr(self, "_memoryLevel"):
            memoryLevel = self._memoryLevel
        else:
            memoryLevel = None

        nodeRep["_memoryLevel"] = memoryLevel

        return nodeRep


class SnitchStructBuffer(StructBuffer):

    initTemplate = BasicAllocateTemplate.referenceStructInitTemplate
    allocTemplate = BasicAllocateTemplate.referenceStructAllocateTemplate
    deallocTemplate = NodeTemplate("")


SnitchOptimizer = TopologyOptimizer([
    SkipEmptyConcatPass(),
    SkipUnityRequantPass(previous_op_regex = "Concat", num_inputs = 2),
    SkipUnityRequantPass(previous_op_regex = "Reshape|Transpose", num_inputs = 1),
    SkipUnityRequantPass(previous_op_regex = "Reshape|Transpose", num_inputs = 1),
    RQSSplitPass(),
    MergeTrueIntegerDivRequantShiftPass(),
    IntegerDivRequantMergePass(),
    iGELURequantMergePass(),
    iHardswishRequantMergePass(),
    MergeConstAddAndRequantPass(),
])

_includeList = [
    "snrt.h",
    "DeeploySnitchMath.h",
]


class SnitchClusterEngine(DeploymentEngine):

    def __init__(self, name: str, Mapping = SnitchMapping, initCode = "", includeList = _includeList) -> None:
        super().__init__(name, Mapping, initCode, includeList)


class SnitchPlatform(DeploymentPlatform):

    def __init__(self,
                 engines = [SnitchClusterEngine("SnitchCluster")],
                 variableBuffer = SnitchVariableBuffer,
                 constantBuffer = SnitchConstantBuffer,
                 structBuffer = SnitchStructBuffer,
                 transientBuffer = SnitchTransientBuffer,
                 includeList: List[str] = _includeList):
        super().__init__(engines, variableBuffer, constantBuffer, structBuffer, transientBuffer)