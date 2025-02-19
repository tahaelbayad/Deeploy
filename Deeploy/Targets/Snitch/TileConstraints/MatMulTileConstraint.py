# from typing import Dict, List, Tuple

# from Deeploy.AbstractDataTypes import PointerClass
# from Deeploy.CommonExtensions.DataTypes import uint32_t
# from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
# from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
# from Deeploy.TilingExtension.TileConstraint import TileConstraint
# from Deeploy.TilingExtension.TilerModel import PerformanceHint, TilerModel
# from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
#     VariableReplacementScheme


# class MatMulTileConstraint(TileConstraint):

#     @staticmethod
#     def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

#         # Get to-be-tiled tensor's buffers
#         bufferA = ctxt.lookup(name = parseDict['A'])
#         bufferB = ctxt.lookup(name = parseDict['B'])
#         bufferY = ctxt.lookup(name = parseDict['data_out'])

#         # Add I/O dimensions to the model as variables
#         for bufferName in [bufferA.name, bufferB.name, bufferY.name]:
#             tilerModel.addTensorDimToModel(ctxt, bufferName)

#         dimCountA = len(bufferA.shape)

#         if parseDict['transA'] == 0:
#             heightIdxA, widthIdxA = dimCountA - 2, dimCountA - 1
#         else:
#             heightIdxA, widthIdxA = dimCountA - 1, dimCountA - 2
#         AHeightDimVar = tilerModel.getTensorDimVar(tensorName = bufferA.name, dimIdx = heightIdxA)
#         AWidthDimVar = tilerModel.getTensorDimVar(tensorName = bufferA.name, dimIdx = widthIdxA)

#         dimCountB = len(bufferB.shape)
#         if parseDict['transB'] == 0:
#             heightIdxB, widthIdxB = dimCountB - 2, dimCountB - 1
#         else:
#             heightIdxB, widthIdxB = dimCountB - 1, dimCountB - 2
#         BHeightDimVar = tilerModel.getTensorDimVar(tensorName = bufferB.name, dimIdx = heightIdxB)
#         BWidthDimVar = tilerModel.getTensorDimVar(tensorName = bufferB.name, dimIdx = widthIdxB)

#         dimCountY = len(bufferY.shape)
#         heightIdxY, widthIdxY = dimCountY - 2, dimCountY - 1
#         YHeightDimVar = tilerModel.getTensorDimVar(tensorName = bufferY.name, dimIdx = heightIdxY)
#         YWidthDimVar = tilerModel.getTensorDimVar(tensorName = bufferY.name, dimIdx = widthIdxY)

#         tilerModel.addConstraint(YHeightDimVar == AHeightDimVar)
#         tilerModel.addConstraint(YWidthDimVar == BWidthDimVar)
#         tilerModel.addConstraint(AWidthDimVar == BHeightDimVar)

#         if 'C' in parseDict:
#             bufferC = ctxt.lookup(name = parseDict['C'])

#             tilerModel.addTensorDimToModel(ctxt, bufferC.name)

#             dimCountC = len(bufferC.shape)
#             heightIdxC, widthIdxC = dimCountC - 2, dimCountC - 1
#             CHeightDimVar = tilerModel.getTensorDimVar(tensorName = bufferC.name, dimIdx = heightIdxC)
#             CWidthDimVar = tilerModel.getTensorDimVar(tensorName = bufferC.name, dimIdx = widthIdxC)

#             tilerModel.addConstraint(CHeightDimVar == YHeightDimVar)
#             tilerModel.addConstraint(CWidthDimVar == YWidthDimVar)

#         return tilerModel

#     @staticmethod
#     def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
#         bufferA = ctxt.lookup(name = parseDict['A'])
#         bufferY = ctxt.lookup(name = parseDict['data_out'])

#         dimCountA = len(bufferA.shape)
#         if parseDict['transA'] == 0:
#             heightIdxA, widthIdxA = dimCountA - 2, dimCountA - 1
#         else:
#             heightIdxA, widthIdxA = dimCountA - 1, dimCountA - 2
#         AHeightDimVar = tilerModel.getTensorDimVar(tensorName = bufferA.name, dimIdx = heightIdxA)
#         AWidthDimVar = tilerModel.getTensorDimVar(tensorName = bufferA.name, dimIdx = widthIdxA)

#         dimCountY = len(bufferY.shape)
#         heightIdxY, widthIdxY = dimCountY - 2, dimCountY - 1
#         YHeightDimVar = tilerModel.getTensorDimVar(tensorName = bufferY.name, dimIdx = heightIdxY)
#         YWidthDimVar = tilerModel.getTensorDimVar(tensorName = bufferY.name, dimIdx = widthIdxY)

#         # Full inner dimension
#         tilerModel.addConstraint(AWidthDimVar == AWidthDimVar.Max())

#         # We parallelize over the output height dimension so try to keep it divisible by the number of cores (8)
#         if parseDict["M"] > 8:
#             tilerModel.addTileSizeDivisibleConstraint(parseDict,
#                                                       "M",
#                                                       YHeightDimVar,
#                                                       8,
#                                                       strategy = PerformanceHint(priority = 1))

#         return tilerModel

#     @classmethod
#     def serializeTilingSolution(
#             cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
#             targetMemLevel: str, ctxt: NetworkContext,
#             operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
#         outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

#         addrNames = ['A', 'B', 'data_out']
#         inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
#                                                                   operatorRepresentation, addrNames)

#         NOffset = 0
#         NSize = operatorRepresentation["N"]

#         replacements = {
#             "M": [],
#             "O": [],
#             "batch": [],
#         }

#         replacementTypes = {
#             "M": PointerClass(uint32_t),
#             "O": PointerClass(uint32_t),
#             "batch": PointerClass(uint32_t),
#         }

#         inputLoadSchedule = []
#         outputLoadSchedule = []

#         for YCube in outputCubes:
#             assert len(YCube.offset) >= 2 or len(
#                 YCube.offset) <= 3, f"Unsupported YCube dimensionality: {len(YCube.offset)}"

#             MOffset, OOffset = YCube.offset[-2:]
#             MSize, OSize = YCube.dims[-2:]

#             replacements["M"].append(MSize)
#             replacements["O"].append(OSize)

#             if len(YCube.offset) == 3:
#                 BatchOffset = YCube.offset[0]
#                 BatchSize = YCube.dims[0]
#             else:
#                 BatchOffset = 0
#                 BatchSize = 1

#             replacements["batch"].append(BatchSize)

#             if operatorRepresentation['transA'] == 0:
#                 ACube = HyperRectangle((BatchOffset, MOffset, NOffset), (BatchSize, MSize, NSize))
#             else:
#                 ACube = HyperRectangle((BatchOffset, NOffset, MOffset), (BatchSize, NSize, MSize))

#             if operatorRepresentation['transB'] == 0:
#                 BCube = HyperRectangle((BatchOffset, NOffset, OOffset), (BatchSize, NSize, OSize))
#             else:
#                 BCube = HyperRectangle((BatchOffset, OOffset, NOffset), (BatchSize, OSize, NSize))

#             inputLoadSchedule.append({"A": ACube, "B": BCube})
#             outputLoadSchedule.append({"data_out": YCube})

#         schedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)

#         return VariableReplacementScheme(replacements, replacementTypes), schedule
# ----------------------------------------------------------------------
#
# File: MatMulTileConstraint.py
#
# Last edited: 04.07.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author:
# - Victor Jung, jungvi@iis.ee.ethz.ch, ETH Zurich
# - Moritz Scherer, scheremo@iis.ee.ethz.ch, ETH Zurich
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

from typing import Dict, List, Tuple

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import int8_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class MatMulTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        # Get to-be-tiled tensor's buffers
        bufferA = ctxt.lookup(name = parseDict['A'])
        bufferB = ctxt.lookup(name = parseDict['B'])
        outputBuffer = ctxt.lookup(name = parseDict['data_out'])

        # Add I/O dimensions to the model as variables
        for _buffer in [bufferA, bufferB, outputBuffer]:
            tilerModel.addTensorDimToModel(ctxt, _buffer.name)

        tensorsShapeLen = len(bufferA.shape)

        AFirstDimVar = tilerModel.getTensorDimVar(tensorName = bufferA.name,
                                                  dimIdx = (tensorsShapeLen - 2) + parseDict['transA'])
        ASecondDimVar = tilerModel.getTensorDimVar(tensorName = bufferA.name,
                                                   dimIdx = (tensorsShapeLen - 1) - parseDict['transA'])
        BFirstDimVar = tilerModel.getTensorDimVar(tensorName = bufferB.name,
                                                  dimIdx = (tensorsShapeLen - 2) + parseDict['transB'])
        BSecondDimVar = tilerModel.getTensorDimVar(tensorName = bufferB.name,
                                                   dimIdx = (tensorsShapeLen - 1) - parseDict['transB'])
        outputFirstDimVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = (tensorsShapeLen - 2))
        outputSecondDimVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = (tensorsShapeLen - 1))

        # Map output dims to inputs dims
        for idx in range(tensorsShapeLen - 2):
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = idx) == tilerModel.getTensorDimVar(
                    tensorName = bufferA.name, dimIdx = idx))
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = idx) == tilerModel.getTensorDimVar(
                    tensorName = bufferB.name, dimIdx = idx))

        tilerModel.addConstraint(outputFirstDimVar == AFirstDimVar)
        tilerModel.addConstraint(outputSecondDimVar == BSecondDimVar)

        # Add GEMM Geometrical constraints
        tilerModel.addConstraint(ASecondDimVar == BFirstDimVar)

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        bufferA = ctxt.lookup(name = parseDict['A'])
        bufferB = ctxt.lookup(name = parseDict['B'])

        tensorsShapeLen = len(bufferA.shape)

        ASecondDimVar = tilerModel.getTensorDimVar(tensorName = bufferA.name,
                                                   dimIdx = (tensorsShapeLen - 1) - parseDict['transA'])
        BFirstDimVar = tilerModel.getTensorDimVar(tensorName = bufferB.name,
                                                  dimIdx = (tensorsShapeLen - 2) + parseDict['transB'])
        BSecondDimVar = tilerModel.getTensorDimVar(tensorName = bufferB.name,
                                                   dimIdx = (tensorsShapeLen - 1) - parseDict['transB'])

        # VIC: We don't want to deal with intermediate results between kernel calls
        tilerModel.addConstraint(ASecondDimVar == parseDict['N'])
        tilerModel.addConstraint(BFirstDimVar == parseDict['N'])

        # VIC: For now we tile only one of the inputs
        tilerModel.addConstraint(BSecondDimVar == parseDict['O'])

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['A', 'B', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        varA = operatorRepresentation['A']

        NSize = ctxt.lookup(varA).shape[-1]
        NOffset = 0

        inputACubes = []
        inputBCubes = []

        replacements = {"M": [], "O": [], "batch": []}

        # Every output is constructed by a pair of inputs. Reconstruct this pair.
        for cube in outputCubes:
            # (BatchOffset, BOffset, MOffset, OOffset) = cube.offset
            # (BatchSize, BSize, MSize, OSize) = cube.dims

            BSize = 1
            BOffset = 0
            BatchSize = 1
            BatchOffset = 0

            if len(cube.offset) == 2:
                (MOffset, OOffset) = cube.offset
                (MSize, OSize) = cube.dims
            elif len(cube.offset) == 3:
                (BatchOffset, MOffset, OOffset) = cube.offset
                (BatchSize, MSize, OSize) = cube.dims
            else:
                (BatchOffset, BOffset, MOffset, OOffset) = cube.offset
                (BatchSize, BSize, MSize, OSize) = cube.dims

            replacements["M"].append(MSize)
            replacements["O"].append(OSize)
            replacements["batch"].append(BSize)

            ACube = HyperRectangle((BatchOffset, BOffset, MOffset, NOffset), (BatchSize, BSize, MSize, NSize))
            BCube = HyperRectangle((BatchOffset, BOffset, NOffset, OOffset), (BatchSize, BSize, NSize, OSize))

            inputACubes.append(ACube)
            inputBCubes.append(BCube)

        inputLoadSchedule = []
        outputLoadSchedule = []

        replacements["N"] = [NSize] * len(outputCubes)

        replacementTypes = {
            "M": PointerClass(int8_t),
            "N": PointerClass(int8_t),
            "O": PointerClass(int8_t),
            "batch": PointerClass(int8_t)
        }

        for a, b in zip(inputACubes, inputBCubes):
            inputLoadSchedule.append({"A": a, "B": b})

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        schedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)

        return VariableReplacementScheme(replacements, replacementTypes), schedule
