"""Tensor-material RCWA solver for z-stacked, x/y-periodic structures."""

from .analytic import (
    AnalyticDisk,
    AnalyticRectangle,
    analyticDiskConvolution,
    analyticRectangleConvolution,
    diskIndicatorConvolution,
    rectangleIndicatorConvolution,
)
from .backend import ArrayBackend, resolveBackend
from .builder import LayerStack, PatternLayer
from .factorization import TensorConvolutionData, liFactorizedSystemMatrix, tensorConvolutionData
from .geometry import (
    SampledPattern,
    analyticCircularPostLayer,
    analyticRectangularPostLayer,
    circularPostLayer,
    ellipticalPostLayer,
    polygonPostLayer,
    rectangularHollowPostLayer,
    rectangularPostLayer,
    slicedTaperStack,
    stack,
)
from .materials import (
    gyrotropicXzTensor,
    reciprocalXzTensor,
    rotateTensor,
    rotationMatrix,
    twistXzCoupling,
    xzTensor,
)
from .project import AnisotropicRCWA, Project
from .simulation import LayerSpec, RCWASimulation, homogeneousLayer
from .solver import (
    CompiledLayer,
    DiffractionOrder,
    Layer,
    LayerEigTiming,
    LayerFieldSolution,
    RCWAResult,
    compileLayers,
    solveStack,
    solveStackBatch,
    solveStackBatchPowers,
)

__all__ = [
    "CompiledLayer",
    "AnalyticDisk",
    "AnalyticRectangle",
    "AnisotropicRCWA",
    "ArrayBackend",
    "DiffractionOrder",
    "Layer",
    "LayerEigTiming",
    "LayerFieldSolution",
    "LayerStack",
    "RCWAResult",
    "RCWASimulation",
    "LayerSpec",
    "PatternLayer",
    "Project",
    "SampledPattern",
    "TensorConvolutionData",
    "analyticCircularPostLayer",
    "analyticDiskConvolution",
    "analyticRectangularPostLayer",
    "analyticRectangleConvolution",
    "circularPostLayer",
    "compileLayers",
    "diskIndicatorConvolution",
    "ellipticalPostLayer",
    "gyrotropicXzTensor",
    "homogeneousLayer",
    "liFactorizedSystemMatrix",
    "polygonPostLayer",
    "reciprocalXzTensor",
    "rectangleIndicatorConvolution",
    "rectangularHollowPostLayer",
    "rectangularPostLayer",
    "resolveBackend",
    "rotateTensor",
    "rotationMatrix",
    "solveStack",
    "solveStackBatch",
    "solveStackBatchPowers",
    "slicedTaperStack",
    "stack",
    "tensorConvolutionData",
    "twistXzCoupling",
    "xzTensor",
]
