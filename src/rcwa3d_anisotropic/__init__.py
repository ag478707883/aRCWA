"""Tensor-material RCWA solver for sampled z-stacked, x/y-periodic structures."""

from .backend import ArrayBackend, resolveBackend
from .builder import LayerStack, PatternLayer
from .geometry import (
    SampledPattern,
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
from .simulation import LayerSpec, RCWASimulation, SimulationConfig, buildSimulation, homogeneousLayer, solveSpectrum
from .solver import (
    DiffractionOrder,
    Layer,
    LayerEigTiming,
    LayerFieldSolution,
    RCWAResult,
    StackTiming,
)

__all__ = [
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
    "SimulationConfig",
    "StackTiming",
    "PatternLayer",
    "Project",
    "SampledPattern",
    "buildSimulation",
    "circularPostLayer",
    "ellipticalPostLayer",
    "gyrotropicXzTensor",
    "homogeneousLayer",
    "polygonPostLayer",
    "reciprocalXzTensor",
    "rectangularHollowPostLayer",
    "rectangularPostLayer",
    "resolveBackend",
    "rotateTensor",
    "rotationMatrix",
    "slicedTaperStack",
    "solveSpectrum",
    "stack",
    "twistXzCoupling",
    "xzTensor",
]
