"""
Research module for novel antenna optimization algorithms.
"""

from .novel_algorithms import *
from .comparative_study import *
from .benchmarks import *

__all__ = [
    'QuantumInspiredOptimizer',
    'DifferentialEvolutionSurrogate', 
    'HybridGradientFreeSampling',
    'MultiObjectivePareto',
    'AdaptiveSamplingOptimizer',
    'ResearchBenchmarks',
    'ComparativeStudy',
    'PublicationGenerator'
]