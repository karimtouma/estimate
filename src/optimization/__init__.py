"""
Optimization module for GEPA-based prompt evolution and specialization.

This module provides GEPA (Genetic Evolution Prompt Architecture) capabilities
specifically focused on pattern extraction and prompt specialization.
"""

from .pattern_extraction_gepa import (
    PatternExtractionGEPA,
    PromptEvolutionResult,
    PatternExtractionMetrics,
    create_pattern_extraction_gepa,
    evolve_discovery_prompts
)
from .gepa_classification_enhancer import (
    GEPAClassificationEnhancer,
    ClassificationCandidate,
    EnhancedClassificationResult,
    create_gepa_classification_enhancer,
    enhance_classification_with_gepa
)
from .comprehensive_gepa_system import (
    ComprehensiveGEPASystem,
    ComprehensiveGEPAResult,
    PromptTestResult,
    create_comprehensive_gepa_system,
    optimize_system_prompts
)

__all__ = [
    'PatternExtractionGEPA',
    'PromptEvolutionResult', 
    'PatternExtractionMetrics',
    'create_pattern_extraction_gepa',
    'evolve_discovery_prompts',
    'GEPAClassificationEnhancer',
    'ClassificationCandidate',
    'EnhancedClassificationResult',
    'create_gepa_classification_enhancer',
    'enhance_classification_with_gepa',
    'ComprehensiveGEPASystem',
    'ComprehensiveGEPAResult',
    'PromptTestResult',
    'create_comprehensive_gepa_system',
    'optimize_system_prompts'
]