"""
Optimization module for GEPA-based prompt evolution and specialization.

This module provides GEPA (Genetic Evolution Prompt Architecture) capabilities
specifically focused on pattern extraction and prompt specialization.
"""

from .comprehensive_gepa_system import (
    ComprehensiveGEPAResult,
    ComprehensiveGEPASystem,
    PromptTestResult,
    create_comprehensive_gepa_system,
    optimize_system_prompts,
)
from .gepa_classification_enhancer import (
    ClassificationCandidate,
    EnhancedClassificationResult,
    GEPAClassificationEnhancer,
    create_gepa_classification_enhancer,
    enhance_classification_with_gepa,
)
from .pattern_extraction_gepa import (
    PatternExtractionGEPA,
    PatternExtractionMetrics,
    PromptEvolutionResult,
    create_pattern_extraction_gepa,
    evolve_discovery_prompts,
)

__all__ = [
    "PatternExtractionGEPA",
    "PromptEvolutionResult",
    "PatternExtractionMetrics",
    "create_pattern_extraction_gepa",
    "evolve_discovery_prompts",
    "GEPAClassificationEnhancer",
    "ClassificationCandidate",
    "EnhancedClassificationResult",
    "create_gepa_classification_enhancer",
    "enhance_classification_with_gepa",
    "ComprehensiveGEPASystem",
    "ComprehensiveGEPAResult",
    "PromptTestResult",
    "create_comprehensive_gepa_system",
    "optimize_system_prompts",
]
