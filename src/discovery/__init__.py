"""
Discovery module for adaptive document analysis.

This module implements the FASE 1 of the adaptive system:
Dynamic discovery of document structure without predefined taxonomies.
"""

from .dynamic_discovery import DynamicPlanoDiscovery
from .nomenclature_parser import NomenclatureParser
from .pattern_analyzer import PatternAnalyzer

__all__ = ["DynamicPlanoDiscovery", "PatternAnalyzer", "NomenclatureParser"]
