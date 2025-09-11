"""
Discovery module for adaptive document analysis.

This module implements the FASE 1 of the adaptive system:
Dynamic discovery of document structure without predefined taxonomies.
"""

from .dynamic_discovery import DynamicPlanoDiscovery
from .pattern_analyzer import PatternAnalyzer
from .nomenclature_parser import NomenclatureParser

__all__ = [
    'DynamicPlanoDiscovery',
    'PatternAnalyzer', 
    'NomenclatureParser'
]
