"""
Enhanced Dynamic Discovery System with Dynamic Schema Integration.

This module extends the existing discovery system to integrate with the dynamic
schema registry for autonomous element type discovery and registration.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from ..core.config import Config
    from ..models.dynamic_schemas import (
        AdaptiveElementType,
        CoreElementCategory,
        DiscoveryMethod,
        DynamicElementRegistry,
        get_dynamic_registry,
        initialize_dynamic_registry,
    )
    from ..models.intelligent_classifier import IntelligentTypeClassifier
    from ..services.gemini_client import GeminiClient
    from ..utils.logging_config import get_logger
    from .dynamic_discovery import DiscoveryResult, DynamicPlanoDiscovery
    from .nomenclature_parser import NomenclatureParser
    from .pattern_analyzer import PatternAnalyzer
except ImportError:
    # Fallback for testing or direct execution
    import logging

    logger = logging.getLogger(__name__)


logger = get_logger(__name__) if "get_logger" in globals() else logging.getLogger(__name__)


@dataclass
class EnhancedDiscoveryResult(DiscoveryResult):
    """Enhanced discovery result with dynamic schema integration."""

    # Dynamic schema specific results
    discovered_element_types: List[AdaptiveElementType] = field(default_factory=list)
    type_classification_confidence: Dict[str, float] = field(default_factory=dict)
    auto_registered_types: List[str] = field(default_factory=list)

    # Registry statistics
    registry_stats: Dict[str, Any] = field(default_factory=dict)
    classification_performance: Dict[str, Any] = field(default_factory=dict)


class EnhancedDynamicDiscovery(DynamicPlanoDiscovery):
    """
    Enhanced discovery system with dynamic schema integration.

    Extends the base discovery system to automatically discover, classify,
    and register new element types using the dynamic schema system.
    """

    def __init__(self, config: Config, pdf_path: Path):
        super().__init__(config, pdf_path)

        # Initialize dynamic schema components
        self.dynamic_registry = get_dynamic_registry()
        self.intelligent_classifier = IntelligentTypeClassifier(config, self.dynamic_registry)

        # Enhanced discovery tracking
        self.discovered_elements = []
        self.classification_results = []
        self.auto_registered_count = 0

        logger.info("Enhanced Dynamic Discovery initialized with dynamic schema integration")

    async def enhanced_initial_exploration(
        self, sample_size: int = 10, pdf_uri: str = None, analyze_all_pages: bool = False
    ) -> EnhancedDiscoveryResult:
        """
        Enhanced initial exploration with dynamic element discovery.

        Args:
            sample_size: Number of pages to sample for discovery
            pdf_uri: URI of uploaded PDF for Gemini analysis
            analyze_all_pages: Whether to analyze all pages instead of sampling

        Returns:
            EnhancedDiscoveryResult with dynamic schema discoveries
        """
        logger.info("🔍 Starting enhanced discovery with dynamic schema integration...")

        # Run base discovery first
        base_result = await self.initial_exploration(sample_size, pdf_uri, analyze_all_pages)

        # Enhance with dynamic element discovery
        enhanced_result = await self._enhance_discovery_with_dynamic_schemas(base_result, pdf_uri)

        logger.info(f"✅ Enhanced discovery complete:")
        logger.info(f"  - Base elements found: {len(base_result.element_types)}")
        logger.info(
            f"  - Dynamic types discovered: {len(enhanced_result.discovered_element_types)}"
        )
        logger.info(f"  - Auto-registered types: {len(enhanced_result.auto_registered_types)}")
        logger.info(
            f"  - Classification confidence: {enhanced_result.classification_performance.get('average_confidence', 0):.3f}"
        )

        return enhanced_result

    async def _enhance_discovery_with_dynamic_schemas(
        self, base_result: DiscoveryResult, pdf_uri: str = None
    ) -> EnhancedDiscoveryResult:
        """
        Enhance base discovery results with dynamic schema analysis.

        Args:
            base_result: Base discovery result from parent class
            pdf_uri: PDF URI for additional analysis

        Returns:
            Enhanced result with dynamic schema information
        """
        logger.info("Enhancing discovery with dynamic schema analysis...")

        # Create enhanced result from base result
        enhanced_result = EnhancedDiscoveryResult(
            document_type=base_result.document_type,
            industry_domain=base_result.industry_domain,
            discovered_patterns=base_result.discovered_patterns,
            nomenclature_system=base_result.nomenclature_system,
            page_organization=base_result.page_organization,
            cross_references=base_result.cross_references,
            element_types=base_result.element_types,
            confidence_score=base_result.confidence_score,
            discovery_metadata=base_result.discovery_metadata,
        )

        # Extract and classify unique elements
        unique_elements = await self._extract_unique_elements_from_discovery(base_result)

        # Classify each unique element
        classified_elements = []
        classification_confidences = {}
        auto_registered = []

        for element in unique_elements:
            try:
                # Create element data for classification
                element_data = self._prepare_element_for_classification(element, base_result)

                # Create context for classification
                context = {
                    "document_type": base_result.document_type,
                    "document_domain": base_result.industry_domain,
                    "discovery_confidence": base_result.confidence_score,
                }

                # Classify the element
                classification_result = await self.intelligent_classifier.classify_element(
                    element_data, context
                )

                # Create adaptive element type
                adaptive_type = AdaptiveElementType(
                    base_category=classification_result.base_category,
                    specific_type=classification_result.classified_type,
                    domain_context=classification_result.domain_context,
                    industry_context=classification_result.industry_context,
                    discovery_confidence=classification_result.confidence,
                    discovery_method=classification_result.discovery_method,
                    is_dynamically_discovered=classification_result.is_new_discovery,
                    reliability_score=classification_result.confidence,
                )

                classified_elements.append(adaptive_type)
                classification_confidences[classification_result.classified_type] = (
                    classification_result.confidence
                )

                # Track auto-registered types
                if (
                    classification_result.is_new_discovery
                    and not classification_result.requires_validation
                ):
                    auto_registered.append(classification_result.classified_type)
                    self.auto_registered_count += 1

                logger.debug(
                    f"Classified element: {classification_result.classified_type} "
                    f"(confidence: {classification_result.confidence:.3f})"
                )

            except Exception as e:
                logger.warning(f"Failed to classify element {element}: {e}")
                continue

        # Update enhanced result
        enhanced_result.discovered_element_types = classified_elements
        enhanced_result.type_classification_confidence = classification_confidences
        enhanced_result.auto_registered_types = auto_registered
        enhanced_result.registry_stats = self.dynamic_registry.get_registry_stats()
        enhanced_result.classification_performance = (
            self.intelligent_classifier.get_classification_stats()
        )

        # Update overall confidence based on classification results
        if classification_confidences:
            avg_classification_confidence = sum(classification_confidences.values()) / len(
                classification_confidences
            )
            # Combine with base confidence (weighted average)
            enhanced_result.confidence_score = (base_result.confidence_score * 0.6) + (
                avg_classification_confidence * 0.4
            )

        return enhanced_result

    async def _extract_unique_elements_from_discovery(
        self, base_result: DiscoveryResult
    ) -> List[Dict[str, Any]]:
        """
        Extract unique elements from base discovery for classification.

        Args:
            base_result: Base discovery result

        Returns:
            List of unique elements for classification
        """
        unique_elements = []

        # Extract from element_types list
        for element_type in base_result.element_types:
            if element_type and element_type.strip():
                unique_elements.append(
                    {
                        "name": element_type,
                        "source": "element_types",
                        "confidence": 0.7,  # Base confidence for discovered types
                    }
                )

        # Extract from discovered patterns
        if "visual_elements" in base_result.discovered_patterns:
            for element in base_result.discovered_patterns["visual_elements"]:
                if isinstance(element, dict) and "type" in element:
                    unique_elements.append(
                        {
                            "name": element["type"],
                            "source": "visual_patterns",
                            "visual_features": element.get("features", {}),
                            "confidence": element.get("confidence", 0.6),
                        }
                    )
                elif isinstance(element, str):
                    unique_elements.append(
                        {"name": element, "source": "visual_patterns", "confidence": 0.6}
                    )

        # Extract from nomenclature system
        if "patterns" in base_result.nomenclature_system:
            for pattern_name, pattern_data in base_result.nomenclature_system["patterns"].items():
                if isinstance(pattern_data, dict) and "inferred_type" in pattern_data:
                    unique_elements.append(
                        {
                            "name": pattern_data["inferred_type"],
                            "source": "nomenclature",
                            "nomenclature_pattern": pattern_name,
                            "confidence": pattern_data.get("confidence", 0.5),
                        }
                    )

        # Extract from cross-references (these often contain specific element names)
        for ref in base_result.cross_references:
            if isinstance(ref, dict) and "element_type" in ref:
                unique_elements.append(
                    {
                        "name": ref["element_type"],
                        "source": "cross_references",
                        "reference_context": ref.get("context", ""),
                        "confidence": ref.get("confidence", 0.5),
                    }
                )

        # Deduplicate based on normalized names
        seen_names = set()
        deduplicated_elements = []

        for element in unique_elements:
            normalized_name = self._normalize_element_name(element["name"])
            if normalized_name not in seen_names and len(normalized_name) > 2:
                seen_names.add(normalized_name)
                element["normalized_name"] = normalized_name
                deduplicated_elements.append(element)

        logger.info(f"Extracted {len(deduplicated_elements)} unique elements for classification")
        return deduplicated_elements

    def _prepare_element_for_classification(
        self, element: Dict[str, Any], base_result: DiscoveryResult
    ) -> Dict[str, Any]:
        """
        Prepare element data for intelligent classification.

        Args:
            element: Element data from discovery
            base_result: Base discovery result for context

        Returns:
            Formatted element data for classifier
        """
        element_data = {
            "extracted_text": element["name"],
            "label": element.get("normalized_name", element["name"]),
            "description": f"Element discovered from {element['source']}",
            "visual_features": element.get("visual_features", {}),
            "textual_features": {
                "source": element["source"],
                "confidence": element.get("confidence", 0.5),
            },
            "location": {},  # Not available from discovery
            "annotations": [],
        }

        # Add source-specific information
        if element["source"] == "nomenclature":
            element_data["nomenclature_pattern"] = element.get("nomenclature_pattern", "")
            element_data["textual_features"]["nomenclature_based"] = True

        elif element["source"] == "visual_patterns":
            element_data["visual_features"].update(element.get("visual_features", {}))
            element_data["textual_features"]["visually_identified"] = True

        elif element["source"] == "cross_references":
            element_data["description"] += f" - {element.get('reference_context', '')}"
            element_data["textual_features"]["cross_referenced"] = True

        return element_data

    def _normalize_element_name(self, name: str) -> str:
        """Normalize element name for deduplication."""
        if not name:
            return ""

        # Convert to lowercase, replace spaces and dashes with underscores
        normalized = name.lower().replace(" ", "_").replace("-", "_")

        # Remove special characters except underscores and alphanumeric
        import re

        normalized = re.sub(r"[^a-z0-9_]", "", normalized)

        # Remove multiple consecutive underscores
        normalized = re.sub(r"_+", "_", normalized)

        # Remove leading/trailing underscores
        normalized = normalized.strip("_")

        return normalized

    async def analyze_element_relationships(
        self, elements: List[AdaptiveElementType]
    ) -> Dict[str, List[str]]:
        """
        Analyze relationships between discovered elements.

        Args:
            elements: List of discovered adaptive element types

        Returns:
            Dictionary mapping element types to their related types
        """
        relationships = {}

        # Group elements by category for relationship analysis
        category_groups = {}
        for element in elements:
            category = element.base_category.value
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(element)

        # Analyze relationships within and across categories
        for element in elements:
            element_name = element.specific_type
            related_types = []

            # Find related types in same category
            same_category_elements = category_groups.get(element.base_category.value, [])
            for other_element in same_category_elements:
                if other_element.specific_type != element_name:
                    # Simple similarity check (can be enhanced)
                    if self._are_elements_related(element, other_element):
                        related_types.append(other_element.specific_type)

            # Find related types in other categories (structural relationships)
            if element.base_category == CoreElementCategory.STRUCTURAL:
                # Structural elements often relate to architectural elements
                architectural_elements = category_groups.get(
                    CoreElementCategory.ARCHITECTURAL.value, []
                )
                for arch_element in architectural_elements:
                    if "wall" in arch_element.specific_type and "beam" in element_name:
                        related_types.append(arch_element.specific_type)

            elif element.base_category == CoreElementCategory.MEP:
                # MEP elements often relate to architectural spaces
                architectural_elements = category_groups.get(
                    CoreElementCategory.ARCHITECTURAL.value, []
                )
                for arch_element in architectural_elements:
                    if "room" in arch_element.specific_type:
                        related_types.append(arch_element.specific_type)

            if related_types:
                relationships[element_name] = related_types

        return relationships

    def _are_elements_related(
        self, element1: AdaptiveElementType, element2: AdaptiveElementType
    ) -> bool:
        """
        Determine if two elements are related based on naming patterns.

        Args:
            element1: First element
            element2: Second element

        Returns:
            True if elements appear to be related
        """
        name1 = element1.specific_type.lower()
        name2 = element2.specific_type.lower()

        # Check for common base words
        words1 = set(name1.split("_"))
        words2 = set(name2.split("_"))

        # If they share significant words, they might be related
        common_words = words1.intersection(words2)
        if len(common_words) > 0:
            # Filter out very common words
            significant_common = common_words - {"element", "component", "system", "unit"}
            if significant_common:
                return True

        # Check for material-based relationships
        materials = ["steel", "concrete", "wood", "aluminum", "composite"]
        for material in materials:
            if material in name1 and material in name2:
                return True

        return False

    async def update_registry_with_relationships(self, relationships: Dict[str, List[str]]):
        """
        Update the dynamic registry with discovered element relationships.

        Args:
            relationships: Dictionary of element relationships
        """
        for element_name, related_types in relationships.items():
            try:
                # Update the element in registry with relationship information
                new_evidence = {
                    "related_types": related_types,
                    "confidence": 0.7,  # Moderate confidence for discovered relationships
                }

                success, message = self.dynamic_registry.evolve_type_definition(
                    element_name, new_evidence
                )

                if success:
                    logger.debug(f"Updated relationships for {element_name}: {related_types}")
                else:
                    logger.warning(f"Failed to update relationships for {element_name}: {message}")

            except Exception as e:
                logger.error(f"Error updating relationships for {element_name}: {e}")

    def get_enhanced_discovery_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about enhanced discovery."""
        base_stats = {
            "total_pages_analyzed": getattr(self, "total_pages", 0),
            "cache_efficiency": len(self.page_cache) / max(1, getattr(self, "total_pages", 1)),
        }

        enhanced_stats = {
            "dynamic_elements_discovered": len(self.discovered_elements),
            "classification_results_count": len(self.classification_results),
            "auto_registered_types": self.auto_registered_count,
            "registry_stats": self.dynamic_registry.get_registry_stats(),
            "classifier_stats": self.intelligent_classifier.get_classification_stats(),
        }

        return {**base_stats, **enhanced_stats}


# Factory function for creating enhanced discovery instances
def create_enhanced_discovery(config: Config, pdf_path: Path) -> EnhancedDynamicDiscovery:
    """
    Factory function to create enhanced discovery instances.

    Args:
        config: Application configuration
        pdf_path: Path to PDF file

    Returns:
        EnhancedDynamicDiscovery instance
    """
    # Initialize dynamic registry if not already done
    registry_path = getattr(config, "registry_persistence_path", None)
    if registry_path:
        initialize_dynamic_registry(Path(registry_path))

    return EnhancedDynamicDiscovery(config, pdf_path)
