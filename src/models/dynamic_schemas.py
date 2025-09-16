"""
Dynamic Schema System for Adaptive Element Type Discovery.

This module implements a dynamic schema system that allows the discovery and
registration of new element types without losing data validation, eliminating
the contradiction between autonomous discovery and static schemas.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field, validator
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    from ..utils.logging_config import get_logger

    logger = get_logger(__name__)
except ImportError:
    # Fallback for direct execution or testing
    import logging

    logger = logging.getLogger(__name__)


class CoreElementCategory(str, Enum):
    """Core immutable categories for structural organization."""

    STRUCTURAL = "structural"
    ARCHITECTURAL = "architectural"
    MEP = "mep"  # Mechanical, Electrical, Plumbing
    ANNOTATION = "annotation"
    SPECIALIZED = "specialized"  # Industry-specific elements


class DiscoveryMethod(str, Enum):
    """Methods used to discover element types."""

    AI_CLASSIFICATION = "ai_classification"
    PATTERN_ANALYSIS = "pattern_analysis"
    NOMENCLATURE_PARSING = "nomenclature_parsing"
    VISUAL_RECOGNITION = "visual_recognition"
    HYBRID_ANALYSIS = "hybrid_analysis"


@dataclass
class ElementTypeDefinition:
    """Complete definition of a discovered element type."""

    # Core identification
    type_name: str
    base_category: CoreElementCategory

    # Contextual information
    domain_context: Optional[str] = None  # e.g., "industrial", "residential"
    industry_context: Optional[str] = None  # e.g., "petrochemical", "aerospace"
    regional_context: Optional[str] = None  # e.g., "US", "EU", "international"

    # Discovery metadata
    discovery_method: DiscoveryMethod = DiscoveryMethod.HYBRID_ANALYSIS
    discovery_confidence: float = 0.0
    first_seen_timestamp: float = field(default_factory=time.time)
    last_seen_timestamp: float = field(default_factory=time.time)
    occurrence_count: int = 1

    # Type relationships
    parent_types: List[str] = field(default_factory=list)
    child_types: List[str] = field(default_factory=list)
    related_types: List[str] = field(default_factory=list)

    # Semantic information
    description: Optional[str] = None
    typical_properties: Dict[str, Any] = field(default_factory=dict)
    common_patterns: List[str] = field(default_factory=list)

    # Quality metrics
    validation_count: int = 0
    correction_count: int = 0

    @property
    def accuracy_score(self) -> float:
        """Calculate accuracy score based on validations and corrections."""
        if self.validation_count == 0:
            return self.discovery_confidence
        return max(0.0, (self.validation_count - self.correction_count) / self.validation_count)

    @property
    def reliability_score(self) -> float:
        """Calculate reliability score based on usage frequency."""
        # Higher occurrence count increases reliability
        frequency_factor = min(1.0, self.occurrence_count / 10.0)
        return (self.accuracy_score * 0.7) + (frequency_factor * 0.3)


class AdaptiveElementType(BaseModel):
    """
    Adaptive element type that combines base categories with dynamic specificity.

    This replaces static enums with a flexible system that maintains structure
    while allowing for discovered element types.
    """

    # Base category (immutable, for structure)
    base_category: CoreElementCategory = Field(
        description="Core category for structural organization"
    )

    # Specific type (dynamic, discovered)
    specific_type: str = Field(description="Specific element type, can be discovered dynamically")

    # Contextual information
    domain_context: Optional[str] = Field(
        description="Domain context (industrial, residential, etc.)", default=None
    )
    industry_context: Optional[str] = Field(
        description="Industry context (petrochemical, aerospace, etc.)", default=None
    )

    # Discovery metadata
    discovery_confidence: float = Field(
        description="Confidence in type classification", ge=0.0, le=1.0, default=1.0
    )
    discovery_method: DiscoveryMethod = Field(
        description="Method used to discover/classify type", default=DiscoveryMethod.HYBRID_ANALYSIS
    )
    is_dynamically_discovered: bool = Field(
        description="Whether this type was discovered dynamically", default=False
    )

    # Type hierarchy
    parent_type: Optional[str] = Field(description="Parent type in hierarchy", default=None)
    specificity_level: int = Field(
        description="Level of specificity (0=general, higher=more specific)", default=0
    )

    # Quality indicators
    reliability_score: float = Field(
        description="Reliability of this classification", ge=0.0, le=1.0, default=1.0
    )

    @validator("specific_type")
    def validate_specific_type(cls, v):
        """Ensure specific type is not empty and follows naming conventions."""
        if not v or not v.strip():
            raise ValueError("specific_type cannot be empty")

        # Convert to lowercase with underscores
        normalized = v.lower().replace(" ", "_").replace("-", "_")
        # Remove any non-alphanumeric characters except underscores
        import re

        normalized = re.sub(r"[^a-z0-9_]", "", normalized)

        return normalized

    @property
    def full_type_name(self) -> str:
        """Get full hierarchical type name."""
        if self.parent_type:
            return f"{self.parent_type}.{self.specific_type}"
        return self.specific_type

    @property
    def is_reliable(self) -> bool:
        """Check if this type classification is reliable."""
        return self.reliability_score >= 0.7 and self.discovery_confidence >= 0.8


class DynamicElementRegistry:
    """
    Central registry for dynamically discovered element types.

    Maintains consistency and allows evolution of knowledge while preserving
    data validation capabilities.
    """

    def __init__(self, persistence_path: Optional[Path] = None):
        self.persistence_path = persistence_path or Path("data/dynamic_registry.json")

        # Core registry data
        self.discovered_types: Dict[str, ElementTypeDefinition] = {}
        self.type_hierarchy: Dict[str, List[str]] = {}  # parent -> children
        self.category_counts: Dict[CoreElementCategory, int] = {
            cat: 0 for cat in CoreElementCategory
        }

        # Performance optimization
        self._type_lookup_cache: Dict[str, ElementTypeDefinition] = {}
        self._pattern_cache: Dict[str, List[str]] = {}

        # Registry metadata
        self.created_timestamp = time.time()
        self.last_updated_timestamp = time.time()
        self.total_discoveries = 0

        # Load existing registry if available
        self._load_registry()

        logger.info(f"DynamicElementRegistry initialized with {len(self.discovered_types)} types")

    def register_discovered_type(
        self,
        type_name: str,
        base_category: CoreElementCategory,
        discovery_confidence: float,
        discovery_method: DiscoveryMethod = DiscoveryMethod.HYBRID_ANALYSIS,
        **kwargs,
    ) -> Tuple[bool, str]:
        """
        Register a newly discovered element type.

        Args:
            type_name: Name of the discovered type
            base_category: Core category for organization
            discovery_confidence: Confidence in the discovery (0.0-1.0)
            discovery_method: Method used for discovery
            **kwargs: Additional metadata

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Normalize type name
            normalized_name = self._normalize_type_name(type_name)

            # Check if type already exists
            if normalized_name in self.discovered_types:
                existing_type = self.discovered_types[normalized_name]
                # Update existing type
                existing_type.occurrence_count += 1
                existing_type.last_seen_timestamp = time.time()

                # Update confidence if new confidence is higher
                if discovery_confidence > existing_type.discovery_confidence:
                    existing_type.discovery_confidence = discovery_confidence
                    existing_type.discovery_method = discovery_method

                self._invalidate_cache()
                logger.debug(f"Updated existing type: {normalized_name}")
                return True, f"Updated existing type: {normalized_name}"

            # Create new type definition
            type_def = ElementTypeDefinition(
                type_name=normalized_name,
                base_category=base_category,
                discovery_confidence=discovery_confidence,
                discovery_method=discovery_method,
                **kwargs,
            )

            # Validate the new type
            validation_result = self._validate_type_definition(type_def)
            if not validation_result[0]:
                return False, f"Validation failed: {validation_result[1]}"

            # Register the type
            self.discovered_types[normalized_name] = type_def
            self.category_counts[base_category] += 1
            self.total_discoveries += 1
            self.last_updated_timestamp = time.time()

            # Update hierarchy if parent specified
            parent_type = kwargs.get("parent_type")
            if parent_type:
                self._update_hierarchy(parent_type, normalized_name)

            # Invalidate caches
            self._invalidate_cache()

            # Persist changes
            self._save_registry()

            logger.info(
                f"Registered new type: {normalized_name} (category: {base_category.value}, confidence: {discovery_confidence:.3f})"
            )
            return True, f"Successfully registered type: {normalized_name}"

        except Exception as e:
            logger.error(f"Failed to register type {type_name}: {e}")
            return False, f"Registration failed: {str(e)}"

    def get_type_definition(self, type_name: str) -> Optional[ElementTypeDefinition]:
        """Get definition for a specific type."""
        normalized_name = self._normalize_type_name(type_name)

        # Check cache first
        if normalized_name in self._type_lookup_cache:
            return self._type_lookup_cache[normalized_name]

        # Get from registry
        type_def = self.discovered_types.get(normalized_name)
        if type_def:
            # Cache the result
            self._type_lookup_cache[normalized_name] = type_def

        return type_def

    def create_adaptive_element_type(
        self, type_name: str, base_category: Optional[CoreElementCategory] = None, **kwargs
    ) -> Optional[AdaptiveElementType]:
        """
        Create an AdaptiveElementType from registry data.

        Args:
            type_name: Name of the type to create
            base_category: Override base category if needed
            **kwargs: Additional parameters for AdaptiveElementType

        Returns:
            AdaptiveElementType instance or None if type not found
        """
        type_def = self.get_type_definition(type_name)
        if not type_def:
            # If not in registry, try to infer base category
            if base_category is None:
                logger.warning(f"Type {type_name} not in registry and no base_category provided")
                return None

            # Create with minimal information
            return AdaptiveElementType(
                base_category=base_category,
                specific_type=self._normalize_type_name(type_name),
                discovery_confidence=0.5,
                is_dynamically_discovered=False,
                **kwargs,
            )

        # Create from registry definition
        return AdaptiveElementType(
            base_category=base_category or type_def.base_category,
            specific_type=type_def.type_name,
            domain_context=type_def.domain_context,
            industry_context=type_def.industry_context,
            discovery_confidence=type_def.discovery_confidence,
            discovery_method=type_def.discovery_method,
            is_dynamically_discovered=True,
            parent_type=type_def.parent_types[0] if type_def.parent_types else None,
            reliability_score=type_def.reliability_score,
            **kwargs,
        )

    def search_similar_types(self, query: str, limit: int = 10) -> List[Tuple[str, float]]:
        """
        Search for types similar to the query.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of (type_name, similarity_score) tuples
        """
        query_normalized = self._normalize_type_name(query)
        results = []

        for type_name, type_def in self.discovered_types.items():
            # Simple similarity based on string matching and context
            similarity = self._calculate_similarity(query_normalized, type_name, type_def)
            if similarity > 0.1:  # Minimum threshold
                results.append((type_name, similarity))

        # Sort by similarity score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def get_types_by_category(self, category: CoreElementCategory) -> List[ElementTypeDefinition]:
        """Get all types in a specific category."""
        return [
            type_def
            for type_def in self.discovered_types.values()
            if type_def.base_category == category
        ]

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get comprehensive registry statistics."""
        return {
            "total_types": len(self.discovered_types),
            "category_counts": dict(self.category_counts),
            "total_discoveries": self.total_discoveries,
            "created_timestamp": self.created_timestamp,
            "last_updated_timestamp": self.last_updated_timestamp,
            "most_reliable_types": self._get_most_reliable_types(5),
            "recent_discoveries": self._get_recent_discoveries(10),
        }

    def evolve_type_definition(
        self, type_name: str, new_evidence: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Evolve an existing type definition with new evidence.

        Args:
            type_name: Name of the type to evolve
            new_evidence: New evidence to incorporate

        Returns:
            Tuple of (success: bool, message: str)
        """
        normalized_name = self._normalize_type_name(type_name)
        type_def = self.discovered_types.get(normalized_name)

        if not type_def:
            return False, f"Type {normalized_name} not found in registry"

        try:
            # Update confidence if new evidence supports it
            if "confidence" in new_evidence:
                new_confidence = float(new_evidence["confidence"])
                if new_confidence > type_def.discovery_confidence:
                    type_def.discovery_confidence = new_confidence

            # Update properties
            if "properties" in new_evidence:
                type_def.typical_properties.update(new_evidence["properties"])

            # Update patterns
            if "patterns" in new_evidence:
                for pattern in new_evidence["patterns"]:
                    if pattern not in type_def.common_patterns:
                        type_def.common_patterns.append(pattern)

            # Update relationships
            if "related_types" in new_evidence:
                for related_type in new_evidence["related_types"]:
                    if related_type not in type_def.related_types:
                        type_def.related_types.append(related_type)

            # Update metadata
            type_def.last_seen_timestamp = time.time()
            type_def.validation_count += 1

            # Invalidate caches and save
            self._invalidate_cache()
            self._save_registry()

            logger.info(f"Evolved type definition: {normalized_name}")
            return True, f"Successfully evolved type: {normalized_name}"

        except Exception as e:
            logger.error(f"Failed to evolve type {normalized_name}: {e}")
            return False, f"Evolution failed: {str(e)}"

    def _normalize_type_name(self, type_name: str) -> str:
        """Normalize type name to consistent format."""
        if not type_name:
            raise ValueError("Type name cannot be empty")

        # Convert to lowercase with underscores
        normalized = type_name.lower().replace(" ", "_").replace("-", "_")
        # Remove any non-alphanumeric characters except underscores
        import re

        normalized = re.sub(r"[^a-z0-9_]", "", normalized)

        # Ensure it doesn't start with a number
        if normalized and normalized[0].isdigit():
            normalized = f"type_{normalized}"

        return normalized

    def _validate_type_definition(self, type_def: ElementTypeDefinition) -> Tuple[bool, str]:
        """Validate a type definition before registration."""
        # Check confidence range
        if not (0.0 <= type_def.discovery_confidence <= 1.0):
            return False, "Discovery confidence must be between 0.0 and 1.0"

        # Check type name format
        if not type_def.type_name or not type_def.type_name.strip():
            return False, "Type name cannot be empty"

        # Check category limits (prevent registry bloat)
        max_types_per_category = 1000  # Configurable
        if self.category_counts[type_def.base_category] >= max_types_per_category:
            return False, f"Maximum types per category ({max_types_per_category}) reached"

        return True, "Valid"

    def _update_hierarchy(self, parent_type: str, child_type: str):
        """Update type hierarchy relationships."""
        parent_normalized = self._normalize_type_name(parent_type)
        child_normalized = self._normalize_type_name(child_type)

        if parent_normalized not in self.type_hierarchy:
            self.type_hierarchy[parent_normalized] = []

        if child_normalized not in self.type_hierarchy[parent_normalized]:
            self.type_hierarchy[parent_normalized].append(child_normalized)

        # Update parent reference in child type
        if child_normalized in self.discovered_types:
            child_def = self.discovered_types[child_normalized]
            if parent_normalized not in child_def.parent_types:
                child_def.parent_types.append(parent_normalized)

    def _calculate_similarity(
        self, query: str, type_name: str, type_def: ElementTypeDefinition
    ) -> float:
        """Calculate similarity score between query and type."""
        # Simple similarity calculation (can be enhanced with NLP)
        score = 0.0

        # Exact match
        if query == type_name:
            return 1.0

        # Substring match
        if query in type_name or type_name in query:
            score += 0.8

        # Context similarity
        if type_def.domain_context and query in type_def.domain_context:
            score += 0.3

        # Pattern similarity
        for pattern in type_def.common_patterns:
            if query in pattern:
                score += 0.2

        return min(1.0, score)

    def _get_most_reliable_types(self, limit: int) -> List[Dict[str, Any]]:
        """Get most reliable types for statistics."""
        sorted_types = sorted(
            self.discovered_types.items(), key=lambda x: x[1].reliability_score, reverse=True
        )

        return [
            {
                "type_name": name,
                "reliability_score": type_def.reliability_score,
                "occurrence_count": type_def.occurrence_count,
                "category": type_def.base_category.value,
            }
            for name, type_def in sorted_types[:limit]
        ]

    def _get_recent_discoveries(self, limit: int) -> List[Dict[str, Any]]:
        """Get most recent discoveries for statistics."""
        sorted_types = sorted(
            self.discovered_types.items(), key=lambda x: x[1].first_seen_timestamp, reverse=True
        )

        return [
            {
                "type_name": name,
                "first_seen": type_def.first_seen_timestamp,
                "discovery_method": type_def.discovery_method.value,
                "confidence": type_def.discovery_confidence,
            }
            for name, type_def in sorted_types[:limit]
        ]

    def _invalidate_cache(self):
        """Invalidate all caches."""
        self._type_lookup_cache.clear()
        self._pattern_cache.clear()

    def _load_registry(self):
        """Load registry from persistence."""
        if not self.persistence_path.exists():
            logger.info("No existing registry found, starting fresh")
            return

        try:
            with open(self.persistence_path, "r") as f:
                data = json.load(f)

            # Load discovered types
            for type_name, type_data in data.get("discovered_types", {}).items():
                type_def = ElementTypeDefinition(
                    type_name=type_data["type_name"],
                    base_category=CoreElementCategory(type_data["base_category"]),
                    domain_context=type_data.get("domain_context"),
                    industry_context=type_data.get("industry_context"),
                    discovery_method=DiscoveryMethod(
                        type_data.get("discovery_method", "hybrid_analysis")
                    ),
                    discovery_confidence=type_data.get("discovery_confidence", 0.0),
                    first_seen_timestamp=type_data.get("first_seen_timestamp", time.time()),
                    last_seen_timestamp=type_data.get("last_seen_timestamp", time.time()),
                    occurrence_count=type_data.get("occurrence_count", 1),
                    parent_types=type_data.get("parent_types", []),
                    child_types=type_data.get("child_types", []),
                    related_types=type_data.get("related_types", []),
                    description=type_data.get("description"),
                    typical_properties=type_data.get("typical_properties", {}),
                    common_patterns=type_data.get("common_patterns", []),
                    validation_count=type_data.get("validation_count", 0),
                    correction_count=type_data.get("correction_count", 0),
                )
                self.discovered_types[type_name] = type_def

            # Load metadata
            self.type_hierarchy = data.get("type_hierarchy", {})
            self.category_counts = {
                CoreElementCategory(k): v for k, v in data.get("category_counts", {}).items()
            }
            self.created_timestamp = data.get("created_timestamp", time.time())
            self.last_updated_timestamp = data.get("last_updated_timestamp", time.time())
            self.total_discoveries = data.get("total_discoveries", len(self.discovered_types))

            logger.info(f"Loaded registry with {len(self.discovered_types)} types")

        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            logger.info("Starting with empty registry")

    def _save_registry(self):
        """Save registry to persistence."""
        try:
            # Ensure directory exists
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare data for serialization
            data = {
                "discovered_types": {},
                "type_hierarchy": self.type_hierarchy,
                "category_counts": {k.value: v for k, v in self.category_counts.items()},
                "created_timestamp": self.created_timestamp,
                "last_updated_timestamp": self.last_updated_timestamp,
                "total_discoveries": self.total_discoveries,
                "version": "1.0",
            }

            # Serialize type definitions
            for type_name, type_def in self.discovered_types.items():
                data["discovered_types"][type_name] = {
                    "type_name": type_def.type_name,
                    "base_category": type_def.base_category.value,
                    "domain_context": type_def.domain_context,
                    "industry_context": type_def.industry_context,
                    "discovery_method": type_def.discovery_method.value,
                    "discovery_confidence": type_def.discovery_confidence,
                    "first_seen_timestamp": type_def.first_seen_timestamp,
                    "last_seen_timestamp": type_def.last_seen_timestamp,
                    "occurrence_count": type_def.occurrence_count,
                    "parent_types": type_def.parent_types,
                    "child_types": type_def.child_types,
                    "related_types": type_def.related_types,
                    "description": type_def.description,
                    "typical_properties": type_def.typical_properties,
                    "common_patterns": type_def.common_patterns,
                    "validation_count": type_def.validation_count,
                    "correction_count": type_def.correction_count,
                }

            # Write to file
            with open(self.persistence_path, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)

            logger.debug(f"Registry saved to {self.persistence_path}")

        except Exception as e:
            logger.error(f"Failed to save registry: {e}")


# Global registry instance (singleton pattern)
_global_registry: Optional[DynamicElementRegistry] = None


def get_dynamic_registry() -> DynamicElementRegistry:
    """Get the global dynamic element registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = DynamicElementRegistry()
    return _global_registry


def initialize_dynamic_registry(persistence_path: Optional[Path] = None) -> DynamicElementRegistry:
    """Initialize the global dynamic registry with custom settings."""
    global _global_registry
    _global_registry = DynamicElementRegistry(persistence_path)
    return _global_registry
