"""
Tests for Dynamic Schema System.

These tests validate the dynamic schema system with real data and no mocks.
All tests use actual functionality without fallbacks.
"""

import pytest
import tempfile
import json
import time
from pathlib import Path
from typing import Dict, Any

from src.models.dynamic_schemas import (
    DynamicElementRegistry, AdaptiveElementType, ElementTypeDefinition,
    CoreElementCategory, DiscoveryMethod, get_dynamic_registry, initialize_dynamic_registry
)


class TestElementTypeDefinition:
    """Test ElementTypeDefinition functionality."""
    
    def test_create_basic_definition(self):
        """Test creating basic element type definition."""
        type_def = ElementTypeDefinition(
            type_name="steel_beam",
            base_category=CoreElementCategory.STRUCTURAL,
            discovery_confidence=0.85,
            discovery_method=DiscoveryMethod.AI_CLASSIFICATION
        )
        
        assert type_def.type_name == "steel_beam"
        assert type_def.base_category == CoreElementCategory.STRUCTURAL
        assert type_def.discovery_confidence == 0.85
        assert type_def.occurrence_count == 1
        assert type_def.accuracy_score == 0.85  # No validations yet
    
    def test_accuracy_score_calculation(self):
        """Test accuracy score calculation with validations."""
        type_def = ElementTypeDefinition(
            type_name="concrete_column",
            base_category=CoreElementCategory.STRUCTURAL,
            discovery_confidence=0.9
        )
        
        # Add some validations
        type_def.validation_count = 10
        type_def.correction_count = 2
        
        expected_accuracy = (10 - 2) / 10  # 0.8
        assert type_def.accuracy_score == expected_accuracy
    
    def test_reliability_score_calculation(self):
        """Test reliability score calculation."""
        type_def = ElementTypeDefinition(
            type_name="hvac_duct",
            base_category=CoreElementCategory.MEP,
            discovery_confidence=0.8,
            occurrence_count=15
        )
        
        type_def.validation_count = 10
        type_def.correction_count = 1
        
        accuracy = (10 - 1) / 10  # 0.9
        frequency_factor = min(1.0, 15 / 10.0)  # 1.0 (capped)
        expected_reliability = (accuracy * 0.7) + (frequency_factor * 0.3)
        
        assert abs(type_def.reliability_score - expected_reliability) < 0.001


class TestAdaptiveElementType:
    """Test AdaptiveElementType functionality."""
    
    def test_create_adaptive_type(self):
        """Test creating adaptive element type."""
        adaptive_type = AdaptiveElementType(
            base_category=CoreElementCategory.ARCHITECTURAL,
            specific_type="sliding_door",
            domain_context="commercial",
            discovery_confidence=0.92,
            is_dynamically_discovered=True
        )
        
        assert adaptive_type.base_category == CoreElementCategory.ARCHITECTURAL
        assert adaptive_type.specific_type == "sliding_door"
        assert adaptive_type.domain_context == "commercial"
        assert adaptive_type.is_dynamically_discovered is True
        assert adaptive_type.is_reliable is True  # High confidence and reliability
    
    def test_type_name_normalization(self):
        """Test type name normalization in validation."""
        # Test with spaces and special characters
        adaptive_type = AdaptiveElementType(
            base_category=CoreElementCategory.MEP,
            specific_type="HVAC Unit-01",
        )
        
        assert adaptive_type.specific_type == "hvac_unit_01"
    
    def test_full_type_name_with_parent(self):
        """Test full type name generation with parent."""
        adaptive_type = AdaptiveElementType(
            base_category=CoreElementCategory.STRUCTURAL,
            specific_type="moment_frame",
            parent_type="steel_beam"
        )
        
        assert adaptive_type.full_type_name == "steel_beam.moment_frame"
    
    def test_full_type_name_without_parent(self):
        """Test full type name generation without parent."""
        adaptive_type = AdaptiveElementType(
            base_category=CoreElementCategory.ANNOTATION,
            specific_type="dimension_line"
        )
        
        assert adaptive_type.full_type_name == "dimension_line"
    
    def test_reliability_check(self):
        """Test reliability assessment."""
        # High reliability
        reliable_type = AdaptiveElementType(
            base_category=CoreElementCategory.STRUCTURAL,
            specific_type="reinforced_concrete_beam",
            discovery_confidence=0.9,
            reliability_score=0.85
        )
        assert reliable_type.is_reliable is True
        
        # Low reliability
        unreliable_type = AdaptiveElementType(
            base_category=CoreElementCategory.SPECIALIZED,
            specific_type="unknown_element",
            discovery_confidence=0.5,
            reliability_score=0.6
        )
        assert unreliable_type.is_reliable is False


class TestDynamicElementRegistry:
    """Test DynamicElementRegistry functionality with real persistence."""
    
    def setup_method(self):
        """Setup test registry with temporary persistence."""
        self.temp_dir = tempfile.mkdtemp()
        self.persistence_path = Path(self.temp_dir) / "test_registry.json"
        self.registry = DynamicElementRegistry(self.persistence_path)
    
    def teardown_method(self):
        """Clean up test files."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_registry_initialization(self):
        """Test registry initialization."""
        assert len(self.registry.discovered_types) == 0
        assert self.registry.total_discoveries == 0
        assert all(count == 0 for count in self.registry.category_counts.values())
    
    def test_register_new_type(self):
        """Test registering a new element type."""
        success, message = self.registry.register_discovered_type(
            type_name="precast_concrete_panel",
            base_category=CoreElementCategory.STRUCTURAL,
            discovery_confidence=0.88,
            discovery_method=DiscoveryMethod.AI_CLASSIFICATION,
            domain_context="industrial",
            description="Precast concrete wall panel for industrial construction"
        )
        
        assert success is True
        assert "Successfully registered" in message
        assert len(self.registry.discovered_types) == 1
        assert self.registry.category_counts[CoreElementCategory.STRUCTURAL] == 1
        assert self.registry.total_discoveries == 1
        
        # Verify the registered type
        type_def = self.registry.get_type_definition("precast_concrete_panel")
        assert type_def is not None
        assert type_def.type_name == "precast_concrete_panel"
        assert type_def.domain_context == "industrial"
        assert type_def.discovery_confidence == 0.88
    
    def test_register_existing_type_updates(self):
        """Test that registering existing type updates it."""
        # Register initial type
        self.registry.register_discovered_type(
            type_name="steel_column",
            base_category=CoreElementCategory.STRUCTURAL,
            discovery_confidence=0.7
        )
        
        # Register same type with higher confidence
        success, message = self.registry.register_discovered_type(
            type_name="steel_column",
            base_category=CoreElementCategory.STRUCTURAL,
            discovery_confidence=0.9
        )
        
        assert success is True
        assert "Updated existing type" in message
        assert len(self.registry.discovered_types) == 1  # Still only one type
        
        # Verify updates
        type_def = self.registry.get_type_definition("steel_column")
        assert type_def.occurrence_count == 2
        assert type_def.discovery_confidence == 0.9  # Updated to higher confidence
    
    def test_type_name_normalization(self):
        """Test type name normalization during registration."""
        success, _ = self.registry.register_discovered_type(
            type_name="Steel Beam W12x26",
            base_category=CoreElementCategory.STRUCTURAL,
            discovery_confidence=0.8
        )
        
        assert success is True
        
        # Should be normalized
        type_def = self.registry.get_type_definition("steel_beam_w12x26")
        assert type_def is not None
        assert type_def.type_name == "steel_beam_w12x26"
    
    def test_create_adaptive_element_type_from_registry(self):
        """Test creating AdaptiveElementType from registry data."""
        # Register a type first
        self.registry.register_discovered_type(
            type_name="fire_damper",
            base_category=CoreElementCategory.MEP,
            discovery_confidence=0.85,
            domain_context="commercial",
            industry_context="hvac"
        )
        
        # Create adaptive type from registry
        adaptive_type = self.registry.create_adaptive_element_type("fire_damper")
        
        assert adaptive_type is not None
        assert adaptive_type.specific_type == "fire_damper"
        assert adaptive_type.base_category == CoreElementCategory.MEP
        assert adaptive_type.domain_context == "commercial"
        assert adaptive_type.industry_context == "hvac"
        assert adaptive_type.is_dynamically_discovered is True
    
    def test_create_adaptive_type_not_in_registry(self):
        """Test creating AdaptiveElementType for unknown type."""
        adaptive_type = self.registry.create_adaptive_element_type(
            "unknown_element",
            base_category=CoreElementCategory.SPECIALIZED
        )
        
        assert adaptive_type is not None
        assert adaptive_type.specific_type == "unknown_element"
        assert adaptive_type.base_category == CoreElementCategory.SPECIALIZED
        assert adaptive_type.is_dynamically_discovered is False
        assert adaptive_type.discovery_confidence == 0.5  # Default low confidence
    
    def test_search_similar_types(self):
        """Test searching for similar types."""
        # Register several types
        types_to_register = [
            ("steel_beam", CoreElementCategory.STRUCTURAL),
            ("steel_column", CoreElementCategory.STRUCTURAL),
            ("concrete_beam", CoreElementCategory.STRUCTURAL),
            ("wooden_beam", CoreElementCategory.STRUCTURAL),
            ("hvac_duct", CoreElementCategory.MEP)
        ]
        
        for type_name, category in types_to_register:
            self.registry.register_discovered_type(
                type_name=type_name,
                base_category=category,
                discovery_confidence=0.8
            )
        
        # Search for beam-related types
        results = self.registry.search_similar_types("beam", limit=5)
        
        assert len(results) >= 3  # Should find at least the beam types
        beam_results = [result for result in results if "beam" in result[0]]
        assert len(beam_results) == 3  # steel_beam, concrete_beam, wooden_beam
        
        # Results should be sorted by similarity (all beam types should have high similarity)
        for type_name, similarity in beam_results:
            assert similarity > 0.5  # High similarity for exact substring match
    
    def test_get_types_by_category(self):
        """Test getting types by category."""
        # Register types in different categories
        structural_types = ["steel_beam", "concrete_column", "foundation_wall"]
        mep_types = ["hvac_duct", "electrical_panel"]
        
        for type_name in structural_types:
            self.registry.register_discovered_type(
                type_name=type_name,
                base_category=CoreElementCategory.STRUCTURAL,
                discovery_confidence=0.8
            )
        
        for type_name in mep_types:
            self.registry.register_discovered_type(
                type_name=type_name,
                base_category=CoreElementCategory.MEP,
                discovery_confidence=0.8
            )
        
        # Get structural types
        structural_results = self.registry.get_types_by_category(CoreElementCategory.STRUCTURAL)
        assert len(structural_results) == 3
        assert all(type_def.base_category == CoreElementCategory.STRUCTURAL for type_def in structural_results)
        
        # Get MEP types
        mep_results = self.registry.get_types_by_category(CoreElementCategory.MEP)
        assert len(mep_results) == 2
        assert all(type_def.base_category == CoreElementCategory.MEP for type_def in mep_results)
    
    def test_evolve_type_definition(self):
        """Test evolving type definition with new evidence."""
        # Register initial type
        self.registry.register_discovered_type(
            type_name="composite_beam",
            base_category=CoreElementCategory.STRUCTURAL,
            discovery_confidence=0.7
        )
        
        # Evolve with new evidence
        new_evidence = {
            "confidence": 0.9,
            "properties": {"material": "steel_concrete", "span_range": "20-40ft"},
            "patterns": ["W-shape", "concrete_slab"],
            "related_types": ["steel_beam", "concrete_slab"]
        }
        
        success, message = self.registry.evolve_type_definition("composite_beam", new_evidence)
        
        assert success is True
        assert "Successfully evolved" in message
        
        # Verify evolution
        type_def = self.registry.get_type_definition("composite_beam")
        assert type_def.discovery_confidence == 0.9
        assert "material" in type_def.typical_properties
        assert "W-shape" in type_def.common_patterns
        assert "steel_beam" in type_def.related_types
        assert type_def.validation_count == 1
    
    def test_registry_persistence(self):
        """Test registry persistence and loading."""
        # Register some types
        test_types = [
            ("reinforced_concrete_wall", CoreElementCategory.STRUCTURAL, 0.85),
            ("sliding_glass_door", CoreElementCategory.ARCHITECTURAL, 0.9),
            ("variable_air_volume_unit", CoreElementCategory.MEP, 0.8)
        ]
        
        for type_name, category, confidence in test_types:
            self.registry.register_discovered_type(
                type_name=type_name,
                base_category=category,
                discovery_confidence=confidence,
                domain_context="commercial"
            )
        
        # Verify persistence file was created
        assert self.persistence_path.exists()
        
        # Create new registry with same persistence path
        new_registry = DynamicElementRegistry(self.persistence_path)
        
        # Verify all types were loaded
        assert len(new_registry.discovered_types) == 3
        assert new_registry.total_discoveries == 3
        
        # Verify specific types
        for type_name, category, confidence in test_types:
            type_def = new_registry.get_type_definition(type_name)
            assert type_def is not None
            assert type_def.base_category == category
            assert type_def.discovery_confidence == confidence
            assert type_def.domain_context == "commercial"
    
    def test_registry_stats(self):
        """Test registry statistics generation."""
        # Register types with different characteristics
        self.registry.register_discovered_type(
            type_name="high_reliability_beam",
            base_category=CoreElementCategory.STRUCTURAL,
            discovery_confidence=0.95
        )
        
        # Make it highly reliable
        type_def = self.registry.get_type_definition("high_reliability_beam")
        type_def.occurrence_count = 20
        type_def.validation_count = 18
        type_def.correction_count = 1
        
        self.registry.register_discovered_type(
            type_name="recent_discovery",
            base_category=CoreElementCategory.MEP,
            discovery_confidence=0.8
        )
        
        stats = self.registry.get_registry_stats()
        
        assert stats["total_types"] == 2
        assert stats["total_discoveries"] == 2
        assert stats["category_counts"][CoreElementCategory.STRUCTURAL.value] == 1
        assert stats["category_counts"][CoreElementCategory.MEP.value] == 1
        assert len(stats["most_reliable_types"]) <= 2
        assert len(stats["recent_discoveries"]) <= 2
        
        # Most reliable should be the high reliability beam
        most_reliable = stats["most_reliable_types"][0]
        assert most_reliable["type_name"] == "high_reliability_beam"
        assert most_reliable["reliability_score"] > 0.9
    
    def test_validation_prevents_invalid_types(self):
        """Test that validation prevents invalid type registration."""
        # Test empty type name
        success, message = self.registry.register_discovered_type(
            type_name="",
            base_category=CoreElementCategory.STRUCTURAL,
            discovery_confidence=0.8
        )
        assert success is False
        assert "cannot be empty" in message.lower()
        
        # Test invalid confidence
        success, message = self.registry.register_discovered_type(
            type_name="valid_name",
            base_category=CoreElementCategory.STRUCTURAL,
            discovery_confidence=1.5  # Invalid: > 1.0
        )
        assert success is False
        assert "between 0.0 and 1.0" in message


class TestGlobalRegistry:
    """Test global registry functionality."""
    
    def test_get_global_registry_singleton(self):
        """Test that global registry is singleton."""
        registry1 = get_dynamic_registry()
        registry2 = get_dynamic_registry()
        
        assert registry1 is registry2  # Same instance
    
    def test_initialize_global_registry(self):
        """Test initializing global registry with custom settings."""
        temp_dir = tempfile.mkdtemp()
        custom_path = Path(temp_dir) / "custom_registry.json"
        
        try:
            registry = initialize_dynamic_registry(custom_path)
            
            # Should be the new global instance
            assert get_dynamic_registry() is registry
            assert registry.persistence_path == custom_path
            
        finally:
            import shutil
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)


class TestRealWorldScenarios:
    """Test real-world scenarios without mocks."""
    
    def setup_method(self):
        """Setup test registry."""
        self.temp_dir = tempfile.mkdtemp()
        self.persistence_path = Path(self.temp_dir) / "realworld_registry.json"
        self.registry = DynamicElementRegistry(self.persistence_path)
    
    def teardown_method(self):
        """Clean up."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_industrial_blueprint_elements(self):
        """Test with realistic industrial blueprint elements."""
        industrial_elements = [
            # Process equipment
            ("centrifugal_pump_p101", CoreElementCategory.MEP, 0.9, "petrochemical"),
            ("heat_exchanger_e201", CoreElementCategory.MEP, 0.85, "petrochemical"),
            ("pressure_vessel_v301", CoreElementCategory.MEP, 0.88, "petrochemical"),
            
            # Structural elements
            ("steel_pipe_rack", CoreElementCategory.STRUCTURAL, 0.9, "industrial"),
            ("concrete_pipe_support", CoreElementCategory.STRUCTURAL, 0.85, "industrial"),
            ("steel_platform_grating", CoreElementCategory.STRUCTURAL, 0.8, "industrial"),
            
            # Specialized elements
            ("flame_arrestor", CoreElementCategory.SPECIALIZED, 0.75, "petrochemical"),
            ("explosion_proof_junction_box", CoreElementCategory.SPECIALIZED, 0.8, "petrochemical")
        ]
        
        # Register all elements
        for type_name, category, confidence, domain in industrial_elements:
            success, _ = self.registry.register_discovered_type(
                type_name=type_name,
                base_category=category,
                discovery_confidence=confidence,
                domain_context=domain,
                industry_context="petrochemical"
            )
            assert success is True
        
        # Verify all registered
        assert len(self.registry.discovered_types) == len(industrial_elements)
        
        # Test searching for related elements
        pump_results = self.registry.search_similar_types("pump")
        assert len(pump_results) >= 1
        assert any("pump" in result[0] for result in pump_results)
        
        # Test category-based retrieval
        mep_elements = self.registry.get_types_by_category(CoreElementCategory.MEP)
        assert len(mep_elements) == 3  # pump, heat exchanger, pressure vessel
        
        specialized_elements = self.registry.get_types_by_category(CoreElementCategory.SPECIALIZED)
        assert len(specialized_elements) == 2  # flame arrestor, junction box
    
    def test_residential_blueprint_elements(self):
        """Test with realistic residential blueprint elements."""
        residential_elements = [
            # Architectural
            ("bifold_closet_door", CoreElementCategory.ARCHITECTURAL, 0.85, "residential"),
            ("casement_window", CoreElementCategory.ARCHITECTURAL, 0.9, "residential"),
            ("interior_partition_wall", CoreElementCategory.ARCHITECTURAL, 0.88, "residential"),
            ("hardwood_flooring", CoreElementCategory.ARCHITECTURAL, 0.8, "residential"),
            
            # MEP
            ("recessed_led_fixture", CoreElementCategory.MEP, 0.85, "residential"),
            ("gfci_outlet", CoreElementCategory.MEP, 0.9, "residential"),
            ("forced_air_register", CoreElementCategory.MEP, 0.8, "residential"),
            
            # Structural
            ("engineered_lumber_joist", CoreElementCategory.STRUCTURAL, 0.85, "residential"),
            ("concrete_footing", CoreElementCategory.STRUCTURAL, 0.9, "residential")
        ]
        
        for type_name, category, confidence, domain in residential_elements:
            success, _ = self.registry.register_discovered_type(
                type_name=type_name,
                base_category=category,
                discovery_confidence=confidence,
                domain_context=domain,
                industry_context="construction"
            )
            assert success is True
        
        # Test domain-specific searches
        door_results = self.registry.search_similar_types("door")
        assert len(door_results) >= 1
        
        window_results = self.registry.search_similar_types("window")
        assert len(window_results) >= 1
        
        # Verify domain context is preserved
        door_type = self.registry.get_type_definition("bifold_closet_door")
        assert door_type.domain_context == "residential"
        assert door_type.industry_context == "construction"
    
    def test_type_evolution_over_time(self):
        """Test how types evolve with more evidence over time."""
        # Initial discovery with limited confidence
        self.registry.register_discovered_type(
            type_name="composite_metal_deck",
            base_category=CoreElementCategory.STRUCTURAL,
            discovery_confidence=0.6,  # Initial low confidence
            description="Unknown metal decking system"
        )
        
        # First evolution: better understanding
        evidence_1 = {
            "confidence": 0.75,
            "properties": {"material": "steel", "profile": "corrugated"},
            "patterns": ["metal_deck", "composite_action"]
        }
        success, _ = self.registry.evolve_type_definition("composite_metal_deck", evidence_1)
        assert success is True
        
        # Second evolution: detailed specifications
        evidence_2 = {
            "confidence": 0.9,
            "properties": {"thickness": "20_gauge", "span": "8_feet", "concrete_topping": True},
            "patterns": ["shear_studs", "concrete_composite"],
            "related_types": ["steel_beam", "concrete_slab"]
        }
        success, _ = self.registry.evolve_type_definition("composite_metal_deck", evidence_2)
        assert success is True
        
        # Verify evolution
        final_type = self.registry.get_type_definition("composite_metal_deck")
        assert final_type.discovery_confidence == 0.9
        assert final_type.validation_count == 2
        assert "thickness" in final_type.typical_properties
        assert "shear_studs" in final_type.common_patterns
        assert "steel_beam" in final_type.related_types
        assert final_type.reliability_score > 0.8
    
    def test_cross_domain_type_recognition(self):
        """Test that similar types across domains are handled correctly."""
        # Register "beam" in different domains
        beam_types = [
            ("steel_beam", CoreElementCategory.STRUCTURAL, "commercial", "construction"),
            ("laminated_timber_beam", CoreElementCategory.STRUCTURAL, "residential", "construction"),
            ("precast_concrete_beam", CoreElementCategory.STRUCTURAL, "industrial", "construction"),
            ("laser_beam_alignment", CoreElementCategory.SPECIALIZED, "industrial", "surveying")
        ]
        
        for type_name, category, domain, industry in beam_types:
            self.registry.register_discovered_type(
                type_name=type_name,
                base_category=category,
                discovery_confidence=0.85,
                domain_context=domain,
                industry_context=industry
            )
        
        # Search should find all beam types
        beam_results = self.registry.search_similar_types("beam")
        assert len(beam_results) == 4
        
        # But they should be distinguished by context
        steel_beam = self.registry.get_type_definition("steel_beam")
        laser_beam = self.registry.get_type_definition("laser_beam_alignment")
        
        assert steel_beam.base_category == CoreElementCategory.STRUCTURAL
        assert laser_beam.base_category == CoreElementCategory.SPECIALIZED
        assert steel_beam.industry_context == "construction"
        assert laser_beam.industry_context == "surveying"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
