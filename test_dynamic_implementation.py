#!/usr/bin/env python3
"""
Test script to validate dynamic schema implementation.
This script tests the core functionality without pytest dependencies.
"""

import sys
import tempfile
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import with absolute path to avoid relative import issues
import importlib.util
import os

# Load the dynamic_schemas module directly
spec = importlib.util.spec_from_file_location(
    "dynamic_schemas", 
    Path(__file__).parent / "src" / "models" / "dynamic_schemas.py"
)
dynamic_schemas = importlib.util.module_from_spec(spec)

# Mock the logging import to avoid dependency issues
class MockLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def debug(self, msg): print(f"DEBUG: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")

def get_logger(name):
    return MockLogger()

# Patch the logging import
import sys
sys.modules['src.utils.logging_config'] = type('module', (), {'get_logger': get_logger})()

# Now load the module
spec.loader.exec_module(dynamic_schemas)

# Import the classes we need
DynamicElementRegistry = dynamic_schemas.DynamicElementRegistry
AdaptiveElementType = dynamic_schemas.AdaptiveElementType
ElementTypeDefinition = dynamic_schemas.ElementTypeDefinition
CoreElementCategory = dynamic_schemas.CoreElementCategory
DiscoveryMethod = dynamic_schemas.DiscoveryMethod

def test_element_type_definition():
    """Test ElementTypeDefinition functionality."""
    print("Testing ElementTypeDefinition...")
    
    # Test basic creation
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
    print("✓ ElementTypeDefinition basic creation works")
    
    # Test accuracy score calculation
    type_def.validation_count = 10
    type_def.correction_count = 2
    expected_accuracy = (10 - 2) / 10  # 0.8
    assert type_def.accuracy_score == expected_accuracy
    print("✓ Accuracy score calculation works")
    
    # Test reliability score
    type_def.occurrence_count = 15
    accuracy = (10 - 2) / 10  # 0.8
    frequency_factor = min(1.0, 15 / 10.0)  # 1.0 (capped)
    expected_reliability = (accuracy * 0.7) + (frequency_factor * 0.3)
    assert abs(type_def.reliability_score - expected_reliability) < 0.001
    print("✓ Reliability score calculation works")


def test_adaptive_element_type():
    """Test AdaptiveElementType functionality."""
    print("\nTesting AdaptiveElementType...")
    
    # Test basic creation
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
    print("✓ AdaptiveElementType creation works")
    
    # Test name normalization
    adaptive_type2 = AdaptiveElementType(
        base_category=CoreElementCategory.MEP,
        specific_type="HVAC Unit-01",
    )
    assert adaptive_type2.specific_type == "hvac_unit_01"
    print("✓ Type name normalization works")
    
    # Test full type name
    adaptive_type3 = AdaptiveElementType(
        base_category=CoreElementCategory.STRUCTURAL,
        specific_type="moment_frame",
        parent_type="steel_beam"
    )
    assert adaptive_type3.full_type_name == "steel_beam.moment_frame"
    print("✓ Full type name generation works")


def test_dynamic_element_registry():
    """Test DynamicElementRegistry functionality."""
    print("\nTesting DynamicElementRegistry...")
    
    # Create temporary registry
    temp_dir = tempfile.mkdtemp()
    persistence_path = Path(temp_dir) / "test_registry.json"
    registry = DynamicElementRegistry(persistence_path)
    
    # Test initialization
    assert len(registry.discovered_types) == 0
    assert registry.total_discoveries == 0
    print("✓ Registry initialization works")
    
    # Test registering new type
    success, message = registry.register_discovered_type(
        type_name="precast_concrete_panel",
        base_category=CoreElementCategory.STRUCTURAL,
        discovery_confidence=0.88,
        discovery_method=DiscoveryMethod.AI_CLASSIFICATION,
        domain_context="industrial",
        description="Precast concrete wall panel for industrial construction"
    )
    
    assert success is True
    assert "Successfully registered" in message
    assert len(registry.discovered_types) == 1
    assert registry.category_counts[CoreElementCategory.STRUCTURAL] == 1
    assert registry.total_discoveries == 1
    print("✓ Type registration works")
    
    # Test retrieving type
    type_def = registry.get_type_definition("precast_concrete_panel")
    assert type_def is not None
    assert type_def.type_name == "precast_concrete_panel"
    assert type_def.domain_context == "industrial"
    assert type_def.discovery_confidence == 0.88
    print("✓ Type retrieval works")
    
    # Test updating existing type
    success, message = registry.register_discovered_type(
        type_name="precast_concrete_panel",
        base_category=CoreElementCategory.STRUCTURAL,
        discovery_confidence=0.95
    )
    
    assert success is True
    assert "Updated existing type" in message
    updated_type = registry.get_type_definition("precast_concrete_panel")
    assert updated_type.occurrence_count == 2
    assert updated_type.discovery_confidence == 0.95
    print("✓ Type updating works")
    
    # Test creating adaptive type from registry
    adaptive_type = registry.create_adaptive_element_type("precast_concrete_panel")
    assert adaptive_type is not None
    assert adaptive_type.specific_type == "precast_concrete_panel"
    assert adaptive_type.base_category == CoreElementCategory.STRUCTURAL
    assert adaptive_type.is_dynamically_discovered is True
    print("✓ Adaptive type creation from registry works")
    
    # Test search functionality
    registry.register_discovered_type(
        type_name="steel_beam",
        base_category=CoreElementCategory.STRUCTURAL,
        discovery_confidence=0.9
    )
    
    registry.register_discovered_type(
        type_name="concrete_beam",
        base_category=CoreElementCategory.STRUCTURAL,
        discovery_confidence=0.85
    )
    
    results = registry.search_similar_types("beam", limit=5)
    assert len(results) >= 2
    beam_results = [result for result in results if "beam" in result[0]]
    assert len(beam_results) >= 2
    print("✓ Type search works")
    
    # Test persistence
    registry._save_registry()
    assert persistence_path.exists()
    
    # Create new registry and load
    new_registry = DynamicElementRegistry(persistence_path)
    assert len(new_registry.discovered_types) == 3  # precast_concrete_panel, steel_beam, concrete_beam
    assert new_registry.total_discoveries == 3
    print("✓ Registry persistence works")
    
    # Test evolution
    new_evidence = {
        "confidence": 0.98,
        "properties": {"material": "high_strength_concrete", "thickness": "8_inches"},
        "patterns": ["precast", "architectural"],
        "related_types": ["concrete_wall", "structural_panel"]
    }
    
    success, message = new_registry.evolve_type_definition("precast_concrete_panel", new_evidence)
    assert success is True
    
    evolved_type = new_registry.get_type_definition("precast_concrete_panel")
    assert evolved_type.discovery_confidence == 0.98
    assert "material" in evolved_type.typical_properties
    assert "precast" in evolved_type.common_patterns
    print("✓ Type evolution works")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


def test_real_world_scenarios():
    """Test real-world scenarios."""
    print("\nTesting real-world scenarios...")
    
    temp_dir = tempfile.mkdtemp()
    registry = DynamicElementRegistry(Path(temp_dir) / "realworld_registry.json")
    
    # Industrial elements
    industrial_elements = [
        ("centrifugal_pump_p101", CoreElementCategory.MEP, 0.9, "petrochemical"),
        ("heat_exchanger_e201", CoreElementCategory.MEP, 0.85, "petrochemical"),
        ("steel_pipe_rack", CoreElementCategory.STRUCTURAL, 0.9, "industrial"),
        ("flame_arrestor", CoreElementCategory.SPECIALIZED, 0.75, "petrochemical")
    ]
    
    for type_name, category, confidence, domain in industrial_elements:
        success, _ = registry.register_discovered_type(
            type_name=type_name,
            base_category=category,
            discovery_confidence=confidence,
            domain_context=domain,
            industry_context="petrochemical"
        )
        assert success is True
    
    assert len(registry.discovered_types) == 4
    print("✓ Industrial elements registration works")
    
    # Test category retrieval
    mep_elements = registry.get_types_by_category(CoreElementCategory.MEP)
    assert len(mep_elements) == 2
    
    specialized_elements = registry.get_types_by_category(CoreElementCategory.SPECIALIZED)
    assert len(specialized_elements) == 1
    print("✓ Category-based retrieval works")
    
    # Test stats
    stats = registry.get_registry_stats()
    assert stats["total_types"] == 4
    assert stats["category_counts"][CoreElementCategory.MEP.value] == 2
    assert stats["category_counts"][CoreElementCategory.SPECIALIZED.value] == 1
    print("✓ Registry statistics work")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


def main():
    """Run all tests."""
    print("🚀 Testing Dynamic Schema Implementation")
    print("=" * 50)
    
    try:
        test_element_type_definition()
        test_adaptive_element_type()
        test_dynamic_element_registry()
        test_real_world_scenarios()
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("Dynamic schema system is working correctly.")
        print("Ready for integration with discovery system.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
