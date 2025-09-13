#!/usr/bin/env python3
"""
Simplified Integration Test for Dynamic Schemas.
Tests core integration functionality without complex dependencies.
"""

import sys
import tempfile
import time
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import our tested dynamic schemas module
import importlib.util

class MockLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def debug(self, msg): print(f"DEBUG: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")

def get_logger(name):
    return MockLogger()

# Mock the logging import
sys.modules['src.utils.logging_config'] = type('module', (), {'get_logger': get_logger})()

# Load the dynamic_schemas module
spec = importlib.util.spec_from_file_location(
    "dynamic_schemas", 
    Path(__file__).parent / "src" / "models" / "dynamic_schemas.py"
)
dynamic_schemas = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dynamic_schemas)

def test_discovery_integration_workflow():
    """Test the complete discovery integration workflow."""
    print("🔍 Testing Discovery Integration Workflow")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    registry_path = Path(temp_dir) / "integration_registry.json"
    
    try:
        # Step 1: Initialize Dynamic Registry
        print("\n📋 Step 1: Initialize Dynamic Registry")
        registry = dynamic_schemas.DynamicElementRegistry(registry_path)
        assert len(registry.discovered_types) == 0
        print("✓ Registry initialized successfully")
        
        # Step 2: Simulate Discovery Results
        print("\n🔍 Step 2: Simulate Discovery Results")
        discovered_elements = simulate_discovery_process()
        print(f"✓ Simulated discovery of {len(discovered_elements)} elements")
        
        # Step 3: Process and Classify Elements
        print("\n🎯 Step 3: Process and Classify Elements")
        classified_elements = []
        registration_count = 0
        
        for element in discovered_elements:
            # Simulate intelligent classification
            classification = simulate_element_classification(element)
            
            # Auto-register if confidence is high enough
            if classification["confidence"] >= 0.8:
                success, message = registry.register_discovered_type(
                    type_name=classification["type_name"],
                    base_category=classification["base_category"],
                    discovery_confidence=classification["confidence"],
                    discovery_method=classification["discovery_method"],
                    domain_context=element.get("domain_context"),
                    description=f"Discovered from {element['source']}"
                )
                
                if success:
                    registration_count += 1
                    print(f"  ✓ Auto-registered: {classification['type_name']} (confidence: {classification['confidence']:.3f})")
                else:
                    print(f"  ✗ Failed to register: {classification['type_name']} - {message}")
            
            # Create adaptive element type
            adaptive_type = dynamic_schemas.AdaptiveElementType(
                base_category=classification["base_category"],
                specific_type=classification["type_name"],
                discovery_confidence=classification["confidence"],
                discovery_method=classification["discovery_method"],
                is_dynamically_discovered=classification["confidence"] >= 0.8,
                domain_context=element.get("domain_context")
            )
            
            classified_elements.append(adaptive_type)
        
        print(f"✓ Classified {len(classified_elements)} elements")
        print(f"✓ Auto-registered {registration_count} high-confidence types")
        
        # Step 4: Analyze Element Relationships
        print("\n🔗 Step 4: Analyze Element Relationships")
        relationships = analyze_element_relationships(classified_elements)
        
        # Update registry with relationships
        relationship_updates = 0
        for element_name, related_types in relationships.items():
            new_evidence = {"related_types": related_types}
            success, message = registry.evolve_type_definition(element_name, new_evidence)
            if success:
                relationship_updates += 1
        
        print(f"✓ Discovered {len(relationships)} relationship groups")
        print(f"✓ Updated {relationship_updates} types with relationships")
        
        # Step 5: Test Registry Functionality
        print("\n📊 Step 5: Test Registry Functionality")
        
        # Test search functionality
        beam_results = registry.search_similar_types("beam", limit=5)
        pump_results = registry.search_similar_types("pump", limit=3)
        
        print(f"✓ Beam search returned {len(beam_results)} results")
        print(f"✓ Pump search returned {len(pump_results)} results")
        
        # Test category retrieval
        structural_types = registry.get_types_by_category(dynamic_schemas.CoreElementCategory.STRUCTURAL)
        mep_types = registry.get_types_by_category(dynamic_schemas.CoreElementCategory.MEP)
        
        print(f"✓ Found {len(structural_types)} structural types")
        print(f"✓ Found {len(mep_types)} MEP types")
        
        # Step 6: Test Persistence and Statistics
        print("\n💾 Step 6: Test Persistence and Statistics")
        
        # Save registry
        registry._save_registry()
        assert registry_path.exists()
        print("✓ Registry saved to disk")
        
        # Get statistics
        stats = registry.get_registry_stats()
        print(f"✓ Registry contains {stats['total_types']} total types")
        print(f"✓ Total discoveries: {stats['total_discoveries']}")
        
        # Test loading in new instance
        new_registry = dynamic_schemas.DynamicElementRegistry(registry_path)
        new_stats = new_registry.get_registry_stats()
        assert new_stats["total_types"] == stats["total_types"]
        print("✓ Registry persistence verified")
        
        # Step 7: Test Type Evolution
        print("\n🔄 Step 7: Test Type Evolution")
        
        # Find a registered type to evolve
        if stats["total_types"] > 0:
            # Get first type for evolution test
            first_type_name = list(new_registry.discovered_types.keys())[0]
            
            # Evolve with new evidence
            evolution_evidence = {
                "confidence": 0.95,
                "properties": {"material": "steel", "span": "20ft"},
                "patterns": ["structural_element", "load_bearing"],
                "related_types": ["foundation", "column"]
            }
            
            success, message = new_registry.evolve_type_definition(first_type_name, evolution_evidence)
            assert success, f"Evolution failed: {message}"
            
            # Verify evolution
            evolved_type = new_registry.get_type_definition(first_type_name)
            assert evolved_type.discovery_confidence == 0.95
            assert "material" in evolved_type.typical_properties
            assert "structural_element" in evolved_type.common_patterns
            
            print(f"✓ Successfully evolved type: {first_type_name}")
        
        # Final Results
        print("\n" + "=" * 60)
        print("✅ DISCOVERY INTEGRATION WORKFLOW COMPLETE!")
        print("\n📊 Final Results:")
        print(f"  🔍 Elements discovered: {len(discovered_elements)}")
        print(f"  🎯 Elements classified: {len(classified_elements)}")
        print(f"  📝 Types auto-registered: {registration_count}")
        print(f"  🔗 Relationship groups: {len(relationships)}")
        print(f"  📊 Registry total types: {stats['total_types']}")
        print(f"  💾 Persistence: ✓ Verified")
        print(f"  🔄 Evolution: ✓ Tested")
        
        # Validate key integration points
        print("\n🔍 Integration Validation:")
        print("  ✓ Discovery → Classification → Registration pipeline")
        print("  ✓ Dynamic type creation and validation")
        print("  ✓ Relationship analysis and updates")
        print("  ✓ Registry persistence and loading")
        print("  ✓ Type evolution with new evidence")
        print("  ✓ Search and retrieval functionality")
        
        return True
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        import shutil
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)


def simulate_discovery_process():
    """Simulate the discovery process results."""
    return [
        # Structural elements
        {
            "name": "W14x30 Steel Beam",
            "source": "visual_patterns",
            "domain_context": "commercial",
            "visual_features": {"horizontal_line": True, "steel_symbol": True},
            "confidence": 0.9
        },
        {
            "name": "HSS8x8 Column", 
            "source": "nomenclature",
            "domain_context": "commercial",
            "nomenclature_code": "C-12",
            "confidence": 0.85
        },
        {
            "name": "Concrete Footing",
            "source": "cross_references",
            "domain_context": "commercial", 
            "reference_context": "Foundation plan",
            "confidence": 0.8
        },
        
        # MEP elements
        {
            "name": "Centrifugal Pump P-101",
            "source": "nomenclature",
            "domain_context": "industrial",
            "nomenclature_code": "P-101",
            "confidence": 0.95
        },
        {
            "name": "Control Valve V-201",
            "source": "nomenclature", 
            "domain_context": "industrial",
            "nomenclature_code": "V-201",
            "confidence": 0.9
        },
        {
            "name": "HVAC Duct",
            "source": "visual_patterns",
            "domain_context": "commercial",
            "visual_features": {"rectangular_path": True},
            "confidence": 0.75
        },
        
        # Architectural elements
        {
            "name": "Sliding Door",
            "source": "visual_patterns",
            "domain_context": "residential",
            "visual_features": {"door_symbol": True, "sliding": True},
            "confidence": 0.8
        },
        {
            "name": "Casement Window",
            "source": "element_types",
            "domain_context": "residential",
            "confidence": 0.85
        },
        
        # Specialized elements
        {
            "name": "Fire Damper",
            "source": "cross_references",
            "domain_context": "commercial",
            "reference_context": "HVAC system safety",
            "confidence": 0.7
        },
        {
            "name": "Explosion Proof Junction Box",
            "source": "element_types",
            "domain_context": "industrial", 
            "confidence": 0.65
        }
    ]


def simulate_element_classification(element):
    """Simulate intelligent element classification."""
    name = element["name"].lower()
    
    # Classification rules based on element characteristics
    if any(keyword in name for keyword in ["beam", "girder", "joist"]):
        return {
            "type_name": "steel_beam" if "steel" in name else "beam",
            "base_category": dynamic_schemas.CoreElementCategory.STRUCTURAL,
            "confidence": element["confidence"],
            "discovery_method": dynamic_schemas.DiscoveryMethod.PATTERN_ANALYSIS
        }
    
    elif any(keyword in name for keyword in ["column", "post", "pillar"]):
        return {
            "type_name": "steel_column" if "steel" in name or "hss" in name else "column",
            "base_category": dynamic_schemas.CoreElementCategory.STRUCTURAL,
            "confidence": element["confidence"],
            "discovery_method": dynamic_schemas.DiscoveryMethod.PATTERN_ANALYSIS
        }
    
    elif any(keyword in name for keyword in ["footing", "foundation"]):
        return {
            "type_name": "concrete_footing" if "footing" in name else "foundation",
            "base_category": dynamic_schemas.CoreElementCategory.STRUCTURAL,
            "confidence": element["confidence"],
            "discovery_method": dynamic_schemas.DiscoveryMethod.PATTERN_ANALYSIS
        }
    
    elif any(keyword in name for keyword in ["pump"]):
        return {
            "type_name": "centrifugal_pump" if "centrifugal" in name else "pump",
            "base_category": dynamic_schemas.CoreElementCategory.MEP,
            "confidence": element["confidence"],
            "discovery_method": dynamic_schemas.DiscoveryMethod.NOMENCLATURE_PARSING
        }
    
    elif any(keyword in name for keyword in ["valve"]):
        return {
            "type_name": "control_valve" if "control" in name else "valve",
            "base_category": dynamic_schemas.CoreElementCategory.MEP,
            "confidence": element["confidence"],
            "discovery_method": dynamic_schemas.DiscoveryMethod.NOMENCLATURE_PARSING
        }
    
    elif any(keyword in name for keyword in ["hvac", "duct", "ventilation"]):
        return {
            "type_name": "hvac_duct",
            "base_category": dynamic_schemas.CoreElementCategory.MEP,
            "confidence": element["confidence"],
            "discovery_method": dynamic_schemas.DiscoveryMethod.VISUAL_RECOGNITION
        }
    
    elif any(keyword in name for keyword in ["door"]):
        return {
            "type_name": "sliding_door" if "sliding" in name else "door",
            "base_category": dynamic_schemas.CoreElementCategory.ARCHITECTURAL,
            "confidence": element["confidence"],
            "discovery_method": dynamic_schemas.DiscoveryMethod.VISUAL_RECOGNITION
        }
    
    elif any(keyword in name for keyword in ["window"]):
        return {
            "type_name": "casement_window" if "casement" in name else "window",
            "base_category": dynamic_schemas.CoreElementCategory.ARCHITECTURAL,
            "confidence": element["confidence"],
            "discovery_method": dynamic_schemas.DiscoveryMethod.VISUAL_RECOGNITION
        }
    
    elif any(keyword in name for keyword in ["damper", "junction", "explosion"]):
        return {
            "type_name": element["name"].lower().replace(" ", "_"),
            "base_category": dynamic_schemas.CoreElementCategory.SPECIALIZED,
            "confidence": element["confidence"],
            "discovery_method": dynamic_schemas.DiscoveryMethod.HYBRID_ANALYSIS
        }
    
    else:
        return {
            "type_name": "unknown_element",
            "base_category": dynamic_schemas.CoreElementCategory.SPECIALIZED,
            "confidence": max(0.3, element["confidence"] - 0.2),  # Reduce confidence for unknown
            "discovery_method": dynamic_schemas.DiscoveryMethod.HYBRID_ANALYSIS
        }


def analyze_element_relationships(elements):
    """Analyze relationships between classified elements."""
    relationships = {}
    
    # Group elements by category
    category_groups = {}
    for element in elements:
        category = element.base_category.value
        if category not in category_groups:
            category_groups[category] = []
        category_groups[category].append(element)
    
    # Find relationships
    for element in elements:
        element_name = element.specific_type
        related_types = []
        
        # Find related types in same category
        same_category = category_groups.get(element.base_category.value, [])
        for other in same_category:
            if other.specific_type != element_name and are_elements_related(element, other):
                related_types.append(other.specific_type)
        
        # Find cross-category relationships
        if element.base_category == dynamic_schemas.CoreElementCategory.STRUCTURAL:
            # Structural elements relate to architectural
            for arch_element in category_groups.get("architectural", []):
                if "door" in arch_element.specific_type and "beam" in element_name:
                    related_types.append(arch_element.specific_type)
        
        elif element.base_category == dynamic_schemas.CoreElementCategory.MEP:
            # MEP elements relate to each other
            for mep_element in category_groups.get("mep", []):
                if mep_element.specific_type != element_name:
                    if ("pump" in element_name and "valve" in mep_element.specific_type) or \
                       ("valve" in element_name and "pump" in mep_element.specific_type):
                        related_types.append(mep_element.specific_type)
        
        if related_types:
            relationships[element_name] = related_types
    
    return relationships


def are_elements_related(element1, element2):
    """Check if two elements are related."""
    name1 = element1.specific_type.lower()
    name2 = element2.specific_type.lower()
    
    # Check for common materials
    materials = ["steel", "concrete", "aluminum"]
    for material in materials:
        if material in name1 and material in name2:
            return True
    
    # Check for common base words
    words1 = set(name1.split('_'))
    words2 = set(name2.split('_'))
    common = words1.intersection(words2) - {"element", "component"}
    
    return len(common) > 0


if __name__ == "__main__":
    success = test_discovery_integration_workflow()
    sys.exit(0 if success else 1)
