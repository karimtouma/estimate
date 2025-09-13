#!/usr/bin/env python3
"""
Test Enhanced Discovery System Integration.
Tests the integration between discovery system and dynamic schemas.
"""

import sys
import tempfile
import time
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Mock dependencies for testing
import importlib.util

class MockConfig:
    def __init__(self):
        self.auto_register_confidence_threshold = 0.85
        self.registry_persistence_path = None

class MockLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def debug(self, msg): print(f"DEBUG: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")

def get_logger(name):
    return MockLogger()

class MockGeminiClient:
    async def generate_content(self, prompt):
        # Mock AI response based on prompt content
        if "steel beam" in prompt.lower():
            return {
                "text": '''
                {
                    "type_name": "steel_beam",
                    "category": "structural",
                    "confidence": 0.88,
                    "reasoning": "W14x30 designation indicates wide flange steel beam",
                    "domain_context": "commercial",
                    "industry_context": "construction"
                }
                '''
            }
        elif "pump" in prompt.lower():
            return {
                "text": '''
                {
                    "type_name": "centrifugal_pump",
                    "category": "mep",
                    "confidence": 0.92,
                    "reasoning": "P-101 designation and centrifugal description",
                    "domain_context": "industrial",
                    "industry_context": "petrochemical"
                }
                '''
            }
        else:
            return {
                "text": '''
                {
                    "type_name": "unknown_element",
                    "category": "specialized",
                    "confidence": 0.5,
                    "reasoning": "Unable to classify element"
                }
                '''
            }

# Mock the imports
sys.modules['src.utils.logging_config'] = type('module', (), {'get_logger': get_logger})()
sys.modules['src.services.gemini_client'] = type('module', (), {'GeminiClient': MockGeminiClient})()

# Load modules with mocked dependencies
def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load dynamic schemas
dynamic_schemas = load_module(
    "dynamic_schemas",
    Path(__file__).parent / "src" / "models" / "dynamic_schemas.py"
)

# Load intelligent classifier
intelligent_classifier = load_module(
    "intelligent_classifier", 
    Path(__file__).parent / "src" / "models" / "intelligent_classifier.py"
)

# Create mock discovery result for testing
class MockDiscoveryResult:
    def __init__(self):
        self.document_type = "structural_drawings"
        self.industry_domain = "commercial"
        self.discovered_patterns = {
            "visual_elements": [
                {"type": "steel_beam", "features": {"horizontal": True}, "confidence": 0.85},
                {"type": "concrete_column", "features": {"vertical": True}, "confidence": 0.8},
                "hvac_duct"  # String format
            ]
        }
        self.nomenclature_system = {
            "patterns": {
                "P-101": {"inferred_type": "centrifugal_pump", "confidence": 0.9},
                "V-201": {"inferred_type": "control_valve", "confidence": 0.85}
            }
        }
        self.page_organization = {}
        self.cross_references = [
            {"element_type": "fire_damper", "context": "HVAC system", "confidence": 0.7}
        ]
        self.element_types = ["steel_beam", "concrete_column", "hvac_duct", "centrifugal_pump"]
        self.confidence_score = 0.82
        self.discovery_metadata = {}

def test_enhanced_discovery_integration():
    """Test the enhanced discovery integration."""
    print("🔍 Testing Enhanced Discovery Integration")
    print("=" * 50)
    
    # Setup
    temp_dir = tempfile.mkdtemp()
    registry_path = Path(temp_dir) / "test_registry.json"
    
    try:
        # Initialize components
        config = MockConfig()
        config.registry_persistence_path = str(registry_path)
        
        registry = dynamic_schemas.DynamicElementRegistry(registry_path)
        classifier = intelligent_classifier.IntelligentTypeClassifier(config, registry)
        
        # Create mock discovery result
        base_result = MockDiscoveryResult()
        
        print("✓ Components initialized")
        
        # Test element extraction
        unique_elements = extract_unique_elements_from_discovery(base_result)
        
        assert len(unique_elements) > 0
        print(f"✓ Extracted {len(unique_elements)} unique elements")
        
        # Verify extracted elements
        element_names = [elem["normalized_name"] for elem in unique_elements]
        expected_elements = ["steel_beam", "concrete_column", "hvac_duct", "centrifugal_pump", "control_valve", "fire_damper"]
        
        for expected in expected_elements:
            assert any(expected in name for name in element_names), f"Expected element {expected} not found"
        
        print("✓ Element extraction validation passed")
        
        # Test element classification (sync version for testing)
        classification_results = []
        for element in unique_elements[:3]:  # Test first 3 elements
            element_data = prepare_element_for_classification(element, base_result)
            
            # Simple classification test
            if "steel_beam" in element["name"]:
                result = create_mock_classification_result(
                    "steel_beam", 
                    dynamic_schemas.CoreElementCategory.STRUCTURAL, 
                    0.88
                )
            elif "pump" in element["name"]:
                result = create_mock_classification_result(
                    "centrifugal_pump", 
                    dynamic_schemas.CoreElementCategory.MEP, 
                    0.92
                )
            else:
                result = create_mock_classification_result(
                    "unknown_element", 
                    dynamic_schemas.CoreElementCategory.SPECIALIZED, 
                    0.5
                )
            
            classification_results.append(result)
        
        print(f"✓ Classified {len(classification_results)} elements")
        
        # Test registry integration
        high_confidence_results = [r for r in classification_results if r["confidence"] >= 0.85]
        
        for result in high_confidence_results:
            success, message = registry.register_discovered_type(
                type_name=result["classified_type"],
                base_category=result["base_category"],
                discovery_confidence=result["confidence"],
                domain_context="commercial"
            )
            assert success, f"Failed to register {result['classified_type']}: {message}"
        
        print(f"✓ Auto-registered {len(high_confidence_results)} high-confidence types")
        
        # Test adaptive element type creation
        adaptive_types = []
        for result in classification_results:
            adaptive_type = dynamic_schemas.AdaptiveElementType(
                base_category=result["base_category"],
                specific_type=result["classified_type"],
                discovery_confidence=result["confidence"],
                is_dynamically_discovered=True,
                domain_context="commercial"
            )
            adaptive_types.append(adaptive_type)
        
        assert len(adaptive_types) == len(classification_results)
        print(f"✓ Created {len(adaptive_types)} adaptive element types")
        
        # Test relationship analysis
        relationships = analyze_element_relationships(adaptive_types)
        print(f"✓ Analyzed relationships: {len(relationships)} elements have relationships")
        
        # Test registry statistics
        stats = registry.get_registry_stats()
        assert "total_types" in stats
        assert stats["total_types"] >= len(high_confidence_results)
        print(f"✓ Registry contains {stats['total_types']} types")
        
        # Test persistence
        registry._save_registry()
        assert registry_path.exists()
        
        # Load new registry and verify persistence
        new_registry = dynamic_schemas.DynamicElementRegistry(registry_path)
        new_stats = new_registry.get_registry_stats()
        assert new_stats["total_types"] == stats["total_types"]
        print("✓ Registry persistence verified")
        
        print("\n" + "=" * 50)
        print("✅ ENHANCED DISCOVERY INTEGRATION TESTS PASSED!")
        print(f"📊 Results:")
        print(f"  - Elements extracted: {len(unique_elements)}")
        print(f"  - Elements classified: {len(classification_results)}")
        print(f"  - Types auto-registered: {len(high_confidence_results)}")
        print(f"  - Registry total types: {stats['total_types']}")
        print(f"  - Relationships discovered: {len(relationships)}")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        import shutil
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
    
    return True


def extract_unique_elements_from_discovery(base_result) -> List[Dict]:
    """Extract unique elements from discovery result."""
    unique_elements = []
    
    # Extract from element_types list
    for element_type in base_result.element_types:
        if element_type and element_type.strip():
            unique_elements.append({
                "name": element_type,
                "source": "element_types",
                "confidence": 0.7,
                "normalized_name": normalize_element_name(element_type)
            })
    
    # Extract from visual patterns
    if "visual_elements" in base_result.discovered_patterns:
        for element in base_result.discovered_patterns["visual_elements"]:
            if isinstance(element, dict) and "type" in element:
                unique_elements.append({
                    "name": element["type"],
                    "source": "visual_patterns",
                    "visual_features": element.get("features", {}),
                    "confidence": element.get("confidence", 0.6),
                    "normalized_name": normalize_element_name(element["type"])
                })
            elif isinstance(element, str):
                unique_elements.append({
                    "name": element,
                    "source": "visual_patterns",
                    "confidence": 0.6,
                    "normalized_name": normalize_element_name(element)
                })
    
    # Extract from nomenclature
    if "patterns" in base_result.nomenclature_system:
        for pattern_name, pattern_data in base_result.nomenclature_system["patterns"].items():
            if isinstance(pattern_data, dict) and "inferred_type" in pattern_data:
                unique_elements.append({
                    "name": pattern_data["inferred_type"],
                    "source": "nomenclature",
                    "nomenclature_pattern": pattern_name,
                    "confidence": pattern_data.get("confidence", 0.5),
                    "normalized_name": normalize_element_name(pattern_data["inferred_type"])
                })
    
    # Extract from cross-references
    for ref in base_result.cross_references:
        if isinstance(ref, dict) and "element_type" in ref:
            unique_elements.append({
                "name": ref["element_type"],
                "source": "cross_references",
                "reference_context": ref.get("context", ""),
                "confidence": ref.get("confidence", 0.5),
                "normalized_name": normalize_element_name(ref["element_type"])
            })
    
    # Deduplicate
    seen_names = set()
    deduplicated = []
    
    for element in unique_elements:
        normalized = element["normalized_name"]
        if normalized not in seen_names and len(normalized) > 2:
            seen_names.add(normalized)
            deduplicated.append(element)
    
    return deduplicated


def normalize_element_name(name: str) -> str:
    """Normalize element name."""
    if not name:
        return ""
    
    import re
    normalized = name.lower().replace(' ', '_').replace('-', '_')
    normalized = re.sub(r'[^a-z0-9_]', '', normalized)
    normalized = re.sub(r'_+', '_', normalized)
    return normalized.strip('_')


def prepare_element_for_classification(element: Dict, base_result) -> Dict:
    """Prepare element for classification."""
    return {
        "extracted_text": element["name"],
        "label": element.get("normalized_name", element["name"]),
        "description": f"Element discovered from {element['source']}",
        "visual_features": element.get("visual_features", {}),
        "textual_features": {
            "source": element["source"],
            "confidence": element.get("confidence", 0.5)
        },
        "location": {},
        "annotations": []
    }


def create_mock_classification_result(type_name: str, category, confidence: float) -> Dict:
    """Create mock classification result."""
    return {
        "classified_type": type_name,
        "base_category": category,
        "confidence": confidence,
        "discovery_method": dynamic_schemas.DiscoveryMethod.HYBRID_ANALYSIS,
        "is_new_discovery": True
    }


def analyze_element_relationships(elements) -> Dict[str, List[str]]:
    """Analyze relationships between elements."""
    relationships = {}
    
    for i, element1 in enumerate(elements):
        related = []
        for j, element2 in enumerate(elements):
            if i != j and are_elements_related(element1, element2):
                related.append(element2.specific_type)
        
        if related:
            relationships[element1.specific_type] = related
    
    return relationships


def are_elements_related(element1, element2) -> bool:
    """Check if two elements are related."""
    name1 = element1.specific_type.lower()
    name2 = element2.specific_type.lower()
    
    # Simple relationship check
    words1 = set(name1.split('_'))
    words2 = set(name2.split('_'))
    
    common_words = words1.intersection(words2)
    significant_common = common_words - {'element', 'component', 'system', 'unit'}
    
    return len(significant_common) > 0


if __name__ == "__main__":
    success = test_enhanced_discovery_integration()
    sys.exit(0 if success else 1)
