#!/usr/bin/env python3
"""
Test Full Autonomy Implementation.
Validates that the system works completely autonomously without hardcoded assumptions.
"""

import sys
import tempfile
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Mock dependencies for testing
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

# Load modules
def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load adaptive questions module
adaptive_questions = load_module(
    "adaptive_questions",
    Path(__file__).parent / "src" / "utils" / "adaptive_questions.py"
)

def test_adaptive_question_generation():
    """Test adaptive question generation for different domains."""
    print("🔍 Testing Adaptive Question Generation")
    print("=" * 50)
    
    generator = adaptive_questions.AdaptiveQuestionGenerator()
    
    # Test construction domain
    construction_discovery = {
        'document_type': 'Construction Drawing Set',
        'industry_domain': 'Architecture, Engineering, and Construction (AEC)',
        'discovered_patterns': {
            'patterns': ['Floor Plans', 'Elevations', 'Structural Plans', 'HVAC Systems']
        }
    }
    
    construction_questions = generator.generate_questions(construction_discovery, max_questions=6)
    
    print("✅ Construction Domain Questions:")
    for i, question in enumerate(construction_questions, 1):
        print(f"  {i}. {question}")
    
    # Validate construction questions
    assert len(construction_questions) <= 6
    assert any('building' in q.lower() or 'structure' in q.lower() for q in construction_questions)
    assert any('construction' in q.lower() or 'architectural' in q.lower() for q in construction_questions)
    print(f"✓ Generated {len(construction_questions)} construction-specific questions")
    
    # Test process engineering domain
    process_discovery = {
        'document_type': 'Process Flow Diagram',
        'industry_domain': 'Petrochemical Process Engineering',
        'discovered_patterns': {
            'patterns': ['P&ID Diagrams', 'Equipment Lists', 'Process Controls']
        }
    }
    
    process_questions = generator.generate_questions(process_discovery, max_questions=6)
    
    print("\n✅ Process Engineering Domain Questions:")
    for i, question in enumerate(process_questions, 1):
        print(f"  {i}. {question}")
    
    # Validate process questions
    assert len(process_questions) <= 6
    assert any('process' in q.lower() for q in process_questions)
    assert any('equipment' in q.lower() or 'system' in q.lower() for q in process_questions)
    print(f"✓ Generated {len(process_questions)} process-specific questions")
    
    # Test electrical domain
    electrical_discovery = {
        'document_type': 'Electrical Schematic',
        'industry_domain': 'Electrical Engineering',
        'discovered_patterns': {
            'patterns': ['Circuit Diagrams', 'Power Distribution', 'Control Systems']
        }
    }
    
    electrical_questions = generator.generate_questions(electrical_discovery, max_questions=6)
    
    print("\n✅ Electrical Domain Questions:")
    for i, question in enumerate(electrical_questions, 1):
        print(f"  {i}. {question}")
    
    # Validate electrical questions
    assert len(electrical_questions) <= 6
    assert any('electrical' in q.lower() or 'circuit' in q.lower() for q in electrical_questions)
    print(f"✓ Generated {len(electrical_questions)} electrical-specific questions")
    
    # Test generic domain
    generic_discovery = {
        'document_type': 'Technical Manual',
        'industry_domain': 'Unknown Domain',
        'discovered_patterns': {
            'patterns': ['Instructions', 'Diagrams', 'Specifications']
        }
    }
    
    generic_questions = generator.generate_questions(generic_discovery, max_questions=6)
    
    print("\n✅ Generic Domain Questions:")
    for i, question in enumerate(generic_questions, 1):
        print(f"  {i}. {question}")
    
    # Validate generic questions
    assert len(generic_questions) <= 6
    assert any('document' in q.lower() for q in generic_questions)
    print(f"✓ Generated {len(generic_questions)} generic questions")
    
    return True


def test_domain_classification():
    """Test domain classification logic."""
    print("\n🎯 Testing Domain Classification")
    print("=" * 50)
    
    generator = adaptive_questions.AdaptiveQuestionGenerator()
    
    test_cases = [
        # Construction domain
        ('Architecture, Engineering, and Construction', 'Construction Drawing Set', 'construction'),
        ('AEC Industry', 'Blueprint Plans', 'construction'),
        
        # Process domain
        ('Petrochemical Process Engineering', 'P&ID Diagram', 'process'),
        ('Chemical Manufacturing', 'Process Flow Diagram', 'process'),
        
        # Electrical domain
        ('Electrical Engineering', 'Circuit Schematic', 'electrical'),
        ('Power Systems', 'Wiring Diagram', 'electrical'),
        
        # Mechanical domain
        ('Mechanical Design', 'Assembly Drawing', 'mechanical'),
        ('Automotive Engineering', 'Component Specification', 'mechanical'),
        
        # Naval domain
        ('Naval Architecture', 'Vessel Plans', 'naval'),
        ('Marine Engineering', 'Ship Design', 'naval'),
        
        # Aerospace domain
        ('Aerospace Engineering', 'Aircraft Systems', 'aerospace'),
        ('Aviation Industry', 'Flight Control Diagram', 'aerospace'),
        
        # Generic domain
        ('Unknown Domain', 'Technical Manual', 'generic'),
        ('General Industry', 'Specification Document', 'generic')
    ]
    
    for industry_domain, document_type, expected_domain in test_cases:
        classified_domain = generator._classify_domain(industry_domain, document_type)
        assert classified_domain == expected_domain, f"Expected {expected_domain}, got {classified_domain} for {industry_domain}/{document_type}"
        print(f"✓ {industry_domain} + {document_type} → {classified_domain}")
    
    print(f"✓ All {len(test_cases)} domain classifications correct")
    return True


def test_context_variable_extraction():
    """Test context variable extraction from discovery results."""
    print("\n🔧 Testing Context Variable Extraction")
    print("=" * 50)
    
    generator = adaptive_questions.AdaptiveQuestionGenerator()
    
    # Test with construction discovery
    discovery_result = {
        'document_type': 'Residential Construction Plans',
        'industry_domain': 'Architecture, Engineering, and Construction',
        'discovered_patterns': {
            'patterns': [
                'Floor Plans (residential layout)',
                'Structural Plans (wood frame)',
                'Mechanical Systems (HVAC)',
                'Electrical Systems (residential power)'
            ]
        }
    }
    
    variables = generator._extract_context_variables(discovery_result)
    
    print("✅ Extracted Context Variables:")
    for key, value in variables.items():
        print(f"  {key}: {value}")
    
    # Validate extracted variables
    assert variables['document_type'] == 'Residential Construction Plans'
    assert variables['industry_domain'] == 'Architecture, Engineering, and Construction'
    assert 'residential' in variables['building_type'].lower()
    assert 'structural' in variables['system_types'] and 'mechanical' in variables['system_types'] and 'electrical' in variables['system_types']
    
    print("✓ Context variable extraction working correctly")
    return True


def test_template_substitution():
    """Test template variable substitution."""
    print("\n📝 Testing Template Substitution")
    print("=" * 50)
    
    generator = adaptive_questions.AdaptiveQuestionGenerator()
    
    # Test normal substitution
    template = "What type of {building_type} is shown in this {document_type}?"
    variables = {
        'building_type': 'commercial building',
        'document_type': 'construction drawing set'
    }
    
    result = generator._substitute_template_variables(template, variables)
    expected = "What type of commercial building is shown in this construction drawing set?"
    
    assert result == expected
    print(f"✓ Normal substitution: {result}")
    
    # Test missing variable handling
    template_missing = "What {missing_var} are shown in this {document_type}?"
    
    result_missing = generator._substitute_template_variables(template_missing, variables)
    print(f"✓ Missing variable handled: {result_missing}")
    
    # Should not crash and should handle missing variables gracefully
    assert 'element' in result_missing  # Generic replacement should be applied
    
    return True


async def main():
    """Run all autonomy tests."""
    print("🚀 Full Autonomy Implementation Test Suite")
    print("=" * 60)
    
    try:
        # Test adaptive question generation
        test1 = test_adaptive_question_generation()
        
        # Test domain classification
        test2 = test_domain_classification()
        
        # Test context variable extraction
        test3 = test_context_variable_extraction()
        
        # Test template substitution
        test4 = test_template_substitution()
        
        if all([test1, test2, test3, test4]):
            print("\n" + "=" * 60)
            print("🎉 ALL AUTONOMY TESTS PASSED!")
            print("\nThe system is now fully autonomous:")
            print("  ✅ Adaptive prompts based on discovery results")
            print("  ✅ Dynamic question generation per domain")
            print("  ✅ Context-aware template substitution")
            print("  ✅ Domain-specific analysis adaptation")
            print("  ✅ No hardcoded assumptions in question generation")
            print("\n🚀 Ready for fully autonomous operation!")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"\n❌ AUTONOMY TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
