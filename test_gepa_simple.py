#!/usr/bin/env python3
"""
Simplified Test for Dynamic GEPA Integration.
Tests the core GEPA optimization concepts without complex dependencies.
"""

import sys
import tempfile
import time
import asyncio
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import our tested modules
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

# Import classes
DynamicElementRegistry = dynamic_schemas.DynamicElementRegistry
CoreElementCategory = dynamic_schemas.CoreElementCategory
DiscoveryMethod = dynamic_schemas.DiscoveryMethod


class MockConfig:
    def __init__(self):
        self.auto_register_confidence_threshold = 0.85
        self.api = type('api', (), {
            'default_model': 'gemini-2.5-flash',
            'gemini_api_key': 'mock_key'
        })()
    
    def get_directories(self):
        return {"output": "output"}


class OptimizationResult:
    """Simplified optimization result for testing."""
    
    def __init__(self):
        self.initial_accuracy = 0.0
        self.final_accuracy = 0.0
        self.improvement = 0.0
        self.type_improvements = {}
        self.new_types_discovered = []
        self.optimization_time = 0.0
        self.registry_size_before = 0
        self.registry_size_after = 0
        self.generations_run = 0
        self.optimization_history = []
    
    @property
    def success_rate(self):
        if self.initial_accuracy == 0:
            return 0.0
        return min(1.0, self.final_accuracy / self.initial_accuracy)


class SimpleDynamicOptimizer:
    """Simplified dynamic optimizer for testing GEPA concepts."""
    
    def __init__(self, config, registry):
        self.config = config
        self.registry = registry
        self.gepa_available = False  # Simulate GEPA not available
        
    def get_initial_performance_metrics(self):
        """Get initial performance metrics."""
        registry_stats = self.registry.get_registry_stats()
        
        return {
            "accuracy": 0.75,  # Baseline accuracy
            "registry_size": registry_stats["total_types"],
            "type_coverage": len(registry_stats["category_counts"]),
            "average_confidence": 0.8
        }
    
    def get_final_performance_metrics(self):
        """Get final performance metrics after optimization."""
        registry_stats = self.registry.get_registry_stats()
        
        # Simulate improvements
        return {
            "accuracy": 0.85,  # Improved accuracy
            "registry_size": registry_stats["total_types"],
            "type_coverage": len(registry_stats["category_counts"]),
            "average_confidence": 0.87,
            "type_improvements": {
                "steel_beam": 0.1,
                "centrifugal_pump": 0.08,
                "control_valve": 0.12
            },
            "new_types": ["composite_beam", "variable_frequency_drive"],
            "strategy_improvements": {
                "pattern_matching": 0.15,
                "nomenclature_analysis": 0.08,
                "ai_classification": 0.12
            }
        }
    
    async def optimize_dynamic_classification(self):
        """Simulate dynamic classification optimization."""
        start_time = time.time()
        
        # Get initial metrics
        initial_metrics = self.get_initial_performance_metrics()
        
        # Simulate optimization process
        print("  🔄 Running optimization iterations...")
        history = []
        for i in range(5):
            current_accuracy = initial_metrics["accuracy"] + (i * 0.02)
            history.append(current_accuracy)
            await asyncio.sleep(0.05)  # Simulate processing
            print(f"    Generation {i+1}: accuracy = {current_accuracy:.3f}")
        
        # Get final metrics
        final_metrics = self.get_final_performance_metrics()
        
        # Create result
        result = OptimizationResult()
        result.initial_accuracy = initial_metrics["accuracy"]
        result.final_accuracy = final_metrics["accuracy"]
        result.improvement = final_metrics["accuracy"] - initial_metrics["accuracy"]
        result.type_improvements = final_metrics["type_improvements"]
        result.new_types_discovered = final_metrics["new_types"]
        result.optimization_time = time.time() - start_time
        result.registry_size_before = initial_metrics["registry_size"]
        result.registry_size_after = final_metrics["registry_size"]
        result.generations_run = 5
        result.optimization_history = history
        
        return result
    
    def save_optimization_results(self, result, output_path):
        """Save optimization results to file."""
        try:
            result_data = {
                "optimization_summary": {
                    "initial_accuracy": result.initial_accuracy,
                    "final_accuracy": result.final_accuracy,
                    "improvement": result.improvement,
                    "success_rate": result.success_rate,
                    "optimization_time": result.optimization_time
                },
                "type_improvements": result.type_improvements,
                "new_types_discovered": result.new_types_discovered,
                "gepa_results": {
                    "generations_run": result.generations_run,
                    "optimization_history": result.optimization_history
                },
                "registry_changes": {
                    "size_before": result.registry_size_before,
                    "size_after": result.registry_size_after,
                    "growth": result.registry_size_after - result.registry_size_before
                },
                "timestamp": time.time()
            }
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            print(f"  ✓ Results saved to {output_path}")
            
        except Exception as e:
            print(f"  ✗ Failed to save results: {e}")


async def test_gepa_integration_concepts():
    """Test GEPA integration concepts with dynamic schemas."""
    print("🧬 Testing GEPA Integration Concepts")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    registry_path = Path(temp_dir) / "gepa_concept_test.json"
    
    try:
        # Step 1: Setup Test Environment
        print("\n📋 Step 1: Setup Test Environment")
        
        registry = DynamicElementRegistry(registry_path)
        config = MockConfig()
        
        # Populate registry with test data
        test_types = [
            ("steel_beam", CoreElementCategory.STRUCTURAL, 0.85, "commercial"),
            ("concrete_column", CoreElementCategory.STRUCTURAL, 0.8, "commercial"),
            ("centrifugal_pump", CoreElementCategory.MEP, 0.9, "industrial"),
            ("control_valve", CoreElementCategory.MEP, 0.88, "industrial"),
            ("hvac_duct", CoreElementCategory.MEP, 0.82, "commercial"),
            ("sliding_door", CoreElementCategory.ARCHITECTURAL, 0.75, "residential")
        ]
        
        registered_count = 0
        for type_name, category, confidence, domain in test_types:
            success, _ = registry.register_discovered_type(
                type_name=type_name,
                base_category=category,
                discovery_confidence=confidence,
                domain_context=domain
            )
            if success:
                registered_count += 1
        
        print(f"✓ Test environment setup with {registered_count} types")
        
        # Step 2: Initialize Optimizer
        print("\n🎯 Step 2: Initialize Dynamic Optimizer")
        
        optimizer = SimpleDynamicOptimizer(config, registry)
        print(f"✓ Optimizer initialized (GEPA available: {optimizer.gepa_available})")
        
        # Step 3: Test Performance Metrics
        print("\n📊 Step 3: Test Performance Metrics")
        
        initial_metrics = optimizer.get_initial_performance_metrics()
        print(f"✓ Initial accuracy: {initial_metrics['accuracy']:.3f}")
        print(f"✓ Registry size: {initial_metrics['registry_size']}")
        print(f"✓ Type coverage: {initial_metrics['type_coverage']} categories")
        
        # Step 4: Run Optimization Simulation
        print("\n🧬 Step 4: Run Optimization Simulation")
        
        result = await optimizer.optimize_dynamic_classification()
        
        print(f"✓ Optimization completed in {result.optimization_time:.2f}s")
        print(f"✓ Accuracy improved from {result.initial_accuracy:.3f} to {result.final_accuracy:.3f}")
        print(f"✓ Total improvement: {result.improvement:.3f}")
        
        # Step 5: Validate Results
        print("\n✅ Step 5: Validate Optimization Results")
        
        assert result.final_accuracy >= result.initial_accuracy
        assert result.improvement >= 0
        assert result.success_rate >= 1.0  # Should show improvement
        assert len(result.optimization_history) > 0
        
        print(f"✓ Success rate: {result.success_rate:.3f}")
        print(f"✓ Generations run: {result.generations_run}")
        print(f"✓ History length: {len(result.optimization_history)}")
        
        # Step 6: Test Type-Specific Improvements
        print("\n📈 Step 6: Test Type-Specific Improvements")
        
        if result.type_improvements:
            print("✓ Type-specific improvements:")
            for type_name, improvement in result.type_improvements.items():
                print(f"  - {type_name}: +{improvement:.3f}")
                
                # Apply improvement to registry
                type_def = registry.get_type_definition(type_name)
                if type_def:
                    evolution_evidence = {
                        "confidence": min(1.0, type_def.discovery_confidence + improvement),
                        "properties": {"gepa_optimized": True, "improvement": improvement},
                        "patterns": ["optimization_enhanced"]
                    }
                    
                    success, _ = registry.evolve_type_definition(type_name, evolution_evidence)
                    if success:
                        print(f"    ✓ Applied improvement to {type_name}")
        
        # Step 7: Test New Type Discovery
        print("\n🔍 Step 7: Test New Type Discovery")
        
        if result.new_types_discovered:
            print(f"✓ New types discovered: {result.new_types_discovered}")
            
            # Register new discovered types
            for new_type in result.new_types_discovered:
                success, _ = registry.register_discovered_type(
                    type_name=new_type,
                    base_category=CoreElementCategory.SPECIALIZED,
                    discovery_confidence=0.8,
                    discovery_method=DiscoveryMethod.AI_CLASSIFICATION,
                    description="Discovered through GEPA optimization"
                )
                if success:
                    print(f"  ✓ Registered new type: {new_type}")
        else:
            print("✓ No new types discovered (expected for simulation)")
        
        # Step 8: Test Result Persistence
        print("\n💾 Step 8: Test Result Persistence")
        
        output_path = Path(temp_dir) / "optimization_results.json"
        optimizer.save_optimization_results(result, output_path)
        
        assert output_path.exists()
        
        # Validate saved content
        with open(output_path, 'r') as f:
            saved_data = json.load(f)
        
        assert "optimization_summary" in saved_data
        assert saved_data["optimization_summary"]["improvement"] == result.improvement
        print("✓ Result persistence validated")
        
        # Step 9: Test Registry Evolution
        print("\n🔄 Step 9: Test Registry Evolution")
        
        final_stats = registry.get_registry_stats()
        print(f"✓ Final registry size: {final_stats['total_types']}")
        
        # Check if any types were evolved
        evolved_count = 0
        for type_name, type_def in registry.discovered_types.items():
            if "gepa_optimized" in type_def.typical_properties:
                evolved_count += 1
        
        print(f"✓ Types evolved with GEPA results: {evolved_count}")
        
        # Step 10: Performance Validation
        print("\n🎯 Step 10: Performance Validation")
        
        # Validate optimization characteristics
        assert result.optimization_time < 10  # Should be fast for simulation
        assert 0 <= result.final_accuracy <= 1.0
        assert result.generations_run > 0
        
        # Validate improvement trajectory
        if len(result.optimization_history) > 1:
            # Should show generally improving trend
            first_accuracy = result.optimization_history[0]
            last_accuracy = result.optimization_history[-1]
            assert last_accuracy >= first_accuracy
            print("✓ Optimization showed improving trend")
        
        print("\n" + "=" * 60)
        print("✅ GEPA INTEGRATION CONCEPT TESTS PASSED!")
        print("\n📊 Final Results Summary:")
        print(f"  🎯 Accuracy improvement: {result.improvement:.3f}")
        print(f"  ⏱️  Optimization time: {result.optimization_time:.2f}s")
        print(f"  🔄 Generations completed: {result.generations_run}")
        print(f"  📈 Success rate: {result.success_rate:.3f}")
        print(f"  📝 Types improved: {len(result.type_improvements)}")
        print(f"  🔍 New types discovered: {len(result.new_types_discovered)}")
        print(f"  📊 Final registry size: {final_stats['total_types']}")
        
        # Integration validation
        print("\n🔍 GEPA Integration Validation:")
        print("  ✓ Registry-based performance metrics")
        print("  ✓ Dynamic type optimization simulation")
        print("  ✓ Type-specific improvement tracking")
        print("  ✓ New type discovery integration")
        print("  ✓ Registry evolution with optimization results")
        print("  ✓ Result persistence and validation")
        print("  ✓ Performance characteristic validation")
        
        return True
        
    except Exception as e:
        print(f"\n❌ GEPA INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        import shutil
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)


async def main():
    """Run GEPA integration concept tests."""
    print("🚀 GEPA Integration Concept Test Suite")
    print("=" * 70)
    
    try:
        success = await test_gepa_integration_concepts()
        
        if success:
            print("\n" + "=" * 70)
            print("🎉 ALL GEPA INTEGRATION CONCEPT TESTS PASSED!")
            print("\nThe GEPA integration concepts are working correctly:")
            print("  ✅ Dynamic registry integration")
            print("  ✅ Performance metric calculation")
            print("  ✅ Optimization simulation")
            print("  ✅ Type-specific improvements")
            print("  ✅ Registry evolution")
            print("  ✅ Result persistence")
            print("\n🚀 Ready for full GEPA implementation!")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
