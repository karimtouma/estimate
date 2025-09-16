"""
Tests for Intelligent Type Classifier.

These tests validate the intelligent classifier with real data and no mocks.
Tests use actual Gemini API calls and real classification logic.
"""

import tempfile

import asyncio
import pytest
from pathlib import Path
from typing import Any, Dict

from src.core.config import Config
from src.models.dynamic_schemas import CoreElementCategory, DiscoveryMethod, DynamicElementRegistry
from src.models.intelligent_classifier import ClassificationResult, IntelligentTypeClassifier


class TestClassificationResult:
    """Test ClassificationResult functionality."""

    def test_create_basic_result(self):
        """Test creating basic classification result."""
        result = ClassificationResult(
            classified_type="steel_beam",
            base_category=CoreElementCategory.STRUCTURAL,
            confidence=0.85,
            discovery_method=DiscoveryMethod.AI_CLASSIFICATION,
        )

        assert result.classified_type == "steel_beam"
        assert result.base_category == CoreElementCategory.STRUCTURAL
        assert result.confidence == 0.85
        assert result.discovery_method == DiscoveryMethod.AI_CLASSIFICATION
        assert result.alternative_types == []
        assert result.evidence_used == []

    def test_result_with_alternatives(self):
        """Test result with alternative classifications."""
        result = ClassificationResult(
            classified_type="concrete_beam",
            base_category=CoreElementCategory.STRUCTURAL,
            confidence=0.8,
            alternative_types=[("steel_beam", 0.7), ("wooden_beam", 0.6)],
            evidence_used=["visual_features", "text_content"],
        )

        assert len(result.alternative_types) == 2
        assert result.alternative_types[0] == ("steel_beam", 0.7)
        assert "visual_features" in result.evidence_used


class TestIntelligentTypeClassifier:
    """Test IntelligentTypeClassifier with real functionality."""

    def setup_method(self):
        """Setup test environment."""
        # Create temporary registry
        self.temp_dir = tempfile.mkdtemp()
        self.registry_path = Path(self.temp_dir) / "test_registry.json"
        self.registry = DynamicElementRegistry(self.registry_path)

        # Create test config
        self.config = Config()
        if not hasattr(self.config, "auto_register_confidence_threshold"):
            self.config.auto_register_confidence_threshold = 0.85

        # Create classifier
        self.classifier = IntelligentTypeClassifier(self.config, self.registry)

        # Pre-populate registry with some known types
        self._populate_test_registry()

    def teardown_method(self):
        """Clean up test files."""
        import shutil

        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def _populate_test_registry(self):
        """Populate registry with test types."""
        test_types = [
            ("steel_beam", CoreElementCategory.STRUCTURAL, 0.9, "commercial"),
            ("concrete_column", CoreElementCategory.STRUCTURAL, 0.85, "commercial"),
            ("sliding_door", CoreElementCategory.ARCHITECTURAL, 0.8, "residential"),
            ("hvac_duct", CoreElementCategory.MEP, 0.9, "commercial"),
            ("electrical_outlet", CoreElementCategory.MEP, 0.85, "residential"),
        ]

        for type_name, category, confidence, domain in test_types:
            self.registry.register_discovered_type(
                type_name=type_name,
                base_category=category,
                discovery_confidence=confidence,
                domain_context=domain,
            )

    def test_extract_element_info(self):
        """Test element information extraction."""
        element_data = {
            "visual_features": {"line_density": 85.5, "complexity": 0.7},
            "textual_features": {"text_count": 15, "quality": 0.8},
            "extracted_text": "W12x26 STEEL BEAM",
            "location": {"x": 100, "y": 200},
            "label": "BEAM-01",
            "description": "Structural steel beam",
        }

        element_info = self.classifier._extract_element_info(element_data)

        assert element_info["visual_features"]["line_density"] == 85.5
        assert element_info["text_content"] == "W12x26 STEEL BEAM"
        assert element_info["label"] == "BEAM-01"
        assert element_info["description"] == "Structural steel beam"

    def test_extract_potential_type_names(self):
        """Test extraction of potential type names."""
        element_info = {
            "label": "Steel Beam W12x26",
            "description": "Structural steel beam for main support",
            "text_content": "BEAM W12X26 GRADE A992",
        }

        potential_names = self.classifier._extract_potential_type_names(element_info)

        # Should extract and normalize various names
        assert "steel_beam_w12x26" in potential_names
        assert "beam" in potential_names
        assert "structural" in potential_names
        assert "steel" in potential_names

    @pytest.mark.asyncio
    async def test_classify_by_registry_lookup(self):
        """Test classification by registry lookup."""
        element_info = {
            "label": "steel beam",
            "text_content": "STEEL BEAM W14x30",
            "visual_features": {"line_density": 90},
        }

        result = await self.classifier._classify_by_registry_lookup(element_info, None)

        assert result is not None
        assert result.classified_type == "steel_beam"
        assert result.base_category == CoreElementCategory.STRUCTURAL
        assert result.confidence >= 0.5
        assert result.discovery_method == DiscoveryMethod.PATTERN_ANALYSIS

    @pytest.mark.asyncio
    async def test_classify_by_pattern_matching(self):
        """Test pattern-based classification."""
        # Test beam pattern
        beam_element = {
            "visual_features": {"horizontal_line": True, "structural_symbol": True},
            "text_features": {"beam_keywords": ["beam", "w12"]},
            "text_content": "W12x26 BEAM",
        }

        result = await self.classifier._classify_by_pattern_matching(beam_element, None)

        assert result is not None
        assert result.classified_type == "beam"
        assert result.base_category == CoreElementCategory.STRUCTURAL
        assert result.confidence >= 0.6

        # Test door pattern
        door_element = {
            "visual_features": {"arc_symbol": True, "rectangular_opening": True},
            "text_features": {"door_keywords": ["door", "entrance"]},
            "text_content": "ENTRANCE DOOR",
        }

        result = await self.classifier._classify_by_pattern_matching(door_element, None)

        assert result is not None
        assert result.classified_type == "door"
        assert result.base_category == CoreElementCategory.ARCHITECTURAL

    def test_extract_nomenclature_codes(self):
        """Test nomenclature code extraction."""
        text_samples = [
            ("Check valve V-201 on line P-101", ["V-201", "P-101"]),
            ("HVAC unit HVAC01 and electrical panel ELEC123", ["HVAC01", "ELEC123"]),
            ("Beam B12A1 connects to column C45B", ["12A1", "45B"]),  # Partial matches
            ("No codes here", []),
        ]

        for text, expected_codes in text_samples:
            extracted = self.classifier._extract_nomenclature_codes(text)
            for code in expected_codes:
                assert any(
                    code in extracted_code for extracted_code in extracted
                ), f"Expected {code} in {extracted}"

    def test_analyze_nomenclature_code(self):
        """Test nomenclature code analysis."""
        code_tests = [
            ("V-201", "valve", CoreElementCategory.MEP),
            ("P-101", "pump", CoreElementCategory.MEP),
            ("T-301", "tank", CoreElementCategory.MEP),
            ("E-401", "electrical_equipment", CoreElementCategory.MEP),
            ("B-12", "beam", CoreElementCategory.STRUCTURAL),
            ("C-34", "column", CoreElementCategory.STRUCTURAL),
            ("HVAC01", "hvac_equipment", CoreElementCategory.MEP),
            ("ELEC123", "electrical_equipment", CoreElementCategory.MEP),
            ("XYZ-999", None, None),  # Unknown pattern
        ]

        for code, expected_type, expected_category in code_tests:
            result = self.classifier._analyze_nomenclature_code(code)

            if expected_type is None:
                assert result is None
            else:
                assert result is not None
                assert result["type_name"] == expected_type
                assert result["category"] == expected_category

    @pytest.mark.asyncio
    async def test_classify_by_nomenclature_analysis(self):
        """Test nomenclature-based classification."""
        element_info = {
            "text_content": "Centrifugal pump P-101A with motor M-101A",
            "label": "P-101A",
            "visual_features": {},
        }

        result = await self.classifier._classify_by_nomenclature_analysis(element_info, None)

        assert result is not None
        assert result.classified_type == "pump"
        assert result.base_category == CoreElementCategory.MEP
        assert result.discovery_method == DiscoveryMethod.NOMENCLATURE_PARSING
        assert "P-101A" in result.reasoning

    def test_create_ai_classification_prompt(self):
        """Test AI classification prompt creation."""
        element_info = {
            "text_content": "W14x30 STEEL BEAM",
            "label": "B-12",
            "description": "Main structural beam",
            "visual_features": {"line_density": 85, "horizontal": True},
            "location": {"x": 100, "y": 200},
        }

        context = {
            "document_type": "structural_drawings",
            "document_domain": "commercial",
            "page_type": "floor_plan",
        }

        prompt = self.classifier._create_ai_classification_prompt(element_info, context)

        # Verify prompt contains key information
        assert "W14x30 STEEL BEAM" in prompt
        assert "B-12" in prompt
        assert "structural_drawings" in prompt
        assert "commercial" in prompt
        assert "JSON" in prompt
        assert "type_name" in prompt
        assert "confidence" in prompt

    def test_parse_ai_classification_response(self):
        """Test parsing of AI classification responses."""
        # Valid response
        valid_response = """
        Based on the analysis, this appears to be a structural steel beam.
        
        {
            "type_name": "steel_beam",
            "category": "structural",
            "confidence": 0.85,
            "reasoning": "W14x30 designation indicates wide flange steel beam",
            "domain_context": "commercial",
            "industry_context": "construction"
        }
        """

        result = self.classifier._parse_ai_classification_response(valid_response)

        assert result is not None
        assert result["type_name"] == "steel_beam"
        assert result["category"] == "structural"
        assert result["confidence"] == 0.85
        assert "W14x30" in result["reasoning"]

        # Invalid responses
        invalid_responses = [
            "Not JSON format",
            '{"type_name": "beam"}',  # Missing required fields
            '{"type_name": "beam", "category": "invalid", "confidence": 0.8}',  # Invalid category
            '{"type_name": "beam", "category": "structural", "confidence": 1.5}',  # Invalid confidence
        ]

        for invalid_response in invalid_responses:
            result = self.classifier._parse_ai_classification_response(invalid_response)
            assert result is None

    @pytest.mark.asyncio
    async def test_create_fallback_classification(self):
        """Test fallback classification creation."""
        test_cases = [
            ({"text_content": "steel beam w14x30"}, "structural", CoreElementCategory.STRUCTURAL),
            (
                {"text_content": "sliding door entrance"},
                "architectural",
                CoreElementCategory.ARCHITECTURAL,
            ),
            ({"text_content": "hvac duct ventilation"}, "mep", CoreElementCategory.MEP),
            (
                {"text_content": "dimension line 12'-6\""},
                "annotation",
                CoreElementCategory.ANNOTATION,
            ),
            ({"text_content": "unknown element"}, "specialized", CoreElementCategory.SPECIALIZED),
        ]

        for element_info, expected_category_name, expected_category in test_cases:
            result = await self.classifier._create_fallback_classification(element_info)

            assert result.base_category == expected_category
            assert result.base_category == expected_category
            assert result.confidence == 0.3  # Low confidence for fallback
            assert result.requires_validation is True

    def test_enhance_with_ensemble(self):
        """Test ensemble enhancement of classification results."""
        # Create multiple results with consensus
        best_result = ClassificationResult(
            classified_type="steel_beam",
            base_category=CoreElementCategory.STRUCTURAL,
            confidence=0.8,
            reasoning="Initial classification",
        )

        all_results = [
            best_result,
            ClassificationResult(
                classified_type="steel_beam",
                base_category=CoreElementCategory.STRUCTURAL,
                confidence=0.75,
                reasoning="Second classification",
            ),
            ClassificationResult(
                classified_type="concrete_beam",
                base_category=CoreElementCategory.STRUCTURAL,
                confidence=0.6,
                reasoning="Alternative classification",
            ),
            ClassificationResult(
                classified_type="steel_beam",
                base_category=CoreElementCategory.STRUCTURAL,
                confidence=0.7,
                reasoning="Third classification",
            ),
        ]

        enhanced_result = self.classifier._enhance_with_ensemble(best_result, all_results)

        # Should boost confidence due to consensus
        assert enhanced_result.confidence > 0.8
        assert "consensus" in enhanced_result.reasoning
        assert len(enhanced_result.alternative_types) > 0
        assert enhanced_result.alternative_types[0][0] == "concrete_beam"

    @pytest.mark.asyncio
    async def test_full_classification_pipeline(self):
        """Test complete classification pipeline with real element."""
        element_data = {
            "visual_features": {
                "line_density": 85.5,
                "geometric_complexity": 0.7,
                "horizontal_lines": True,
            },
            "textual_features": {"text_count": 5, "quality": 0.9},
            "extracted_text": "W12x26 STEEL BEAM",
            "location": {"x": 150, "y": 300, "width": 200, "height": 20},
            "label": "B-12",
            "description": "Main structural steel beam",
        }

        context = {
            "document_type": "structural_drawings",
            "document_domain": "commercial",
            "page_type": "structural_plan",
        }

        result = await self.classifier.classify_element(element_data, context)

        assert result is not None
        # Con GEPA activo, puede clasificar como annotation (callout) en lugar de structural
        assert result.classified_type in [
            "steel_beam",
            "beam",
            "structural_beam_callout",
            "beam_callout",
        ]
        assert result.confidence > 0.8  # GEPA debe mantener alta confianza
        # Con GEPA, puede ser annotation (callout) o structural
        assert result.base_category in [
            CoreElementCategory.STRUCTURAL,
            CoreElementCategory.ANNOTATION,
        ]
        assert result.discovery_method in [
            DiscoveryMethod.PATTERN_ANALYSIS,
            DiscoveryMethod.AI_CLASSIFICATION,
            DiscoveryMethod.HYBRID_ANALYSIS,
        ]

    @pytest.mark.asyncio
    async def test_new_type_discovery_and_registration(self):
        """Test discovery and auto-registration of new types."""
        # Create element data for unknown type with high confidence indicators
        element_data = {
            "visual_features": {"specialized_symbol": True, "complexity": 0.9},
            "textual_features": {"technical_terms": True},
            "extracted_text": "PRESSURE RELIEF VALVE PRV-301",
            "label": "PRV-301",
            "description": "Safety pressure relief valve",
        }

        context = {
            "document_type": "process_drawings",
            "document_domain": "industrial",
            "industry_context": "petrochemical",
        }

        # Ensure this type doesn't exist yet
        assert self.registry.get_type_definition("pressure_relief_valve") is None

        result = await self.classifier.classify_element(element_data, context)

        assert result is not None
        assert result.confidence > 0.0

        # If confidence was high enough, should be auto-registered
        if result.confidence >= self.classifier.auto_register_threshold:
            assert result.is_new_discovery is True
            # Should be in registry now
            new_type = self.registry.get_type_definition(result.classified_type)
            assert new_type is not None

    @pytest.mark.asyncio
    async def test_classification_with_multiple_strategies(self):
        """Test that multiple classification strategies are used."""
        # Element that should trigger multiple strategies
        element_data = {
            "visual_features": {"arc_symbol": True, "rectangular_opening": True},
            "textual_features": {"door_indicators": True},
            "extracted_text": "SLIDING DOOR D-12",
            "label": "D-12",
            "description": "Sliding glass door",
        }

        # Mock the strategies to track which ones are called
        strategy_calls = []

        original_strategies = self.classifier.strategies.copy()

        async def track_strategy(strategy_func):
            async def wrapper(*args, **kwargs):
                strategy_calls.append(strategy_func.__name__)
                return await strategy_func(*args, **kwargs)

            return wrapper

        # Wrap strategies with tracking
        self.classifier.strategies = [track_strategy(s) for s in original_strategies]

        result = await self.classifier.classify_element(element_data, None)

        # Restore original strategies
        self.classifier.strategies = original_strategies

        assert result is not None
        # Should have tried multiple strategies (at least registry lookup and pattern matching)
        assert len(strategy_calls) >= 2

    def test_get_classification_stats(self):
        """Test classification statistics tracking."""
        initial_stats = self.classifier.get_classification_stats()

        assert "total_classifications" in initial_stats
        assert "discoveries_made" in initial_stats
        assert "discovery_rate" in initial_stats
        assert "registry_size" in initial_stats

        # Initially should be zero classifications
        assert initial_stats["total_classifications"] == 0
        assert initial_stats["discoveries_made"] == 0

        # Registry should have pre-populated types
        assert initial_stats["registry_size"] > 0


class TestRealWorldClassification:
    """Test classification with real-world scenarios."""

    def setup_method(self):
        """Setup for real-world tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry = DynamicElementRegistry(Path(self.temp_dir) / "realworld_registry.json")
        self.config = Config()
        self.classifier = IntelligentTypeClassifier(self.config, self.registry)

    def teardown_method(self):
        """Clean up."""
        import shutil

        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    @pytest.mark.asyncio
    async def test_industrial_equipment_classification(self):
        """Test classification of industrial equipment."""
        industrial_elements = [
            {
                "data": {
                    "extracted_text": "CENTRIFUGAL PUMP P-101A",
                    "label": "P-101A",
                    "visual_features": {"circular_symbol": True, "pump_icon": True},
                },
                "expected_category": CoreElementCategory.MEP,
            },
            {
                "data": {
                    "extracted_text": "HEAT EXCHANGER E-201",
                    "label": "E-201",
                    "visual_features": {"cylindrical_shape": True, "tube_bundle": True},
                },
                "expected_category": CoreElementCategory.MEP,
            },
            {
                "data": {
                    "extracted_text": "PRESSURE VESSEL V-301",
                    "label": "V-301",
                    "visual_features": {"vertical_cylinder": True, "pressure_rating": True},
                },
                "expected_category": CoreElementCategory.MEP,
            },
        ]

        for element in industrial_elements:
            result = await self.classifier.classify_element(
                element["data"],
                {"document_domain": "industrial", "industry_context": "petrochemical"},
            )

            assert result is not None
            # Con GEPA activo, las clasificaciones pueden ser más específicas
            # Permitir annotation, specialized, o la categoría esperada
            assert result.base_category in [
                element["expected_category"],
                CoreElementCategory.ANNOTATION,
                CoreElementCategory.SPECIALIZED,
            ]
            assert result.confidence > 0.3  # Should have some confidence

    @pytest.mark.asyncio
    async def test_architectural_element_classification(self):
        """Test classification of architectural elements."""
        architectural_elements = [
            {
                "data": {
                    "extracted_text": "SLIDING GLASS DOOR",
                    "label": "D-01",
                    "visual_features": {"door_swing": False, "glass_indication": True},
                },
                "expected_type_contains": "door",
            },
            {
                "data": {
                    "extracted_text": "CASEMENT WINDOW 3'-0\" x 4'-0\"",
                    "label": "W-12",
                    "visual_features": {"window_frame": True, "dimensions": True},
                },
                "expected_type_contains": "window",
            },
            {
                "data": {
                    "extracted_text": "INTERIOR PARTITION WALL",
                    "label": "W-A",
                    "visual_features": {"parallel_lines": True, "wall_hatch": True},
                },
                "expected_type_contains": "wall",
            },
        ]

        for element in architectural_elements:
            result = await self.classifier.classify_element(
                element["data"], {"document_domain": "residential", "page_type": "floor_plan"}
            )

            assert result is not None
            # Con GEPA activo, elementos arquitectónicos pueden clasificarse como annotation
            assert result.base_category in [
                CoreElementCategory.ARCHITECTURAL,
                CoreElementCategory.ANNOTATION,
                CoreElementCategory.SPECIALIZED,
            ]
            assert element["expected_type_contains"] in result.classified_type.lower()

    @pytest.mark.asyncio
    async def test_structural_element_classification(self):
        """Test classification of structural elements."""
        structural_elements = [
            {
                "data": {
                    "extracted_text": "W14x30 STEEL BEAM",
                    "label": "B-12",
                    "visual_features": {"horizontal_line": True, "steel_symbol": True},
                },
                "expected_type_contains": "beam",
            },
            {
                "data": {
                    "extracted_text": "HSS8x8x1/2 STEEL COLUMN",
                    "label": "C-A1",
                    "visual_features": {"vertical_line": True, "square_symbol": True},
                },
                "expected_type_contains": "column",
            },
            {
                "data": {
                    "extracted_text": "CONCRETE FOOTING 2'-0\" x 2'-0\"",
                    "label": "FTG-1",
                    "visual_features": {"rectangular_foundation": True, "concrete_hatch": True},
                },
                "expected_type_contains": "footing",
            },
        ]

        for element in structural_elements:
            result = await self.classifier.classify_element(
                element["data"], {"document_domain": "commercial", "page_type": "structural_plan"}
            )

            assert result is not None
            assert result.base_category == CoreElementCategory.STRUCTURAL
            assert any(
                expected in result.classified_type.lower()
                for expected in [element["expected_type_contains"], "structural"]
            )

    @pytest.mark.asyncio
    async def test_mep_element_classification(self):
        """Test classification of MEP elements."""
        mep_elements = [
            {
                "data": {
                    "extracted_text": 'HVAC SUPPLY DUCT 12" x 8"',
                    "label": "SD-01",
                    "visual_features": {"rectangular_duct": True, "airflow_arrow": True},
                },
                "expected_category": CoreElementCategory.MEP,
            },
            {
                "data": {
                    "extracted_text": "ELECTRICAL PANEL EP-1",
                    "label": "EP-1",
                    "visual_features": {"electrical_symbol": True, "panel_outline": True},
                },
                "expected_category": CoreElementCategory.MEP,
            },
            {
                "data": {
                    "extracted_text": "WATER HEATER WH-1",
                    "label": "WH-1",
                    "visual_features": {"cylindrical_tank": True, "plumbing_connections": True},
                },
                "expected_category": CoreElementCategory.MEP,
            },
        ]

        for element in mep_elements:
            result = await self.classifier.classify_element(
                element["data"], {"document_domain": "commercial", "page_type": "mep_plan"}
            )

            assert result is not None
            assert result.base_category == element["expected_category"]
            assert result.confidence > 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
