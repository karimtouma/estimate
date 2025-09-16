"""
Tests for GEPA Optimization System.

Tests validate the genetic evolution prompt architecture,
multiple candidate generation, and judge evaluation system.
"""

import tempfile
from unittest.mock import Mock, patch

import asyncio
import pytest
from pathlib import Path
from typing import Any, Dict

from src.core.config import Config
from src.models.dynamic_schemas import CoreElementCategory, DiscoveryMethod
from src.optimization.comprehensive_gepa_system import (
    ComprehensiveGEPAResult,
    ComprehensiveGEPASystem,
    PromptTestResult,
    create_comprehensive_gepa_system,
)
from src.optimization.gepa_classification_enhancer import (
    ClassificationCandidate,
    EnhancedClassificationResult,
    GEPAClassificationEnhancer,
    create_gepa_classification_enhancer,
)


class TestClassificationCandidate:
    """Test ClassificationCandidate data structure."""

    def test_create_basic_candidate(self):
        """Test creating basic classification candidate."""
        candidate = ClassificationCandidate(
            candidate_id="candidate_1",
            type_name="steel_beam",
            category="structural",
            confidence=0.85,
            reasoning="Strong structural indicators detected",
        )

        assert candidate.candidate_id == "candidate_1"
        assert candidate.type_name == "steel_beam"
        assert candidate.category == "structural"
        assert candidate.confidence == 0.85
        assert "structural indicators" in candidate.reasoning
        assert candidate.judge_score == 0.0  # Default
        assert candidate.strengths == []
        assert candidate.weaknesses == []

    def test_candidate_with_judge_evaluation(self):
        """Test candidate with judge evaluation."""
        candidate = ClassificationCandidate(
            candidate_id="candidate_2",
            type_name="concrete_column",
            category="structural",
            confidence=0.9,
            reasoning="Clear structural element",
            judge_score=0.95,
            strengths=["specific type", "high confidence"],
            weaknesses=["minor context issues"],
        )

        assert candidate.judge_score == 0.95
        assert len(candidate.strengths) == 2
        assert len(candidate.weaknesses) == 1
        assert "specific type" in candidate.strengths


class TestEnhancedClassificationResult:
    """Test EnhancedClassificationResult structure."""

    def test_create_enhanced_result(self):
        """Test creating enhanced classification result."""
        best_candidate = ClassificationCandidate(
            candidate_id="best",
            type_name="project_title",
            category="annotation",
            confidence=0.99,
            reasoning="Clear project identification",
        )

        all_candidates = [best_candidate]

        result = EnhancedClassificationResult(
            best_candidate=best_candidate,
            all_candidates=all_candidates,
            consensus_score=0.95,
            judge_score=1.0,
            processing_time=45.2,
            improvement_over_base=0.15,
        )

        assert result.best_candidate == best_candidate
        assert len(result.all_candidates) == 1
        assert result.consensus_score == 0.95
        assert result.judge_score == 1.0
        assert result.processing_time == 45.2
        assert result.improvement_over_base == 0.15

    def test_to_classification_result_conversion(self):
        """Test conversion to standard ClassificationResult."""
        candidate = ClassificationCandidate(
            candidate_id="test",
            type_name="architectural_note",
            category="annotation",
            confidence=0.88,
            reasoning="Architectural annotation detected",
            domain_context="commercial",
            industry_context="construction",
        )

        enhanced_result = EnhancedClassificationResult(
            best_candidate=candidate,
            all_candidates=[candidate],
            consensus_score=0.9,
            judge_score=0.98,
        )

        standard_result = enhanced_result.to_classification_result()

        assert standard_result.classified_type == "architectural_note"
        assert standard_result.base_category == CoreElementCategory.ANNOTATION
        assert standard_result.confidence == 0.88
        assert "GEPA-enhanced" in standard_result.reasoning
        assert standard_result.discovery_method == DiscoveryMethod.AI_CLASSIFICATION
        assert "gepa_multi_candidate_analysis" in standard_result.evidence_used


class TestGEPAClassificationEnhancerBasic:
    """Test GEPAClassificationEnhancer basic functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.config = Config()
        # Mock gemini client to avoid API calls
        self.mock_gemini_client = Mock()
        self.enhancer = GEPAClassificationEnhancer(self.config, self.mock_gemini_client)

    def test_enhancer_initialization(self):
        """Test GEPA enhancer initialization."""
        assert self.enhancer.config is not None
        assert self.enhancer.gemini_client is not None
        assert self.enhancer.enhancement_count == 0
        assert self.enhancer.consensus_history == []
        assert self.enhancer.judge_score_history == []
        assert self.enhancer.processing_time_history == []

    def test_generate_classification_schema(self):
        """Test classification schema generation."""
        schema = self.enhancer._get_classification_schema()

        assert schema is not None
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "type_name" in schema["properties"]
        assert "category" in schema["properties"]
        assert "confidence" in schema["properties"]
        assert "reasoning" in schema["properties"]

    def test_judge_schema_generation(self):
        """Test judge evaluation schema generation."""
        schema = self.enhancer._get_judge_schema()

        assert schema is not None
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "best_candidate_id" in schema["properties"]
        assert "candidate_evaluations" in schema["properties"]
        assert "consensus_analysis" in schema["properties"]
        assert "judge_reasoning" in schema["properties"]

        # Verify candidate_evaluations is array (not object)
        candidate_eval = schema["properties"]["candidate_evaluations"]
        assert candidate_eval["type"] == "array"
        assert "items" in candidate_eval

    def test_calculate_consensus_score_single_candidate(self):
        """Test consensus calculation with single candidate."""
        candidates = [
            ClassificationCandidate(
                candidate_id="only", type_name="test_type", category="annotation", confidence=0.9
            )
        ]

        consensus = self.enhancer._calculate_consensus_score(candidates)
        assert consensus == 1.0  # Perfect consensus with single candidate

    def test_calculate_consensus_score_multiple_candidates(self):
        """Test consensus calculation with multiple candidates."""
        candidates = [
            ClassificationCandidate("c1", "steel_beam", "structural", 0.9),
            ClassificationCandidate("c2", "steel_beam", "structural", 0.85),
            ClassificationCandidate("c3", "concrete_beam", "structural", 0.7),
            ClassificationCandidate("c4", "steel_beam", "structural", 0.8),
            ClassificationCandidate("c5", "steel_beam", "structural", 0.88),
        ]

        consensus = self.enhancer._calculate_consensus_score(candidates)

        # Should have high consensus (4/5 agree on steel_beam)
        assert consensus > 0.7
        assert consensus <= 1.0

    def test_select_best_candidate_with_judge(self):
        """Test best candidate selection with judge evaluation."""
        candidates = [
            ClassificationCandidate("c1", "type_a", "annotation", 0.8, judge_score=0.9),
            ClassificationCandidate("c2", "type_b", "annotation", 0.85, judge_score=0.95),
            ClassificationCandidate("c3", "type_c", "annotation", 0.9, judge_score=0.85),
        ]

        judge_evaluation = {
            "best_candidate_id": "c2",
            "candidate_evaluations": [
                {"candidate_id": "c1", "score": 0.9},
                {"candidate_id": "c2", "score": 0.95},
                {"candidate_id": "c3", "score": 0.85},
            ],
            "judge_reasoning": "c2 has best balance of confidence and specificity",
        }

        best = self.enhancer._select_best_candidate(candidates, judge_evaluation)

        assert best.candidate_id == "c2"
        assert best.type_name == "type_b"
        assert best.judge_score == 0.95

    def test_select_best_candidate_fallback(self):
        """Test best candidate selection fallback without judge."""
        candidates = [
            ClassificationCandidate("c1", "type_a", "annotation", 0.8),
            ClassificationCandidate("c2", "type_b", "annotation", 0.95),  # Highest confidence
            ClassificationCandidate("c3", "type_c", "annotation", 0.7),
        ]

        # Empty judge evaluation (fallback case)
        judge_evaluation = {}

        best = self.enhancer._select_best_candidate(candidates, judge_evaluation)

        assert best.candidate_id == "c2"  # Should select highest confidence
        assert best.confidence == 0.95


class TestComprehensiveGEPASystemBasic:
    """Test ComprehensiveGEPASystem basic functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.config = Config()
        # Mock gemini client
        self.mock_gemini_client = Mock()
        self.gepa_system = ComprehensiveGEPASystem(self.config, self.mock_gemini_client)

    def test_gepa_system_initialization(self):
        """Test GEPA system initialization."""
        assert self.gepa_system.config is not None
        assert self.gepa_system.gemini_client is not None
        assert self.gepa_system.optimization_history == []
        assert self.gepa_system.prompt_performance == {}
        assert self.gepa_system.evolution_generation == 0

    def test_prompt_test_result_creation(self):
        """Test PromptTestResult creation."""
        test_result = PromptTestResult(
            prompt_id="test_prompt_1",
            test_score=0.85,
            execution_time=12.5,
            error_rate=0.1,
            sample_results=[{"accuracy": 0.9}, {"accuracy": 0.8}],
        )

        assert test_result.prompt_id == "test_prompt_1"
        assert test_result.test_score == 0.85
        assert test_result.execution_time == 12.5
        assert test_result.error_rate == 0.1
        assert len(test_result.sample_results) == 2
        assert test_result.needs_improvement() is False  # Good score, low error rate

    def test_prompt_test_result_needs_improvement(self):
        """Test PromptTestResult needs improvement detection."""
        # Poor performance result
        poor_result = PromptTestResult(
            prompt_id="poor_prompt",
            test_score=0.4,  # Below threshold
            execution_time=60.0,
            error_rate=0.3,  # High error rate
            sample_results=[],
        )

        assert poor_result.needs_improvement() is True

        # Good performance result
        good_result = PromptTestResult(
            prompt_id="good_prompt",
            test_score=0.9,
            execution_time=15.0,
            error_rate=0.05,
            sample_results=[],
        )

        assert good_result.needs_improvement() is False


class TestGEPAFactoryFunctions:
    """Test GEPA factory functions."""

    def setup_method(self):
        """Setup test environment."""
        self.config = Config()

    def test_create_gepa_classification_enhancer(self):
        """Test GEPA classification enhancer factory."""
        # Mock gemini client
        mock_gemini_client = Mock()

        enhancer = create_gepa_classification_enhancer(self.config, mock_gemini_client)

        assert enhancer is not None
        assert isinstance(enhancer, GEPAClassificationEnhancer)
        assert enhancer.config == self.config
        assert enhancer.gemini_client == mock_gemini_client

    def test_create_comprehensive_gepa_system(self):
        """Test comprehensive GEPA system factory."""
        # Mock gemini client
        mock_gemini_client = Mock()

        gepa_system = create_comprehensive_gepa_system(self.config, mock_gemini_client)

        assert gepa_system is not None
        assert isinstance(gepa_system, ComprehensiveGEPASystem)
        assert gepa_system.config == self.config
        assert gepa_system.gemini_client == mock_gemini_client


class TestGEPAStatistics:
    """Test GEPA statistics tracking and calculation."""

    def setup_method(self):
        """Setup test environment."""
        self.config = Config()
        self.mock_gemini_client = Mock()
        self.enhancer = GEPAClassificationEnhancer(self.config, self.mock_gemini_client)

    def test_statistics_tracking_initialization(self):
        """Test statistics tracking initialization."""
        stats = self.enhancer.get_enhancement_statistics()

        assert stats["total_enhancements"] == 0
        assert stats["average_consensus"] == 0.0
        assert stats["average_judge_score"] == 0.0
        assert stats["average_improvement"] == 0.0
        assert stats["average_processing_time"] == 0.0
        assert stats["category_distribution"] == {}
        assert stats["confidence_distribution"]["high"] == 0
        assert stats["confidence_distribution"]["medium"] == 0
        assert stats["confidence_distribution"]["low"] == 0

    def test_statistics_after_enhancements(self):
        """Test statistics after simulated enhancements."""
        # Simulate enhancement history
        self.enhancer.enhancement_count = 5
        self.enhancer.consensus_history = [0.95, 0.92, 0.98, 0.90, 0.96]
        self.enhancer.judge_score_history = [1.0, 0.98, 1.0, 0.95, 0.99]
        self.enhancer.processing_time_history = [35.2, 42.1, 38.5, 45.0, 39.8]

        stats = self.enhancer.get_enhancement_statistics()

        assert stats["total_enhancements"] == 5
        assert 0.9 < stats["average_consensus"] < 1.0
        assert 0.95 < stats["average_judge_score"] <= 1.0
        assert 35.0 < stats["average_processing_time"] < 50.0

    def test_confidence_distribution_calculation(self):
        """Test confidence distribution calculation."""
        # Mock some results with different confidence levels
        mock_results = [
            Mock(confidence=0.95),  # High
            Mock(confidence=0.88),  # High
            Mock(confidence=0.75),  # Medium
            Mock(confidence=0.92),  # High
            Mock(confidence=0.65),  # Medium
            Mock(confidence=0.45),  # Low
        ]

        # Simulate processing these results
        distribution = {"high": 0, "medium": 0, "low": 0}

        for result in mock_results:
            if result.confidence >= 0.8:
                distribution["high"] += 1
            elif result.confidence >= 0.6:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1

        assert distribution["high"] == 3
        assert distribution["medium"] == 2
        assert distribution["low"] == 1


class TestGEPAErrorHandling:
    """Test GEPA system error handling and resilience."""

    def setup_method(self):
        """Setup test environment."""
        self.config = Config()
        self.mock_gemini_client = Mock()
        self.enhancer = GEPAClassificationEnhancer(self.config, self.mock_gemini_client)

    def test_fallback_judge_evaluation(self):
        """Test fallback judge evaluation when main judge fails."""
        candidates = [
            ClassificationCandidate("c1", "type_a", "annotation", 0.8),
            ClassificationCandidate("c2", "type_b", "annotation", 0.9),
            ClassificationCandidate("c3", "type_c", "annotation", 0.7),
        ]

        fallback_result = self.enhancer._fallback_judge_evaluation(candidates)

        assert fallback_result is not None
        assert "best_candidate_id" in fallback_result
        assert "judge_reasoning" in fallback_result
        assert "consensus_analysis" in fallback_result

        # Should select highest confidence candidate
        assert fallback_result["best_candidate_id"] == "c2"

    def test_apply_judge_scores_with_missing_evaluations(self):
        """Test applying judge scores when some evaluations are missing."""
        candidates = [
            ClassificationCandidate("c1", "type_a", "annotation", 0.8),
            ClassificationCandidate("c2", "type_b", "annotation", 0.9),
            ClassificationCandidate("c3", "type_c", "annotation", 0.7),
        ]

        # Judge result with only partial evaluations
        judge_result = {
            "candidate_evaluations": [
                {"candidate_id": "c1", "score": 0.85, "strengths": ["good"], "weaknesses": []},
                {"candidate_id": "c3", "score": 0.75, "strengths": [], "weaknesses": ["weak"]},
                # c2 missing
            ]
        }

        self.enhancer._apply_judge_scores(candidates, judge_result)

        assert candidates[0].judge_score == 0.85  # c1
        assert candidates[1].judge_score == 0.5  # c2 (default)
        assert candidates[2].judge_score == 0.75  # c3

        assert candidates[0].strengths == ["good"]
        assert candidates[2].weaknesses == ["weak"]

    def test_consensus_calculation_edge_cases(self):
        """Test consensus calculation edge cases."""
        # All candidates identical
        identical_candidates = [
            ClassificationCandidate("c1", "same_type", "annotation", 0.9),
            ClassificationCandidate("c2", "same_type", "annotation", 0.9),
            ClassificationCandidate("c3", "same_type", "annotation", 0.9),
        ]

        consensus = self.enhancer._calculate_consensus_score(identical_candidates)
        assert consensus == 1.0  # Perfect consensus

        # All candidates different
        different_candidates = [
            ClassificationCandidate("c1", "type_a", "annotation", 0.9),
            ClassificationCandidate("c2", "type_b", "structural", 0.8),
            ClassificationCandidate("c3", "type_c", "mep", 0.7),
        ]

        consensus = self.enhancer._calculate_consensus_score(different_candidates)
        assert consensus < 0.5  # Low consensus


class TestGEPAPromptEvolution:
    """Test GEPA prompt evolution and genetic algorithms."""

    def setup_method(self):
        """Setup test environment."""
        self.config = Config()
        self.mock_gemini_client = Mock()
        self.gepa_system = ComprehensiveGEPASystem(self.config, self.mock_gemini_client)

    def test_prompt_performance_tracking(self):
        """Test prompt performance tracking."""
        # Simulate prompt performance data
        prompt_id = "test_prompt_1"
        performance_data = {"accuracy": 0.85, "processing_time": 25.0, "error_rate": 0.1}

        # Add to performance tracking
        self.gepa_system.prompt_performance[prompt_id] = performance_data

        assert prompt_id in self.gepa_system.prompt_performance
        assert self.gepa_system.prompt_performance[prompt_id]["accuracy"] == 0.85

    def test_optimization_history_management(self):
        """Test optimization history management."""
        # Add optimization record
        optimization_record = {
            "timestamp": 12345,
            "generation": 1,
            "improvement": 0.05,
            "best_prompt": "optimized_prompt",
        }

        self.gepa_system.optimization_history.append(optimization_record)

        assert len(self.gepa_system.optimization_history) == 1
        assert self.gepa_system.optimization_history[0]["generation"] == 1

    def test_evolution_generation_tracking(self):
        """Test evolution generation tracking."""
        assert self.gepa_system.evolution_generation == 0

        # Simulate evolution
        self.gepa_system.evolution_generation += 1
        assert self.gepa_system.evolution_generation == 1


class TestGEPAIntegration:
    """Test GEPA system integration scenarios."""

    def setup_method(self):
        """Setup integration test environment."""
        self.config = Config()

    def test_gepa_enhancer_factory_integration(self):
        """Test GEPA enhancer factory integration."""
        mock_gemini_client = Mock()

        enhancer = create_gepa_classification_enhancer(self.config, mock_gemini_client)

        assert enhancer is not None
        assert enhancer.config is self.config
        assert enhancer.gemini_client is mock_gemini_client
        assert enhancer.enhancement_count == 0

    def test_comprehensive_gepa_factory_integration(self):
        """Test comprehensive GEPA factory integration."""
        mock_gemini_client = Mock()

        gepa_system = create_comprehensive_gepa_system(self.config, mock_gemini_client)

        assert gepa_system is not None
        assert gepa_system.config is self.config
        assert gepa_system.gemini_client is mock_gemini_client
        assert gepa_system.evolution_generation == 0
