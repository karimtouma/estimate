"""
Tests for Language Router System.

Tests validate the language detection and prompt optimization functionality
with real scenarios and edge cases.
"""

import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any

from src.core.config import Config
from src.utils.language_router import (
    LanguageRouter, LanguageDetectionResult
)
from src.services.gemini_client import GeminiClient


class TestLanguageDetectionResult:
    """Test LanguageDetectionResult functionality."""
    
    def test_create_basic_result(self):
        """Test creating basic language detection result."""
        result = LanguageDetectionResult(
            primary_language="spanish",
            secondary_languages=["english"],
            confidence=0.85,
            language_distribution={"spanish": 0.7, "english": 0.3},
            mixed_language=True,
            technical_terminology=["plano", "construcción"],
            region_indicators=["méxico", "cdmx"],
            optimal_prompt_language="spanish"
        )
        
        assert result.primary_language == "spanish"
        assert result.confidence == 0.85
        assert result.mixed_language is True
        assert "plano" in result.technical_terminology
        assert result.optimal_prompt_language == "spanish"


class TestLanguageRouter:
    """Test LanguageRouter with real functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        # Create test config
        self.config = Config()
        self.router = LanguageRouter(self.config)
    
    def test_initialization(self):
        """Test router initialization."""
        assert self.router.config is not None
        assert self.router.language_patterns is not None
        assert "spanish" in self.router.language_patterns
        assert "english" in self.router.language_patterns
        assert "portuguese" in self.router.language_patterns
    
    def test_analyze_text_patterns_spanish(self):
        """Test pattern analysis for Spanish text."""
        spanish_texts = [
            "Plano de construcción del edificio",
            "Especificaciones técnicas de ingeniería",
            "Detalles de cimentación y estructura"
        ]
        
        scores = self.router._analyze_text_patterns(spanish_texts, "spanish")
        assert scores["pattern_score"] > 0.5
        assert scores["keyword_matches"] > 0
    
    def test_analyze_text_patterns_english(self):
        """Test pattern analysis for English text."""
        english_texts = [
            "Construction blueprint and drawings",
            "Technical specifications for building",
            "Foundation details and structure"
        ]
        
        scores = self.router._analyze_text_patterns(english_texts, "english")
        assert scores["pattern_score"] > 0.5
        assert scores["keyword_matches"] > 0
    
    def test_analyze_text_patterns_mixed(self):
        """Test pattern analysis for mixed language text."""
        mixed_texts = [
            "Construction plano with detalles",
            "Building specifications y especificaciones",
            "Foundation cimentación details"
        ]
        
        spanish_scores = self.router._analyze_text_patterns(mixed_texts, "spanish")
        english_scores = self.router._analyze_text_patterns(mixed_texts, "english")
        
        assert spanish_scores["pattern_score"] > 0
        assert english_scores["pattern_score"] > 0
    
    def test_extract_technical_terms_spanish(self):
        """Test technical term extraction for Spanish."""
        text_samples = [
            "Plano arquitectónico con detalles de construcción",
            "Especificaciones de ingeniería estructural",
            "Cimentación y zapatas de concreto"
        ]
        
        terms = self.router._extract_technical_terms(text_samples, "spanish")
        expected_terms = ["plano", "construcción", "ingeniería", "cimentación"]
        
        found_terms = [term for term in expected_terms if any(term in t.lower() for t in terms)]
        assert len(found_terms) > 0
    
    def test_extract_technical_terms_english(self):
        """Test technical term extraction for English."""
        text_samples = [
            "Architectural blueprint with construction details",
            "Structural engineering specifications", 
            "Foundation and concrete footings"
        ]
        
        terms = self.router._extract_technical_terms(text_samples, "english")
        expected_terms = ["blueprint", "construction", "engineering", "foundation"]
        
        found_terms = [term for term in expected_terms if any(term in t.lower() for t in terms)]
        assert len(found_terms) > 0
    
    def test_detect_region_indicators_spanish(self):
        """Test region indicator detection for Spanish."""
        text_samples = [
            "Proyecto en Ciudad de México, CDMX",
            "Construcción en Guadalajara, Jalisco",
            "Obra en Monterrey, Nuevo León"
        ]
        
        indicators = self.router._detect_region_indicators(text_samples, "spanish")
        expected_indicators = ["méxico", "guadalajara", "monterrey"]
        
        found_indicators = [ind for ind in expected_indicators if any(ind in i.lower() for i in indicators)]
        assert len(found_indicators) > 0
    
    def test_detect_region_indicators_english(self):
        """Test region indicator detection for English."""
        text_samples = [
            "Project in New York, NY",
            "Construction in Los Angeles, CA",
            "Building in Chicago, IL"
        ]
        
        indicators = self.router._detect_region_indicators(text_samples, "english")
        expected_indicators = ["new york", "los angeles", "chicago"]
        
        found_indicators = [ind for ind in expected_indicators if any(ind in i.lower() for i in indicators)]
        assert len(found_indicators) > 0
    
    def test_calculate_language_confidence(self):
        """Test language confidence calculation."""
        # Strong Spanish indicators
        spanish_scores = {
            "pattern_score": 0.9,
            "keyword_matches": 5,
            "technical_terms": ["plano", "construcción"],
            "region_indicators": ["méxico"]
        }
        
        confidence = self.router._calculate_language_confidence("spanish", spanish_scores)
        assert confidence > 0.7
        
        # Weak indicators
        weak_scores = {
            "pattern_score": 0.2,
            "keyword_matches": 1,
            "technical_terms": [],
            "region_indicators": []
        }
        
        weak_confidence = self.router._calculate_language_confidence("spanish", weak_scores)
        assert weak_confidence < 0.5
    
    def test_optimize_prompt_for_language_spanish(self):
        """Test prompt optimization for Spanish."""
        base_prompt = "Analyze this technical document and provide insights"
        
        detection_result = LanguageDetectionResult(
            primary_language="spanish",
            secondary_languages=[],
            confidence=0.9,
            language_distribution={"spanish": 0.9, "english": 0.1},
            mixed_language=False,
            technical_terminology=["plano", "construcción"],
            region_indicators=["méxico"],
            optimal_prompt_language="spanish"
        )
        
        optimized = self.router.optimize_prompt_for_language(base_prompt, detection_result)
        
        assert "spanish" in optimized.lower() or "español" in optimized.lower()
        assert len(optimized) > len(base_prompt)  # Should add language-specific instructions
    
    def test_optimize_prompt_for_language_english(self):
        """Test prompt optimization for English."""
        base_prompt = "Analyze this technical document and provide insights"
        
        detection_result = LanguageDetectionResult(
            primary_language="english",
            secondary_languages=[],
            confidence=0.95,
            language_distribution={"english": 0.95, "spanish": 0.05},
            mixed_language=False,
            technical_terminology=["blueprint", "construction"],
            region_indicators=["usa"],
            optimal_prompt_language="english"
        )
        
        optimized = self.router.optimize_prompt_for_language(base_prompt, detection_result)
        
        assert "english" in optimized.lower()
        assert len(optimized) > len(base_prompt)
    
    def test_optimize_prompt_for_mixed_language(self):
        """Test prompt optimization for mixed language documents."""
        base_prompt = "Analyze this technical document"
        
        detection_result = LanguageDetectionResult(
            primary_language="spanish",
            secondary_languages=["english"],
            confidence=0.6,
            language_distribution={"spanish": 0.6, "english": 0.4},
            mixed_language=True,
            technical_terminology=["plano", "construction"],
            region_indicators=["méxico"],
            optimal_prompt_language="spanish"
        )
        
        optimized = self.router.optimize_prompt_for_language(base_prompt, detection_result)
        
        assert "mixed" in optimized.lower() or "bilingual" in optimized.lower()
        assert len(optimized) > len(base_prompt)


class TestLanguageRouterOptimalLanguage:
    """Test optimal language determination within LanguageRouter."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = Config()
        self.router = LanguageRouter(self.config)
    
    def test_language_distribution_analysis(self):
        """Test language distribution analysis."""
        # Test Spanish dominant
        spanish_samples = [
            "Plano arquitectónico de construcción",
            "Especificaciones técnicas del proyecto",
            "Detalles de cimentación"
        ]
        
        spanish_scores = self.router._analyze_text_patterns(spanish_samples, "spanish")
        english_scores = self.router._analyze_text_patterns(spanish_samples, "english")
        
        assert spanish_scores["pattern_score"] > english_scores["pattern_score"]
    
    def test_mixed_language_detection(self):
        """Test mixed language detection capability."""
        mixed_samples = [
            "Construction plano arquitectónico",
            "Building especificaciones técnicas",
            "Foundation detalles de cimentación"
        ]
        
        spanish_scores = self.router._analyze_text_patterns(mixed_samples, "spanish")
        english_scores = self.router._analyze_text_patterns(mixed_samples, "english")
        
        # Both should have some score for mixed content
        assert spanish_scores["pattern_score"] > 0.2
        assert english_scores["pattern_score"] > 0.2


class TestLanguageRouterIntegration:
    """Test Language Router with realistic scenarios."""
    
    def setup_method(self):
        """Setup for integration tests."""
        self.config = Config()
        self.router = LanguageRouter(self.config)
    
    def test_spanish_construction_document(self):
        """Test with Spanish construction document samples."""
        text_samples = [
            "PLANO ARQUITECTÓNICO - CASA HABITACIÓN",
            "Especificaciones técnicas de construcción",
            "Detalles de cimentación y estructura de concreto",
            "Instalaciones eléctricas y sanitarias",
            "Acabados y materiales especificados"
        ]
        
        # Test pattern analysis
        spanish_scores = self.router._analyze_text_patterns(text_samples, "spanish")
        english_scores = self.router._analyze_text_patterns(text_samples, "english")
        
        assert spanish_scores["pattern_score"] > english_scores["pattern_score"]
        assert spanish_scores["keyword_matches"] > 0
    
    def test_english_construction_document(self):
        """Test with English construction document samples."""
        text_samples = [
            "ARCHITECTURAL BLUEPRINT - RESIDENTIAL BUILDING",
            "Technical construction specifications",
            "Foundation and concrete structure details", 
            "Electrical and plumbing installations",
            "Finishes and specified materials"
        ]
        
        # Test pattern analysis
        spanish_scores = self.router._analyze_text_patterns(text_samples, "spanish")
        english_scores = self.router._analyze_text_patterns(text_samples, "english")
        
        assert english_scores["pattern_score"] > spanish_scores["pattern_score"]
        assert english_scores["keyword_matches"] > 0
    
    def test_mixed_language_document(self):
        """Test with mixed language document samples."""
        text_samples = [
            "PLANO ARQUITECTÓNICO - Architectural Plan",
            "Construction especificaciones técnicas",
            "Foundation cimentación details",
            "Electrical instalaciones eléctricas",
            "Materials materiales specified"
        ]
        
        # Both languages should have some score
        spanish_scores = self.router._analyze_text_patterns(text_samples, "spanish")
        english_scores = self.router._analyze_text_patterns(text_samples, "english")
        
        assert spanish_scores["pattern_score"] > 0.2
        assert english_scores["pattern_score"] > 0.2
        assert spanish_scores["keyword_matches"] > 0
        assert english_scores["keyword_matches"] > 0


class TestLanguageRouterEdgeCases:
    """Test Language Router edge cases and error handling."""
    
    def setup_method(self):
        """Setup for edge case tests."""
        self.config = Config()
        self.router = LanguageRouter(self.config)
    
    def test_empty_text_samples(self):
        """Test with empty text samples."""
        empty_samples = []
        
        scores = self.router._analyze_text_patterns(empty_samples, "spanish")
        assert scores["pattern_score"] == 0.0
        assert scores["keyword_matches"] == 0
    
    def test_very_short_text_samples(self):
        """Test with very short text samples."""
        short_samples = ["A", "B", "C"]
        
        scores = self.router._analyze_text_patterns(short_samples, "spanish")
        assert scores["pattern_score"] >= 0.0  # Should not crash
    
    def test_non_technical_text(self):
        """Test with non-technical text."""
        non_technical = [
            "The quick brown fox jumps",
            "Lorem ipsum dolor sit amet",
            "Random text without technical terms"
        ]
        
        spanish_scores = self.router._analyze_text_patterns(non_technical, "spanish")
        english_scores = self.router._analyze_text_patterns(non_technical, "english")
        
        # Should have low scores but not crash
        assert spanish_scores["pattern_score"] >= 0.0
        assert english_scores["pattern_score"] >= 0.0
    
    def test_special_characters_and_numbers(self):
        """Test with special characters and numbers."""
        special_samples = [
            "12'-6\" x 8'-0\" opening",
            "Ø25mm reinforcement bars", 
            "45° angle connection",
            "R-19 insulation specified",
            "2x8 @ 16\" O.C. framing"
        ]
        
        scores = self.router._analyze_text_patterns(special_samples, "english")
        assert scores["pattern_score"] >= 0.0  # Should handle special chars
    
    def test_determine_optimal_language_edge_cases(self):
        """Test optimal language determination edge cases."""
        # Empty distribution
        result = determine_optimal_language({}, [])
        assert result == "english"  # Default fallback
        
        # All equal distribution
        result = determine_optimal_language({
            "spanish": 0.33,
            "english": 0.33, 
            "portuguese": 0.34
        }, [])
        assert result in ["spanish", "english", "portuguese"]
        
        # Single language
        result = determine_optimal_language({"spanish": 1.0}, ["méxico"])
        assert result == "spanish"


class TestLanguageRouterPerformance:
    """Test Language Router performance and efficiency."""
    
    def setup_method(self):
        """Setup for performance tests."""
        self.config = Config()
        self.router = LanguageRouter(self.config)
    
    def test_large_text_samples_performance(self):
        """Test performance with large number of text samples."""
        import time
        
        # Generate large set of text samples
        large_samples = []
        for i in range(100):
            large_samples.extend([
                f"Plano arquitectónico número {i}",
                f"Especificación técnica {i}",
                f"Detalle constructivo {i}"
            ])
        
        start_time = time.time()
        scores = self.router._analyze_text_patterns(large_samples, "spanish")
        end_time = time.time()
        
        processing_time = end_time - start_time
        assert processing_time < 5.0  # Should process quickly
        assert scores["pattern_score"] > 0.5  # Should detect Spanish
    
    def test_pattern_caching(self):
        """Test that pattern analysis results are consistent."""
        text_samples = [
            "Plano de construcción residencial",
            "Especificaciones técnicas de proyecto"
        ]
        
        # Run analysis twice
        scores1 = self.router._analyze_text_patterns(text_samples, "spanish")
        scores2 = self.router._analyze_text_patterns(text_samples, "spanish")
        
        # Results should be identical (deterministic)
        assert scores1["pattern_score"] == scores2["pattern_score"]
        assert scores1["keyword_matches"] == scores2["keyword_matches"]


class TestLanguageRouterRealWorldScenarios:
    """Test Language Router with real-world document scenarios."""
    
    def setup_method(self):
        """Setup for real-world tests."""
        self.config = Config()
        self.router = LanguageRouter(self.config)
    
    def test_mexican_construction_project(self):
        """Test with Mexican construction project text."""
        mexican_samples = [
            "PROYECTO: CASA HABITACIÓN EN CDMX",
            "PLANO ARQUITECTÓNICO - PLANTA BAJA",
            "Especificaciones técnicas de construcción",
            "Detalles de cimentación con zapatas corridas",
            "Instalaciones hidráulicas y sanitarias",
            "Acabados: azulejo, pintura, cancelería de aluminio"
        ]
        
        scores = self.router._analyze_text_patterns(mexican_samples, "spanish")
        
        assert scores["pattern_score"] > 0.7
        assert scores["keyword_matches"] >= 3
        
        # Should detect Mexican technical terms
        terms = self.router._extract_technical_terms(mexican_samples, "spanish")
        expected_mexican_terms = ["plano", "construcción", "cimentación", "instalaciones"]
        found_terms = [term for term in expected_mexican_terms if any(term in t.lower() for t in terms)]
        assert len(found_terms) >= 2
    
    def test_us_construction_project(self):
        """Test with US construction project text."""
        us_samples = [
            "PROJECT: COMMERCIAL BUILDING IN CALIFORNIA",
            "ARCHITECTURAL BLUEPRINT - GROUND FLOOR",
            "Technical construction specifications",
            "Foundation details with continuous footings",
            "MEP installations and systems",
            "Finishes: tile, paint, aluminum storefront"
        ]
        
        scores = self.router._analyze_text_patterns(us_samples, "english")
        
        assert scores["pattern_score"] > 0.7
        assert scores["keyword_matches"] >= 3
        
        # Should detect US technical terms
        terms = self.router._extract_technical_terms(us_samples, "english")
        expected_us_terms = ["blueprint", "construction", "foundation", "mep"]
        found_terms = [term for term in expected_us_terms if any(term in t.lower() for t in terms)]
        assert len(found_terms) >= 2
    
    def test_international_project_mixed(self):
        """Test with international project (mixed language)."""
        international_samples = [
            "PROYECTO INTERNACIONAL - International Project",
            "Planos arquitectónicos / Architectural Plans",
            "Specifications especificaciones técnicas",
            "Foundation cimentación requirements",
            "MEP sistemas installations"
        ]
        
        spanish_scores = self.router._analyze_text_patterns(international_samples, "spanish")
        english_scores = self.router._analyze_text_patterns(international_samples, "english")
        
        # Both should have reasonable scores
        assert spanish_scores["pattern_score"] > 0.3
        assert english_scores["pattern_score"] > 0.3
        
        # Both should find some technical terms
        spanish_terms = self.router._extract_technical_terms(international_samples, "spanish")
        english_terms = self.router._extract_technical_terms(international_samples, "english")
        
        assert len(spanish_terms) > 0
        assert len(english_terms) > 0
