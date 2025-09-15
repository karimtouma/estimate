"""
Tests for Core PDF Processor.

Tests validate the main processor functionality, configuration,
and integration with services.
"""

import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch

from src.core.processor import PDFProcessor, ProcessorError, ValidationError
from src.core.config import Config
from src.models.schemas import AnalysisType, ComprehensiveAnalysisResult


class TestProcessorError:
    """Test ProcessorError exception."""
    
    def test_processor_error_creation(self):
        """Test creating ProcessorError."""
        error = ProcessorError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)


class TestValidationError:
    """Test ValidationError exception."""
    
    def test_validation_error_creation(self):
        """Test creating ValidationError."""
        error = ValidationError("Validation failed")
        assert str(error) == "Validation failed"
        assert isinstance(error, ProcessorError)
        assert isinstance(error, Exception)


class TestPDFProcessorInitialization:
    """Test PDFProcessor initialization and configuration."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Config()
    
    def teardown_method(self):
        """Clean up test files."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_processor_initialization_with_config(self):
        """Test processor initialization with provided config."""
        processor = PDFProcessor(self.config)
        
        assert processor.config is not None
        assert processor.gemini_client is not None
        assert processor.file_manager is not None
        assert processor.conversation_history == []
        assert processor._current_file_uri is None
    
    def test_processor_initialization_without_config(self):
        """Test processor initialization without config (should load default)."""
        with patch('src.core.processor.get_config') as mock_get_config:
            mock_get_config.return_value = self.config
            processor = PDFProcessor(None)
            
            assert processor.config is not None
            mock_get_config.assert_called_once()
    
    def test_processor_context_manager(self):
        """Test processor as context manager."""
        with PDFProcessor(self.config) as processor:
            assert processor is not None
            assert processor.config is not None
        
        # After context, cleanup should have been called
        # (We can't easily test this without mocking, but structure is verified)


class TestPDFProcessorConfiguration:
    """Test PDFProcessor configuration and setup."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = Config()
        self.processor = PDFProcessor(self.config)
    
    def test_get_analysis_config(self):
        """Test analysis configuration retrieval."""
        # Test with discovery result
        discovery_result = {
            "document_type": "Construction Documents",
            "industry_domain": "AEC"
        }
        
        prompt, schema = self.processor._get_analysis_config(
            AnalysisType.GENERAL, 
            discovery_result
        )
        
        assert prompt is not None
        assert len(prompt) > 0
        assert schema is not None
    
    def test_get_analysis_config_without_discovery(self):
        """Test analysis configuration without discovery result."""
        prompt, schema = self.processor._get_analysis_config(AnalysisType.GENERAL, None)
        
        assert prompt is not None
        assert len(prompt) > 0
        assert schema is not None
    
    def test_get_schema_for_different_types(self):
        """Test schema retrieval for different analysis types."""
        analysis_types = [
            AnalysisType.GENERAL,
            AnalysisType.SECTIONS,
            AnalysisType.DATA_EXTRACTION
        ]
        
        for analysis_type in analysis_types:
            schema = self.processor._get_schema(analysis_type)
            assert schema is not None
            assert "type" in schema
            assert "properties" in schema


class TestPDFProcessorFileHandling:
    """Test PDFProcessor file handling capabilities."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = Config()
        self.processor = PDFProcessor(self.config)
    
    def test_validate_pdf_path_valid(self):
        """Test PDF path validation with valid path."""
        # Create a temporary PDF file
        temp_pdf = Path(tempfile.mkdtemp()) / "test.pdf"
        temp_pdf.touch()
        
        try:
            # Should not raise exception
            self.processor._validate_pdf_path(temp_pdf)
        except Exception as e:
            pytest.fail(f"Validation should not fail for valid PDF: {e}")
        finally:
            temp_pdf.unlink()
            temp_pdf.parent.rmdir()
    
    def test_validate_pdf_path_nonexistent(self):
        """Test PDF path validation with non-existent file."""
        nonexistent_path = Path("/nonexistent/file.pdf")
        
        with pytest.raises(ValidationError):
            self.processor._validate_pdf_path(nonexistent_path)
    
    def test_validate_pdf_path_wrong_extension(self):
        """Test PDF path validation with wrong file extension."""
        temp_txt = Path(tempfile.mkdtemp()) / "test.txt"
        temp_txt.touch()
        
        try:
            with pytest.raises(ValidationError):
                self.processor._validate_pdf_path(temp_txt)
        finally:
            temp_txt.unlink()
            temp_txt.parent.rmdir()


class TestPDFProcessorLanguageHandling:
    """Test PDFProcessor language detection and optimization."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = Config()
        self.processor = PDFProcessor(self.config)
    
    def test_create_language_instruction_auto(self):
        """Test language instruction creation with auto detection."""
        # Mock language router if available
        if hasattr(self.processor, 'language_router') and self.processor.language_router:
            # Test with detected language
            detected_language = Mock()
            detected_language.primary_language = "spanish"
            detected_language.confidence = 0.85
            
            self.processor.detected_language = detected_language
            
            instruction = self.processor._create_language_instruction()
            assert instruction is not None
            assert len(instruction) > 0
    
    def test_create_language_instruction_force_english(self):
        """Test language instruction with forced English."""
        # Temporarily modify config
        original_output_lang = getattr(self.processor.config.api, 'output_language', 'auto')
        original_force_lang = getattr(self.processor.config.api, 'force_language_output', False)
        
        self.processor.config.api.output_language = 'english'
        self.processor.config.api.force_language_output = True
        
        instruction = self.processor._create_language_instruction()
        
        assert "English" in instruction or "english" in instruction
        
        # Restore original config
        self.processor.config.api.output_language = original_output_lang
        self.processor.config.api.force_language_output = original_force_lang


class TestPDFProcessorStatistics:
    """Test PDFProcessor statistics and tracking."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = Config()
        self.processor = PDFProcessor(self.config)
    
    def test_api_statistics_initialization(self):
        """Test API statistics initialization."""
        assert hasattr(self.processor, 'gemini_client')
        # Statistics are tracked in gemini_client
        assert self.processor.gemini_client is not None
    
    def test_conversation_history_management(self):
        """Test conversation history management."""
        assert self.processor.conversation_history == []
        
        # Add mock conversation entry
        entry = {
            "timestamp": 12345,
            "type": "question",
            "content": "Test question"
        }
        
        self.processor.conversation_history.append(entry)
        assert len(self.processor.conversation_history) == 1
        assert self.processor.conversation_history[0] == entry


class TestPDFProcessorErrorHandling:
    """Test PDFProcessor error handling and resilience."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = Config()
    
    def test_initialization_with_invalid_config(self):
        """Test processor initialization with invalid config."""
        # Test with None config and mocked get_config failure
        with patch('src.core.processor.get_config') as mock_get_config:
            mock_get_config.side_effect = Exception("Config load failed")
            
            with pytest.raises(Exception):
                PDFProcessor(None)
    
    def test_processor_cleanup_on_exception(self):
        """Test processor cleanup when exception occurs."""
        processor = PDFProcessor(self.config)
        
        # Test context manager cleanup on exception
        try:
            with processor:
                raise Exception("Test exception")
        except Exception:
            pass  # Expected
        
        # Processor should have cleaned up (structure verified)


class TestPDFProcessorIntegration:
    """Test PDFProcessor integration with other components."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = Config()
        self.processor = PDFProcessor(self.config)
    
    def test_processor_has_required_services(self):
        """Test that processor has all required services."""
        assert hasattr(self.processor, 'gemini_client')
        assert hasattr(self.processor, 'file_manager')
        assert hasattr(self.processor, 'config')
        
        assert self.processor.gemini_client is not None
        assert self.processor.file_manager is not None
        assert self.processor.config is not None
    
    def test_processor_language_router_integration(self):
        """Test processor integration with language router."""
        # Language router may or may not be available
        if hasattr(self.processor, 'language_router'):
            if self.processor.language_router is not None:
                assert hasattr(self.processor.language_router, 'config')
                assert hasattr(self.processor.language_router, 'gemini_client')
    
    def test_processor_state_management(self):
        """Test processor state management."""
        assert self.processor._current_file_uri is None
        assert self.processor.conversation_history == []
        assert self.processor.detected_language is None
        
        # Test state modification
        self.processor._current_file_uri = "test_uri"
        assert self.processor._current_file_uri == "test_uri"


class TestPDFProcessorMethods:
    """Test specific PDFProcessor methods."""
    
    def setup_method(self):
        """Setup test environment.""" 
        self.config = Config()
        self.processor = PDFProcessor(self.config)
    
    def test_create_adaptive_prompt_structure(self):
        """Test adaptive prompt creation structure."""
        discovery_result = {
            "document_type": "Construction Documents",
            "industry_domain": "AEC (Architecture, Engineering, Construction)"
        }
        
        # Test that method exists and returns reasonable structure
        analysis_type = AnalysisType.GENERAL
        
        # This tests the method structure without requiring actual API calls
        prompt = self.processor._create_adaptive_prompt(
            analysis_type, 
            discovery_result, 
            "English language instruction."
        )
        
        assert prompt is not None
        assert len(prompt) > 0
        assert "Construction Documents" in prompt
        assert "AEC" in prompt
    
    def test_cleanup_async_clients_method_exists(self):
        """Test that async cleanup method exists."""
        assert hasattr(self.processor, '_cleanup_async_clients')
        
        # Should not raise exception when called
        try:
            self.processor._cleanup_async_clients()
        except Exception as e:
            pytest.fail(f"Cleanup method should not fail: {e}")


# Mock tests that don't require API calls
class TestPDFProcessorMocked:
    """Test PDFProcessor with mocked dependencies."""
    
    def setup_method(self):
        """Setup mocked test environment."""
        self.config = Config()
        
        # Mock gemini client to avoid API calls
        with patch('src.core.processor.GeminiClient') as mock_gemini:
            mock_gemini.return_value = Mock()
            with patch('src.core.processor.FileManager') as mock_file_manager:
                mock_file_manager.return_value = Mock()
                self.processor = PDFProcessor(self.config)
    
    def test_mocked_processor_initialization(self):
        """Test processor initialization with mocked services."""
        assert self.processor is not None
        assert self.processor.config is not None
        assert self.processor.gemini_client is not None
        assert self.processor.file_manager is not None
