"""
DSPy-based Hallucination Detection System.

This module uses DSPy's typed signatures and chain-of-thought reasoning
to detect and fix hallucinations in AI-generated content.
"""

import dspy
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Initialize DSPy with Gemini
def initialize_dspy_gemini(api_key: str, model: str = "gemini-2.0-flash-exp"):
    """Initialize DSPy with Google Gemini backend."""
    try:
        # Configure DSPy to use Gemini
        lm = dspy.Google(
            model=model,
            api_key=api_key,
            temperature=0.1,  # Low temperature for consistent detection
            max_output_tokens=1000
        )
        dspy.settings.configure(lm=lm)
        logger.info(f"✅ DSPy initialized with {model}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize DSPy: {e}")
        return False


# Typed Signatures for Hallucination Detection
class DetectRepetitiveHallucination(dspy.Signature):
    """Detect if text contains repetitive patterns indicating hallucination."""
    
    text: str = dspy.InputField(desc="Text to analyze for repetitive patterns")
    has_repetition: bool = dspy.OutputField(desc="True if repetitive hallucination detected")
    pattern_found: str = dspy.OutputField(desc="The repetitive pattern if found, empty string otherwise")
    confidence: float = dspy.OutputField(desc="Confidence score between 0 and 1")


class CleanHallucinatedText(dspy.Signature):
    """Clean text that contains hallucinations, preserving valid content."""
    
    text: str = dspy.InputField(desc="Text containing hallucinations")
    pattern: str = dspy.InputField(desc="The hallucination pattern to remove")
    cleaned_text: str = dspy.OutputField(desc="Cleaned text with hallucinations removed")
    items_preserved: int = dspy.OutputField(desc="Number of valid items preserved")


class ValidateDataExtraction(dspy.Signature):
    """Validate data extraction results for quality and hallucinations."""
    
    data: str = dspy.InputField(desc="JSON data extraction results")
    field_name: str = dspy.InputField(desc="Name of the field being validated")
    max_items: int = dspy.InputField(desc="Maximum allowed items for this field")
    is_valid: bool = dspy.OutputField(desc="True if data is valid and hallucination-free")
    issues: str = dspy.OutputField(desc="Description of issues found, empty if valid")
    cleaned_data: str = dspy.OutputField(desc="Cleaned version of the data if needed")


# DSPy Modules for Hallucination Detection
class HallucinationDetector(dspy.Module):
    """Main hallucination detection module using DSPy."""
    
    def __init__(self):
        super().__init__()
        self.detect_repetition = dspy.ChainOfThought(DetectRepetitiveHallucination)
        self.clean_text = dspy.ChainOfThought(CleanHallucinatedText)
        self.validate_data = dspy.ChainOfThought(ValidateDataExtraction)
    
    def detect_and_clean(self, text: str) -> Tuple[bool, str]:
        """
        Detect hallucinations and return cleaned text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (has_hallucination, cleaned_text)
        """
        try:
            # First, detect if there's a repetitive pattern
            detection = self.detect_repetition(text=text)
            
            if detection.has_repetition and detection.pattern_found:
                logger.warning(f"🔴 Hallucination detected: {detection.pattern_found[:50]}...")
                
                # Clean the text
                cleaning = self.clean_text(
                    text=text,
                    pattern=detection.pattern_found
                )
                
                return True, cleaning.cleaned_text
            
            return False, text
            
        except Exception as e:
            logger.error(f"Error in hallucination detection: {e}")
            # Fallback to simple regex-based detection
            return self._fallback_detection(text)
    
    def _fallback_detection(self, text: str) -> Tuple[bool, str]:
        """Fallback regex-based detection if DSPy fails."""
        # Check for repetitive patterns like "-item-item-item"
        pattern = r'(-\w+){10,}'  # Word repeated 10+ times with dash
        match = re.search(pattern, text)
        
        if match:
            logger.warning("🔴 Fallback: Repetitive pattern detected")
            # Remove the repetitive part
            cleaned = re.sub(pattern, '', text)
            return True, cleaned.strip()
        
        # Check for extremely long strings (>1000 chars without spaces)
        if len(text) > 1000 and ' ' not in text[-500:]:
            logger.warning("🔴 Fallback: Abnormally long text detected")
            return True, text[:500] + "..."
        
        return False, text
    
    def validate_field(self, field_name: str, data: Any, max_items: int = 50) -> Dict[str, Any]:
        """
        Validate a specific field for hallucinations.
        
        Args:
            field_name: Name of the field
            data: The data to validate
            max_items: Maximum allowed items
            
        Returns:
            Dict with 'valid', 'data', and 'issues' keys
        """
        try:
            # Convert data to string for validation
            import json
            data_str = json.dumps(data) if not isinstance(data, str) else data
            
            validation = self.validate_data(
                data=data_str,
                field_name=field_name,
                max_items=max_items
            )
            
            if not validation.is_valid:
                logger.warning(f"🔴 Field '{field_name}' validation failed: {validation.issues}")
                
                # Parse cleaned data back
                try:
                    cleaned = json.loads(validation.cleaned_data)
                except:
                    cleaned = validation.cleaned_data
                
                return {
                    'valid': False,
                    'data': cleaned,
                    'issues': validation.issues
                }
            
            return {
                'valid': True,
                'data': data,
                'issues': None
            }
            
        except Exception as e:
            logger.error(f"Error validating field {field_name}: {e}")
            return {
                'valid': False,
                'data': data,
                'issues': str(e)
            }


# Pydantic Models for Typed Validation
class ValidationResult(BaseModel):
    """Result of hallucination validation."""
    
    has_hallucinations: bool = Field(description="Whether hallucinations were detected")
    cleaned_data: Dict[str, Any] = Field(description="Cleaned data after removing hallucinations")
    issues_found: List[str] = Field(default_factory=list, description="List of issues found")
    confidence: float = Field(default=1.0, description="Confidence in the validation")


def validate_data_extraction_with_dspy(data: Dict[str, Any]) -> ValidationResult:
    """
    Validate data extraction results using DSPy.
    
    Args:
        data: Data extraction dictionary
        
    Returns:
        ValidationResult with cleaned data and issues
    """
    detector = HallucinationDetector()
    cleaned_data = data.copy()
    issues = []
    has_hallucinations = False
    
    # Fields to validate with their max items
    field_limits = {
        'entities': 50,
        'dates': 30,
        'numbers': 40,
        'references': 30,
        'technical_terms': 50,
        'specifications': 40
    }
    
    for field, max_items in field_limits.items():
        if field in data and data[field]:
            # Check each item in the field
            if isinstance(data[field], list):
                cleaned_items = []
                for item in data[field]:
                    if isinstance(item, str):
                        had_hallucination, cleaned_item = detector.detect_and_clean(item)
                        if had_hallucination:
                            has_hallucinations = True
                            issues.append(f"Hallucination in {field}: {item[:50]}...")
                        cleaned_items.append(cleaned_item)
                    else:
                        cleaned_items.append(item)
                
                # Limit number of items
                if len(cleaned_items) > max_items:
                    cleaned_items = cleaned_items[:max_items]
                    issues.append(f"Field {field} truncated to {max_items} items")
                
                cleaned_data[field] = cleaned_items
            
            elif isinstance(data[field], str):
                had_hallucination, cleaned_text = detector.detect_and_clean(data[field])
                if had_hallucination:
                    has_hallucinations = True
                    issues.append(f"Hallucination in {field}")
                cleaned_data[field] = cleaned_text
    
    return ValidationResult(
        has_hallucinations=has_hallucinations,
        cleaned_data=cleaned_data,
        issues_found=issues,
        confidence=0.95 if not has_hallucinations else 0.7
    )


def validate_page_classification(classification: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a single page classification for hallucinations.
    
    Args:
        classification: Page classification dictionary
        
    Returns:
        Cleaned classification dictionary
    """
    detector = HallucinationDetector()
    cleaned = classification.copy()
    
    # Fields that might contain hallucinations
    text_fields = ['content_summary', 'primary_category']
    list_fields = ['secondary_categories', 'key_elements']
    
    for field in text_fields:
        if field in cleaned and isinstance(cleaned[field], str):
            had_hallucination, cleaned_text = detector.detect_and_clean(cleaned[field])
            if had_hallucination:
                logger.warning(f"🔴 Hallucination in page {cleaned.get('page_number', '?')} {field}")
            cleaned[field] = cleaned_text
    
    for field in list_fields:
        if field in cleaned and isinstance(cleaned[field], list):
            cleaned_items = []
            for item in cleaned[field]:
                if isinstance(item, str):
                    had_hallucination, cleaned_item = detector.detect_and_clean(item)
                    if not had_hallucination and len(cleaned_item) > 0:
                        cleaned_items.append(cleaned_item)
            cleaned[field] = cleaned_items[:10]  # Limit to 10 items
    
    return cleaned


# Export main functions
__all__ = [
    'initialize_dspy_gemini',
    'HallucinationDetector',
    'ValidationResult',
    'validate_data_extraction_with_dspy',
    'validate_page_classification'
]
