"""
Hallucination detection and cleanup utilities.

This module provides functions to detect and clean up hallucinations
in AI-generated content, particularly repetitive patterns.
"""

import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def detect_repetitive_pattern(text: str, min_repetitions: int = 5) -> bool:
    """
    Detect if a string contains repetitive patterns (hallucination indicator).
    
    Args:
        text: Text to analyze
        min_repetitions: Minimum number of repetitions to consider as hallucination
        
    Returns:
        True if repetitive pattern detected
    """
    # Check for repeated words or phrases
    # Pattern: word repeated 5+ times with separators
    patterns = [
        r'(\b\w+\b)(?:\W+\1){' + str(min_repetitions - 1) + r',}',  # Repeated words
        r'(.{2,20})(?:\1){' + str(min_repetitions - 1) + r',}',  # Repeated sequences
        r'(-item){' + str(min_repetitions) + r',}',  # Specific pattern found in hallucination
    ]
    
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            logger.warning(f"Detected repetitive pattern in text: {text[:100]}...")
            return True
    
    # Check if text is abnormally long (potential runaway generation)
    if len(text) > 1000:
        logger.warning(f"Detected abnormally long text: {len(text)} characters")
        return True
    
    return False


def clean_hallucinated_text(text: str) -> str:
    """
    Clean up hallucinated text by removing repetitive patterns.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    # Remove repetitive patterns like "-item-item-item..."
    cleaned = re.sub(r'(-item){3,}.*', '', text)
    
    # Remove other repetitive sequences
    cleaned = re.sub(r'(\b\w+\b)(?:\W+\1){5,}', r'\1', cleaned)
    
    # Truncate if still too long
    if len(cleaned) > 200:
        cleaned = cleaned[:200] + "..."
        
    return cleaned.strip()


def validate_data_extraction(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean data extraction results.
    
    Args:
        data: Data extraction dictionary
        
    Returns:
        Cleaned data dictionary
    """
    if not isinstance(data, dict):
        return data
    
    # Fields to check for hallucinations
    list_fields = ['entities', 'dates', 'numbers', 'references', 'key_terms']
    
    for field in list_fields:
        if field in data and isinstance(data[field], list):
            cleaned_items = []
            
            for item in data[field]:
                if isinstance(item, str):
                    # Check for hallucination
                    if detect_repetitive_pattern(item):
                        # Clean the item
                        cleaned_item = clean_hallucinated_text(item)
                        if cleaned_item and len(cleaned_item) > 3:
                            cleaned_items.append(cleaned_item)
                        else:
                            logger.warning(f"Removed hallucinated item from {field}: {item[:50]}...")
                    else:
                        # Keep valid items under reasonable length
                        if len(item) < 500:
                            cleaned_items.append(item)
                        else:
                            # Truncate overly long items
                            cleaned_items.append(item[:200] + "...")
            
            # Update with cleaned list
            data[field] = cleaned_items
            
            # Enforce maximum items as additional safety
            max_items = {
                'entities': 50,
                'dates': 30,
                'numbers': 40,
                'references': 25,
                'key_terms': 30
            }
            
            if field in max_items and len(data[field]) > max_items[field]:
                data[field] = data[field][:max_items[field]]
                logger.info(f"Truncated {field} to {max_items[field]} items")
    
    return data


def validate_comprehensive_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean comprehensive analysis results.
    
    Args:
        result: Complete analysis result dictionary
        
    Returns:
        Cleaned result dictionary
    """
    # Check data_extraction section specifically
    if 'data_extraction' in result:
        result['data_extraction'] = validate_data_extraction(result['data_extraction'])
    
    # Check for hallucinations in other text fields
    text_fields = ['summary', 'content_summary', 'description']
    
    def check_dict_recursively(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in text_fields and isinstance(value, str):
                    if detect_repetitive_pattern(value):
                        obj[key] = clean_hallucinated_text(value)
                elif isinstance(value, (dict, list)):
                    check_dict_recursively(value)
        elif isinstance(obj, list):
            for item in obj:
                check_dict_recursively(item)
    
    check_dict_recursively(result)
    
    return result
