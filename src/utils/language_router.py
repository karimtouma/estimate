"""
Language Router for Automatic Language Detection and Prompt Optimization.

This module implements intelligent language detection and prompt adaptation
based on the detected language(s) in the document content.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter

try:
    from ..services.gemini_client import GeminiClient
    from ..core.config import Config
    from ..utils.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class LanguageDetectionResult:
    """Result of language detection analysis."""
    
    primary_language: str
    secondary_languages: List[str]
    confidence: float
    language_distribution: Dict[str, float]
    mixed_language: bool
    technical_terminology: List[str]
    region_indicators: List[str]
    
    def get_optimal_prompt_language(self) -> str:
        """Get the optimal language for prompts based on detection."""
        # If high confidence in Spanish, use Spanish
        if self.primary_language == 'spanish' and self.confidence > 0.7:
            return 'spanish'
        # If mixed with significant Spanish content, use Spanish
        elif self.mixed_language and self.language_distribution.get('spanish', 0) > 0.3:
            return 'spanish'
        # Default to English for international compatibility
        else:
            return 'english'


class LanguageRouter:
    """
    Intelligent language detection and prompt optimization system.
    
    Detects the language(s) used in technical documents and optimizes
    prompts accordingly for better analysis quality.
    """
    
    def __init__(self, config: Config, gemini_client: Optional[GeminiClient] = None):
        self.config = config
        self.gemini_client = gemini_client or GeminiClient(config)
        
        # Language patterns for quick detection
        self.language_patterns = {
            'spanish': {
                'keywords': [
                    'plano', 'proyecto', 'construcción', 'edificio', 'estructura',
                    'arquitectura', 'ingeniería', 'diseño', 'especificación',
                    'detalle', 'sección', 'elevación', 'planta', 'corte',
                    'materiales', 'acabados', 'instalaciones', 'cimentación',
                    'muros', 'losa', 'columna', 'viga', 'escalera'
                ],
                'indicators': [
                    r'\b(de|del|la|las|los|el|en|con|por|para|desde|hasta)\b',
                    r'\b(mm|cm|m|kg|ton|°C)\b',
                    r'\bN\d+[°º]\s*E\b',  # Coordinates in Spanish format
                ]
            },
            'english': {
                'keywords': [
                    'plan', 'project', 'construction', 'building', 'structure',
                    'architecture', 'engineering', 'design', 'specification',
                    'detail', 'section', 'elevation', 'floor', 'section',
                    'materials', 'finishes', 'systems', 'foundation',
                    'walls', 'slab', 'column', 'beam', 'stair'
                ],
                'indicators': [
                    r'\b(of|the|and|or|in|with|by|for|from|to)\b',
                    r'\b(ft|in|lb|psf|psi|°F)\b',
                    r'\bN\d+[°º]\s*W\b',  # Coordinates in English format
                ]
            },
            'portuguese': {
                'keywords': [
                    'planta', 'projeto', 'construção', 'edifício', 'estrutura',
                    'arquitetura', 'engenharia', 'desenho', 'especificação',
                    'detalhe', 'seção', 'elevação', 'pavimento', 'corte'
                ],
                'indicators': [
                    r'\b(da|das|do|dos|na|nas|no|nos|em|com|por|para)\b',
                ]
            }
        }
        
        logger.info("LanguageRouter initialized")
    
    async def detect_document_language(
        self, 
        text_samples: List[str], 
        use_ai_analysis: bool = True
    ) -> LanguageDetectionResult:
        """
        Detect the primary language(s) used in the document.
        
        Args:
            text_samples: List of text samples from different pages
            use_ai_analysis: Whether to use AI for enhanced detection
            
        Returns:
            LanguageDetectionResult with detection details
        """
        # Combine all text samples
        combined_text = ' '.join(text_samples).lower()
        
        # Quick pattern-based detection
        pattern_scores = self._analyze_language_patterns(combined_text)
        
        # AI-enhanced detection if enabled
        if use_ai_analysis and self.gemini_client:
            ai_detection = await self._ai_language_detection(text_samples[:5])  # Limit samples for API
            if ai_detection:
                # Combine pattern and AI results
                pattern_scores = self._merge_detection_results(pattern_scores, ai_detection)
        
        # Determine primary and secondary languages
        sorted_languages = sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True)
        primary_language = sorted_languages[0][0] if sorted_languages else 'english'
        primary_confidence = sorted_languages[0][1] if sorted_languages else 0.5
        
        secondary_languages = [lang for lang, score in sorted_languages[1:] if score > 0.2]
        
        # Check if it's a mixed-language document
        mixed_language = len([score for score in pattern_scores.values() if score > 0.3]) > 1
        
        # Extract technical terminology
        technical_terms = self._extract_technical_terminology(combined_text)
        
        # Extract region indicators
        region_indicators = self._extract_region_indicators(combined_text)
        
        result = LanguageDetectionResult(
            primary_language=primary_language,
            secondary_languages=secondary_languages,
            confidence=primary_confidence,
            language_distribution=pattern_scores,
            mixed_language=mixed_language,
            technical_terminology=technical_terms,
            region_indicators=region_indicators
        )
        
        logger.info(f"Language detection complete: {primary_language} ({primary_confidence:.3f})")
        if mixed_language:
            logger.info(f"Mixed-language document detected: {list(pattern_scores.keys())}")
        
        return result
    
    def _analyze_language_patterns(self, text: str) -> Dict[str, float]:
        """Analyze language patterns using regex and keyword matching."""
        
        scores = {}
        text_length = len(text)
        
        if text_length == 0:
            return {'english': 0.5}  # Default fallback
        
        for language, patterns in self.language_patterns.items():
            score = 0.0
            
            # Keyword matching
            keyword_matches = sum(1 for keyword in patterns['keywords'] if keyword in text)
            keyword_score = min(1.0, keyword_matches / len(patterns['keywords']))
            
            # Pattern matching
            pattern_matches = 0
            for pattern in patterns['indicators']:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                pattern_matches += matches
            
            pattern_score = min(1.0, pattern_matches / max(1, text_length / 100))
            
            # Combined score
            score = (keyword_score * 0.6) + (pattern_score * 0.4)
            scores[language] = score
        
        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {lang: score / total_score for lang, score in scores.items()}
        
        return scores
    
    async def _ai_language_detection(self, text_samples: List[str]) -> Optional[Dict[str, Any]]:
        """Use AI for enhanced language detection."""
        
        if not text_samples:
            return None
        
        # Create AI detection prompt
        prompt = f"""
        LANGUAGE DETECTION ANALYSIS
        
        Analyze these text samples from a technical document:
        
        {chr(10).join([f"Sample {i+1}: {sample[:200]}" for i, sample in enumerate(text_samples)])}
        
        Detect:
        1. Primary language used
        2. Secondary languages (if any)
        3. Regional indicators (Mexico, Spain, Argentina, etc.)
        4. Technical terminology language
        5. Mixed-language usage patterns
        
        RESPONSE FORMAT (JSON):
        {{
            "primary_language": "spanish|english|portuguese|french",
            "confidence": 0.85,
            "secondary_languages": ["english"],
            "regional_indicators": ["Mexico", "Latin America"],
            "mixed_language": true,
            "technical_terms_language": "spanish",
            "reasoning": "Brief explanation of detection"
        }}
        """
        
        try:
            response_text = self.gemini_client.generate_text_only_content(
                prompt=prompt,
                response_schema=self._get_language_detection_schema()
            )
            
            if response_text:
                return self._parse_language_detection_response(response_text)
                
        except Exception as e:
            logger.warning(f"AI language detection failed: {e}")
        
        return None
    
    def _get_language_detection_schema(self) -> Dict[str, Any]:
        """Get JSON schema for language detection response."""
        return {
            "type": "object",
            "properties": {
                "primary_language": {
                    "type": "string",
                    "enum": ["spanish", "english", "portuguese", "french", "italian", "german"],
                    "description": "Primary language detected"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence in primary language detection"
                },
                "secondary_languages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Secondary languages detected"
                },
                "regional_indicators": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "Regional or country indicators found"
                },
                "mixed_language": {
                    "type": "boolean",
                    "description": "Whether document uses multiple languages"
                },
                "technical_terms_language": {
                    "type": "string",
                    "description": "Language of technical terminology"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explanation of language detection"
                }
            },
            "required": ["primary_language", "confidence"]
        }
    
    def _parse_language_detection_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse AI language detection response."""
        try:
            import json
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                # Validate required fields
                if 'primary_language' in data and 'confidence' in data:
                    return data
                    
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse language detection response: {e}")
        
        return None
    
    def _merge_detection_results(
        self, 
        pattern_scores: Dict[str, float], 
        ai_detection: Dict[str, Any]
    ) -> Dict[str, float]:
        """Merge pattern-based and AI-based detection results."""
        
        ai_primary = ai_detection.get('primary_language', 'english')
        ai_confidence = ai_detection.get('confidence', 0.5)
        
        # Boost the AI-detected primary language
        if ai_primary in pattern_scores:
            pattern_scores[ai_primary] = max(pattern_scores[ai_primary], ai_confidence * 0.8)
        else:
            pattern_scores[ai_primary] = ai_confidence * 0.6
        
        # Add secondary languages from AI
        secondary_languages = ai_detection.get('secondary_languages', [])
        for lang in secondary_languages:
            if lang in pattern_scores:
                pattern_scores[lang] = max(pattern_scores[lang], 0.3)
            else:
                pattern_scores[lang] = 0.2
        
        # Renormalize
        total_score = sum(pattern_scores.values())
        if total_score > 0:
            pattern_scores = {lang: score / total_score for lang, score in pattern_scores.items()}
        
        return pattern_scores
    
    def _extract_technical_terminology(self, text: str) -> List[str]:
        """Extract technical terminology from text."""
        
        # Common technical terms patterns
        technical_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms like HVAC, MEP, AEC
            r'\b\d+\s*(mm|cm|m|ft|in|kg|lb|psi|psf|°[CF])\b',  # Measurements
            r'\b[A-Z]\d+[A-Z]?\b',  # Codes like A101, S1.1
            r'\b\d+/\d+\b',  # Fractions like 1/4, 3/8
        ]
        
        technical_terms = []
        for pattern in technical_patterns:
            matches = re.findall(pattern, text)
            technical_terms.extend(matches)
        
        # Return most common terms
        term_counts = Counter(technical_terms)
        return [term for term, count in term_counts.most_common(20)]
    
    def _extract_region_indicators(self, text: str) -> List[str]:
        """Extract regional indicators from text."""
        
        region_patterns = {
            'mexico': r'\b(mexico|méxico|cdmx|guadalajara|monterrey)\b',
            'spain': r'\b(españa|madrid|barcelona|valencia)\b',
            'argentina': r'\b(argentina|buenos aires|córdoba)\b',
            'colombia': r'\b(colombia|bogotá|medellín|cali)\b',
            'chile': r'\b(chile|santiago|valparaíso)\b',
            'peru': r'\b(perú|lima|arequipa)\b',
            'usa': r'\b(usa|united states|california|texas|florida)\b',
            'canada': r'\b(canada|ontario|quebec|alberta)\b'
        }
        
        indicators = []
        for region, pattern in region_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                indicators.append(region)
        
        return indicators
    
    def optimize_prompt_for_language(
        self, 
        base_prompt: str, 
        language_result: LanguageDetectionResult,
        analysis_type: str = "general"
    ) -> str:
        """
        Optimize prompt based on detected language and regional context.
        
        Args:
            base_prompt: Base analysis prompt
            language_result: Language detection result
            analysis_type: Type of analysis being performed
            
        Returns:
            Optimized prompt adapted for detected language
        """
        
        optimal_language = language_result.get_optimal_prompt_language()
        
        # Language-specific optimizations
        if optimal_language == 'spanish':
            language_instruction = self._get_spanish_instruction(language_result)
        else:
            language_instruction = self._get_english_instruction(language_result)
        
        # Technical terminology adaptation
        if language_result.technical_terminology:
            terminology_instruction = self._get_terminology_instruction(
                language_result.technical_terminology, 
                optimal_language
            )
        else:
            terminology_instruction = ""
        
        # Regional context adaptation
        regional_instruction = ""
        if language_result.region_indicators:
            regional_instruction = self._get_regional_instruction(
                language_result.region_indicators,
                optimal_language
            )
        
        # Combine all instructions
        optimized_prompt = f"{language_instruction}\n\n"
        
        if terminology_instruction:
            optimized_prompt += f"{terminology_instruction}\n\n"
            
        if regional_instruction:
            optimized_prompt += f"{regional_instruction}\n\n"
            
        optimized_prompt += base_prompt
        
        logger.info(f"Prompt optimized for {optimal_language} with {len(language_result.technical_terminology)} technical terms")
        
        return optimized_prompt
    
    def _get_spanish_instruction(self, language_result: LanguageDetectionResult) -> str:
        """Get Spanish-specific instruction."""
        
        base_instruction = "IMPORTANTE: Responde completamente en español. Proporciona todo el análisis, resúmenes e insights en idioma español."
        
        if language_result.mixed_language:
            base_instruction += " Este documento contiene múltiples idiomas, pero mantén todas las respuestas en español para consistencia."
        
        if 'mexico' in language_result.region_indicators:
            base_instruction += " Usa terminología técnica mexicana cuando sea apropiada."
        elif any(region in language_result.region_indicators for region in ['spain', 'argentina', 'colombia']):
            base_instruction += " Usa terminología técnica latinoamericana/española cuando sea apropiada."
        
        return base_instruction
    
    def _get_english_instruction(self, language_result: LanguageDetectionResult) -> str:
        """Get English-specific instruction."""
        
        base_instruction = "IMPORTANT: Respond completely in English. Provide all analysis, summaries, and insights in English language only."
        
        if language_result.mixed_language:
            base_instruction += " This document contains multiple languages, but maintain all responses in English for consistency."
        
        if 'usa' in language_result.region_indicators:
            base_instruction += " Use US technical terminology and standards when appropriate."
        elif 'canada' in language_result.region_indicators:
            base_instruction += " Use Canadian technical terminology and standards when appropriate."
        
        return base_instruction
    
    def _get_terminology_instruction(self, technical_terms: List[str], language: str) -> str:
        """Get instruction for handling technical terminology."""
        
        if language == 'spanish':
            return f"""
            TERMINOLOGÍA TÉCNICA DETECTADA: {', '.join(technical_terms[:10])}
            
            Usa esta terminología específica del documento cuando sea relevante.
            Mantén los términos técnicos originales cuando sean códigos o referencias específicas.
            """
        else:
            return f"""
            DETECTED TECHNICAL TERMINOLOGY: {', '.join(technical_terms[:10])}
            
            Use this document-specific terminology when relevant.
            Preserve original technical terms when they are specific codes or references.
            """
    
    def _get_regional_instruction(self, regions: List[str], language: str) -> str:
        """Get instruction for regional context."""
        
        if language == 'spanish':
            return f"""
            CONTEXTO REGIONAL DETECTADO: {', '.join(regions)}
            
            Considera las prácticas de construcción y estándares específicos de esta región.
            Usa unidades de medida y terminología apropiadas para el contexto regional.
            """
        else:
            return f"""
            DETECTED REGIONAL CONTEXT: {', '.join(regions)}
            
            Consider construction practices and standards specific to this region.
            Use appropriate units of measurement and terminology for the regional context.
            """


# Convenience functions
def create_language_router(config: Config, gemini_client: Optional[GeminiClient] = None) -> LanguageRouter:
    """Create a language router instance."""
    return LanguageRouter(config, gemini_client)


async def detect_and_optimize_prompt(
    base_prompt: str,
    text_samples: List[str],
    config: Config,
    gemini_client: Optional[GeminiClient] = None,
    analysis_type: str = "general"
) -> Tuple[str, LanguageDetectionResult]:
    """
    Convenience function to detect language and optimize prompt.
    
    Returns:
        Tuple of (optimized_prompt, language_detection_result)
    """
    router = create_language_router(config, gemini_client)
    language_result = await router.detect_document_language(text_samples)
    optimized_prompt = router.optimize_prompt_for_language(base_prompt, language_result, analysis_type)
    
    return optimized_prompt, language_result
