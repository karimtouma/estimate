"""
Intelligent Type Classifier for Dynamic Schema System.

This module implements AI-powered classification of element types using
Gemini and GEPA optimization for autonomous type discovery and registration.
"""

import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from pathlib import Path

try:
    from ..services.gemini_client import GeminiClient
    from ..core.config import Config
    from ..utils.logging_config import get_logger
    from .dynamic_schemas import (
        DynamicElementRegistry, AdaptiveElementType, ElementTypeDefinition,
        CoreElementCategory, DiscoveryMethod, get_dynamic_registry
    )
    from ..optimization.gepa_classification_enhancer import create_gepa_classification_enhancer
    GEPA_ENHANCER_AVAILABLE = True
except ImportError:
    # Fallback for direct execution or testing
    import logging
    logger = logging.getLogger(__name__)
    GEPA_ENHANCER_AVAILABLE = False
    
    # Mock classes for testing
    class GeminiClient:
        def __init__(self, config): pass
        async def generate_content(self, prompt): return {"text": "mock response"}
    
    class Config:
        def __init__(self): 
            self.auto_register_confidence_threshold = 0.85

try:
    logger = get_logger(__name__)
except NameError:
    logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of element type classification."""
    
    # Primary classification
    classified_type: str
    base_category: CoreElementCategory
    confidence: float
    
    # Alternative classifications
    alternative_types: List[Tuple[str, float]] = None
    
    # Classification metadata
    discovery_method: DiscoveryMethod = DiscoveryMethod.HYBRID_ANALYSIS
    reasoning: Optional[str] = None
    evidence_used: List[str] = None
    
    # Context information
    domain_context: Optional[str] = None
    industry_context: Optional[str] = None
    
    # Quality indicators
    is_new_discovery: bool = False
    requires_validation: bool = False
    
    def __post_init__(self):
        if self.alternative_types is None:
            self.alternative_types = []
        if self.evidence_used is None:
            self.evidence_used = []


class IntelligentTypeClassifier:
    """
    AI-powered classifier that determines element types using multiple strategies.
    
    Combines pattern recognition, nomenclature analysis, visual features,
    and AI reasoning to classify elements accurately and discover new types.
    """
    
    def __init__(self, config: Config, registry: Optional[DynamicElementRegistry] = None):
        self.config = config
        self.gemini_client = GeminiClient(config)
        self.registry = registry or get_dynamic_registry()
        
        # Classification thresholds
        self.auto_register_threshold = getattr(config, 'auto_register_confidence_threshold', 0.85)
        self.validation_threshold = 0.7
        self.new_discovery_threshold = 0.6
        
        # Classification strategies
        self.strategies = [
            self._classify_by_registry_lookup,
            self._classify_by_pattern_matching,
            self._classify_by_nomenclature_analysis,
            self._classify_by_ai_reasoning
        ]
        
        # Initialize GEPA classification enhancer for always-on improvement
        if GEPA_ENHANCER_AVAILABLE:
            self.gepa_enhancer = create_gepa_classification_enhancer(config, self.gemini_client)
            logger.info("GEPA classification enhancer initialized - ALWAYS ACTIVE")
        else:
            self.gepa_enhancer = None
            logger.warning("GEPA enhancer not available - using standard classification")
        
        # Performance tracking
        self.classification_count = 0
        self.discovery_count = 0
        self.accuracy_history = []
        self.gepa_enhancement_count = 0
        
        logger.info("IntelligentTypeClassifier initialized")
    
    async def classify_element(
        self, 
        element_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ClassificationResult:
        """
        Classify an element using intelligent multi-strategy approach.
        
        Args:
            element_data: Element data including visual features, text, location, etc.
            context: Additional context like document type, page context, etc.
            
        Returns:
            ClassificationResult with type, confidence, and metadata
        """
        start_time = time.time()
        self.classification_count += 1
        
        try:
            # Extract key information from element data
            element_info = self._extract_element_info(element_data)
            
            # Apply classification strategies in order of efficiency
            best_result = None
            all_results = []
            
            for strategy in self.strategies:
                try:
                    result = await strategy(element_info, context)
                    if result:
                        all_results.append(result)
                        
                        # Use first high-confidence result
                        if result.confidence >= self.auto_register_threshold:
                            best_result = result
                            break
                        
                        # Keep track of best result so far
                        if best_result is None or result.confidence > best_result.confidence:
                            best_result = result
                            
                except Exception as e:
                    logger.warning(f"Classification strategy {strategy.__name__} failed: {e}")
                    continue
            
            # If no result, create fallback classification using AI reasoning
            if best_result is None:
                best_result = await self._create_fallback_classification(element_info)
            
            # ALWAYS enhance with GEPA using multiple candidates and judge
            if self.gepa_enhancer:
                logger.debug("🧬 GEPA: Always enhancing classification with multiple candidates...")
                
                gepa_result = await self.gepa_enhancer.enhance_classification(
                    element_info=element_info,
                    context=context,
                    base_classification=best_result
                )
                
                # Use GEPA-enhanced result
                best_result = gepa_result.get_classification_result()
                self.gepa_enhancement_count += 1
                
                logger.debug(f"🧬 GEPA enhanced: {best_result.classified_type} (consensus: {gepa_result.consensus_score:.3f})")
            
            # Enhance result with ensemble information
            if len(all_results) > 1:
                best_result = self._enhance_with_ensemble(best_result, all_results)
            
            # Determine if this is a new discovery
            existing_type = self.registry.get_type_definition(best_result.classified_type)
            best_result.is_new_discovery = existing_type is None
            
            # Auto-register if confidence is high enough
            if (best_result.is_new_discovery and 
                best_result.confidence >= self.auto_register_threshold):
                
                await self._auto_register_type(best_result, element_info, context)
            
            # Set validation requirement
            best_result.requires_validation = (
                best_result.confidence < self.validation_threshold or
                best_result.is_new_discovery
            )
            
            # Log classification
            processing_time = time.time() - start_time
            logger.debug(
                f"Classified element: {best_result.classified_type} "
                f"(confidence: {best_result.confidence:.3f}, "
                f"method: {best_result.discovery_method.value}, "
                f"time: {processing_time:.3f}s)"
            )
            
            return best_result
            
        except Exception as e:
            logger.error(f"Element classification failed: {e}")
            # Return minimal fallback result
            return ClassificationResult(
                classified_type="unknown_element",
                base_category=CoreElementCategory.SPECIALIZED,
                confidence=0.1,
                discovery_method=DiscoveryMethod.HYBRID_ANALYSIS,
                reasoning=f"Classification failed: {str(e)}"
            )
    
    async def _classify_by_registry_lookup(
        self, 
        element_info: Dict[str, Any], 
        context: Optional[Dict[str, Any]]
    ) -> Optional[ClassificationResult]:
        """Strategy 1: Look up element in existing registry."""
        
        # Extract potential type names from element info
        potential_names = self._extract_potential_type_names(element_info)
        
        best_match = None
        best_confidence = 0.0
        
        for name in potential_names:
            type_def = self.registry.get_type_definition(name)
            if type_def:
                # Calculate confidence based on type reliability and context match
                confidence = self._calculate_registry_confidence(type_def, element_info, context)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = type_def
        
        if best_match and best_confidence >= 0.5:
            return ClassificationResult(
                classified_type=best_match.type_name,
                base_category=best_match.base_category,
                confidence=best_confidence,
                discovery_method=DiscoveryMethod.PATTERN_ANALYSIS,
                reasoning=f"Found in registry with {best_match.occurrence_count} occurrences",
                domain_context=best_match.domain_context,
                industry_context=best_match.industry_context
            )
        
        return None
    
    async def _classify_by_pattern_matching(
        self, 
        element_info: Dict[str, Any], 
        context: Optional[Dict[str, Any]]
    ) -> Optional[ClassificationResult]:
        """Strategy 2: Pattern-based classification using known patterns."""
        
        # Get visual and textual features
        visual_features = element_info.get('visual_features', {})
        text_features = element_info.get('text_features', {})
        
        # Pattern matching rules (can be enhanced with ML)
        patterns = {
            # Structural patterns
            'beam': {
                'visual': ['horizontal_line', 'structural_symbol'],
                'text': ['beam', 'girder', 'joist', 'w12', 'w14', 'w16'],
                'category': CoreElementCategory.STRUCTURAL,
                'confidence_base': 0.8
            },
            'column': {
                'visual': ['vertical_line', 'circular_symbol', 'square_symbol'],
                'text': ['column', 'post', 'pillar', 'hss', 'pipe'],
                'category': CoreElementCategory.STRUCTURAL,
                'confidence_base': 0.8
            },
            'wall': {
                'visual': ['parallel_lines', 'hatched_area'],
                'text': ['wall', 'partition', 'barrier'],
                'category': CoreElementCategory.ARCHITECTURAL,
                'confidence_base': 0.7
            },
            'door': {
                'visual': ['arc_symbol', 'rectangular_opening'],
                'text': ['door', 'entrance', 'exit', 'opening'],
                'category': CoreElementCategory.ARCHITECTURAL,
                'confidence_base': 0.75
            },
            'window': {
                'visual': ['parallel_lines', 'glass_symbol'],
                'text': ['window', 'glazing', 'opening'],
                'category': CoreElementCategory.ARCHITECTURAL,
                'confidence_base': 0.75
            },
            # MEP patterns
            'electrical_outlet': {
                'visual': ['circular_symbol', 'electrical_symbol'],
                'text': ['outlet', 'receptacle', 'plug'],
                'category': CoreElementCategory.MEP,
                'confidence_base': 0.8
            },
            'hvac_duct': {
                'visual': ['rectangular_path', 'airflow_symbol'],
                'text': ['duct', 'hvac', 'air', 'ventilation'],
                'category': CoreElementCategory.MEP,
                'confidence_base': 0.8
            }
        }
        
        best_match = None
        best_confidence = 0.0
        
        for type_name, pattern in patterns.items():
            confidence = self._calculate_pattern_confidence(
                pattern, visual_features, text_features
            )
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = (type_name, pattern)
        
        if best_match and best_confidence >= 0.6:
            type_name, pattern = best_match
            return ClassificationResult(
                classified_type=type_name,
                base_category=pattern['category'],
                confidence=best_confidence,
                discovery_method=DiscoveryMethod.PATTERN_ANALYSIS,
                reasoning=f"Matched pattern with confidence {best_confidence:.3f}",
                evidence_used=['visual_features', 'text_features']
            )
        
        return None
    
    async def _classify_by_nomenclature_analysis(
        self, 
        element_info: Dict[str, Any], 
        context: Optional[Dict[str, Any]]
    ) -> Optional[ClassificationResult]:
        """Strategy 3: Nomenclature-based classification."""
        
        # Extract nomenclature codes from text
        text_content = element_info.get('text_content', '')
        codes = self._extract_nomenclature_codes(text_content)
        
        if not codes:
            return None
        
        # Analyze codes for type hints
        for code in codes:
            type_hint = self._analyze_nomenclature_code(code)
            if type_hint:
                confidence = self._calculate_nomenclature_confidence(code, type_hint)
                
                if confidence >= 0.6:
                    return ClassificationResult(
                        classified_type=type_hint['type_name'],
                        base_category=type_hint['category'],
                        confidence=confidence,
                        discovery_method=DiscoveryMethod.NOMENCLATURE_PARSING,
                        reasoning=f"Inferred from nomenclature code: {code}",
                        evidence_used=['nomenclature_codes']
                    )
        
        return None
    
    async def _classify_by_ai_reasoning(
        self, 
        element_info: Dict[str, Any], 
        context: Optional[Dict[str, Any]]
    ) -> Optional[ClassificationResult]:
        """Strategy 4: AI-powered reasoning using Gemini."""
        
        try:
            # Create comprehensive analysis prompt
            prompt = self._create_ai_classification_prompt(element_info, context)
            
            # Get AI analysis using text-only method
            response_text = self.gemini_client.generate_text_only_content(
                prompt=prompt,
                response_schema=self._get_classification_schema()
            )
            
            if not response_text:
                return None
            
            # Parse AI response
            ai_result = self._parse_ai_classification_response(response_text)
            
            if ai_result and ai_result['confidence'] >= 0.5:
                return ClassificationResult(
                    classified_type=ai_result['type_name'],
                    base_category=CoreElementCategory(ai_result['category']),
                    confidence=ai_result['confidence'],
                    discovery_method=DiscoveryMethod.AI_CLASSIFICATION,
                    reasoning=ai_result.get('reasoning', 'AI classification'),
                    evidence_used=['ai_analysis'],
                    domain_context=ai_result.get('domain_context'),
                    industry_context=ai_result.get('industry_context')
                )
            
        except Exception as e:
            logger.warning(f"AI classification failed: {e}")
        
        return None
    
    def _extract_element_info(self, element_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant information from element data."""
        return {
            'visual_features': element_data.get('visual_features', {}),
            'text_features': element_data.get('textual_features', {}),
            'text_content': element_data.get('extracted_text', ''),
            'location': element_data.get('location', {}),
            'dimensions': element_data.get('dimensions', {}),
            'annotations': element_data.get('annotations', []),
            'label': element_data.get('label', ''),
            'description': element_data.get('description', '')
        }
    
    def _extract_potential_type_names(self, element_info: Dict[str, Any]) -> List[str]:
        """Extract potential type names from element information."""
        potential_names = []
        
        # From label
        if element_info.get('label'):
            potential_names.append(element_info['label'].lower().strip())
        
        # From description
        if element_info.get('description'):
            # Extract key words from description
            desc_words = element_info['description'].lower().split()
            potential_names.extend(desc_words)
        
        # From text content
        text_content = element_info.get('text_content', '')
        if text_content:
            # Extract meaningful words (skip common words)
            words = text_content.lower().split()
            meaningful_words = [w for w in words if len(w) > 2 and w.isalpha()]
            potential_names.extend(meaningful_words[:5])  # Limit to avoid noise
        
        # Clean and normalize names
        cleaned_names = []
        for name in potential_names:
            if name and len(name.strip()) > 1:
                # Use the same normalization as registry
                normalized = self.registry._normalize_type_name(name)
                if normalized and normalized not in cleaned_names:
                    cleaned_names.append(normalized)
        
        return cleaned_names
    
    def _calculate_registry_confidence(
        self, 
        type_def: ElementTypeDefinition, 
        element_info: Dict[str, Any], 
        context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate confidence for registry-based classification."""
        base_confidence = type_def.reliability_score
        
        # Boost confidence based on context match
        context_boost = 0.0
        if context:
            doc_domain = context.get('document_domain', '')
            if type_def.domain_context and doc_domain in type_def.domain_context:
                context_boost += 0.1
            
            doc_industry = context.get('industry_context', '')
            if type_def.industry_context and doc_industry in type_def.industry_context:
                context_boost += 0.1
        
        # Boost based on occurrence frequency
        frequency_boost = min(0.1, type_def.occurrence_count / 100.0)
        
        return min(1.0, base_confidence + context_boost + frequency_boost)
    
    def _calculate_pattern_confidence(
        self, 
        pattern: Dict[str, Any], 
        visual_features: Dict[str, Any], 
        text_features: Dict[str, Any]
    ) -> float:
        """Calculate confidence for pattern-based classification."""
        base_confidence = pattern['confidence_base']
        
        # Check visual pattern matches
        visual_matches = 0
        visual_patterns = pattern.get('visual', [])
        if visual_patterns:
            for visual_pattern in visual_patterns:
                if visual_pattern in str(visual_features):
                    visual_matches += 1
            visual_score = visual_matches / len(visual_patterns)
        else:
            visual_score = 0.5  # Neutral if no visual patterns defined
        
        # Check text pattern matches
        text_matches = 0
        text_patterns = pattern.get('text', [])
        text_content = str(text_features).lower()
        if text_patterns:
            for text_pattern in text_patterns:
                if text_pattern.lower() in text_content:
                    text_matches += 1
            text_score = text_matches / len(text_patterns)
        else:
            text_score = 0.5  # Neutral if no text patterns defined
        
        # Combine scores
        combined_score = (visual_score * 0.6) + (text_score * 0.4)
        return base_confidence * combined_score
    
    def _extract_nomenclature_codes(self, text_content: str) -> List[str]:
        """Extract nomenclature codes from text content."""
        import re
        
        # Common nomenclature patterns
        patterns = [
            r'[A-Z]{1,3}-\d{2,4}[A-Z]?',  # V-201, P-101-A
            r'[A-Z]{2,4}\d{2,4}',         # HVAC01, ELEC123
            r'\d{2,3}[A-Z]{1,2}\d{1,3}',  # 12A1, 345BC
        ]
        
        codes = []
        for pattern in patterns:
            matches = re.findall(pattern, text_content)
            codes.extend(matches)
        
        return list(set(codes))  # Remove duplicates
    
    def _analyze_nomenclature_code(self, code: str) -> Optional[Dict[str, Any]]:
        """Analyze a nomenclature code to infer element type."""
        code_upper = code.upper()
        
        # Common prefix mappings
        prefix_mappings = {
            'V': {'type_name': 'valve', 'category': CoreElementCategory.MEP},
            'P': {'type_name': 'pump', 'category': CoreElementCategory.MEP},
            'T': {'type_name': 'tank', 'category': CoreElementCategory.MEP},
            'E': {'type_name': 'electrical_equipment', 'category': CoreElementCategory.MEP},
            'H': {'type_name': 'hvac_equipment', 'category': CoreElementCategory.MEP},
            'W': {'type_name': 'wall', 'category': CoreElementCategory.ARCHITECTURAL},
            'D': {'type_name': 'door', 'category': CoreElementCategory.ARCHITECTURAL},
            'B': {'type_name': 'beam', 'category': CoreElementCategory.STRUCTURAL},
            'C': {'type_name': 'column', 'category': CoreElementCategory.STRUCTURAL},
        }
        
        # Check prefixes
        for prefix, mapping in prefix_mappings.items():
            if code_upper.startswith(prefix):
                return mapping
        
        # Check for specific patterns
        if 'HVAC' in code_upper:
            return {'type_name': 'hvac_equipment', 'category': CoreElementCategory.MEP}
        elif 'ELEC' in code_upper:
            return {'type_name': 'electrical_equipment', 'category': CoreElementCategory.MEP}
        
        return None
    
    def _calculate_nomenclature_confidence(self, code: str, type_hint: Dict[str, Any]) -> float:
        """Calculate confidence for nomenclature-based classification."""
        # Base confidence depends on code pattern clarity
        base_confidence = 0.7
        
        # Boost for well-known prefixes
        well_known_prefixes = ['V', 'P', 'T', 'E', 'H', 'W', 'D', 'B', 'C']
        if any(code.upper().startswith(prefix) for prefix in well_known_prefixes):
            base_confidence += 0.1
        
        # Boost for complete codes (prefix + numbers + suffix)
        import re
        if re.match(r'[A-Z]{1,3}-\d{2,4}[A-Z]?', code):
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _create_ai_classification_prompt(
        self, 
        element_info: Dict[str, Any], 
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Create comprehensive AI classification prompt."""
        
        context_str = ""
        if context:
            context_str = f"""
            DOCUMENT CONTEXT:
            - Document type: {context.get('document_type', 'unknown')}
            - Industry domain: {context.get('document_domain', 'unknown')}
            - Page type: {context.get('page_type', 'unknown')}
            """
        
        prompt = f"""
        You are an expert structural engineer and architect analyzing a technical drawing element.
        
        {context_str}
        
        ELEMENT INFORMATION:
        - Text content: "{element_info.get('text_content', 'No text')}"
        - Label: "{element_info.get('label', 'No label')}"
        - Description: "{element_info.get('description', 'No description')}"
        - Visual features: {json.dumps(element_info.get('visual_features', {}), indent=2)}
        - Location: {json.dumps(element_info.get('location', {}), indent=2)}
        - Annotations: {element_info.get('annotations', [])}
        
        CLASSIFICATION TASK:
        Classify this element into a specific type. Be as specific as possible while maintaining accuracy.
        
        CATEGORIES:
        - structural: beams, columns, foundations, slabs, etc.
        - architectural: walls, doors, windows, rooms, stairs, etc.
        - mep: electrical outlets, HVAC ducts, plumbing fixtures, etc.
        - annotation: dimensions, text labels, symbols, grid lines, etc.
        - specialized: industry-specific elements not in other categories
        
        RESPONSE FORMAT (JSON):
        {{
            "type_name": "specific_element_type",
            "category": "structural|architectural|mep|annotation|specialized",
            "confidence": 0.85,
            "reasoning": "Brief explanation of classification logic",
            "domain_context": "residential|commercial|industrial|infrastructure",
            "industry_context": "construction|petrochemical|aerospace|naval|etc"
        }}
        
        Be specific with type names (e.g., "steel_moment_frame" not just "beam").
        Only respond with valid JSON.
        """
        
        return prompt
    
    def _get_classification_schema(self) -> Dict[str, Any]:
        """Get JSON schema for AI classification response."""
        return {
            "type": "object",
            "properties": {
                "type_name": {
                    "type": "string",
                    "description": "Specific element type name"
                },
                "category": {
                    "type": "string",
                    "enum": ["structural", "architectural", "mep", "annotation", "specialized"],
                    "description": "Core category classification"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Classification confidence score"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of classification logic"
                },
                "domain_context": {
                    "type": "string",
                    "description": "Domain context (residential, commercial, industrial, etc.)"
                },
                "industry_context": {
                    "type": "string",
                    "description": "Industry context (construction, petrochemical, aerospace, etc.)"
                }
            },
            "required": ["type_name", "category", "confidence"]
        }
    
    def _parse_ai_classification_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse AI classification response."""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                return None
            
            json_str = json_match.group(0)
            result = json.loads(json_str)
            
            # Validate required fields
            required_fields = ['type_name', 'category', 'confidence']
            if not all(field in result for field in required_fields):
                return None
            
            # Validate confidence range
            if not (0.0 <= result['confidence'] <= 1.0):
                return None
            
            # Validate category
            valid_categories = [cat.value for cat in CoreElementCategory]
            if result['category'] not in valid_categories:
                return None
            
            return result
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse AI classification response: {e}")
            return None
    
    async def _create_fallback_classification(self, element_info: Dict[str, Any]) -> ClassificationResult:
        """Create fallback classification using pure AI reasoning when all strategies fail."""
        
        try:
            # Create a simplified AI reasoning prompt for fallback
            fallback_prompt = self._create_fallback_ai_prompt(element_info)
            
            # Use AI reasoning even for fallback
            response_text = self.gemini_client.generate_text_only_content(
                prompt=fallback_prompt,
                response_schema=self._get_classification_schema()
            )
            
            if response_text:
                ai_result = self._parse_ai_classification_response(response_text)
                if ai_result:
                    return ClassificationResult(
                        classified_type=ai_result['type_name'],
                        base_category=CoreElementCategory(ai_result['category']),
                        confidence=max(0.3, ai_result['confidence'] * 0.7),  # Reduced confidence for fallback
                        discovery_method=DiscoveryMethod.AI_CLASSIFICATION,
                        reasoning=f"Fallback AI reasoning: {ai_result.get('reasoning', 'AI fallback classification')}",
                        evidence_used=['fallback_ai_analysis'],
                        requires_validation=True
                    )
        
        except Exception as e:
            logger.warning(f"Fallback AI classification failed: {e}")
        
        # Ultimate fallback - use generic classification without keywords
        return ClassificationResult(
            classified_type="unclassified_element",
            base_category=CoreElementCategory.SPECIALIZED,
            confidence=0.1,  # Very low confidence
            discovery_method=DiscoveryMethod.PATTERN_ANALYSIS,
            reasoning="Ultimate fallback - insufficient information for classification",
            requires_validation=True
        )
    
    def _create_fallback_ai_prompt(self, element_info: Dict[str, Any]) -> str:
        """Create a simplified AI prompt for fallback classification."""
        
        text_content = element_info.get('text_content', 'No text available')
        visual_features = element_info.get('visual_features', {})
        location = element_info.get('location', {})
        
        prompt = f"""
        SIMPLIFIED ELEMENT CLASSIFICATION
        
        Analyze this element with limited information available:
        
        TEXT CONTENT: {text_content[:200]}
        VISUAL FEATURES: {visual_features}
        LOCATION: {location}
        
        Based on this limited information, classify this element into one of these categories:
        - structural: load-bearing elements, support systems
        - architectural: building envelope, spaces, aesthetic elements  
        - mep: mechanical, electrical, plumbing systems
        - annotation: text, labels, dimensions, notes
        - specialized: industry-specific or unique elements
        
        Provide your best analysis even with limited information.
        
        RESPONSE FORMAT (JSON):
        {{
            "type_name": "specific_element_type",
            "category": "structural|architectural|mep|annotation|specialized",
            "confidence": 0.5,
            "reasoning": "Brief explanation based on available evidence"
        }}
        """
        
        return prompt
    
    def _enhance_with_ensemble(
        self, 
        best_result: ClassificationResult, 
        all_results: List[ClassificationResult]
    ) -> ClassificationResult:
        """Enhance result using ensemble of multiple classifications."""
        
        # If multiple results agree on type, boost confidence
        type_votes = {}
        for result in all_results:
            if result.classified_type in type_votes:
                type_votes[result.classified_type] += result.confidence
            else:
                type_votes[result.classified_type] = result.confidence
        
        # If consensus exists, boost confidence
        if best_result.classified_type in type_votes:
            vote_strength = type_votes[best_result.classified_type]
            total_votes = sum(type_votes.values())
            consensus_factor = vote_strength / total_votes if total_votes > 0 else 0
            
            if consensus_factor > 0.6:  # Strong consensus
                best_result.confidence = min(1.0, best_result.confidence * 1.1)
                best_result.reasoning += f" (consensus factor: {consensus_factor:.2f})"
        
        # Collect alternative types
        alternatives = []
        for result in all_results:
            if result.classified_type != best_result.classified_type:
                alternatives.append((result.classified_type, result.confidence))
        
        # Sort alternatives by confidence and keep top 3
        alternatives.sort(key=lambda x: x[1], reverse=True)
        best_result.alternative_types = alternatives[:3]
        
        return best_result
    
    async def _auto_register_type(
        self, 
        result: ClassificationResult, 
        element_info: Dict[str, Any], 
        context: Optional[Dict[str, Any]]
    ):
        """Auto-register a newly discovered type with high confidence."""
        
        try:
            success, message = self.registry.register_discovered_type(
                type_name=result.classified_type,
                base_category=result.base_category,
                discovery_confidence=result.confidence,
                discovery_method=result.discovery_method,
                domain_context=result.domain_context,
                industry_context=result.industry_context,
                description=result.reasoning,
                typical_properties=element_info.get('visual_features', {}),
                common_patterns=[element_info.get('text_content', '')[:100]]
            )
            
            if success:
                self.discovery_count += 1
                logger.info(f"Auto-registered new type: {result.classified_type}")
            else:
                logger.warning(f"Failed to auto-register type: {message}")
                
        except Exception as e:
            logger.error(f"Auto-registration failed: {e}")
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get classifier performance statistics."""
        stats = {
            "total_classifications": self.classification_count,
            "discoveries_made": self.discovery_count,
            "discovery_rate": self.discovery_count / max(1, self.classification_count),
            "average_accuracy": sum(self.accuracy_history) / max(1, len(self.accuracy_history)),
            "registry_size": len(self.registry.discovered_types)
        }
        
        # Add GEPA enhancement statistics
        if self.gepa_enhancer:
            gepa_stats = self.gepa_enhancer.get_enhancement_statistics()
            stats.update({
                "gepa_enhancements": self.gepa_enhancement_count,
                "gepa_enhancement_rate": self.gepa_enhancement_count / max(1, self.classification_count),
                "gepa_statistics": gepa_stats
            })
        
        return stats
