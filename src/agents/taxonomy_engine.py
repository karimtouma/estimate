"""
Intelligent taxonomy engine for structural blueprint analysis.

This module implements advanced taxonomy generation using DSPy optimization
and multi-modal analysis with Gemini 2.5 Flash.
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import dspy
from google.genai import types

from ..models.blueprint_schemas import (
    PageTaxonomy, BlueprintPageType, StructuralElement, StructuralElementType,
    VisualFeatures, TextualFeatures, SpatialRelationship, BoundingBox, Coordinates
)
from ..services.gemini_client import GeminiClient
from ..core.config import Config
from ..optimization.gepa_optimizer import OptimizedBlueprintAnalyzer
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class BlueprintAnalysisSignature(dspy.Signature):
    """DSPy signature for structural blueprint analysis."""
    
    # Input fields
    page_image_description: str = dspy.InputField(
        desc="Detailed description of the page image content"
    )
    extracted_text: str = dspy.InputField(
        desc="Text extracted from the page via OCR"
    )
    visual_features: str = dspy.InputField(
        desc="JSON string of visual features (colors, lines, complexity)"
    )
    context_from_previous_pages: str = dspy.InputField(
        desc="Context and patterns from previously analyzed pages"
    )
    
    # Output fields
    page_type: str = dspy.OutputField(
        desc="Primary classification of the blueprint page type"
    )
    page_subtype: str = dspy.OutputField(
        desc="Secondary classification providing more specific categorization"
    )
    structural_elements_detected: str = dspy.OutputField(
        desc="JSON list of structural elements identified in the page"
    )
    spatial_relationships: str = dspy.OutputField(
        desc="JSON list of spatial relationships between elements"
    )
    technical_specifications: str = dspy.OutputField(
        desc="Technical specifications and standards identified"
    )
    confidence_assessment: str = dspy.OutputField(
        desc="Confidence score and reasoning for the analysis"
    )


class ElementDetectionSignature(dspy.Signature):
    """DSPy signature for detailed element detection."""
    
    page_type: str = dspy.InputField(desc="Type of blueprint page")
    image_analysis: str = dspy.InputField(desc="Detailed image analysis results")
    text_content: str = dspy.InputField(desc="Extracted text content")
    
    detected_elements: str = dspy.OutputField(
        desc="Detailed list of structural elements with properties"
    )
    element_relationships: str = dspy.OutputField(
        desc="Relationships and connections between elements"
    )
    measurement_analysis: str = dspy.OutputField(
        desc="Extracted measurements and dimensional information"
    )


class TaxonomyValidationSignature(dspy.Signature):
    """DSPy signature for taxonomy validation and refinement."""
    
    initial_taxonomy: str = dspy.InputField(desc="Initial taxonomy classification")
    cross_page_context: str = dspy.InputField(desc="Context from other pages")
    domain_knowledge: str = dspy.InputField(desc="Architectural domain knowledge")
    
    validated_taxonomy: str = dspy.OutputField(
        desc="Validated and refined taxonomy classification"
    )
    consistency_check: str = dspy.OutputField(
        desc="Consistency validation with document context"
    )
    confidence_score: str = dspy.OutputField(
        desc="Final confidence score with reasoning"
    )


class IntelligentTaxonomyEngine:
    """
    Advanced taxonomy engine using DSPy and multi-modal analysis.
    """
    
    def __init__(self, config: Config, optimization_results: Optional[Dict[str, Any]] = None):
        self.config = config
        self.gemini_client = GeminiClient(config)
        
        # Initialize GEPA-optimized analyzer
        self.gepa_analyzer = OptimizedBlueprintAnalyzer(config, optimization_results)
        
        # Initialize DSPy components
        self.setup_dspy_components()
        
        # Taxonomy knowledge base
        self.taxonomy_knowledge = self.load_taxonomy_knowledge()
        
        # Context management
        self.context_memory = {}
        
        # GEPA optimization status
        self.gepa_enabled = optimization_results is not None
        
        if self.gepa_enabled:
            logger.info("Intelligent Taxonomy Engine initialized with GEPA optimization")
        else:
            logger.info("Intelligent Taxonomy Engine initialized (baseline mode)")
    
    def setup_dspy_components(self):
        """Initialize DSPy reasoning components."""
        try:
            # Configure DSPy with Gemini
            gemini_lm = dspy.Google(
                model=self.config.api.default_model,
                api_key=self.config.api.gemini_api_key
            )
            dspy.settings.configure(lm=gemini_lm)
            
            # Initialize reasoning modules
            self.blueprint_analyzer = dspy.ChainOfThought(BlueprintAnalysisSignature)
            self.element_detector = dspy.ReAct(ElementDetectionSignature)
            self.taxonomy_validator = dspy.MultiChainComparison(TaxonomyValidationSignature)
            
            logger.info("DSPy components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize DSPy components: {e}")
            # Fallback to direct Gemini usage
            self.blueprint_analyzer = None
            self.element_detector = None
            self.taxonomy_validator = None
    
    def load_taxonomy_knowledge(self) -> Dict[str, Any]:
        """Load architectural taxonomy knowledge base."""
        return {
            "page_types": {
                "floor_plan": {
                    "indicators": ["room labels", "walls", "doors", "windows", "furniture"],
                    "typical_elements": ["wall", "door", "window", "room"],
                    "scale_range": ["1/8\"=1'", "1/4\"=1'", "1/2\"=1'"]
                },
                "elevation": {
                    "indicators": ["height dimensions", "material callouts", "vertical view"],
                    "typical_elements": ["wall", "window", "door", "roof"],
                    "views": ["front", "rear", "side", "interior"]
                },
                "section": {
                    "indicators": ["cut line", "structural details", "vertical cut"],
                    "typical_elements": ["beam", "column", "foundation", "slab"],
                    "types": ["building section", "wall section", "detail section"]
                },
                "detail": {
                    "indicators": ["enlarged view", "connection detail", "assembly"],
                    "typical_elements": ["connection", "fastener", "joint"],
                    "scales": ["1\"=1'", "3\"=1'", "6\"=1'", "full size"]
                }
            },
            "structural_elements": {
                "primary": ["wall", "beam", "column", "foundation", "slab"],
                "secondary": ["door", "window", "stair", "elevator"],
                "annotations": ["dimension", "text_annotation", "symbol", "grid_line"]
            },
            "relationship_types": {
                "spatial": ["adjacent", "parallel", "perpendicular", "intersecting"],
                "structural": ["supported_by", "supports", "connected_to", "spans"],
                "functional": ["accesses", "serves", "contains", "divides"]
            }
        }
    
    async def generate_page_taxonomy(
        self, 
        page_data: Dict[str, Any], 
        document_context: Optional[Dict[str, Any]] = None
    ) -> PageTaxonomy:
        """
        Generate comprehensive taxonomy for a single page using GEPA-optimized prompts.
        
        Args:
            page_data: Processed page data from PageProcessor
            document_context: Context from other pages in document
            
        Returns:
            Complete PageTaxonomy object with GEPA enhancements
        """
        page_number = page_data["page_info"]["page_number"]
        
        if self.gepa_enabled:
            logger.info(f"Generating GEPA-optimized taxonomy for page {page_number}")
        else:
            logger.info(f"Generating baseline taxonomy for page {page_number}")
        
        start_time = time.time()
        
        try:
            # Use GEPA-optimized analysis if available
            if self.gepa_enabled:
                taxonomy = await self.gepa_analyzer.analyze_page_with_optimized_prompts(
                    page_data, document_context
                )
                
                # Add GEPA optimization metadata
                taxonomy.tools_used.append("gepa_optimized_prompts")
                
                logger.info(f"GEPA-optimized taxonomy generated for page {page_number}")
                
            else:
                # Fallback to standard analysis
                # 1. Prepare inputs for analysis
                inputs = self.prepare_analysis_inputs(page_data, document_context)
                
                # 2. Multi-modal analysis with Gemini
                gemini_analysis = await self.perform_gemini_analysis(page_data)
                
                # 3. DSPy reasoning (if available)
                if self.blueprint_analyzer:
                    dspy_analysis = await self.perform_dspy_analysis(inputs)
                else:
                    dspy_analysis = await self.perform_fallback_analysis(inputs)
                
                # 4. Synthesize results
                taxonomy = await self.synthesize_taxonomy(
                    page_data, gemini_analysis, dspy_analysis, document_context
                )
                
                # 5. Validate and refine
                taxonomy = await self.validate_taxonomy(taxonomy, document_context)
            
            processing_time = time.time() - start_time
            taxonomy.processing_time = processing_time
            
            return taxonomy
            
        except Exception as e:
            logger.error(f"Taxonomy generation failed for page {page_number}: {e}")
            
            # Return minimal taxonomy on failure
            return self.create_fallback_taxonomy(page_data, str(e))
    
    def prepare_analysis_inputs(
        self, 
        page_data: Dict[str, Any], 
        document_context: Optional[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Prepare structured inputs for analysis."""
        visual_features = page_data["visual_features"]
        textual_features = page_data["textual_features"]
        
        # Create comprehensive image description
        image_description = self.create_image_description(visual_features, textual_features)
        
        # Format context
        context_str = self.format_document_context(document_context)
        
        return {
            "page_image_description": image_description,
            "extracted_text": textual_features.extracted_text,
            "visual_features": json.dumps(visual_features.model_dump()),
            "context_from_previous_pages": context_str
        }
    
    def create_image_description(
        self, 
        visual_features: VisualFeatures, 
        textual_features: TextualFeatures
    ) -> str:
        """Create comprehensive image description for analysis."""
        description_parts = []
        
        # Visual characteristics
        description_parts.append(f"Drawing complexity: {visual_features.geometric_complexity:.2f}")
        description_parts.append(f"Line density: {visual_features.line_density:.2f}")
        description_parts.append(f"Symmetry score: {visual_features.symmetry_score:.2f}")
        
        if visual_features.dominant_colors:
            description_parts.append(f"Dominant colors: {', '.join(visual_features.dominant_colors[:3])}")
        
        if visual_features.drawing_style:
            description_parts.append(f"Drawing style: {visual_features.drawing_style}")
        
        # Text content indicators
        if textual_features.titles:
            description_parts.append(f"Titles found: {', '.join(textual_features.titles[:3])}")
        
        if textual_features.key_terms:
            description_parts.append(f"Technical terms: {', '.join(textual_features.key_terms[:5])}")
        
        if textual_features.measurements:
            description_parts.append(f"Measurements: {', '.join(textual_features.measurements[:3])}")
        
        return "; ".join(description_parts)
    
    def format_document_context(self, context: Optional[Dict[str, Any]]) -> str:
        """Format document context for analysis."""
        if not context:
            return "No previous context available"
        
        context_parts = []
        
        if "previous_page_types" in context:
            context_parts.append(f"Previous page types: {', '.join(context['previous_page_types'])}")
        
        if "building_type" in context:
            context_parts.append(f"Building type: {context['building_type']}")
        
        if "recurring_elements" in context:
            context_parts.append(f"Recurring elements: {', '.join(context['recurring_elements'][:5])}")
        
        if "document_patterns" in context:
            context_parts.append(f"Document patterns: {', '.join(context['document_patterns'][:3])}")
        
        return "; ".join(context_parts) if context_parts else "Minimal context available"
    
    async def perform_gemini_analysis(self, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform multi-modal analysis using Gemini Vision."""
        try:
            image_path = Path(page_data["image_info"]["image_path"])
            
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Upload image for analysis
            with open(image_path, 'rb') as image_file:
                uploaded_file = self.gemini_client.client.files.upload(
                    file=image_file,
                    config=types.UploadFileConfig(
                        mime_type='image/png',
                        display_name=f"page_{page_data['page_info']['page_number']}"
                    )
                )
            
            # Wait for processing
            self.gemini_client._wait_for_processing(uploaded_file.name)
            
            # Analyze with structured prompt
            analysis_prompt = self.create_gemini_analysis_prompt(page_data)
            
            response = self.gemini_client.client.models.generate_content(
                model=self.config.api.default_model,
                contents=[
                    types.Content(
                        role='user',
                        parts=[
                            types.Part.from_uri(
                                file_uri=uploaded_file.uri,
                                mime_type='image/png'
                            ),
                            types.Part.from_text(text=analysis_prompt)
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "object",
                        "properties": {
                            "page_classification": {"type": "string"},
                            "page_subtype": {"type": "string"},
                            "structural_elements": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {"type": "string"},
                                        "location": {"type": "string"},
                                        "description": {"type": "string"},
                                        "confidence": {"type": "number"}
                                    }
                                }
                            },
                            "spatial_relationships": {
                                "type": "array", 
                                "items": {"type": "string"}
                            },
                            "key_insights": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "confidence_score": {"type": "number"}
                        },
                        "required": ["page_classification", "structural_elements", "confidence_score"]
                    }
                )
            )
            
            # Cleanup uploaded file
            self.gemini_client.client.files.delete(name=uploaded_file.name)
            
            return json.loads(response.text)
            
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            return self.create_fallback_gemini_analysis(page_data)
    
    def create_gemini_analysis_prompt(self, page_data: Dict[str, Any]) -> str:
        """Create comprehensive analysis prompt for Gemini."""
        textual_features = page_data["textual_features"]
        visual_features = page_data["visual_features"]
        
        prompt = f"""
        Analyze this structural blueprint page with expert-level architectural understanding.
        
        CONTEXT:
        - This is page {page_data['page_info']['page_number']} of a structural blueprint document
        - Image quality score: {visual_features.quality_score:.2f}
        - Drawing complexity: {visual_features.geometric_complexity:.2f}
        - Text extracted: "{textual_features.processed_text[:200]}..."
        
        ANALYSIS REQUIREMENTS:
        1. CLASSIFY the page type (floor_plan, elevation, section, detail, site_plan, etc.)
        2. IDENTIFY structural elements visible in the drawing
        3. DETECT spatial relationships between elements
        4. EXTRACT technical specifications and standards
        5. ASSESS confidence in your analysis
        
        FOCUS AREAS:
        - Architectural elements (walls, doors, windows, rooms)
        - Structural components (beams, columns, foundations)
        - MEP elements (electrical, plumbing, HVAC)
        - Annotations and dimensions
        - Scale and measurement indicators
        
        EXPERTISE LEVEL: Provide analysis equivalent to a licensed architect or structural engineer.
        
        RESPONSE FORMAT: Structured JSON with detailed element descriptions and high confidence assessments.
        """
        
        return prompt
    
    async def perform_dspy_analysis(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        """Perform DSPy-optimized analysis."""
        try:
            # Primary analysis using ChainOfThought
            primary_result = self.blueprint_analyzer(**inputs)
            
            # Element detection using ReAct
            element_result = self.element_detector(
                page_type=primary_result.page_type,
                image_analysis=inputs["page_image_description"],
                text_content=inputs["extracted_text"]
            )
            
            # Combine results
            return {
                "page_type": primary_result.page_type,
                "page_subtype": primary_result.page_subtype,
                "structural_elements": json.loads(primary_result.structural_elements_detected),
                "spatial_relationships": json.loads(primary_result.spatial_relationships),
                "technical_specifications": primary_result.technical_specifications,
                "confidence": float(primary_result.confidence_assessment),
                "detected_elements": json.loads(element_result.detected_elements),
                "element_relationships": json.loads(element_result.element_relationships),
                "measurements": element_result.measurement_analysis
            }
            
        except Exception as e:
            logger.warning(f"DSPy analysis failed, using fallback: {e}")
            return await self.perform_fallback_analysis(inputs)
    
    async def perform_fallback_analysis(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        """Fallback analysis when DSPy is not available."""
        # Use rule-based classification
        page_type = self.classify_page_type_heuristic(inputs)
        
        # Extract elements using heuristics
        elements = self.detect_elements_heuristic(inputs, page_type)
        
        return {
            "page_type": page_type,
            "page_subtype": "standard",
            "structural_elements": elements,
            "spatial_relationships": [],
            "technical_specifications": "",
            "confidence": 0.7,  # Lower confidence for heuristic analysis
            "detected_elements": elements,
            "element_relationships": [],
            "measurements": inputs.get("extracted_text", "")
        }
    
    def classify_page_type_heuristic(self, inputs: Dict[str, str]) -> str:
        """Heuristic-based page type classification."""
        text = inputs.get("extracted_text", "").lower()
        image_desc = inputs.get("page_image_description", "").lower()
        
        # Classification rules
        if any(term in text for term in ["floor plan", "plan view", "room"]):
            return "floor_plan"
        elif any(term in text for term in ["elevation", "front", "rear", "side"]):
            return "elevation"
        elif any(term in text for term in ["section", "cut", "vertical"]):
            return "section"
        elif any(term in text for term in ["detail", "enlarged", "connection"]):
            return "detail"
        elif any(term in text for term in ["site", "plot", "landscape"]):
            return "site_plan"
        elif "complexity" in image_desc and "high" in image_desc:
            return "structural_plan"
        else:
            return "unknown"
    
    def detect_elements_heuristic(self, inputs: Dict[str, str], page_type: str) -> List[Dict[str, Any]]:
        """Heuristic-based element detection."""
        text = inputs.get("extracted_text", "").lower()
        elements = []
        
        # Element detection based on text content
        element_keywords = {
            "wall": ["wall", "partition"],
            "door": ["door", "entrance"],
            "window": ["window", "glazing"],
            "beam": ["beam", "girder"],
            "column": ["column", "post"],
            "room": ["room", "space", "area"]
        }
        
        for element_type, keywords in element_keywords.items():
            if any(keyword in text for keyword in keywords):
                elements.append({
                    "type": element_type,
                    "location": "detected in text",
                    "description": f"{element_type} mentioned in page text",
                    "confidence": 0.6
                })
        
        return elements
    
    async def synthesize_taxonomy(
        self,
        page_data: Dict[str, Any],
        gemini_analysis: Dict[str, Any],
        dspy_analysis: Dict[str, Any],
        document_context: Optional[Dict[str, Any]]
    ) -> PageTaxonomy:
        """Synthesize final taxonomy from multiple analysis sources."""
        
        page_number = page_data["page_info"]["page_number"]
        
        # Determine best classification
        page_type = self.determine_best_classification(gemini_analysis, dspy_analysis)
        
        # Synthesize structural elements
        structural_elements = self.synthesize_structural_elements(
            gemini_analysis, dspy_analysis, page_data
        )
        
        # Create spatial relationships
        spatial_relationships = self.create_spatial_relationships(structural_elements)
        
        # Determine confidence
        analysis_confidence = self.calculate_confidence(gemini_analysis, dspy_analysis)
        
        # Create taxonomy
        taxonomy = PageTaxonomy(
            page_number=page_number,
            page_type=BlueprintPageType(page_type),
            primary_category=page_type,
            secondary_category=gemini_analysis.get("page_subtype"),
            tertiary_category=None,
            
            structural_elements=structural_elements,
            spatial_relationships=spatial_relationships,
            
            visual_features=page_data["visual_features"],
            textual_features=page_data["textual_features"],
            
            purpose=self.determine_page_purpose(page_type, textual_features),
            complexity_level=self.assess_complexity_level(page_data["visual_features"]),
            technical_level=self.assess_technical_level(textual_features),
            
            analysis_confidence=analysis_confidence,
            completeness_score=self.calculate_completeness(structural_elements, page_data),
            
            processing_time=0.0,  # Will be set by caller
            tools_used=["gemini_vision", "dspy_reasoning", "ocr_engine"],
            analysis_timestamp=time.time()
        )
        
        return taxonomy
    
    def determine_best_classification(
        self, 
        gemini_analysis: Dict[str, Any], 
        dspy_analysis: Dict[str, Any]
    ) -> str:
        """Determine best page classification from multiple sources."""
        gemini_type = gemini_analysis.get("page_classification", "unknown")
        dspy_type = dspy_analysis.get("page_type", "unknown")
        
        # If both agree, use that
        if gemini_type == dspy_type:
            return gemini_type
        
        # Otherwise, use the one with higher confidence
        gemini_conf = gemini_analysis.get("confidence_score", 0.0)
        dspy_conf = dspy_analysis.get("confidence", 0.0)
        
        if gemini_conf > dspy_conf:
            return gemini_type
        else:
            return dspy_type
    
    def synthesize_structural_elements(
        self,
        gemini_analysis: Dict[str, Any],
        dspy_analysis: Dict[str, Any],
        page_data: Dict[str, Any]
    ) -> List[StructuralElement]:
        """Synthesize structural elements from multiple analysis sources."""
        elements = []
        element_id_counter = 1
        
        # Process Gemini detected elements
        gemini_elements = gemini_analysis.get("structural_elements", [])
        for elem in gemini_elements:
            if isinstance(elem, dict):
                element = self.create_structural_element(
                    elem, f"elem_{element_id_counter}", page_data["page_info"]["page_number"]
                )
                if element:
                    elements.append(element)
                    element_id_counter += 1
        
        # Process DSPy detected elements
        dspy_elements = dspy_analysis.get("detected_elements", [])
        for elem in dspy_elements:
            if isinstance(elem, dict):
                element = self.create_structural_element(
                    elem, f"elem_{element_id_counter}", page_data["page_info"]["page_number"]
                )
                if element:
                    elements.append(element)
                    element_id_counter += 1
        
        return elements
    
    def create_structural_element(
        self, 
        element_data: Dict[str, Any], 
        element_id: str, 
        page_number: int
    ) -> Optional[StructuralElement]:
        """Create StructuralElement from analysis data."""
        try:
            element_type_str = element_data.get("type", "unknown").lower()
            
            # Map to enum
            element_type = StructuralElementType.UNKNOWN
            for enum_type in StructuralElementType:
                if enum_type.value in element_type_str or element_type_str in enum_type.value:
                    element_type = enum_type
                    break
            
            # Create bounding box (placeholder for now)
            location = BoundingBox(
                top_left=Coordinates(x=0, y=0),
                bottom_right=Coordinates(x=100, y=100)
            )
            
            return StructuralElement(
                element_id=element_id,
                element_type=element_type,
                element_subtype=element_data.get("subtype"),
                location=location,
                label=element_data.get("description", ""),
                description=element_data.get("description", ""),
                confidence=float(element_data.get("confidence", 0.5)),
                detection_method="multi_modal_analysis",
                page_number=page_number
            )
            
        except Exception as e:
            logger.warning(f"Failed to create structural element: {e}")
            return None
    
    def create_spatial_relationships(
        self, 
        elements: List[StructuralElement]
    ) -> List[SpatialRelationship]:
        """Create spatial relationships between elements."""
        relationships = []
        
        # Simple proximity-based relationships for now
        for i, elem1 in enumerate(elements):
            for j, elem2 in enumerate(elements[i+1:], i+1):
                # Calculate distance between elements
                distance = self.calculate_element_distance(elem1, elem2)
                
                if distance < 200:  # Arbitrary proximity threshold
                    relationship = SpatialRelationship(
                        relationship_id=f"rel_{i}_{j}",
                        element_1_id=elem1.element_id,
                        element_2_id=elem2.element_id,
                        relationship_type="adjacent",
                        distance=distance,
                        description=f"{elem1.element_type.value} adjacent to {elem2.element_type.value}",
                        confidence=0.7
                    )
                    relationships.append(relationship)
        
        return relationships
    
    def calculate_element_distance(self, elem1: StructuralElement, elem2: StructuralElement) -> float:
        """Calculate distance between two elements."""
        center1 = elem1.location.center
        center2 = elem2.location.center
        
        return ((center1.x - center2.x)**2 + (center1.y - center2.y)**2)**0.5
    
    async def validate_taxonomy(
        self, 
        taxonomy: PageTaxonomy, 
        document_context: Optional[Dict[str, Any]]
    ) -> PageTaxonomy:
        """Validate and refine taxonomy using cross-page context."""
        
        if self.taxonomy_validator and document_context:
            try:
                # Prepare validation inputs
                initial_taxonomy = json.dumps(taxonomy.model_dump(), default=str)
                context_str = json.dumps(document_context, default=str)
                domain_knowledge = json.dumps(self.taxonomy_knowledge, default=str)
                
                # Validate using DSPy
                validation_result = self.taxonomy_validator(
                    initial_taxonomy=initial_taxonomy,
                    cross_page_context=context_str,
                    domain_knowledge=domain_knowledge
                )
                
                # Apply validation results
                validated_data = json.loads(validation_result.validated_taxonomy)
                taxonomy.analysis_confidence = float(validation_result.confidence_score)
                
                logger.info(f"Taxonomy validated for page {taxonomy.page_number}")
                
            except Exception as e:
                logger.warning(f"Taxonomy validation failed: {e}")
        
        return taxonomy
    
    def determine_page_purpose(self, page_type: str, textual_features: TextualFeatures) -> Optional[str]:
        """Determine the purpose and intent of the page."""
        purpose_map = {
            "floor_plan": "Show spatial layout and room arrangements",
            "elevation": "Display vertical building appearance",
            "section": "Reveal internal structural details",
            "detail": "Provide specific construction information",
            "site_plan": "Show building placement and site context"
        }
        
        base_purpose = purpose_map.get(page_type, "Provide technical information")
        
        # Enhance with specific content
        if textual_features.specifications:
            base_purpose += " with technical specifications"
        
        if textual_features.measurements:
            base_purpose += " including dimensional information"
        
        return base_purpose
    
    def assess_complexity_level(self, visual_features: VisualFeatures) -> str:
        """Assess complexity level of the page."""
        complexity = visual_features.geometric_complexity
        line_density = visual_features.line_density
        
        if complexity > 0.8 or line_density > 150:
            return "high"
        elif complexity > 0.5 or line_density > 75:
            return "medium"
        else:
            return "low"
    
    def assess_technical_level(self, textual_features: TextualFeatures) -> str:
        """Assess technical detail level."""
        spec_count = len(textual_features.specifications)
        measurement_count = len(textual_features.measurements)
        term_count = len(textual_features.key_terms)
        
        total_technical_content = spec_count + measurement_count + term_count
        
        if total_technical_content > 15:
            return "advanced"
        elif total_technical_content > 8:
            return "standard"
        else:
            return "basic"
    
    def calculate_confidence(
        self, 
        gemini_analysis: Dict[str, Any], 
        dspy_analysis: Dict[str, Any]
    ) -> float:
        """Calculate overall analysis confidence."""
        gemini_conf = gemini_analysis.get("confidence_score", 0.5)
        dspy_conf = dspy_analysis.get("confidence", 0.5)
        
        # Weight Gemini higher for vision tasks
        weighted_confidence = (gemini_conf * 0.6 + dspy_conf * 0.4)
        
        return round(weighted_confidence, 3)
    
    def calculate_completeness(
        self, 
        elements: List[StructuralElement], 
        page_data: Dict[str, Any]
    ) -> float:
        """Calculate analysis completeness score."""
        # Base completeness on element detection
        element_count = len(elements)
        
        # Expected elements based on page characteristics
        has_text = len(page_data["textual_features"].processed_text) > 10
        has_complex_drawing = page_data["visual_features"].geometric_complexity > 0.5
        
        expected_elements = 5  # Base expectation
        if has_text:
            expected_elements += 3
        if has_complex_drawing:
            expected_elements += 5
        
        completeness = min(element_count / expected_elements, 1.0)
        
        return round(completeness, 3)
    
    def create_fallback_taxonomy(self, page_data: Dict[str, Any], error_msg: str) -> PageTaxonomy:
        """Create minimal taxonomy when analysis fails."""
        page_number = page_data["page_info"]["page_number"]
        
        return PageTaxonomy(
            page_number=page_number,
            page_type=BlueprintPageType.UNKNOWN,
            primary_category="unknown",
            secondary_category=None,
            tertiary_category=None,
            
            structural_elements=[],
            spatial_relationships=[],
            
            visual_features=page_data.get("visual_features", VisualFeatures(
                dominant_colors=[], line_density=0.0, text_regions=[],
                geometric_complexity=0.0, symmetry_score=0.0, scale_indicators=[],
                quality_score=0.0
            )),
            textual_features=page_data.get("textual_features", TextualFeatures(
                extracted_text="", processed_text="", key_terms=[],
                measurements=[], specifications=[], titles=[],
                labels=[], notes=[], language="en", text_quality=0.0
            )),
            
            purpose="Analysis failed",
            complexity_level="unknown",
            technical_level="unknown",
            
            analysis_confidence=0.1,
            completeness_score=0.0,
            
            processing_time=0.0,
            tools_used=["fallback"],
            analysis_timestamp=time.time()
        )
    
    def create_fallback_gemini_analysis(self, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback analysis when Gemini fails."""
        return {
            "page_classification": "unknown",
            "page_subtype": "unknown",
            "structural_elements": [],
            "spatial_relationships": [],
            "key_insights": [],
            "confidence_score": 0.1
        }
