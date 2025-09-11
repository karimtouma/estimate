"""
Gemini multimodal service for structural blueprint analysis.

This module provides advanced multimodal analysis using Gemini 2.5 Flash
native capabilities for vision, OCR, and text understanding without external dependencies.
"""

import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import base64

import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image
from google.genai import types

from ..models.blueprint_schemas import (
    PageTaxonomy, VisualFeatures, TextualFeatures, BlueprintPageType,
    StructuralElement, StructuralElementType, BoundingBox, Coordinates
)
from ..services.gemini_client import GeminiClient
from ..core.config import Config
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class GeminiMultimodalProcessor:
    """
    Advanced multimodal processor using Gemini 2.5 Flash native capabilities.
    
    Leverages Gemini's built-in vision, OCR, and reasoning capabilities
    for comprehensive structural blueprint analysis.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.gemini_client = GeminiClient(config)
        self.temp_dir = Path(config.get_directories()['temp'])
        self.temp_dir.mkdir(exist_ok=True)
        
        # Image processing settings
        self.image_dpi = config.processing.__dict__.get('image_dpi', 300)
        self.enhancement_enabled = config.processing.__dict__.get('image_enhancement', True)
        
        logger.info("Gemini Multimodal Processor initialized")
    
    async def process_document_pages(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Process all pages using Gemini multimodal capabilities.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of processed page data with Gemini analysis
        """
        logger.info(f"Processing document with Gemini multimodal: {pdf_path}")
        start_time = time.time()
        
        try:
            # 1. Extract basic PDF information
            pdf_info = self.extract_pdf_info(pdf_path)
            
            # 2. Convert pages to high-quality images
            page_images = self.convert_pages_to_images(pdf_path)
            
            # 3. Process each page with Gemini multimodal
            processed_pages = []
            
            for i, (page_info, image_path) in enumerate(zip(pdf_info, page_images)):
                logger.info(f"Processing page {i+1}/{len(pdf_info)} with Gemini")
                
                # Analyze page with Gemini multimodal
                page_analysis = await self.analyze_page_with_gemini(
                    image_path, page_info, i + 1
                )
                
                processed_pages.append(page_analysis)
            
            processing_time = time.time() - start_time
            logger.info(f"Document processing completed in {processing_time:.2f} seconds")
            
            return processed_pages
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            raise
    
    def extract_pdf_info(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract basic PDF page information."""
        try:
            pdf_document = fitz.open(str(pdf_path))
            pages_info = []
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                page_info = {
                    "page_number": page_num + 1,
                    "page_rect": {
                        "width": page.rect.width,
                        "height": page.rect.height
                    },
                    "rotation": page.rotation,
                    "has_text": bool(page.get_text().strip()),
                    "has_images": len(page.get_images()) > 0,
                    "has_drawings": len(page.get_drawings()) > 0,
                    "text_length": len(page.get_text()),
                    "basic_text": page.get_text()[:500]  # First 500 chars for context
                }
                
                pages_info.append(page_info)
            
            pdf_document.close()
            return pages_info
            
        except Exception as e:
            logger.error(f"PDF info extraction failed: {e}")
            raise
    
    def convert_pages_to_images(self, pdf_path: Path) -> List[Path]:
        """Convert PDF pages to high-quality images for Gemini analysis."""
        try:
            # Convert with high DPI for better analysis
            images = convert_from_path(
                pdf_path,
                dpi=self.image_dpi,
                fmt='PNG'
            )
            
            image_paths = []
            
            for i, image in enumerate(images):
                # Enhance image if enabled
                if self.enhancement_enabled:
                    image = self.enhance_image_simple(image)
                
                # Save image
                image_path = self.temp_dir / f"page_{i+1:03d}.png"
                image.save(image_path, "PNG", quality=95)
                image_paths.append(image_path)
            
            logger.info(f"Converted {len(images)} pages to images")
            return image_paths
            
        except Exception as e:
            logger.error(f"Image conversion failed: {e}")
            raise
    
    def enhance_image_simple(self, image: Image.Image) -> Image.Image:
        """Simple image enhancement using PIL only."""
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Simple contrast enhancement for better text visibility
            from PIL import ImageEnhance
            
            # Enhance contrast slightly
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.1)
            
            # Enhance sharpness for text clarity
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.05)
            
            return image
            
        except Exception as e:
            logger.warning(f"Image enhancement failed, using original: {e}")
            return image
    
    async def analyze_page_with_gemini(
        self, 
        image_path: Path, 
        page_info: Dict[str, Any], 
        page_number: int
    ) -> Dict[str, Any]:
        """
        Comprehensive page analysis using Gemini 2.5 Flash multimodal capabilities.
        
        Args:
            image_path: Path to page image
            page_info: Basic page information
            page_number: Page number
            
        Returns:
            Complete page analysis using Gemini's native capabilities
        """
        logger.info(f"Analyzing page {page_number} with Gemini multimodal")
        
        try:
            # Upload image to Gemini
            with open(image_path, 'rb') as image_file:
                uploaded_file = self.gemini_client.client.files.upload(
                    file=image_file,
                    config=types.UploadFileConfig(
                        mime_type='image/png',
                        display_name=f"blueprint_page_{page_number}"
                    )
                )
            
            # Wait for processing
            self.gemini_client._wait_for_processing(uploaded_file.name)
            
            # Create comprehensive analysis prompt
            analysis_prompt = self.create_multimodal_analysis_prompt(page_info)
            
            # Execute multimodal analysis
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
                    response_schema=self.get_multimodal_analysis_schema()
                )
            )
            
            # Parse Gemini analysis
            gemini_analysis = json.loads(response.text)
            
            # Cleanup uploaded file
            self.gemini_client.client.files.delete(name=uploaded_file.name)
            
            # Create structured page analysis
            page_analysis = self.create_structured_page_analysis(
                gemini_analysis, page_info, page_number, str(image_path)
            )
            
            logger.info(f"Page {page_number} analysis completed with Gemini")
            
            return page_analysis
            
        except Exception as e:
            logger.error(f"Gemini multimodal analysis failed for page {page_number}: {e}")
            return self.create_fallback_page_analysis(page_info, page_number, str(e))
    
    def create_multimodal_analysis_prompt(self, page_info: Dict[str, Any]) -> str:
        """Create comprehensive multimodal analysis prompt for Gemini."""
        
        prompt = f"""
        Analyze this structural blueprint page with expert-level architectural understanding using your multimodal capabilities.
        
        CONTEXT INFORMATION:
        - Page dimensions: {page_info['page_rect']['width']} x {page_info['page_rect']['height']}
        - Has text content: {page_info['has_text']}
        - Has images: {page_info['has_images']}
        - Has technical drawings: {page_info['has_drawings']}
        - Basic text preview: "{page_info.get('basic_text', '')[:200]}..."
        
        COMPREHENSIVE ANALYSIS REQUIREMENTS:
        
        1. PAGE CLASSIFICATION:
           - Determine page type: floor_plan, elevation, section, detail, site_plan, structural_plan, mechanical_plan, electrical_plan, title_block, legend, notes, schedule, specifications
           - Identify building type: residential, commercial, industrial, institutional
           - Assess complexity level: low, medium, high
           - Determine technical level: basic, standard, advanced
        
        2. VISUAL ANALYSIS (Use your vision capabilities):
           - Analyze line density and geometric complexity
           - Identify dominant visual patterns and symmetries
           - Detect drawing style and quality
           - Assess image clarity and technical drawing standards
        
        3. TEXT EXTRACTION AND ANALYSIS (Use your OCR capabilities):
           - Extract ALL visible text from the image
           - Identify titles, labels, and annotations
           - Extract measurements, dimensions, and scales
           - Find technical specifications and material callouts
           - Detect room names, element labels, and notes
        
        4. STRUCTURAL ELEMENT DETECTION (Use multimodal understanding):
           - Identify walls, doors, windows, rooms (architectural elements)
           - Detect beams, columns, foundations, slabs (structural elements)
           - Find electrical outlets, fixtures, HVAC elements (MEP elements)
           - Locate dimensions, symbols, grid lines (annotation elements)
           - For each element, provide location description and confidence
        
        5. SPATIAL RELATIONSHIP ANALYSIS:
           - Identify how elements connect and relate spatially
           - Determine structural load paths and dependencies
           - Map functional relationships (access, containment, support)
           - Analyze alignment patterns and modular arrangements
        
        6. TECHNICAL SPECIFICATIONS:
           - Extract material specifications and grades
           - Identify construction methods and standards
           - Find code references and compliance notes
           - Extract dimensional information and tolerances
        
        EXPERTISE LEVEL: Provide analysis equivalent to a licensed architect with structural engineering knowledge.
        
        USE YOUR MULTIMODAL CAPABILITIES: Combine visual understanding with text extraction for comprehensive analysis.
        """
        
        return prompt
    
    def get_multimodal_analysis_schema(self) -> Dict[str, Any]:
        """Get JSON schema for Gemini multimodal analysis response."""
        return {
            "type": "object",
            "properties": {
                "page_classification": {
                    "type": "object",
                    "properties": {
                        "primary_type": {"type": "string"},
                        "secondary_type": {"type": "string"},
                        "building_type": {"type": "string"},
                        "complexity_level": {"type": "string"},
                        "technical_level": {"type": "string"},
                        "confidence": {"type": "number"}
                    },
                    "required": ["primary_type", "confidence"]
                },
                "visual_analysis": {
                    "type": "object",
                    "properties": {
                        "line_density": {"type": "string"},
                        "geometric_complexity": {"type": "string"},
                        "symmetry_assessment": {"type": "string"},
                        "drawing_style": {"type": "string"},
                        "image_quality": {"type": "string"},
                        "dominant_features": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "text_extraction": {
                    "type": "object",
                    "properties": {
                        "extracted_text": {"type": "string"},
                        "titles_and_headers": {"type": "array", "items": {"type": "string"}},
                        "room_labels": {"type": "array", "items": {"type": "string"}},
                        "measurements": {"type": "array", "items": {"type": "string"}},
                        "specifications": {"type": "array", "items": {"type": "string"}},
                        "notes_and_annotations": {"type": "array", "items": {"type": "string"}},
                        "scale_indicators": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "structural_elements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "element_type": {"type": "string"},
                            "element_subtype": {"type": "string"},
                            "location_description": {"type": "string"},
                            "description": {"type": "string"},
                            "confidence": {"type": "number"},
                            "material": {"type": "string"},
                            "dimensions": {"type": "string"}
                        },
                        "required": ["element_type", "description", "confidence"]
                    }
                },
                "spatial_relationships": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "relationship_type": {"type": "string"},
                            "description": {"type": "string"},
                            "elements_involved": {"type": "array", "items": {"type": "string"}},
                            "confidence": {"type": "number"}
                        }
                    }
                },
                "technical_analysis": {
                    "type": "object",
                    "properties": {
                        "construction_method": {"type": "string"},
                        "material_specifications": {"type": "array", "items": {"type": "string"}},
                        "code_references": {"type": "array", "items": {"type": "string"}},
                        "compliance_notes": {"type": "array", "items": {"type": "string"}},
                        "dimensional_system": {"type": "string"}
                    }
                },
                "analysis_confidence": {
                    "type": "object",
                    "properties": {
                        "overall_confidence": {"type": "number"},
                        "classification_confidence": {"type": "number"},
                        "element_detection_confidence": {"type": "number"},
                        "text_extraction_confidence": {"type": "number"},
                        "reasoning": {"type": "string"}
                    }
                }
            },
            "required": ["page_classification", "visual_analysis", "text_extraction", "structural_elements"]
        }
    
    async def analyze_single_page(self, pdf_path: Path, page_number: int) -> Dict[str, Any]:
        """
        Analyze a single page using Gemini multimodal.
        
        Args:
            pdf_path: Path to PDF file
            page_number: Page number to analyze (1-based)
            
        Returns:
            Complete page analysis
        """
        logger.info(f"Analyzing single page {page_number} with Gemini multimodal")
        
        try:
            # Extract specific page
            page_images = convert_from_path(
                pdf_path,
                dpi=self.image_dpi,
                first_page=page_number - 1,
                last_page=page_number - 1,
                fmt='PNG'
            )
            
            if not page_images:
                raise ValueError(f"Failed to extract page {page_number}")
            
            # Save page image
            image = page_images[0]
            if self.enhancement_enabled:
                image = self.enhance_image_simple(image)
            
            image_path = self.temp_dir / f"page_{page_number:03d}.png"
            image.save(image_path, "PNG", quality=95)
            
            # Get page info
            pdf_info = self.extract_pdf_info(pdf_path)
            page_info = pdf_info[page_number - 1] if page_number <= len(pdf_info) else {}
            
            # Analyze with Gemini
            analysis = await self.analyze_page_with_gemini(image_path, page_info, page_number)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Single page analysis failed: {e}")
            raise
    
    def enhance_image_simple(self, image: Image.Image) -> Image.Image:
        """Simple image enhancement using PIL only."""
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Simple enhancements for better text visibility
            from PIL import ImageEnhance
            
            # Enhance contrast slightly for better line definition
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.1)
            
            # Enhance sharpness for text clarity
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.05)
            
            return image
            
        except Exception as e:
            logger.warning(f"Image enhancement failed, using original: {e}")
            return image
    
    def create_structured_page_analysis(
        self, 
        gemini_analysis: Dict[str, Any], 
        page_info: Dict[str, Any], 
        page_number: int,
        image_path: str
    ) -> Dict[str, Any]:
        """Create structured page analysis from Gemini response."""
        
        try:
            # Extract classification
            classification = gemini_analysis.get("page_classification", {})
            
            # Create visual features from Gemini analysis
            visual_analysis = gemini_analysis.get("visual_analysis", {})
            visual_features = VisualFeatures(
                dominant_colors=["#000000", "#ffffff"],  # Default for technical drawings
                line_density=self.parse_density_description(visual_analysis.get("line_density", "medium")),
                text_regions=[],  # Gemini handles text detection internally
                geometric_complexity=self.parse_complexity_description(visual_analysis.get("geometric_complexity", "medium")),
                symmetry_score=self.parse_symmetry_description(visual_analysis.get("symmetry_assessment", "moderate")),
                scale_indicators=gemini_analysis.get("text_extraction", {}).get("scale_indicators", []),
                drawing_style=visual_analysis.get("drawing_style", "technical"),
                quality_score=self.parse_quality_description(visual_analysis.get("image_quality", "good"))
            )
            
            # Create textual features from Gemini extraction
            text_extraction = gemini_analysis.get("text_extraction", {})
            textual_features = TextualFeatures(
                extracted_text=text_extraction.get("extracted_text", ""),
                processed_text=text_extraction.get("extracted_text", ""),
                key_terms=self.extract_technical_terms_from_text(text_extraction.get("extracted_text", "")),
                measurements=text_extraction.get("measurements", []),
                specifications=text_extraction.get("specifications", []),
                titles=text_extraction.get("titles_and_headers", []),
                labels=text_extraction.get("room_labels", []),
                notes=text_extraction.get("notes_and_annotations", []),
                language="en",
                text_quality=0.9  # High quality from Gemini OCR
            )
            
            # Create structural elements from Gemini detection
            structural_elements = []
            gemini_elements = gemini_analysis.get("structural_elements", [])
            
            for i, elem_data in enumerate(gemini_elements):
                element = self.create_structural_element_from_gemini(
                    elem_data, f"elem_{page_number}_{i+1}", page_number
                )
                if element:
                    structural_elements.append(element)
            
            # Create page taxonomy
            page_type_str = classification.get("primary_type", "unknown").lower()
            page_type = BlueprintPageType.UNKNOWN
            
            # Map Gemini classification to our enum
            for enum_type in BlueprintPageType:
                if enum_type.value in page_type_str or page_type_str in enum_type.value:
                    page_type = enum_type
                    break
            
            # Create complete page analysis
            page_analysis = {
                "page_info": page_info,
                "image_info": {
                    "image_path": image_path,
                    "dpi": self.image_dpi,
                    "enhancement_applied": self.enhancement_enabled
                },
                "visual_features": visual_features,
                "textual_features": textual_features,
                "structural_elements": structural_elements,
                "gemini_analysis": gemini_analysis,  # Keep raw Gemini response
                "page_type": page_type,
                "classification_confidence": classification.get("confidence", 0.8),
                "processing_timestamp": time.time()
            }
            
            return page_analysis
            
        except Exception as e:
            logger.error(f"Failed to create structured analysis: {e}")
            return self.create_fallback_page_analysis(page_info, page_number, str(e))
    
    def parse_density_description(self, description: str) -> float:
        """Parse line density from Gemini description."""
        description = description.lower()
        if "high" in description or "dense" in description:
            return 150.0
        elif "medium" in description or "moderate" in description:
            return 75.0
        elif "low" in description or "sparse" in description:
            return 25.0
        else:
            return 50.0
    
    def parse_complexity_description(self, description: str) -> float:
        """Parse geometric complexity from Gemini description."""
        description = description.lower()
        if "high" in description or "complex" in description:
            return 0.8
        elif "medium" in description or "moderate" in description:
            return 0.5
        elif "low" in description or "simple" in description:
            return 0.2
        else:
            return 0.5
    
    def parse_symmetry_description(self, description: str) -> float:
        """Parse symmetry assessment from Gemini description."""
        description = description.lower()
        if "highly symmetric" in description or "very symmetric" in description:
            return 0.9
        elif "symmetric" in description or "balanced" in description:
            return 0.7
        elif "somewhat symmetric" in description or "partially" in description:
            return 0.4
        elif "asymmetric" in description or "irregular" in description:
            return 0.1
        else:
            return 0.5
    
    def parse_quality_description(self, description: str) -> float:
        """Parse image quality from Gemini description."""
        description = description.lower()
        if "excellent" in description or "very high" in description:
            return 0.95
        elif "good" in description or "high" in description:
            return 0.8
        elif "fair" in description or "medium" in description:
            return 0.6
        elif "poor" in description or "low" in description:
            return 0.3
        else:
            return 0.7
    
    def extract_technical_terms_from_text(self, text: str) -> List[str]:
        """Extract technical terms from Gemini-extracted text."""
        if not text:
            return []
        
        technical_terms = [
            "beam", "column", "wall", "foundation", "slab", "footing",
            "door", "window", "stair", "elevator", "room", "bathroom",
            "kitchen", "living", "bedroom", "office", "garage",
            "concrete", "steel", "wood", "brick", "stone", "glass",
            "scale", "dimension", "elevation", "section", "detail",
            "north", "south", "east", "west", "arrow", "grid"
        ]
        
        found_terms = []
        text_lower = text.lower()
        
        for term in technical_terms:
            if term in text_lower:
                found_terms.append(term)
        
        return found_terms
    
    def create_structural_element_from_gemini(
        self, 
        elem_data: Dict[str, Any], 
        element_id: str, 
        page_number: int
    ) -> Optional[StructuralElement]:
        """Create StructuralElement from Gemini analysis."""
        try:
            element_type_str = elem_data.get("element_type", "unknown").lower()
            
            # Map to enum
            element_type = StructuralElementType.UNKNOWN
            for enum_type in StructuralElementType:
                if enum_type.value in element_type_str or element_type_str in enum_type.value:
                    element_type = enum_type
                    break
            
            # Create placeholder bounding box (Gemini provides location description)
            location = BoundingBox(
                top_left=Coordinates(x=0, y=0),
                bottom_right=Coordinates(x=100, y=100)
            )
            
            return StructuralElement(
                element_id=element_id,
                element_type=element_type,
                element_subtype=elem_data.get("element_subtype"),
                location=location,
                label=elem_data.get("location_description", ""),
                description=elem_data.get("description", ""),
                annotations=[elem_data.get("location_description", "")],
                confidence=float(elem_data.get("confidence", 0.7)),
                detection_method="gemini_multimodal",
                page_number=page_number
            )
            
        except Exception as e:
            logger.warning(f"Failed to create structural element: {e}")
            return None
    
    def create_fallback_page_analysis(
        self, 
        page_info: Dict[str, Any], 
        page_number: int, 
        error_msg: str
    ) -> Dict[str, Any]:
        """Create minimal page analysis when Gemini analysis fails."""
        
        return {
            "page_info": page_info,
            "image_info": {"error": error_msg},
            "visual_features": VisualFeatures(
                dominant_colors=["#000000"],
                line_density=0.0,
                text_regions=[],
                geometric_complexity=0.0,
                symmetry_score=0.0,
                scale_indicators=[],
                quality_score=0.0
            ),
            "textual_features": TextualFeatures(
                extracted_text=page_info.get("basic_text", ""),
                processed_text=page_info.get("basic_text", ""),
                key_terms=[],
                measurements=[],
                specifications=[],
                titles=[],
                labels=[],
                notes=[],
                language="en",
                text_quality=0.1
            ),
            "structural_elements": [],
            "gemini_analysis": {"error": error_msg},
            "page_type": BlueprintPageType.UNKNOWN,
            "classification_confidence": 0.1,
            "processing_timestamp": time.time()
        }


class GeminiTaxonomyEngine:
    """
    Simplified taxonomy engine using pure Gemini multimodal capabilities.
    """
    
    def __init__(self, config: Config, optimization_results: Optional[Dict[str, Any]] = None):
        self.config = config
        self.gemini_processor = GeminiMultimodalProcessor(config)
        self.optimization_results = optimization_results
        self.gepa_enabled = optimization_results is not None
        
        if self.gepa_enabled:
            logger.info("Gemini Taxonomy Engine initialized with GEPA optimization")
        else:
            logger.info("Gemini Taxonomy Engine initialized (baseline mode)")
    
    async def generate_page_taxonomy(
        self, 
        page_data: Dict[str, Any], 
        document_context: Optional[Dict[str, Any]] = None
    ) -> PageTaxonomy:
        """
        Generate taxonomy using Gemini's native multimodal capabilities.
        
        Args:
            page_data: Page data (can be just PDF path and page number)
            document_context: Context from other pages
            
        Returns:
            Complete PageTaxonomy using Gemini analysis
        """
        start_time = time.time()
        
        # If page_data contains Gemini analysis, use it directly
        if "gemini_analysis" in page_data:
            gemini_analysis = page_data["gemini_analysis"]
            page_number = page_data["page_info"]["page_number"]
        else:
            # Need to run Gemini analysis
            pdf_path = page_data.get("pdf_path")
            page_number = page_data.get("page_number", 1)
            
            if not pdf_path:
                raise ValueError("PDF path required for analysis")
            
            analysis = await self.gemini_processor.analyze_single_page(Path(pdf_path), page_number)
            gemini_analysis = analysis.get("gemini_analysis", {})
            page_data = analysis
        
        try:
            # Create taxonomy from Gemini analysis
            taxonomy = self.create_taxonomy_from_gemini(
                gemini_analysis, page_data, document_context
            )
            
            # Apply GEPA enhancements if available
            if self.gepa_enabled:
                taxonomy = self.apply_gepa_enhancements(taxonomy)
            
            processing_time = time.time() - start_time
            taxonomy.processing_time = processing_time
            
            return taxonomy
            
        except Exception as e:
            logger.error(f"Taxonomy generation failed: {e}")
            return self.create_fallback_taxonomy(page_number, str(e))
    
    def create_taxonomy_from_gemini(
        self,
        gemini_analysis: Dict[str, Any],
        page_data: Dict[str, Any],
        document_context: Optional[Dict[str, Any]]
    ) -> PageTaxonomy:
        """Create PageTaxonomy from Gemini multimodal analysis."""
        
        page_number = page_data["page_info"]["page_number"]
        
        # Extract classification
        classification = gemini_analysis.get("page_classification", {})
        page_type_str = classification.get("primary_type", "unknown").lower()
        
        # Map to enum
        page_type = BlueprintPageType.UNKNOWN
        for enum_type in BlueprintPageType:
            if enum_type.value in page_type_str or page_type_str in enum_type.value:
                page_type = enum_type
                break
        
        # Get confidence
        confidence_data = gemini_analysis.get("analysis_confidence", {})
        overall_confidence = confidence_data.get("overall_confidence", classification.get("confidence", 0.8))
        
        # Create taxonomy
        taxonomy = PageTaxonomy(
            page_number=page_number,
            page_type=page_type,
            primary_category=classification.get("primary_type", "unknown"),
            secondary_category=classification.get("secondary_type"),
            tertiary_category=classification.get("building_type"),
            
            structural_elements=page_data.get("structural_elements", []),
            spatial_relationships=[],  # Will be populated from Gemini relationships
            
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
            
            purpose=self.determine_page_purpose(page_type, classification),
            complexity_level=classification.get("complexity_level", "medium"),
            technical_level=classification.get("technical_level", "standard"),
            
            analysis_confidence=float(overall_confidence),
            completeness_score=self.calculate_completeness_from_gemini(gemini_analysis),
            
            processing_time=0.0,  # Will be set by caller
            tools_used=["gemini_multimodal", "native_ocr", "vision_analysis"],
            analysis_timestamp=time.time()
        )
        
        return taxonomy
    
    def apply_gepa_enhancements(self, taxonomy: PageTaxonomy) -> PageTaxonomy:
        """Apply GEPA optimization enhancements."""
        if not self.gepa_enabled:
            return taxonomy
        
        improvement = self.optimization_results.get("improvement", 0.0)
        
        # Enhance confidence scores based on GEPA learning
        enhanced_confidence = min(0.98, taxonomy.analysis_confidence + improvement)
        taxonomy.analysis_confidence = enhanced_confidence
        
        # Enhance completeness
        enhanced_completeness = min(0.98, taxonomy.completeness_score + improvement * 0.8)
        taxonomy.completeness_score = enhanced_completeness
        
        # Add GEPA metadata
        taxonomy.tools_used.append("gepa_optimized")
        
        return taxonomy
    
    def determine_page_purpose(self, page_type: BlueprintPageType, classification: Dict[str, Any]) -> str:
        """Determine page purpose from classification."""
        purpose_map = {
            BlueprintPageType.FLOOR_PLAN: "Show spatial layout and room arrangements",
            BlueprintPageType.ELEVATION: "Display vertical building appearance and materials",
            BlueprintPageType.SECTION: "Reveal internal structural details and systems",
            BlueprintPageType.DETAIL: "Provide specific construction and connection information",
            BlueprintPageType.SITE_PLAN: "Show building placement and site context",
            BlueprintPageType.STRUCTURAL_PLAN: "Detail structural system and load paths"
        }
        
        base_purpose = purpose_map.get(page_type, "Provide technical information")
        
        # Enhance with building type if available
        building_type = classification.get("building_type")
        if building_type:
            base_purpose += f" for {building_type} construction"
        
        return base_purpose
    
    def calculate_completeness_from_gemini(self, gemini_analysis: Dict[str, Any]) -> float:
        """Calculate completeness score from Gemini analysis quality."""
        
        # Base completeness on content richness
        text_extraction = gemini_analysis.get("text_extraction", {})
        structural_elements = gemini_analysis.get("structural_elements", [])
        
        completeness = 0.0
        
        # Text content completeness (30%)
        if text_extraction.get("extracted_text"):
            completeness += 0.3
        
        # Element detection completeness (40%)
        if structural_elements:
            element_score = min(len(structural_elements) / 5.0, 1.0)  # Expect ~5 elements per page
            completeness += 0.4 * element_score
        
        # Analysis depth completeness (30%)
        analysis_sections = ["visual_analysis", "technical_analysis", "spatial_relationships"]
        filled_sections = sum(1 for section in analysis_sections if section in gemini_analysis)
        completeness += 0.3 * (filled_sections / len(analysis_sections))
        
        return min(completeness, 0.98)
    
    def create_fallback_taxonomy(self, page_number: int, error_msg: str) -> PageTaxonomy:
        """Create fallback taxonomy when analysis fails."""
        
        return PageTaxonomy(
            page_number=page_number,
            page_type=BlueprintPageType.UNKNOWN,
            primary_category="unknown",
            secondary_category=None,
            tertiary_category=None,
            
            structural_elements=[],
            spatial_relationships=[],
            
            visual_features=VisualFeatures(
                dominant_colors=[], line_density=0.0, text_regions=[],
                geometric_complexity=0.0, symmetry_score=0.0, scale_indicators=[],
                quality_score=0.0
            ),
            textual_features=TextualFeatures(
                extracted_text="", processed_text="", key_terms=[],
                measurements=[], specifications=[], titles=[],
                labels=[], notes=[], language="en", text_quality=0.0
            ),
            
            purpose="Analysis failed",
            complexity_level="unknown",
            technical_level="unknown",
            
            analysis_confidence=0.1,
            completeness_score=0.0,
            
            processing_time=0.0,
            tools_used=["fallback"],
            analysis_timestamp=time.time()
        )
