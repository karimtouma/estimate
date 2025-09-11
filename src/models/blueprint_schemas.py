"""
Advanced data models for structural blueprint analysis.

This module defines sophisticated Pydantic models for the intelligent
analysis of structural blueprints with multi-modal understanding.
"""

from typing import List, Dict, Optional, Union, Any, Tuple
from pydantic import BaseModel, Field, validator
from enum import Enum
import time
from pathlib import Path


class BlueprintPageType(str, Enum):
    """Enumeration of structural blueprint page types."""
    FLOOR_PLAN = "floor_plan"
    ELEVATION = "elevation"
    SECTION = "section"
    DETAIL = "detail"
    SITE_PLAN = "site_plan"
    STRUCTURAL_PLAN = "structural_plan"
    MECHANICAL_PLAN = "mechanical_plan"
    ELECTRICAL_PLAN = "electrical_plan"
    PLUMBING_PLAN = "plumbing_plan"
    TITLE_BLOCK = "title_block"
    LEGEND = "legend"
    NOTES = "notes"
    SCHEDULE = "schedule"
    SPECIFICATIONS = "specifications"
    UNKNOWN = "unknown"


class StructuralElementType(str, Enum):
    """Types of structural elements in blueprints."""
    # Structural Elements
    WALL = "wall"
    BEAM = "beam"
    COLUMN = "column"
    FOUNDATION = "foundation"
    SLAB = "slab"
    FOOTING = "footing"
    
    # Architectural Elements
    DOOR = "door"
    WINDOW = "window"
    STAIR = "stair"
    ELEVATOR = "elevator"
    ROOM = "room"
    
    # MEP Elements
    ELECTRICAL_OUTLET = "electrical_outlet"
    LIGHT_FIXTURE = "light_fixture"
    HVAC_DUCT = "hvac_duct"
    PLUMBING_FIXTURE = "plumbing_fixture"
    
    # Annotation Elements
    DIMENSION = "dimension"
    TEXT_ANNOTATION = "text_annotation"
    SYMBOL = "symbol"
    GRID_LINE = "grid_line"
    
    # Other
    FURNITURE = "furniture"
    EQUIPMENT = "equipment"
    LANDSCAPE = "landscape"
    UNKNOWN = "unknown"


class Coordinates(BaseModel):
    """2D coordinates with optional Z-axis for 3D elements."""
    x: float = Field(description="X coordinate")
    y: float = Field(description="Y coordinate")
    z: Optional[float] = Field(description="Z coordinate (elevation)", default=None)


class BoundingBox(BaseModel):
    """Bounding box for element location."""
    top_left: Coordinates = Field(description="Top-left corner")
    bottom_right: Coordinates = Field(description="Bottom-right corner")
    
    @property
    def width(self) -> float:
        """Calculate width of bounding box."""
        return self.bottom_right.x - self.top_left.x
    
    @property
    def height(self) -> float:
        """Calculate height of bounding box."""
        return self.bottom_right.y - self.top_left.y
    
    @property
    def center(self) -> Coordinates:
        """Calculate center point of bounding box."""
        return Coordinates(
            x=(self.top_left.x + self.bottom_right.x) / 2,
            y=(self.top_left.y + self.bottom_right.y) / 2
        )


class Dimensions(BaseModel):
    """Physical dimensions of structural elements."""
    length: Optional[float] = Field(description="Length in units", default=None)
    width: Optional[float] = Field(description="Width in units", default=None)
    height: Optional[float] = Field(description="Height in units", default=None)
    thickness: Optional[float] = Field(description="Thickness in units", default=None)
    diameter: Optional[float] = Field(description="Diameter for circular elements", default=None)
    units: str = Field(description="Measurement units", default="inches")


class MaterialProperties(BaseModel):
    """Material properties for structural elements."""
    material_type: Optional[str] = Field(description="Type of material", default=None)
    grade: Optional[str] = Field(description="Material grade or specification", default=None)
    strength: Optional[float] = Field(description="Material strength", default=None)
    density: Optional[float] = Field(description="Material density", default=None)
    properties: Dict[str, Any] = Field(description="Additional material properties", default_factory=dict)


class StructuralElement(BaseModel):
    """Comprehensive structural element model."""
    element_id: str = Field(description="Unique element identifier")
    element_type: StructuralElementType = Field(description="Type of structural element")
    element_subtype: Optional[str] = Field(description="Specific subtype", default=None)
    
    # Spatial properties
    location: BoundingBox = Field(description="Element location on page")
    dimensions: Optional[Dimensions] = Field(description="Physical dimensions", default=None)
    
    # Material properties
    material: Optional[MaterialProperties] = Field(description="Material properties", default=None)
    
    # Visual properties
    color: Optional[str] = Field(description="Element color", default=None)
    line_weight: Optional[float] = Field(description="Line weight/thickness", default=None)
    pattern: Optional[str] = Field(description="Fill pattern or hatch", default=None)
    
    # Semantic properties
    label: Optional[str] = Field(description="Element label or name", default=None)
    description: Optional[str] = Field(description="Element description", default=None)
    annotations: List[str] = Field(description="Associated annotations", default_factory=list)
    
    # Metadata
    confidence: float = Field(description="Detection confidence", ge=0.0, le=1.0)
    detection_method: str = Field(description="Method used for detection")
    page_number: int = Field(description="Source page number")
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Ensure confidence is within valid range."""
        return round(v, 3)


class SpatialRelationship(BaseModel):
    """Spatial and functional relationships between elements."""
    relationship_id: str = Field(description="Unique relationship identifier")
    element_1_id: str = Field(description="First element ID")
    element_2_id: str = Field(description="Second element ID")
    
    relationship_type: str = Field(description="Type of relationship")
    relationship_subtype: Optional[str] = Field(description="Specific relationship subtype", default=None)
    
    # Spatial properties
    distance: Optional[float] = Field(description="Distance between elements", default=None)
    angle: Optional[float] = Field(description="Angular relationship", default=None)
    direction: Optional[str] = Field(description="Directional relationship", default=None)
    
    # Semantic properties
    functional_relationship: Optional[str] = Field(description="Functional connection", default=None)
    structural_dependency: bool = Field(description="Structural dependency exists", default=False)
    
    description: str = Field(description="Natural language description")
    confidence: float = Field(description="Relationship confidence", ge=0.0, le=1.0)


class VisualFeatures(BaseModel):
    """Visual analysis features extracted from page images."""
    dominant_colors: List[str] = Field(description="Dominant colors in the image")
    line_density: float = Field(description="Density of lines in the drawing")
    text_regions: List[BoundingBox] = Field(description="Detected text regions")
    
    geometric_complexity: float = Field(description="Complexity score of geometric elements")
    symmetry_score: float = Field(description="Symmetry analysis score")
    scale_indicators: List[str] = Field(description="Scale and dimension indicators found")
    
    drawing_style: Optional[str] = Field(description="Drawing style classification", default=None)
    quality_score: float = Field(description="Image quality assessment", ge=0.0, le=1.0)


class TextualFeatures(BaseModel):
    """Textual analysis features from page content."""
    extracted_text: str = Field(description="Raw extracted text")
    processed_text: str = Field(description="Cleaned and processed text")
    
    # Text analysis
    key_terms: List[str] = Field(description="Important technical terms")
    measurements: List[str] = Field(description="Extracted measurements")
    specifications: List[str] = Field(description="Technical specifications")
    
    # Document structure
    titles: List[str] = Field(description="Titles and headings")
    labels: List[str] = Field(description="Element labels")
    notes: List[str] = Field(description="Notes and annotations")
    
    # Metadata
    language: str = Field(description="Detected language", default="en")
    text_quality: float = Field(description="Text extraction quality", ge=0.0, le=1.0)


class PageTaxonomy(BaseModel):
    """Comprehensive taxonomy for individual blueprint pages."""
    page_number: int = Field(description="Page number in document")
    page_type: BlueprintPageType = Field(description="Primary page classification")
    
    # Hierarchical classification
    primary_category: str = Field(description="Primary classification category")
    secondary_category: Optional[str] = Field(description="Secondary classification", default=None)
    tertiary_category: Optional[str] = Field(description="Tertiary classification", default=None)
    
    # Content analysis
    structural_elements: List[StructuralElement] = Field(description="Detected structural elements")
    spatial_relationships: List[SpatialRelationship] = Field(description="Element relationships")
    
    # Feature analysis
    visual_features: VisualFeatures = Field(description="Visual analysis results")
    textual_features: TextualFeatures = Field(description="Text analysis results")
    
    # Semantic understanding
    purpose: Optional[str] = Field(description="Page purpose and intent", default=None)
    complexity_level: str = Field(description="Complexity assessment", default="medium")
    technical_level: str = Field(description="Technical detail level", default="standard")
    
    # Quality metrics
    analysis_confidence: float = Field(description="Overall analysis confidence", ge=0.0, le=1.0)
    completeness_score: float = Field(description="Analysis completeness", ge=0.0, le=1.0)
    
    # Processing metadata
    processing_time: float = Field(description="Time taken for analysis")
    tools_used: List[str] = Field(description="Tools used in analysis")
    analysis_timestamp: float = Field(description="Unix timestamp of analysis")
    
    @validator('analysis_confidence', 'completeness_score')
    def validate_scores(cls, v):
        """Ensure scores are within valid range."""
        return round(v, 3)


class DocumentPattern(BaseModel):
    """Cross-document patterns and insights."""
    pattern_id: str = Field(description="Unique pattern identifier")
    pattern_type: str = Field(description="Type of pattern identified")
    
    affected_pages: List[int] = Field(description="Pages where pattern appears")
    pattern_description: str = Field(description="Description of the pattern")
    
    structural_significance: str = Field(description="Structural importance of pattern")
    frequency: float = Field(description="Pattern frequency in document")
    confidence: float = Field(description="Pattern detection confidence", ge=0.0, le=1.0)


class StructuralSummary(BaseModel):
    """High-level structural analysis summary."""
    building_type: Optional[str] = Field(description="Type of building", default=None)
    structural_system: Optional[str] = Field(description="Structural system type", default=None)
    
    # Quantitative analysis
    total_floors: Optional[int] = Field(description="Number of floors", default=None)
    total_area: Optional[float] = Field(description="Total building area", default=None)
    structural_elements_count: Dict[str, int] = Field(description="Count by element type", default_factory=dict)
    
    # Qualitative analysis
    complexity_assessment: str = Field(description="Overall complexity assessment")
    design_style: Optional[str] = Field(description="Architectural design style", default=None)
    construction_type: Optional[str] = Field(description="Type of construction", default=None)
    
    # Compliance and standards
    applicable_codes: List[str] = Field(description="Applicable building codes", default_factory=list)
    compliance_notes: List[str] = Field(description="Compliance observations", default_factory=list)


class DocumentTaxonomy(BaseModel):
    """Complete document taxonomy with advanced analytics."""
    document_id: str = Field(description="Unique document identifier")
    document_name: str = Field(description="Document name or title")
    
    # Document metadata
    document_metadata: Dict[str, Any] = Field(description="Document-level metadata")
    creation_timestamp: float = Field(description="Document creation timestamp")
    analysis_timestamp: float = Field(description="Analysis completion timestamp")
    
    # Page-level analysis
    page_taxonomies: List[PageTaxonomy] = Field(description="Individual page taxonomies")
    total_pages: int = Field(description="Total number of pages")
    
    # Cross-page analysis
    global_patterns: List[DocumentPattern] = Field(description="Cross-page patterns")
    structural_summary: StructuralSummary = Field(description="Overall structural analysis")
    
    # Knowledge extraction
    semantic_graph: Dict[str, Any] = Field(description="Semantic knowledge graph", default_factory=dict)
    key_insights: List[str] = Field(description="Key insights from analysis", default_factory=list)
    
    # Quality metrics
    overall_confidence: float = Field(description="Overall analysis confidence", ge=0.0, le=1.0)
    completeness_score: float = Field(description="Analysis completeness", ge=0.0, le=1.0)
    consistency_score: float = Field(description="Cross-page consistency", ge=0.0, le=1.0)
    
    # Performance metrics
    total_processing_time: float = Field(description="Total processing time")
    pages_per_second: float = Field(description="Processing rate")
    api_calls_made: int = Field(description="Total API calls made")
    
    @validator('overall_confidence', 'completeness_score', 'consistency_score')
    def validate_scores(cls, v):
        """Ensure scores are within valid range."""
        return round(v, 3)
    
    @property
    def pages_by_type(self) -> Dict[str, int]:
        """Count pages by type."""
        type_counts = {}
        for page in self.page_taxonomies:
            page_type = page.page_type.value
            type_counts[page_type] = type_counts.get(page_type, 0) + 1
        return type_counts
    
    @property
    def elements_by_type(self) -> Dict[str, int]:
        """Count elements by type across all pages."""
        element_counts = {}
        for page in self.page_taxonomies:
            for element in page.structural_elements:
                element_type = element.element_type.value
                element_counts[element_type] = element_counts.get(element_type, 0) + 1
        return element_counts


class AnalysisTask(BaseModel):
    """Task definition for agent processing."""
    task_id: str = Field(description="Unique task identifier")
    task_type: str = Field(description="Type of analysis task")
    
    page_number: int = Field(description="Target page number")
    priority: int = Field(description="Task priority (1-10)", ge=1, le=10)
    
    required_tools: List[str] = Field(description="Required tools for task")
    optional_tools: List[str] = Field(description="Optional tools", default_factory=list)
    
    context_requirements: Dict[str, Any] = Field(description="Context requirements", default_factory=dict)
    expected_duration: float = Field(description="Expected processing time")
    
    dependencies: List[str] = Field(description="Task dependencies", default_factory=list)
    status: str = Field(description="Task status", default="pending")


class AgentResult(BaseModel):
    """Result from individual agent processing."""
    agent_id: str = Field(description="Agent identifier")
    agent_type: str = Field(description="Type of agent")
    
    task_id: str = Field(description="Processed task ID")
    page_number: int = Field(description="Processed page number")
    
    # Results
    primary_result: Dict[str, Any] = Field(description="Primary analysis result")
    secondary_results: Dict[str, Any] = Field(description="Secondary results", default_factory=dict)
    
    # Metadata
    processing_time: float = Field(description="Processing duration")
    tools_used: List[str] = Field(description="Tools utilized")
    confidence: float = Field(description="Result confidence", ge=0.0, le=1.0)
    
    # Quality indicators
    completeness: float = Field(description="Result completeness", ge=0.0, le=1.0)
    accuracy_indicators: Dict[str, float] = Field(description="Accuracy metrics", default_factory=dict)
    
    timestamp: float = Field(description="Result timestamp", default_factory=time.time)


class ContextWindow(BaseModel):
    """Context window for multi-turn reasoning."""
    window_id: str = Field(description="Context window identifier")
    document_id: str = Field(description="Associated document ID")
    
    # Context content
    pages_context: List[PageTaxonomy] = Field(description="Page contexts in window")
    global_context: Dict[str, Any] = Field(description="Global document context", default_factory=dict)
    
    # Window management
    max_pages: int = Field(description="Maximum pages in window", default=10)
    current_focus: int = Field(description="Current page focus")
    
    # Semantic features
    key_patterns: List[str] = Field(description="Key patterns in context")
    recurring_elements: List[str] = Field(description="Recurring elements")
    
    @property
    def context_size(self) -> int:
        """Get current context size."""
        return len(self.pages_context)
    
    def add_page(self, page_taxonomy: PageTaxonomy):
        """Add page to context window with size management."""
        self.pages_context.append(page_taxonomy)
        
        # Maintain window size
        if len(self.pages_context) > self.max_pages:
            self.pages_context.pop(0)  # Remove oldest
    
    def get_relevant_context(self, current_page: int) -> Dict[str, Any]:
        """Extract relevant context for current page."""
        relevant_pages = [
            p for p in self.pages_context 
            if abs(p.page_number - current_page) <= 3
        ]
        
        return {
            "recent_pages": relevant_pages,
            "global_patterns": self.key_patterns,
            "recurring_elements": self.recurring_elements,
            "document_context": self.global_context
        }


class ProcessingSession(BaseModel):
    """Complete processing session tracking."""
    session_id: str = Field(description="Unique session identifier")
    document_path: str = Field(description="Source document path")
    
    # Session configuration
    processing_config: Dict[str, Any] = Field(description="Processing configuration")
    agents_used: List[str] = Field(description="Agents utilized in session")
    
    # Results
    document_taxonomy: Optional[DocumentTaxonomy] = Field(description="Final taxonomy", default=None)
    agent_results: List[AgentResult] = Field(description="Individual agent results", default_factory=list)
    
    # Performance metrics
    start_time: float = Field(description="Session start time")
    end_time: Optional[float] = Field(description="Session end time", default=None)
    total_pages_processed: int = Field(description="Pages processed", default=0)
    
    # Quality metrics
    success_rate: float = Field(description="Processing success rate", ge=0.0, le=1.0, default=0.0)
    average_confidence: float = Field(description="Average confidence", ge=0.0, le=1.0, default=0.0)
    
    # Error tracking
    errors_encountered: List[Dict[str, Any]] = Field(description="Errors during processing", default_factory=list)
    warnings: List[str] = Field(description="Warnings generated", default_factory=list)
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate session duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def pages_per_second(self) -> Optional[float]:
        """Calculate processing rate."""
        if self.duration and self.duration > 0:
            return self.total_pages_processed / self.duration
        return None
