"""
Data models and Pydantic schemas for PDF processing.

This module defines the structured data models used throughout the application
for type safety and validation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum


class DocumentType(str, Enum):
    """Enumeration of supported document types."""
    TECHNICAL_REPORT = "technical_report"
    BUSINESS_PROPOSAL = "business_proposal"
    RESEARCH_PAPER = "research_paper"
    FINANCIAL_STATEMENT = "financial_statement"
    LEGAL_DOCUMENT = "legal_document"
    MANUAL = "manual"
    PRESENTATION = "presentation"
    OTHER = "other"


class AnalysisType(str, Enum):
    """Types of analysis that can be performed."""
    GENERAL = "general"
    SECTIONS = "sections" 
    DATA_EXTRACTION = "data_extraction"
    COMPREHENSIVE = "comprehensive"


class DocumentAnalysis(BaseModel):
    """Schema for general document analysis results."""
    
    summary: str = Field(
        description="Executive summary of the document content",
        min_length=50,
        max_length=1000
    )
    main_topics: List[str] = Field(
        description="Main topics identified in the document",
        min_items=1,
        max_items=10
    )
    key_insights: List[str] = Field(
        description="Key insights extracted from the document",
        min_items=1,
        max_items=15
    )
    document_type: DocumentType = Field(
        description="Identified document type"
    )
    confidence_score: float = Field(
        description="Confidence level of the analysis (0-1)",
        ge=0.0,
        le=1.0
    )
    
    @validator('confidence_score')
    def validate_confidence(cls, v):
        """Ensure confidence score is within valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence score must be between 0.0 and 1.0')
        return round(v, 3)


class SectionAnalysis(BaseModel):
    """Schema for section-specific analysis results."""
    
    section_title: str = Field(
        description="Title or heading of the section",
        min_length=1,
        max_length=200
    )
    content_summary: str = Field(
        description="Summary of the section content",
        min_length=20,
        max_length=500
    )
    important_data: List[str] = Field(
        description="Important data points found in the section",
        max_items=20
    )
    questions_raised: List[str] = Field(
        description="Questions or concerns raised by the content",
        max_items=10
    )
    section_type: Optional[str] = Field(
        description="Type of section (introduction, methodology, results, etc.)",
        default=None
    )


class DataExtraction(BaseModel):
    """Schema for structured data extraction results."""
    
    entities: List[str] = Field(
        description="Named entities (people, organizations, locations)",
        max_items=50
    )
    dates: List[str] = Field(
        description="Relevant dates mentioned in the document",
        max_items=30
    )
    numbers: List[str] = Field(
        description="Important numbers, metrics, and statistics (max 100 chars each)",
        max_items=40
    )
    references: List[str] = Field(
        description="Citations, references, and external sources",
        max_items=25
    )
    key_terms: Optional[List[str]] = Field(
        description="Technical terms and domain-specific vocabulary",
        max_items=30,
        default=[]
    )


class QuestionAnswer(BaseModel):
    """Schema for Q&A analysis results."""
    
    question: str = Field(
        description="The question asked",
        min_length=5
    )
    answer: str = Field(
        description="The generated answer",
        min_length=10
    )
    confidence: float = Field(
        description="Confidence in the answer (0-1)",
        ge=0.0,
        le=1.0
    )
    sources: Optional[List[str]] = Field(
        description="Sources or sections that informed the answer",
        default=[]
    )
    follow_up_questions: Optional[List[str]] = Field(
        description="Suggested follow-up questions",
        max_items=5,
        default=[]
    )
    question_index: Optional[int] = Field(
        description="Index of the question in a sequence",
        ge=1,
        default=None
    )


class ProcessingMetadata(BaseModel):
    """Metadata about the processing operation."""
    
    timestamp: float = Field(description="Unix timestamp of processing")
    processor_version: str = Field(description="Version of the processor")
    model_used: str = Field(description="AI model used for processing")
    config_file: str = Field(description="Configuration file path")
    environment: str = Field(description="Execution environment (local/container)")
    file_info: Optional[dict] = Field(description="Original file information", default={})


class PageClassification(BaseModel):
    """Schema for individual page classification and analysis."""
    
    page_number: int = Field(description="Page number (1-indexed)")
    primary_category: str = Field(description="Primary category from main topics")
    secondary_categories: List[str] = Field(default=[], description="Additional relevant categories")
    content_summary: str = Field(description="Brief summary of page content")
    key_elements: List[str] = Field(default=[], description="Key elements found on this page")
    complexity_score: float = Field(ge=0.0, le=1.0, description="Visual complexity score")
    confidence: float = Field(ge=0.0, le=1.0, description="Classification confidence")


class DocumentPageMap(BaseModel):
    """Complete page-by-page mapping of document structure."""
    
    total_pages: int = Field(description="Total number of pages in document")
    pages: List[PageClassification] = Field(description="Classification for each page")
    category_distribution: dict = Field(description="Distribution of pages by category")
    coverage_analysis: dict = Field(description="Analysis of topic coverage across pages")


class ComprehensiveAnalysisResult(BaseModel):
    """Complete analysis result combining all analysis types."""
    
    file_info: dict = Field(description="Information about the processed file")
    general_analysis: Optional[DocumentAnalysis] = None
    sections_analysis: Optional[List[SectionAnalysis]] = None
    data_extraction: Optional[DataExtraction] = None
    qa_analysis: Optional[List[QuestionAnswer]] = None
    discovery_analysis: Optional[dict] = Field(default=None, description="Discovery phase analysis results")
    dynamic_schema_results: Optional[dict] = Field(default=None, description="Dynamic schema discovery and classification results")
    page_map: Optional[DocumentPageMap] = Field(default=None, description="Complete page-by-page analysis and categorization")
    metadata: Optional[ProcessingMetadata] = None
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
