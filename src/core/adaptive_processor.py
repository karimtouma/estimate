"""
Adaptive Processor with Dynamic Schema Integration.

This module extends the main processor to use dynamic schemas and
fully autonomous analysis without hardcoded assumptions.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

try:
    from ..core.config import Config
    from ..core.processor import PDFProcessor
    from ..models.dynamic_schemas import (
        DynamicElementRegistry, AdaptiveElementType, CoreElementCategory,
        get_dynamic_registry, initialize_dynamic_registry
    )
    from ..models.intelligent_classifier import IntelligentTypeClassifier
    from ..discovery.enhanced_discovery import EnhancedDynamicDiscovery, create_enhanced_discovery
    from ..utils.adaptive_questions import generate_adaptive_questions
    from ..utils.logging_config import get_logger
    from ..models.schemas import ComprehensiveAnalysisResult
except ImportError:
    # Fallback for testing
    import logging
    logger = logging.getLogger(__name__)

try:
    logger = get_logger(__name__)
except NameError:
    logger = logging.getLogger(__name__)


class AdaptiveProcessor(PDFProcessor):
    """
    Fully autonomous processor with dynamic schema integration.
    
    Extends the base PDFProcessor to use dynamic schemas, adaptive prompts,
    and intelligent question generation for complete autonomy.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize adaptive processor with dynamic schema support."""
        super().__init__(config)
        
        # Initialize dynamic schema components
        registry_path = getattr(config, 'registry_persistence_path', None)
        if registry_path:
            self.dynamic_registry = initialize_dynamic_registry(Path(registry_path))
        else:
            self.dynamic_registry = get_dynamic_registry()
        
        self.intelligent_classifier = IntelligentTypeClassifier(config, self.dynamic_registry)
        
        # Enable dynamic schemas in configuration
        self.enable_dynamic_schemas = getattr(config, 'enable_dynamic_schemas', True)
        
        logger.info(f"Adaptive Processor initialized (dynamic schemas: {'enabled' if self.enable_dynamic_schemas else 'disabled'})")
    
    def comprehensive_analysis_adaptive(
        self,
        pdf_path: Union[str, Path],
        questions: Optional[List[str]] = None,
        enable_discovery: bool = True
    ) -> ComprehensiveAnalysisResult:
        """
        Perform fully adaptive comprehensive analysis.
        
        This method uses enhanced discovery with dynamic schemas for
        complete autonomy without hardcoded assumptions.
        
        Args:
            pdf_path: Path to the PDF file
            questions: Optional list of specific questions (if None, generates adaptive questions)
            enable_discovery: Enable enhanced discovery phase
            
        Returns:
            Comprehensive analysis results with dynamic schema integration
        """
        pdf_path = Path(pdf_path)
        logger.info(f"Starting adaptive comprehensive analysis of: {pdf_path}")
        
        # Reset API statistics
        from ..services.gemini_client import GeminiClient
        GeminiClient.reset_statistics()
        
        try:
            # Upload PDF once for all analyses
            file_uri = self.upload_pdf(pdf_path)
            
            # FASE 1: Enhanced Discovery with Dynamic Schemas
            enhanced_discovery_result = None
            if enable_discovery and self.enable_dynamic_schemas:
                logger.info("🔍 FASE 1: Running enhanced discovery with dynamic schema integration...")
                
                enhanced_discovery = create_enhanced_discovery(self.config, pdf_path)
                enhanced_discovery_result = await enhanced_discovery.enhanced_initial_exploration(
                    sample_size=10, 
                    pdf_uri=file_uri
                )
                
                logger.info(f"✅ Enhanced discovery complete:")
                logger.info(f"  - Document type: {enhanced_discovery_result.document_type}")
                logger.info(f"  - Industry domain: {enhanced_discovery_result.industry_domain}")
                logger.info(f"  - Dynamic types discovered: {len(enhanced_discovery_result.discovered_element_types)}")
                logger.info(f"  - Auto-registered types: {len(enhanced_discovery_result.auto_registered_types)}")
                
                # Update registry with discovered relationships
                if hasattr(enhanced_discovery, 'analyze_element_relationships'):
                    relationships = await enhanced_discovery.analyze_element_relationships(
                        enhanced_discovery_result.discovered_element_types
                    )
                    await enhanced_discovery.update_registry_with_relationships(relationships)
                
                # Convert enhanced result to format expected by base processor
                discovery_result_dict = {
                    'document_type': enhanced_discovery_result.document_type,
                    'industry_domain': enhanced_discovery_result.industry_domain,
                    'discovered_patterns': enhanced_discovery_result.discovered_patterns,
                    'nomenclature_system': enhanced_discovery_result.nomenclature_system,
                    'element_types': enhanced_discovery_result.element_types,
                    'confidence_score': enhanced_discovery_result.confidence_score,
                    'discovery_metadata': enhanced_discovery_result.discovery_metadata
                }
                
            else:
                # Fallback to base discovery
                logger.info("🔍 FASE 1: Using base discovery system...")
                if enable_discovery:
                    from ..discovery import DynamicPlanoDiscovery
                    discovery = DynamicPlanoDiscovery(self.config, pdf_path)
                    
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        base_discovery_result = loop.run_until_complete(
                            discovery.initial_exploration(sample_size=10, pdf_uri=file_uri)
                        )
                        
                        discovery_result_dict = {
                            'document_type': base_discovery_result.document_type,
                            'industry_domain': base_discovery_result.industry_domain,
                            'discovered_patterns': base_discovery_result.discovered_patterns,
                            'nomenclature_system': base_discovery_result.nomenclature_system,
                            'element_types': base_discovery_result.element_types,
                            'confidence_score': base_discovery_result.confidence_score,
                            'discovery_metadata': base_discovery_result.discovery_metadata
                        }
                    finally:
                        loop.close()
                        discovery.close()
                else:
                    discovery_result_dict = None
            
            # Generate adaptive questions if none provided
            if questions is None:
                if discovery_result_dict:
                    questions = generate_adaptive_questions(discovery_result_dict, max_questions=8)
                    logger.info(f"Generated {len(questions)} adaptive questions based on discovery")
                else:
                    questions = self.config.analysis.default_questions
                    logger.info("Using default questions (no discovery available)")
            
            # Execute core analysis with adaptive prompts
            logger.info("🚀 Starting adaptive core analysis...")
            
            # Initialize result
            result = ComprehensiveAnalysisResult(
                file_info={
                    "path": str(pdf_path),
                    "uri": file_uri,
                    "timestamp": time.time(),
                    "discovery_enabled": enable_discovery,
                    "dynamic_schemas_enabled": self.enable_dynamic_schemas,
                    "size_bytes": pdf_path.stat().st_size
                }
            )
            
            # Execute analysis types with discovery context
            enabled_types = self.config.analysis.enabled_types
            
            if "general" in enabled_types:
                analysis_data = self.analyze_document(file_uri, "general", discovery_result_dict)
                result.general_analysis = analysis_data
                
            if "sections" in enabled_types:
                analysis_data = self.analyze_document(file_uri, "sections", discovery_result_dict)
                result.sections_analysis = [analysis_data]
                
            if "data_extraction" in enabled_types:
                analysis_data = self.analyze_document(file_uri, "data_extraction", discovery_result_dict)
                result.data_extraction = analysis_data
            
            # Multi-turn Q&A with adaptive questions
            qa_results = self.multi_turn_analysis(file_uri, questions)
            result.qa_analysis = qa_results
            
            # Add discovery results
            if discovery_result_dict:
                result.discovery_analysis = discovery_result_dict
            
            # Add dynamic schema results if available
            if enhanced_discovery_result:
                result.dynamic_schema_results = {
                    "discovered_element_types": [
                        {
                            "specific_type": elem.specific_type,
                            "base_category": elem.base_category.value,
                            "discovery_confidence": elem.discovery_confidence,
                            "is_dynamically_discovered": elem.is_dynamically_discovered,
                            "domain_context": elem.domain_context
                        }
                        for elem in enhanced_discovery_result.discovered_element_types
                    ],
                    "auto_registered_types": enhanced_discovery_result.auto_registered_types,
                    "registry_stats": enhanced_discovery_result.registry_stats,
                    "classification_performance": enhanced_discovery_result.classification_performance
                }
            
            # Add page mapping if enabled
            enable_page_mapping = getattr(self.config.processing, 'enable_complete_page_mapping', True)
            if enable_page_mapping:
                logger.info("🗺️ Creating adaptive page map...")
                
                if enhanced_discovery_result:
                    # Use enhanced discovery for page mapping
                    page_map = await enhanced_discovery.create_complete_page_map(
                        main_topics=list(enhanced_discovery_result.discovered_patterns.get('patterns', [])),
                        pdf_uri=file_uri
                    )
                else:
                    # Fallback to base discovery
                    discovery = DynamicPlanoDiscovery(self.config, pdf_path)
                    try:
                        page_map = await discovery.create_complete_page_map(
                            main_topics=discovery_result_dict.get('discovered_patterns', {}).get('patterns', []) if discovery_result_dict else [],
                            pdf_uri=file_uri
                        )
                    finally:
                        discovery.close()
                
                result.page_map = page_map
            
            logger.info("Adaptive comprehensive analysis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Adaptive comprehensive analysis failed: {e}")
            raise
    
    def get_adaptive_analysis_stats(self) -> Dict[str, Any]:
        """Get statistics about adaptive analysis performance."""
        registry_stats = self.dynamic_registry.get_registry_stats()
        classifier_stats = self.intelligent_classifier.get_classification_stats()
        
        return {
            "dynamic_schemas_enabled": self.enable_dynamic_schemas,
            "registry_stats": registry_stats,
            "classifier_stats": classifier_stats,
            "total_discovered_types": registry_stats.get("total_types", 0),
            "classification_accuracy": classifier_stats.get("average_accuracy", 0),
            "discovery_rate": classifier_stats.get("discovery_rate", 0)
        }


# Factory function for creating adaptive processor
def create_adaptive_processor(config: Optional[Config] = None) -> AdaptiveProcessor:
    """
    Factory function to create adaptive processor with dynamic schemas.
    
    Args:
        config: Optional configuration (loads default if None)
        
    Returns:
        AdaptiveProcessor instance
    """
    if config is None:
        from ..core.config import get_config
        config = get_config()
    
    return AdaptiveProcessor(config)
