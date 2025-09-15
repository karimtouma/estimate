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
    from ..utils.language_router import LanguageRouter, LanguageDetectionResult
    from ..utils.logging_config import get_logger
    from ..models.schemas import ComprehensiveAnalysisResult
    from ..optimization.comprehensive_gepa_system import create_comprehensive_gepa_system
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
    Procesador adaptativo completamente autónomo con integración de esquemas dinámicos.
    
    Esta clase extiende PDFProcessor para proporcionar análisis completamente autónomo
    usando esquemas dinámicos, optimización GEPA, detección de idioma automática y
    generación inteligente de preguntas sin configuración previa.
    
    Características principales:
    - Esquemas dinámicos que se adaptan automáticamente al contenido
    - Sistema GEPA con múltiples candidatos y juez inteligente
    - Language Router para detección automática de idioma
    - Discovery Engine con análisis estratégico de muestras
    - Auto-registro de tipos con evolución continua
    
    Flujo de procesamiento:
    1. Enhanced Discovery con esquemas dinámicos
    2. Language Detection y optimización de prompts
    3. Clasificación GEPA con múltiples candidatos
    4. Análisis core con prompts optimizados
    5. Q&A adaptativo contextual
    6. Page mapping completo
    7. Optimización GEPA en background
    
    Attributes:
        dynamic_registry (DynamicElementRegistry): Registro de tipos dinámicos
        intelligent_classifier (IntelligentTypeClassifier): Clasificador con GEPA
        language_router (LanguageRouter): Router de detección de idioma
        gepa_system (ComprehensiveGEPASystem): Sistema de optimización GEPA
        analysis_history (List): Historial para optimización GEPA
        enable_dynamic_schemas (bool): Flag de esquemas dinámicos habilitados
        
    Example:
        ```python
        from src.core.adaptive_processor import AdaptiveProcessor
        from src.core.config import get_config
        
        config = get_config()
        processor = AdaptiveProcessor(config)
        
        result = processor.comprehensive_analysis_adaptive(
            pdf_path="document.pdf",
            enable_discovery=True
        )
        
        print(f"Tipos descubiertos: {len(result.dynamic_schema_results.discovered_element_types)}")
        print(f"Judge Score GEPA: {result.dynamic_schema_results.gepa_statistics.average_judge_score}")
        ```
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
        
        # Initialize language router for automatic language detection
        self.language_router = LanguageRouter(config, self.gemini_client)
        
        # Initialize comprehensive GEPA system for prompt optimization
        self.gepa_system = create_comprehensive_gepa_system(config, self.gemini_client)
        self.analysis_history = []  # Track analysis results for GEPA optimization
        
        # Enable dynamic schemas in configuration
        self.enable_dynamic_schemas = getattr(config, 'enable_dynamic_schemas', True)
        
        logger.info(f"Adaptive Processor initialized (dynamic schemas: {'enabled' if self.enable_dynamic_schemas else 'disabled'})")
    
    def _extract_text_samples_from_discovery(self, discovery_result) -> List[str]:
        """Extract text samples from discovery result for language detection."""
        
        text_samples = []
        
        # Extract from element types
        if hasattr(discovery_result, 'element_types') and discovery_result.element_types:
            text_samples.extend(discovery_result.element_types[:3])
        
        # Extract from discovered patterns
        if hasattr(discovery_result, 'discovered_patterns') and discovery_result.discovered_patterns:
            patterns = discovery_result.discovered_patterns.get('patterns', [])
            text_samples.extend(patterns[:5])
        
        # Extract from nomenclature system
        if hasattr(discovery_result, 'nomenclature_system') and discovery_result.nomenclature_system:
            nomenclature = discovery_result.nomenclature_system
            if 'patterns' in nomenclature:
                for pattern_info in nomenclature['patterns'].values():
                    if 'examples' in pattern_info:
                        text_samples.extend(pattern_info['examples'][:3])
        
        # Extract from discovered element types
        if hasattr(discovery_result, 'discovered_element_types'):
            for element in discovery_result.discovered_element_types[:5]:
                if hasattr(element, 'specific_type'):
                    text_samples.append(element.specific_type)
                if hasattr(element, 'domain_context') and element.domain_context:
                    text_samples.append(element.domain_context)
        
        # Clean and filter samples
        clean_samples = []
        for sample in text_samples:
            if isinstance(sample, str) and len(sample.strip()) > 3:
                clean_samples.append(sample.strip())
        
        # Return unique samples, limited for API efficiency
        unique_samples = list(dict.fromkeys(clean_samples))[:10]
        
        logger.debug(f"Extracted {len(unique_samples)} text samples for language detection")
        return unique_samples
    
    async def _trigger_gepa_optimization_if_needed(
        self,
        analysis_result: Dict[str, Any],
        discovery_result: Dict[str, Any]
    ) -> None:
        """Trigger GEPA optimization if sufficient analysis history accumulated."""
        
        # Record current analysis for GEPA learning
        analysis_record = {
            "timestamp": time.time(),
            "document_type": discovery_result.get("document_type"),
            "industry_domain": discovery_result.get("industry_domain"),
            "analysis_quality": self._assess_analysis_quality(analysis_result),
            "discovery_patterns": discovery_result.get("discovered_patterns", {}),
            "language_detected": getattr(self, 'detected_language', None),
            "processing_metrics": {
                "types_discovered": len(discovery_result.get("discovered_patterns", {})),
                "confidence_score": discovery_result.get("confidence_score", 0.0)
            }
        }
        
        self.analysis_history.append(analysis_record)
        
        # Keep only last 10 analyses for efficiency
        if len(self.analysis_history) > 10:
            self.analysis_history = self.analysis_history[-10:]
        
        # Check if we should trigger GEPA optimization
        gepa_threshold = getattr(self.config.analysis, 'gepa_evolution_threshold', 3)
        
        if len(self.analysis_history) >= gepa_threshold:
            logger.info(f"🧬 GEPA TRIGGER: {len(self.analysis_history)} analyses accumulated - starting comprehensive prompt optimization...")
            
            # Run GEPA optimization in background
            asyncio.create_task(self._run_comprehensive_gepa_optimization(discovery_result.get("industry_domain", "construction_documents")))
    
    async def _run_comprehensive_gepa_optimization(self, target_domain: str) -> None:
        """Run comprehensive GEPA optimization in background."""
        
        try:
            logger.info("🧬 Starting comprehensive GEPA optimization for all dynamic prompts...")
            
            # Use analysis history as test documents
            optimization_result = await self.gepa_system.optimize_all_dynamic_prompts(
                current_prompts=await self._collect_current_prompts(target_domain),
                test_documents=self.analysis_history[-5:],  # Use last 5 analyses
                target_domain=target_domain
            )
            
            # Update system with optimized prompts
            self._apply_optimized_prompts(optimization_result.optimized_prompts)
            
            logger.info(f"🎉 Comprehensive GEPA optimization complete:")
            summary = optimization_result.get_summary()
            for key, value in summary.items():
                logger.info(f"  - {key}: {value}")
                
        except Exception as e:
            logger.error(f"Comprehensive GEPA optimization failed: {e}")
    
    async def _collect_current_prompts(self, domain: str) -> Dict[str, str]:
        """Collect all current dynamic prompts used in the system."""
        
        prompts = {}
        
        # Get adaptive prompts for each analysis type
        for analysis_type in [AnalysisType.GENERAL, AnalysisType.SECTIONS, AnalysisType.DATA_EXTRACTION]:
            prompt, _ = self._get_analysis_config(analysis_type, {
                'document_type': f'{domain}_document',
                'industry_domain': domain,
                'discovered_patterns': {'patterns': ['sample_pattern']},
                'focus_areas': ['technical', 'structural']
            })
            prompts[f'{analysis_type.value}_analysis'] = prompt
        
        # Add discovery prompts
        prompts['discovery_exploration'] = """
        You are analyzing a technical document. Discover without preconceptions:
        1. Document type and domain
        2. Organization system  
        3. Nomenclature and coding
        4. Visual patterns
        5. Relationships and references
        6. Unique elements
        """
        
        # Add classification prompts
        prompts['element_classification'] = """
        Classify this element based on its characteristics:
        - Text content, visual features, location
        - Provide specific type with confidence and reasoning
        - Use categories: structural, architectural, mep, annotation, specialized
        """
        
        return prompts
    
    def _apply_optimized_prompts(self, optimized_prompts: Dict[str, str]) -> None:
        """Apply optimized prompts to the system."""
        
        # Store optimized prompts for future use
        if not hasattr(self, 'optimized_prompts'):
            self.optimized_prompts = {}
        
        self.optimized_prompts.update(optimized_prompts)
        
        logger.info(f"Applied {len(optimized_prompts)} optimized prompts to system")
    
    def _assess_analysis_quality(self, analysis_result: Dict[str, Any]) -> float:
        """Assess the quality of an analysis result."""
        
        quality_score = 0.0
        factors = 0
        
        # Check general analysis quality
        if 'general_analysis' in analysis_result:
            general = analysis_result['general_analysis']
            if general and general.get('confidence_score', 0) > 0.8:
                quality_score += 0.3
            factors += 1
        
        # Check data extraction completeness
        if 'data_extraction' in analysis_result:
            data = analysis_result['data_extraction']
            if data:
                completeness = (
                    len(data.get('entities', [])) / 30 +  # Normalize to expected counts
                    len(data.get('dates', [])) / 10 +
                    len(data.get('numbers', [])) / 20 +
                    len(data.get('references', [])) / 15
                ) / 4
                quality_score += min(0.3, completeness * 0.3)
            factors += 1
        
        # Check Q&A quality
        if 'qa_analysis' in analysis_result:
            qa = analysis_result['qa_analysis']
            if qa and len(qa) > 0:
                avg_confidence = sum(q.get('confidence', 0) for q in qa) / len(qa)
                quality_score += avg_confidence * 0.4
            factors += 1
        
        return quality_score / max(1, factors)
    
    def comprehensive_analysis_adaptive(
        self,
        pdf_path: Union[str, Path],
        questions: Optional[List[str]] = None,
        enable_discovery: bool = True
    ) -> 'ComprehensiveAnalysisResult':
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
                
                # Run async discovery in sync context
                import asyncio
                
                # Create a single loop for all async operations
                main_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(main_loop)
                
                try:
                    # FASE 1: Enhanced Discovery
                    enhanced_discovery_result = main_loop.run_until_complete(
                        enhanced_discovery.enhanced_initial_exploration(
                            sample_size=10, 
                            pdf_uri=file_uri
                        )
                    )
                    
                    logger.info(f"✅ Enhanced discovery complete:")
                    logger.info(f"  - Document type: {enhanced_discovery_result.document_type}")
                    logger.info(f"  - Industry domain: {enhanced_discovery_result.industry_domain}")
                    logger.info(f"  - Dynamic types discovered: {len(enhanced_discovery_result.discovered_element_types)}")
                    logger.info(f"  - Auto-registered types: {len(enhanced_discovery_result.auto_registered_types)}")
                    
                    # FASE 1.5: Language Detection and Prompt Optimization
                    logger.info("🌐 FASE 1.5: Detecting document language for prompt optimization...")
                    
                    # Extract text samples from discovery for language detection
                    text_samples = self._extract_text_samples_from_discovery(enhanced_discovery_result)
                    
                    # Use the same loop for language detection
                    language_detection_result = main_loop.run_until_complete(
                        self.language_router.detect_document_language(text_samples)
                    )
                    
                    logger.info(f"✅ Language detection complete:")
                    logger.info(f"  - Primary language: {language_detection_result.primary_language}")
                    logger.info(f"  - Confidence: {language_detection_result.confidence:.3f}")
                    logger.info(f"  - Mixed language: {language_detection_result.mixed_language}")
                    logger.info(f"  - Technical terms: {len(language_detection_result.technical_terminology)}")
                    
                    # Store language result for prompt optimization
                    self.detected_language = language_detection_result
                    
                finally:
                    # Close the loop only once, at the end
                    main_loop.close()
                
                # Update registry with discovered relationships (if method exists)
                if hasattr(enhanced_discovery, 'analyze_element_relationships'):
                    rel_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(rel_loop)
                    try:
                        relationships = rel_loop.run_until_complete(
                            enhanced_discovery.analyze_element_relationships(
                                enhanced_discovery_result.discovered_element_types
                            )
                        )
                        rel_loop.run_until_complete(
                            enhanced_discovery.update_registry_with_relationships(relationships)
                        )
                    except Exception as e:
                        logger.warning(f"Element relationship analysis failed: {e}")
                    finally:
                        rel_loop.close()
                
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
            
            # Add language detection results
            if hasattr(self, 'detected_language'):
                result.discovery_analysis = result.discovery_analysis or {}
                result.discovery_analysis['language_detection'] = {
                    "primary_language": self.detected_language.primary_language,
                    "confidence": self.detected_language.confidence,
                    "mixed_language": self.detected_language.mixed_language,
                    "language_distribution": self.detected_language.language_distribution,
                    "technical_terminology": self.detected_language.technical_terminology[:10],
                    "region_indicators": self.detected_language.region_indicators,
                    "optimal_prompt_language": self.detected_language.get_optimal_prompt_language()
                }
            
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
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        page_map = loop.run_until_complete(
                            enhanced_discovery.create_complete_page_map(
                                main_topics=list(enhanced_discovery_result.discovered_patterns.get('patterns', [])),
                                pdf_uri=file_uri
                            )
                        )
                    finally:
                        loop.close()
                else:
                    # Fallback to base discovery
                    from ..discovery import DynamicPlanoDiscovery
                    discovery = DynamicPlanoDiscovery(self.config, pdf_path)
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        page_map = loop.run_until_complete(
                            discovery.create_complete_page_map(
                                main_topics=discovery_result_dict.get('discovered_patterns', {}).get('patterns', []) if discovery_result_dict else [],
                                pdf_uri=file_uri
                            )
                        )
                    finally:
                        loop.close()
                        discovery.close()
                
                result.page_map = page_map
            
            logger.info("Adaptive comprehensive analysis completed successfully")
            
            # FASE 3: Trigger GEPA optimization if needed (background)
            if discovery_result_dict:
                analysis_result_dict = {
                    'general_analysis': result.general_analysis,
                    'sections_analysis': result.sections_analysis,
                    'data_extraction': result.data_extraction,
                    'qa_analysis': result.qa_analysis
                }
                
                # Trigger GEPA optimization in background thread to avoid event loop issues
                import threading
                
                def run_gepa_background():
                    try:
                        # Create new event loop for background GEPA
                        gepa_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(gepa_loop)
                        try:
                            gepa_loop.run_until_complete(
                                self._trigger_gepa_optimization_if_needed(analysis_result_dict, discovery_result_dict)
                            )
                        finally:
                            gepa_loop.close()
                    except Exception as e:
                        logger.warning(f"Background GEPA optimization failed: {e}")
                
                # Run GEPA in background thread
                gepa_thread = threading.Thread(target=run_gepa_background, daemon=True)
                gepa_thread.start()
            
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
