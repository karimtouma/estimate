"""
Main PDF processor with clean architecture and separation of concerns.

This module provides the main PDFProcessor class that orchestrates
the entire PDF analysis workflow using clean code principles.
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from ..core.config import Config
from ..services.gemini_client import GeminiClient, GeminiAPIError
from ..models.schemas import (
    DocumentAnalysis, SectionAnalysis, DataExtraction, QuestionAnswer,
    ComprehensiveAnalysisResult, ProcessingMetadata, AnalysisType
)
from ..utils.file_manager import FileManager
from ..utils.logging_config import setup_logging

# Import the new discovery system (FASE 1)
try:
    from ..discovery import DynamicPlanoDiscovery
    DISCOVERY_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Discovery system (FASE 1) loaded successfully")
except ImportError:
    DISCOVERY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Discovery system not available - using standard analysis")


class ProcessorError(Exception):
    """Base exception for processor errors."""
    pass


class ValidationError(ProcessorError):
    """Raised when input validation fails."""
    pass


class PDFProcessor:
    """
    Main PDF processor class with clean architecture.
    
    Orchestrates the entire PDF analysis workflow using dependency injection
    and separation of concerns for maximum testability and maintainability.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the PDF processor.
        
        Args:
            config: Optional configuration instance. If None, loads from default location.
            
        Raises:
            ProcessorError: If initialization fails
        """
        try:
            # Load configuration
            if config is None:
                from ..core.config import get_config
                config = get_config()
            
            self.config = config
            
            # Setup logging
            setup_logging(config)
            
            # Initialize services
            self.gemini_client = GeminiClient(config)
            self.file_manager = FileManager(config)
            
            # Initialize state
            self.conversation_history: List[Dict[str, Any]] = []
            self._current_file_uri: Optional[str] = None
            
            logger.info("PDFProcessor initialized successfully")
            
            # Validate configuration
            if not self.config.validate():
                logger.warning("Configuration validation failed, but continuing...")
            
        except Exception as e:
            logger.error(f"Failed to initialize PDFProcessor: {e}")
            raise ProcessorError(f"Initialization failed: {e}") from e
    
    def upload_pdf(self, pdf_path: Union[str, Path], display_name: Optional[str] = None) -> str:
        """
        Upload a PDF file for processing.
        
        Args:
            pdf_path: Path to the PDF file
            display_name: Optional display name for the file
            
        Returns:
            URI of the uploaded file
            
        Raises:
            ValidationError: If file validation fails
            ProcessorError: If upload fails
        """
        pdf_path = Path(pdf_path)
        
        # Validate file
        self._validate_pdf_file(pdf_path)
        
        try:
            logger.info(f"Uploading PDF: {pdf_path}")
            file_uri = self.gemini_client.upload_file(pdf_path, display_name)
            self._current_file_uri = file_uri
            
            logger.info(f"PDF uploaded successfully: {file_uri}")
            return file_uri
            
        except GeminiAPIError as e:
            raise ProcessorError(f"Failed to upload PDF: {e}") from e
    
    def _validate_pdf_file(self, pdf_path: Path) -> None:
        """
        Validate PDF file before processing.
        
        Args:
            pdf_path: Path to the PDF file
            
        Raises:
            ValidationError: If validation fails
        """
        if not pdf_path.exists():
            raise ValidationError(f"File not found: {pdf_path}")
        
        if not pdf_path.is_file():
            raise ValidationError(f"Path is not a file: {pdf_path}")
        
        # Check file extension
        if self.config.security.validate_file_types:
            allowed_exts = self.config.security.allowed_extensions
            if pdf_path.suffix.lower() not in allowed_exts:
                raise ValidationError(
                    f"File type not allowed: {pdf_path.suffix}. Allowed: {allowed_exts}"
                )
        
        # Check file size
        file_size = pdf_path.stat().st_size
        max_size = self.config.processing.max_pdf_size_mb * 1024 * 1024
        
        if file_size > max_size:
            raise ValidationError(
                f"File too large: {file_size / (1024*1024):.1f}MB. "
                f"Maximum: {self.config.processing.max_pdf_size_mb}MB"
            )
        
        logger.debug(f"PDF file validation passed: {pdf_path}")
    
    def analyze_document(
        self,
        file_uri: str,
        analysis_type: Union[str, AnalysisType] = AnalysisType.GENERAL
    ) -> Dict[str, Any]:
        """
        Perform structured document analysis.
        
        Args:
            file_uri: URI of the uploaded file
            analysis_type: Type of analysis to perform
            
        Returns:
            Structured analysis results
            
        Raises:
            ProcessorError: If analysis fails
        """
        if isinstance(analysis_type, str):
            try:
                analysis_type = AnalysisType(analysis_type)
            except ValueError:
                raise ValidationError(f"Invalid analysis type: {analysis_type}")
        
        logger.info(f"Starting {analysis_type.value} analysis")
        
        try:
            # Get analysis configuration
            prompt, schema = self._get_analysis_config(analysis_type)
            
            # Generate content
            response_text = self.gemini_client.generate_content(
                file_uri=file_uri,
                prompt=prompt,
                response_schema=schema
            )
            
            # Parse and validate result
            result = json.loads(response_text)
            
            # Add to conversation history
            self._add_to_history({
                "type": "analysis",
                "analysis_type": analysis_type.value,
                "result": result,
                "timestamp": time.time()
            })
            
            logger.info(f"{analysis_type.value} analysis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed for type {analysis_type.value}: {e}")
            raise ProcessorError(f"Analysis failed: {e}") from e
    
    def _get_analysis_config(self, analysis_type: AnalysisType) -> tuple[str, Dict[str, Any]]:
        """
        Get prompt and schema for analysis type with GEPA optimization.
        
        Args:
            analysis_type: Type of analysis
            
        Returns:
            Tuple of (prompt, schema)
        """
        # Get language configuration
        output_language = getattr(self.config.api, 'output_language', 'english')
        force_english = getattr(self.config.api, 'force_english_output', True)
        analysis_instructions = getattr(self.config.analysis, 'analysis_instructions', '')
        
        # Base language instruction
        language_instruction = ""
        if force_english or output_language.lower() == 'english':
            language_instruction = "IMPORTANT: Always respond in English. Provide all analysis, summaries, and insights in English language only. "
        
        # Add analysis instructions if configured
        if analysis_instructions:
            language_instruction += f"{analysis_instructions} "
        
        # Try to get GEPA-optimized prompts
        optimized_prompt = self._get_gepa_optimized_prompt(analysis_type, language_instruction)
        if optimized_prompt:
            logger.info(f"Using GEPA-optimized prompt for {analysis_type.value}")
            return optimized_prompt, self._get_schema(analysis_type)
        
        logger.debug(f"Using baseline prompt for {analysis_type.value}")
        
        prompts = {
            AnalysisType.GENERAL: f"""
            {language_instruction}
            
            Perform a comprehensive analysis of this PDF document. Provide:
            1. A clear and concise executive summary
            2. The main topics addressed in the document
            3. The most important and relevant insights
            4. The identified document type
            5. Your confidence level in the analysis
            
            Be precise, objective and structured in your response. Focus on technical, architectural, structural, mechanical, electrical, and civil engineering aspects if applicable.
            """,
            
            AnalysisType.SECTIONS: f"""
            {language_instruction}
            
            Identify and analyze the main sections of this document.
            For each important section, provide:
            - Section title or heading
            - Content summary (maximum 500 characters)
            - Important data found
            - Questions or concerns arising from the content
            - Section type if identifiable
            
            Focus on technical drawings, specifications, and engineering details.
            """,
            
            AnalysisType.DATA_EXTRACTION: f"""
            {language_instruction}
            
            Extract specific structured data from this document:
            - Entities mentioned (people, organizations, places, companies)
            - Relevant dates and important deadlines
            - Numbers, metrics and key statistics
            - References, citations and external sources
            - Technical terms and specialized vocabulary
            - Engineering specifications and codes
            - Material specifications and standards
            """
        }
        
        schemas = {
            AnalysisType.GENERAL: DocumentAnalysis.model_json_schema(),
            AnalysisType.SECTIONS: SectionAnalysis.model_json_schema(),
            AnalysisType.DATA_EXTRACTION: DataExtraction.model_json_schema()
        }
        
        return prompts[analysis_type], schemas[analysis_type]
    
    def _get_gepa_optimized_prompt(self, analysis_type: AnalysisType, language_instruction: str) -> Optional[str]:
        """
        Get GEPA+DSPy optimized prompt with intelligent learning.
        
        Args:
            analysis_type: Type of analysis
            language_instruction: Language instruction to prepend
            
        Returns:
            Optimized prompt using GEPA+DSPy or None if optimization should be triggered
        """
        try:
            # Initialize GEPA+DSPy system
            gepa_system = self._initialize_gepa_dspy_system()
            if not gepa_system:
                return None
            
            # Get current performance metrics for this analysis type
            current_performance = self._get_analysis_performance_metrics(analysis_type)
            
            # Check if we need to trigger optimization
            if self._should_trigger_optimization(analysis_type, current_performance):
                logger.info(f"Triggering GEPA+DSPy optimization for {analysis_type.value}")
                self._trigger_intelligent_optimization(analysis_type, current_performance)
            
            # Load optimized prompt from GEPA+DSPy system
            optimized_prompt = gepa_system.get_optimized_prompt(analysis_type)
            if optimized_prompt:
                # Enhance with DSPy reasoning chain
                enhanced_prompt = self._enhance_with_dspy_reasoning(
                    optimized_prompt, analysis_type, language_instruction
                )
                
                logger.info(f"Using GEPA+DSPy optimized prompt for {analysis_type.value} (performance boost: {gepa_system.get_improvement_score(analysis_type):.2f})")
                return enhanced_prompt
            
            return None
            
        except Exception as e:
            logger.error(f"GEPA+DSPy optimization failed: {e}")
            # Don't fallback - log error and continue with optimization trigger
            self._trigger_intelligent_optimization(analysis_type, {"error": str(e)})
            return None
    
    def _should_auto_optimize(self) -> bool:
        """
        Determine if GEPA optimization should be auto-generated.
        
        Returns:
            True if auto-optimization should run
        """
        # Check configuration
        auto_optimize = getattr(self.config.analysis, 'auto_gepa_optimization', True)
        if not auto_optimize:
            return False
        
        # Check if we have enough processing history to optimize
        output_dir = Path(self.config.get_directories()["output"])
        analysis_files = list(output_dir.glob("*_comprehensive_analysis.json"))
        
        # Auto-optimize if we have processed at least 3 documents
        return len(analysis_files) >= 3
    
    def _auto_generate_gepa_optimization(self) -> None:
        """
        Auto-generate GEPA optimization in background.
        """
        try:
            # Import GEPA components
            from ..optimization.gepa_optimizer import GEPAPromptOptimizer
            
            logger.info("Starting background GEPA optimization...")
            
            # Create simplified optimization
            gepa_optimizer = GEPAPromptOptimizer(self.config)
            
            # Generate quick optimization (reduced parameters for auto-mode)
            quick_config = {
                "num_generations": 5,  # Reduced from 10
                "population_size": 4,  # Reduced from 8
                "target_improvement": 0.10,  # Reduced from 0.15
                "auto_mode": True
            }
            
            # Run optimization asynchronously (don't block main analysis)
            import asyncio
            import threading
            
            def run_optimization():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    results = loop.run_until_complete(
                        gepa_optimizer.optimize_blueprint_analysis_prompts()
                    )
                    
                    # Save results
                    output_dir = Path(self.config.get_directories()["output"])
                    results_file = output_dir / "gepa_optimization_results.json"
                    gepa_optimizer.save_optimization_results(results, results_file)
                    
                    logger.info("Background GEPA optimization completed successfully")
                    
                except Exception as e:
                    logger.warning(f"Background GEPA optimization failed: {e}")
                finally:
                    loop.close()
            
            # Start optimization in background thread
            optimization_thread = threading.Thread(target=run_optimization, daemon=True)
            optimization_thread.start()
            
        except Exception as e:
            logger.warning(f"Failed to start background GEPA optimization: {e}")
    
    def _get_schema(self, analysis_type: AnalysisType) -> Dict[str, Any]:
        """
        Get schema for analysis type.
        
        Args:
            analysis_type: Type of analysis
            
        Returns:
            JSON schema for the analysis type
        """
        schemas = {
            AnalysisType.GENERAL: DocumentAnalysis.model_json_schema(),
            AnalysisType.SECTIONS: SectionAnalysis.model_json_schema(),
            AnalysisType.DATA_EXTRACTION: DataExtraction.model_json_schema()
        }
        
        return schemas[analysis_type]
    
    def _get_gepa_usage_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about GEPA usage in current analysis.
        
        Returns:
            Dictionary with GEPA usage information
        """
        try:
            if not getattr(self.config.analysis, 'enable_dspy_optimization', False):
                return None
            
            output_dir = Path(self.config.get_directories()["output"])
            
            # Check for GEPA optimization results
            gepa_file = output_dir / "gepa_optimization_results.json"
            intelligent_file = output_dir / "intelligent_optimization_state.json"
            
            gepa_info = {
                "gepa_enabled": True,
                "optimization_available": False,
                "intelligent_system_active": False,
                "performance_tracking": True,
                "last_optimization": None
            }
            
            if gepa_file.exists():
                gepa_info["optimization_available"] = True
                stat = gepa_file.stat()
                gepa_info["last_optimization"] = stat.st_mtime
            
            if intelligent_file.exists():
                gepa_info["intelligent_system_active"] = True
                try:
                    with open(intelligent_file, 'r', encoding='utf-8') as f:
                        state_data = json.load(f)
                    
                    optimizations = state_data.get("optimizations", {})
                    gepa_info["cached_optimizations"] = len(optimizations)
                    gepa_info["last_updated"] = state_data.get("last_updated")
                    
                except Exception:
                    pass
            
            return gepa_info
            
        except Exception as e:
            logger.warning(f"Failed to get GEPA usage info: {e}")
            return None
    
    def _initialize_gepa_dspy_system(self):
        """
        Initialize intelligent GEPA+DSPy optimization system.
        
        Returns:
            GEPA+DSPy system instance or None if not available
        """
        try:
            # Check if optimization is enabled
            if not getattr(self.config.analysis, 'enable_dspy_optimization', False):
                return None
            
            # Import and initialize the intelligent system
            from ..optimization.intelligent_gepa_system import IntelligentGEPADSPySystem
            
            # Create or load existing system
            gepa_system = IntelligentGEPADSPySystem(self.config)
            
            return gepa_system
            
        except ImportError:
            logger.warning("GEPA+DSPy system not available - creating intelligent system")
            return self._create_intelligent_system()
        except Exception as e:
            logger.error(f"Failed to initialize GEPA+DSPy system: {e}")
            return None
    
    def _create_intelligent_system(self):
        """Create a lightweight intelligent optimization system."""
        try:
            from ..optimization.intelligent_gepa_system import IntelligentPromptSystem
            return IntelligentPromptSystem(self.config)
        except ImportError:
            logger.warning("IntelligentPromptSystem not available")
            return None
    
    def _get_analysis_performance_metrics(self, analysis_type: AnalysisType) -> Dict[str, Any]:
        """
        Get performance metrics for specific analysis type.
        
        Args:
            analysis_type: Type of analysis
            
        Returns:
            Performance metrics dictionary
        """
        try:
            output_dir = Path(self.config.get_directories()["output"])
            
            # Analyze recent results to calculate performance
            recent_files = sorted(
                output_dir.glob("*_comprehensive_analysis.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )[:10]  # Last 10 analyses
            
            metrics = {
                "accuracy_scores": [],
                "confidence_scores": [],
                "processing_times": [],
                "error_rates": [],
                "total_analyses": len(recent_files)
            }
            
            for file_path in recent_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Extract metrics based on analysis type
                    if analysis_type == AnalysisType.GENERAL and "general_analysis" in data:
                        confidence = data["general_analysis"].get("confidence_score", 0)
                        metrics["confidence_scores"].append(confidence)
                    
                    elif analysis_type == AnalysisType.SECTIONS and "sections_analysis" in data:
                        # Calculate section analysis quality
                        sections = data["sections_analysis"]
                        if sections:
                            avg_quality = sum(len(s.get("important_data", [])) for s in sections) / len(sections)
                            metrics["accuracy_scores"].append(min(avg_quality / 10, 1.0))
                    
                    elif analysis_type == AnalysisType.DATA_EXTRACTION and "data_extraction" in data:
                        # Calculate extraction completeness
                        extraction = data["data_extraction"]
                        completeness = sum([
                            len(extraction.get("entities", [])),
                            len(extraction.get("dates", [])),
                            len(extraction.get("numbers", [])),
                            len(extraction.get("references", [])),
                            len(extraction.get("key_terms", []))
                        ]) / 50  # Normalize to 0-1
                        metrics["accuracy_scores"].append(min(completeness, 1.0))
                
                except Exception as e:
                    metrics["error_rates"].append(1)
                    continue
            
            # Calculate aggregate metrics
            if metrics["confidence_scores"]:
                metrics["avg_confidence"] = sum(metrics["confidence_scores"]) / len(metrics["confidence_scores"])
            if metrics["accuracy_scores"]:
                metrics["avg_accuracy"] = sum(metrics["accuracy_scores"]) / len(metrics["accuracy_scores"])
            
            metrics["error_rate"] = len(metrics["error_rates"]) / max(len(recent_files), 1)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate performance metrics: {e}")
            return {"error": str(e), "total_analyses": 0}
    
    def _should_trigger_optimization(self, analysis_type: AnalysisType, performance: Dict[str, Any]) -> bool:
        """
        Determine if optimization should be triggered based on performance.
        
        Args:
            analysis_type: Type of analysis
            performance: Current performance metrics
            
        Returns:
            True if optimization should be triggered
        """
        # Trigger optimization if:
        # 1. We have enough data (5+ analyses)
        # 2. Performance is below threshold
        # 3. Error rate is too high
        # 4. No recent optimization
        
        min_analyses = 5
        confidence_threshold = 0.8
        accuracy_threshold = 0.75
        max_error_rate = 0.2
        
        if performance.get("total_analyses", 0) < min_analyses:
            return False
        
        # Check performance thresholds
        avg_confidence = performance.get("avg_confidence", 1.0)
        avg_accuracy = performance.get("avg_accuracy", 1.0)
        error_rate = performance.get("error_rate", 0.0)
        
        should_optimize = (
            avg_confidence < confidence_threshold or
            avg_accuracy < accuracy_threshold or
            error_rate > max_error_rate
        )
        
        if should_optimize:
            logger.info(f"Optimization triggered for {analysis_type.value}: confidence={avg_confidence:.2f}, accuracy={avg_accuracy:.2f}, error_rate={error_rate:.2f}")
        
        return should_optimize
    
    def _trigger_intelligent_optimization(self, analysis_type: AnalysisType, performance: Dict[str, Any]):
        """
        Trigger intelligent GEPA+DSPy optimization in background.
        
        Args:
            analysis_type: Type of analysis to optimize
            performance: Current performance metrics
        """
        try:
            import threading
            import asyncio
            
            def run_optimization():
                try:
                    # Create new event loop for background thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # Run intelligent optimization
                    from ..optimization.intelligent_gepa_system import run_intelligent_optimization
                    
                    optimization_result = loop.run_until_complete(
                        run_intelligent_optimization(
                            self.config, 
                            analysis_type, 
                            performance
                        )
                    )
                    
                    logger.info(f"Background optimization completed for {analysis_type.value}: improvement={optimization_result.get('improvement', 0):.2f}")
                    
                except Exception as e:
                    logger.error(f"Background optimization failed: {e}")
                finally:
                    loop.close()
            
            # Start optimization in background
            optimization_thread = threading.Thread(target=run_optimization, daemon=True)
            optimization_thread.start()
            
        except Exception as e:
            logger.error(f"Failed to trigger intelligent optimization: {e}")
    
    def _enhance_with_dspy_reasoning(self, base_prompt: str, analysis_type: AnalysisType, language_instruction: str) -> str:
        """
        Enhance prompt with DSPy reasoning chain.
        
        Args:
            base_prompt: Base optimized prompt from GEPA
            analysis_type: Type of analysis
            language_instruction: Language instruction
            
        Returns:
            Enhanced prompt with DSPy reasoning
        """
        # DSPy reasoning enhancement based on analysis type
        reasoning_enhancements = {
            AnalysisType.GENERAL: """
Think step by step:
1. First, identify the document type and primary purpose
2. Extract key structural and architectural elements
3. Analyze technical specifications and standards
4. Synthesize insights about construction methodology
5. Assess overall project scope and complexity
""",
            AnalysisType.SECTIONS: """
Use systematic section analysis:
1. Identify distinct document sections and their purposes
2. Extract critical data from each section
3. Note relationships between sections
4. Flag any inconsistencies or missing information
5. Summarize section-specific technical requirements
""",
            AnalysisType.DATA_EXTRACTION: """
Apply comprehensive data mining:
1. Scan for all entities (companies, people, locations)
2. Extract numerical data (dimensions, loads, specifications)
3. Identify dates and project timeline information
4. Collect technical references and standards
5. Gather specialized terminology and key concepts
"""
        }
        
        reasoning_chain = reasoning_enhancements.get(analysis_type, "")
        
        return f"""{language_instruction}

{reasoning_chain}

{base_prompt}

Remember: Provide precise, technical analysis with high confidence scores for accurate information."""
    
    def multi_turn_analysis(
        self,
        file_uri: str,
        questions: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Perform multi-turn Q&A analysis.
        
        Args:
            file_uri: URI of the uploaded file
            questions: List of questions to ask
            
        Returns:
            List of question-answer results
            
        Raises:
            ProcessorError: If multi-turn analysis fails
        """
        if not questions:
            raise ValidationError("No questions provided for multi-turn analysis")
        
        logger.info(f"Starting multi-turn analysis with {len(questions)} questions")
        
        try:
            # Build context from conversation history
            context_parts = self._build_context_parts()
            
            # Generate multi-turn content
            results = self.gemini_client.generate_multi_turn_content(
                file_uri=file_uri,
                questions=questions,
                context_parts=context_parts
            )
            
            # Add to conversation history
            for result in results:
                if "error" not in result:
                    self._add_to_history({
                        "type": "question",
                        "question": result["question"],
                        "result": result,
                        "timestamp": time.time()
                    })
            
            logger.info(f"Multi-turn analysis completed: {len(results)} responses")
            return results
            
        except Exception as e:
            logger.error(f"Multi-turn analysis failed: {e}")
            raise ProcessorError(f"Multi-turn analysis failed: {e}") from e
    
    def comprehensive_analysis(
        self,
        pdf_path: Union[str, Path],
        questions: Optional[List[str]] = None,
        enable_discovery: bool = True
    ) -> ComprehensiveAnalysisResult:
        """
        Perform comprehensive analysis of a PDF document.
        
        Now includes FASE 1: Discovery system for adaptive analysis
        without predefined taxonomies.
        
        Args:
            pdf_path: Path to the PDF file
            questions: Optional list of specific questions
            enable_discovery: Enable discovery phase for adaptive analysis
            
        Returns:
            Comprehensive analysis results
            
        Raises:
            ProcessorError: If comprehensive analysis fails
        """
        pdf_path = Path(pdf_path)
        
        logger.info(f"Starting comprehensive analysis of: {pdf_path}")
        
        # Upload PDF once for all analyses
        file_uri = None
        
        # FASE 1: Run discovery phase if enabled and available
        discovery_result = None
        if enable_discovery and DISCOVERY_AVAILABLE:
            logger.info("🔍 FASE 1: Running discovery phase for adaptive analysis...")
            try:
                # Upload PDF if not already uploaded
                if file_uri is None:
                    file_uri = self.upload_pdf(pdf_path)
                
                import asyncio
                discovery = DynamicPlanoDiscovery(self.config, pdf_path)
                
                # Run async discovery in sync context with the uploaded PDF URI
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    discovery_result = loop.run_until_complete(
                        discovery.initial_exploration(sample_size=10, pdf_uri=file_uri)
                    )
                    
                    logger.info(f"✅ Discovery complete:")
                    logger.info(f"  - Document type: {discovery_result.document_type}")
                    logger.info(f"  - Industry domain: {discovery_result.industry_domain}")
                    logger.info(f"  - Patterns found: {len(discovery_result.discovered_patterns)}")
                    logger.info(f"  - Nomenclature codes: {len(discovery_result.nomenclature_system.get('patterns', {}))}")
                    
                    # Save discovery results
                    discovery_output = {
                        "document_type": discovery_result.document_type,
                        "industry_domain": discovery_result.industry_domain,
                        "discovered_patterns": discovery_result.discovered_patterns,
                        "nomenclature_system": discovery_result.nomenclature_system,
                        "page_organization": discovery_result.page_organization,
                        "element_types": discovery_result.element_types,
                        "confidence_score": discovery_result.confidence_score,
                        "metadata": discovery_result.discovery_metadata
                    }
                    
                finally:
                    loop.close()
                    discovery.close()
                
            except Exception as e:
                logger.warning(f"Discovery phase failed, continuing with standard analysis: {e}")
                discovery_result = None
        
        try:
            # Upload file if not already uploaded
            if file_uri is None:
                file_uri = self.upload_pdf(pdf_path)
            
            # Use default questions if none provided
            if questions is None:
                questions = self.config.analysis.default_questions
            
            # Initialize result
            result = ComprehensiveAnalysisResult(
                file_info={
                    "path": str(pdf_path),
                    "uri": file_uri,
                    "timestamp": time.time(),
                    "discovery_enabled": enable_discovery and DISCOVERY_AVAILABLE,
                    "size_bytes": pdf_path.stat().st_size
                }
            )
            
            # Execute enabled analysis types IN PARALLEL for maximum performance
            enabled_types = self.config.analysis.enabled_types
            
            # PARALLEL PROCESSING: Run all core analyses simultaneously
            parallel_analyses = []
            analysis_tasks = []
            
            if "general" in enabled_types:
                parallel_analyses.append(("general", AnalysisType.GENERAL))
            if "sections" in enabled_types:
                parallel_analyses.append(("sections", AnalysisType.SECTIONS))
            if "data_extraction" in enabled_types:
                parallel_analyses.append(("data_extraction", AnalysisType.DATA_EXTRACTION))
            
            if parallel_analyses:
                # Check if parallel processing is enabled
                enable_parallel = getattr(self.config.processing, 'enable_parallel_core_analysis', True)
                max_workers = getattr(self.config.processing, 'core_analysis_threads', 3)
                
                if enable_parallel and len(parallel_analyses) > 1:
                    import asyncio
                    import concurrent.futures
                    from functools import partial
                    
                    logger.info(f"🚀 Starting PARALLEL core analysis: {len(parallel_analyses)} phases simultaneously")
                    start_parallel = time.time()
                    
                    # Create thread pool for parallel execution
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # Submit all analysis tasks
                        future_to_analysis = {}
                        for analysis_name, analysis_type in parallel_analyses:
                            future = executor.submit(self.analyze_document, file_uri, analysis_type)
                            future_to_analysis[future] = (analysis_name, analysis_type)
                        
                        # Collect results as they complete
                        for future in concurrent.futures.as_completed(future_to_analysis):
                            analysis_name, analysis_type = future_to_analysis[future]
                            try:
                                analysis_data = future.result()
                                
                                # Process results based on analysis type
                                if analysis_name == "general":
                                    result.general_analysis = DocumentAnalysis(**analysis_data)
                                    logger.info("✅ General analysis completed in parallel")
                                elif analysis_name == "sections":
                                    result.sections_analysis = [SectionAnalysis(**analysis_data)]
                                    logger.info("✅ Sections analysis completed in parallel")
                                elif analysis_name == "data_extraction":
                                    result.data_extraction = DataExtraction(**analysis_data)
                                    logger.info("✅ Data extraction completed in parallel")
                                    
                            except Exception as e:
                                logger.error(f"❌ {analysis_name} analysis failed in parallel execution: {e}")
                    
                    parallel_time = time.time() - start_parallel
                    logger.info(f"🎯 PARALLEL core analysis completed in {parallel_time:.1f}s (vs. {len(parallel_analyses)*30:.0f}s sequential)")
                    
                else:
                    # Fallback to sequential processing
                    logger.info("📋 Using SEQUENTIAL core analysis (parallel disabled or single analysis)")
                    for analysis_name, analysis_type in parallel_analyses:
                        try:
                            analysis_data = self.analyze_document(file_uri, analysis_type)
                            
                            if analysis_name == "general":
                                result.general_analysis = DocumentAnalysis(**analysis_data)
                            elif analysis_name == "sections":
                                result.sections_analysis = [SectionAnalysis(**analysis_data)]
                            elif analysis_name == "data_extraction":
                                result.data_extraction = DataExtraction(**analysis_data)
                                
                        except Exception as e:
                            logger.error(f"❌ {analysis_name} analysis failed: {e}")
            else:
                logger.warning("No core analysis types enabled")
            
            # Q&A analysis
            if questions:
                try:
                    qa_results = self.multi_turn_analysis(file_uri, questions)
                    result.qa_analysis = [
                        QuestionAnswer(**qa_data) for qa_data in qa_results
                        if "error" not in qa_data
                    ]
                except Exception as e:
                    logger.error(f"Q&A analysis failed: {e}")
            
            # Add discovery results if available
            if discovery_result and 'discovery_output' in locals():
                result.discovery_analysis = discovery_output
                logger.info("📊 Discovery results added to comprehensive analysis")
            
            # CREATE COMPLETE PAGE MAP if enabled and general analysis is available
            enable_page_mapping = getattr(self.config.processing, 'enable_complete_page_mapping', True)
            if enable_page_mapping and result.general_analysis and result.general_analysis.main_topics:
                try:
                    logger.info("🗺️ Creating complete page-by-page map...")
                    
                    # Re-initialize discovery for page mapping (reuse existing PDF connection if possible)
                    page_discovery = DynamicPlanoDiscovery(self.config, pdf_path)
                    
                    # Create complete page map using main topics from general analysis
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        page_map_data = loop.run_until_complete(
                            page_discovery.create_complete_page_map(
                                pdf_uri=file_uri,
                                main_topics=result.general_analysis.main_topics
                            )
                        )
                        
                        # Convert to Pydantic model
                        from ..models.schemas import DocumentPageMap, PageClassification
                        
                        page_classifications = []
                        for page_data in page_map_data["pages"]:
                            page_classifications.append(PageClassification(**page_data))
                        
                        result.page_map = DocumentPageMap(
                            total_pages=page_map_data["total_pages"],
                            pages=page_classifications,
                            category_distribution=page_map_data["category_distribution"],
                            coverage_analysis=page_map_data["coverage_analysis"]
                        )
                        
                        logger.info(f"✅ Page map created: {len(page_classifications)} pages classified")
                        logger.info(f"📊 Categories found: {list(page_map_data['category_distribution']['_summary']['pages_per_category'].keys())}")
                        
                    finally:
                        loop.close()
                        page_discovery.close()
                        
                except Exception as e:
                    logger.warning(f"Page map creation failed: {e}")
                    # Continue without page map
            
            # Add metadata with GEPA information
            gepa_info = self._get_gepa_usage_info()
            result.metadata = ProcessingMetadata(
                timestamp=time.time(),
                processor_version="2.0.0",
                model_used=self.config.api.default_model,
                config_file=str(self.config.config_path),
                environment="container" if self.config.is_container_environment() else "local",
                file_info=result.file_info,
                discovery_enabled=enable_discovery and DISCOVERY_AVAILABLE
            )
            
            # Add GEPA information to metadata if available
            if gepa_info:
                result.metadata.__dict__.update(gepa_info)
            
            logger.info("Comprehensive analysis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            raise ProcessorError(f"Comprehensive analysis failed: {e}") from e
    
    def save_results(
        self,
        results: Union[Dict[str, Any], ComprehensiveAnalysisResult],
        output_path: Union[str, Path]
    ) -> Path:
        """
        Save analysis results to file.
        
        Args:
            results: Analysis results to save
            output_path: Output file path
            
        Returns:
            Path to saved file
            
        Raises:
            ProcessorError: If saving fails
        """
        try:
            # Convert Pydantic model to dict if needed
            if isinstance(results, ComprehensiveAnalysisResult):
                results_dict = results.model_dump(exclude_none=True)
            else:
                results_dict = results
            
            # Use file manager to save
            saved_path = self.file_manager.save_results(results_dict, output_path)
            
            logger.info(f"Results saved to: {saved_path}")
            return saved_path
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise ProcessorError(f"Failed to save results: {e}") from e
    
    def _build_context_parts(self) -> List:
        """Build context parts from conversation history."""
        from google.genai import types
        
        context_parts = []
        
        # Get recent history entries
        recent_history = self.conversation_history[-3:] if self.conversation_history else []
        
        if recent_history:
            context_text = "Contexto de análisis previo:\n"
            for entry in recent_history:
                if entry["type"] == "analysis":
                    context_text += f"- Análisis {entry['analysis_type']}: {json.dumps(entry['result'], indent=2)}\n"
                elif entry["type"] == "question":
                    context_text += f"- Q&A: {entry['question']}\n"
            
            context_parts.append(types.Part.from_text(text=context_text))
        
        return context_parts
    
    def _add_to_history(self, entry: Dict[str, Any]) -> None:
        """Add entry to conversation history with size management."""
        self.conversation_history.append(entry)
        
        # Limit history size
        max_items = self.config.analysis.max_history_items
        if len(self.conversation_history) > max_items:
            self.conversation_history = self.conversation_history[-max_items:]
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self.conversation_history.copy()
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        import sys
        import os
        
        return {
            "processor_version": "2.0.0",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform,
            "working_directory": os.getcwd(),
            "container_environment": self.config.is_container_environment(),
            "config_file": str(self.config.config_path),
            "model": self.config.api.default_model,
            "history_size": len(self.conversation_history)
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Cleanup resources
        if hasattr(self, 'gemini_client'):
            self.gemini_client.cleanup_uploaded_files()
        
        logger.info("PDFProcessor cleanup completed")
