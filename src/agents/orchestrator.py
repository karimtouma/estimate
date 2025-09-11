"""
Multi-agent orchestration system for structural blueprint analysis.

This module implements the master orchestrator that coordinates multiple
specialized agents for comprehensive blueprint analysis.
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

from ..models.blueprint_schemas import (
    DocumentTaxonomy, PageTaxonomy, StructuralSummary, DocumentPattern,
    ProcessingSession, AnalysisTask, AgentResult, ContextWindow
)
from ..services.gemini_multimodal import GeminiMultimodalProcessor, GeminiTaxonomyEngine
from ..core.config import Config
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class AgentPool:
    """Pool of available agents for task processing."""
    page_analyzers: List[Any] = field(default_factory=list)
    taxonomy_engines: List[Any] = field(default_factory=list)
    element_detectors: List[Any] = field(default_factory=list)
    context_managers: List[Any] = field(default_factory=list)
    
    max_concurrent_tasks: int = 4
    active_tasks: Dict[str, Any] = field(default_factory=dict)


class TaskScheduler:
    """
    Intelligent task scheduler for optimal agent utilization.
    """
    
    def __init__(self, agent_pool: AgentPool):
        self.agent_pool = agent_pool
        self.task_queue = asyncio.PriorityQueue()
        self.completed_tasks = {}
        
    def create_task_plan(self, document_pages: List[Dict[str, Any]]) -> List[AnalysisTask]:
        """
        Create optimal task plan for document processing.
        
        Args:
            document_pages: List of processed page data
            
        Returns:
            List of analysis tasks ordered by priority
        """
        tasks = []
        
        for i, page_data in enumerate(document_pages):
            page_number = page_data["page_info"]["page_number"]
            
            # Determine task priority based on page characteristics
            priority = self.calculate_task_priority(page_data)
            
            # Determine required tools
            required_tools = self.determine_required_tools(page_data)
            
            # Estimate processing duration
            estimated_duration = self.estimate_processing_time(page_data)
            
            task = AnalysisTask(
                task_id=f"page_analysis_{page_number}",
                task_type="page_taxonomy_generation",
                page_number=page_number,
                priority=priority,
                required_tools=required_tools,
                expected_duration=estimated_duration,
                dependencies=self.get_task_dependencies(page_number, document_pages)
            )
            
            tasks.append(task)
        
        # Sort by priority (higher priority first)
        tasks.sort(key=lambda t: t.priority, reverse=True)
        
        logger.info(f"Created task plan with {len(tasks)} tasks")
        return tasks
    
    def calculate_task_priority(self, page_data: Dict[str, Any]) -> int:
        """Calculate task priority based on page characteristics."""
        priority = 5  # Base priority
        
        # Increase priority for complex pages
        visual_features = getattr(page_data, 'visual_features', None)
        if visual_features and hasattr(visual_features, 'geometric_complexity'):
            complexity = visual_features.geometric_complexity
        else:
            complexity = 0.5
            
        if complexity > 0.8:
            priority += 3
        elif complexity > 0.6:
            priority += 2
        
        # Increase priority for pages with rich content
        textual_features = getattr(page_data, 'textual_features', None)
        if textual_features and hasattr(textual_features, 'processed_text'):
            text_length = len(textual_features.processed_text)
        else:
            text_length = 0
            
        if text_length > 200:
            priority += 2
        elif text_length > 100:
            priority += 1
        
        # First and last pages often have important context
        page_info = getattr(page_data, 'page_info', None)
        if page_info and hasattr(page_info, 'page_number'):
            page_num = page_info.page_number
            if page_num <= 2:  # Title pages, covers
                priority += 2
        
        return min(priority, 10)  # Cap at maximum priority
    
    def determine_required_tools(self, page_data: Dict[str, Any]) -> List[str]:
        """Determine required tools based on page characteristics."""
        tools = ["gemini_vision", "taxonomy_engine"]  # Always required
        
        # Add OCR if text is present
        page_info = getattr(page_data, 'page_info', None)
        if page_info and hasattr(page_info, 'has_text') and page_info.has_text:
            tools.append("ocr_engine")
        
        # Add image processing for complex drawings
        visual_features = getattr(page_data, 'visual_features', None)
        if visual_features and hasattr(visual_features, 'geometric_complexity'):
            complexity = visual_features.geometric_complexity
        else:
            complexity = 0.0
            
        if complexity > 0.6:
            tools.append("image_processor")
        
        # Add element detector for technical drawings
        if visual_features and hasattr(visual_features, 'line_density'):
            line_density = visual_features.line_density
        else:
            line_density = 0.0
            
        if line_density > 50:
            tools.append("element_detector")
        
        return tools
    
    def estimate_processing_time(self, page_data: Dict[str, Any]) -> float:
        """Estimate processing time for a page."""
        base_time = 15.0  # Base processing time in seconds
        
        # Add time for complexity
        visual_features = getattr(page_data, 'visual_features', None)
        if visual_features and hasattr(visual_features, 'geometric_complexity'):
            complexity = visual_features.geometric_complexity
        else:
            complexity = 0.5
        base_time += complexity * 20.0
        
        # Add time for text processing
        textual_features = getattr(page_data, 'textual_features', None)
        if textual_features and hasattr(textual_features, 'processed_text'):
            text_length = len(textual_features.processed_text)
        else:
            text_length = 0
        base_time += (text_length / 100) * 2.0
        
        return base_time
    
    def get_task_dependencies(self, page_number: int, document_pages: List[Dict[str, Any]]) -> List[str]:
        """Determine task dependencies."""
        dependencies = []
        
        # Pages depend on previous pages for context
        if page_number > 1:
            dependencies.append(f"page_analysis_{page_number - 1}")
        
        # Title pages should be processed first
        if page_number > 2:
            dependencies.append("page_analysis_1")
        
        return dependencies


class MasterOrchestrator:
    """
    Master orchestrator for multi-agent structural blueprint analysis.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.page_processor = GeminiMultimodalProcessor(config)
        self.taxonomy_engine = GeminiTaxonomyEngine(config)
        
        # Agent management
        self.agent_pool = AgentPool()
        self.task_scheduler = TaskScheduler(self.agent_pool)
        
        # Context management
        self.global_context = {}
        self.context_window = None
        
        # Performance tracking
        self.performance_metrics = {
            "total_pages_processed": 0,
            "total_processing_time": 0.0,
            "average_confidence": 0.0,
            "success_rate": 0.0
        }
        
        logger.info("Master Orchestrator initialized")
    
    async def analyze_structural_blueprint(self, pdf_path: Path) -> DocumentTaxonomy:
        """
        Orchestrate complete analysis of structural blueprint document.
        
        Args:
            pdf_path: Path to the structural blueprint PDF
            
        Returns:
            Complete DocumentTaxonomy with intelligent analysis
        """
        logger.info(f"Starting structural blueprint analysis: {pdf_path}")
        session_start = time.time()
        
        # Create processing session
        session = ProcessingSession(
            session_id=f"session_{int(time.time())}",
            document_path=str(pdf_path),
            processing_config={"config_loaded": True},  # Simplified config representation
            agents_used=["master_orchestrator", "page_processor", "taxonomy_engine"],
            start_time=session_start
        )
        
        try:
            # Phase 1: Document preprocessing
            logger.info("Phase 1: Document preprocessing and page extraction")
            document_pages = await self.page_processor.process_document_pages(pdf_path)
            
            # Phase 2: Task planning and scheduling
            logger.info("Phase 2: Creating optimal task plan")
            tasks = self.task_scheduler.create_task_plan(document_pages)
            
            # Phase 3: Initialize context management
            logger.info("Phase 3: Initializing context management")
            self.context_window = ContextWindow(
                window_id=f"context_{session.session_id}",
                document_id=session.session_id,
                pages_context=[],
                max_pages=10,
                current_focus=1
            )
            
            # Phase 4: Distributed page analysis
            logger.info("Phase 4: Executing distributed page analysis")
            page_taxonomies = await self.execute_distributed_analysis(
                tasks, document_pages, session
            )
            
            # Phase 5: Cross-page pattern analysis
            logger.info("Phase 5: Analyzing cross-page patterns")
            global_patterns = await self.analyze_global_patterns(page_taxonomies)
            
            # Phase 6: Structural summary generation
            logger.info("Phase 6: Generating structural summary")
            structural_summary = await self.generate_structural_summary(
                page_taxonomies, global_patterns
            )
            
            # Phase 7: Final synthesis
            logger.info("Phase 7: Synthesizing final document taxonomy")
            document_taxonomy = await self.synthesize_document_taxonomy(
                pdf_path, page_taxonomies, global_patterns, structural_summary, session
            )
            
            # Complete session
            session.end_time = time.time()
            session.document_taxonomy = document_taxonomy
            session.total_pages_processed = len(page_taxonomies)
            
            # Update performance metrics
            self.update_performance_metrics(session)
            
            logger.info(f"Structural blueprint analysis completed in {session.duration:.2f} seconds")
            
            return document_taxonomy
            
        except Exception as e:
            logger.error(f"Structural blueprint analysis failed: {e}")
            session.errors_encountered.append({
                "error": str(e),
                "timestamp": time.time(),
                "phase": "analysis"
            })
            
            # Return partial results if available
            if hasattr(session, 'partial_results'):
                return session.partial_results
            
            raise
    
    async def execute_distributed_analysis(
        self,
        tasks: List[AnalysisTask],
        document_pages: List[Dict[str, Any]],
        session: ProcessingSession
    ) -> List[PageTaxonomy]:
        """Execute page analysis tasks in optimal distribution."""
        
        page_taxonomies = []
        semaphore = asyncio.Semaphore(self.agent_pool.max_concurrent_tasks)
        
        async def process_page_task(task: AnalysisTask) -> PageTaxonomy:
            """Process individual page task."""
            async with semaphore:
                try:
                    # Find corresponding page data
                    page_data = next(
                        p for p in document_pages 
                        if p["page_info"]["page_number"] == task.page_number
                    )
                    
                    # Get current document context
                    document_context = self.build_document_context(page_taxonomies)
                    
                    # Generate taxonomy
                    taxonomy = await self.taxonomy_engine.generate_page_taxonomy(
                        page_data, document_context
                    )
                    
                    # Update context window
                    self.context_window.add_page(taxonomy)
                    
                    # Record agent result
                    agent_result = AgentResult(
                        agent_id="taxonomy_engine_1",
                        agent_type="taxonomy_engine",
                        task_id=task.task_id,
                        page_number=task.page_number,
                        primary_result=taxonomy.model_dump(),
                        processing_time=taxonomy.processing_time,
                        tools_used=taxonomy.tools_used,
                        confidence=taxonomy.analysis_confidence,
                        completeness=taxonomy.completeness_score
                    )
                    
                    session.agent_results.append(agent_result)
                    
                    logger.info(f"Completed analysis for page {task.page_number}")
                    
                    return taxonomy
                    
                except Exception as e:
                    logger.error(f"Task {task.task_id} failed: {e}")
                    # Return minimal taxonomy on failure
                    return self.taxonomy_engine.create_fallback_taxonomy(
                        {"page_info": {"page_number": task.page_number}}, str(e)
                    )
        
        # Execute all tasks concurrently
        taxonomies = await asyncio.gather(*[
            process_page_task(task) for task in tasks
        ], return_exceptions=True)
        
        # Filter out exceptions and sort by page number
        valid_taxonomies = [
            t for t in taxonomies 
            if isinstance(t, PageTaxonomy)
        ]
        
        valid_taxonomies.sort(key=lambda t: t.page_number)
        
        return valid_taxonomies
    
    def build_document_context(self, existing_taxonomies: List[PageTaxonomy]) -> Dict[str, Any]:
        """Build document context from existing page analyses."""
        if not existing_taxonomies:
            return {}
        
        # Extract patterns from existing pages
        page_types = [t.page_type.value for t in existing_taxonomies]
        element_types = []
        
        for taxonomy in existing_taxonomies:
            for element in taxonomy.structural_elements:
                element_types.append(element.element_type.value)
        
        # Count occurrences
        from collections import Counter
        page_type_counts = Counter(page_types)
        element_type_counts = Counter(element_types)
        
        return {
            "previous_page_types": list(page_type_counts.keys()),
            "most_common_page_type": page_type_counts.most_common(1)[0][0] if page_type_counts else "unknown",
            "recurring_elements": [elem for elem, count in element_type_counts.most_common(5)],
            "total_pages_analyzed": len(existing_taxonomies),
            "average_confidence": sum(t.analysis_confidence for t in existing_taxonomies) / len(existing_taxonomies),
            "document_complexity": self.assess_document_complexity(existing_taxonomies)
        }
    
    def assess_document_complexity(self, taxonomies: List[PageTaxonomy]) -> str:
        """Assess overall document complexity."""
        if not taxonomies:
            return "unknown"
        
        # Calculate average complexity
        avg_complexity = sum(
            1 if t.complexity_level == "high" else 0.5 if t.complexity_level == "medium" else 0
            for t in taxonomies
        ) / len(taxonomies)
        
        if avg_complexity > 0.7:
            return "high"
        elif avg_complexity > 0.4:
            return "medium"
        else:
            return "low"
    
    async def analyze_global_patterns(self, page_taxonomies: List[PageTaxonomy]) -> List[DocumentPattern]:
        """Analyze patterns across all pages."""
        logger.info("Analyzing global patterns across document")
        
        patterns = []
        pattern_id_counter = 1
        
        try:
            # Pattern 1: Page type sequences
            page_types = [t.page_type.value for t in page_taxonomies]
            type_patterns = self.find_sequence_patterns(page_types)
            
            for pattern_seq in type_patterns:
                pattern = DocumentPattern(
                    pattern_id=f"pattern_{pattern_id_counter}",
                    pattern_type="page_sequence",
                    affected_pages=[i+1 for i, pt in enumerate(page_types) if pt in pattern_seq],
                    pattern_description=f"Sequence pattern: {' → '.join(pattern_seq)}",
                    structural_significance="Document organization pattern",
                    frequency=len([i for i, pt in enumerate(page_types) if pt in pattern_seq]) / len(page_types),
                    confidence=0.8
                )
                patterns.append(pattern)
                pattern_id_counter += 1
            
            # Pattern 2: Recurring structural elements
            all_elements = []
            for taxonomy in page_taxonomies:
                all_elements.extend([e.element_type.value for e in taxonomy.structural_elements])
            
            from collections import Counter
            element_counts = Counter(all_elements)
            
            for element_type, count in element_counts.most_common(5):
                if count >= len(page_taxonomies) * 0.3:  # Appears in 30%+ of pages
                    affected_pages = [
                        t.page_number for t in page_taxonomies
                        if any(e.element_type.value == element_type for e in t.structural_elements)
                    ]
                    
                    pattern = DocumentPattern(
                        pattern_id=f"pattern_{pattern_id_counter}",
                        pattern_type="recurring_element",
                        affected_pages=affected_pages,
                        pattern_description=f"Recurring structural element: {element_type}",
                        structural_significance="Key structural component",
                        frequency=count / len(page_taxonomies),
                        confidence=0.9
                    )
                    patterns.append(pattern)
                    pattern_id_counter += 1
            
            # Pattern 3: Complexity progression
            complexity_scores = [
                1 if t.complexity_level == "high" else 0.5 if t.complexity_level == "medium" else 0
                for t in page_taxonomies
            ]
            
            if len(complexity_scores) > 3:
                # Check for increasing complexity
                increasing_trend = all(
                    complexity_scores[i] <= complexity_scores[i+1] 
                    for i in range(len(complexity_scores)-1)
                )
                
                if increasing_trend:
                    pattern = DocumentPattern(
                        pattern_id=f"pattern_{pattern_id_counter}",
                        pattern_type="complexity_progression",
                        affected_pages=list(range(1, len(page_taxonomies) + 1)),
                        pattern_description="Increasing complexity from general to detailed views",
                        structural_significance="Document follows standard architectural progression",
                        frequency=1.0,
                        confidence=0.85
                    )
                    patterns.append(pattern)
            
            logger.info(f"Identified {len(patterns)} global patterns")
            return patterns
            
        except Exception as e:
            logger.error(f"Global pattern analysis failed: {e}")
            return []
    
    def find_sequence_patterns(self, sequence: List[str]) -> List[List[str]]:
        """Find common subsequence patterns."""
        patterns = []
        
        # Look for common 2-3 element sequences
        for length in [2, 3]:
            for i in range(len(sequence) - length + 1):
                subseq = sequence[i:i+length]
                
                # Count occurrences of this subsequence
                count = 0
                for j in range(len(sequence) - length + 1):
                    if sequence[j:j+length] == subseq:
                        count += 1
                
                # If it appears multiple times, it's a pattern
                if count > 1 and subseq not in patterns:
                    patterns.append(subseq)
        
        return patterns
    
    async def generate_structural_summary(
        self, 
        page_taxonomies: List[PageTaxonomy], 
        global_patterns: List[DocumentPattern]
    ) -> StructuralSummary:
        """Generate comprehensive structural summary."""
        logger.info("Generating structural summary")
        
        try:
            # Analyze building characteristics
            building_type = self.infer_building_type(page_taxonomies)
            structural_system = self.infer_structural_system(page_taxonomies)
            
            # Count elements and analyze
            all_elements = []
            for taxonomy in page_taxonomies:
                all_elements.extend(taxonomy.structural_elements)
            
            from collections import Counter
            element_counts = Counter(e.element_type.value for e in all_elements)
            
            # Assess complexity
            complexity_levels = [t.complexity_level for t in page_taxonomies]
            overall_complexity = max(set(complexity_levels), key=complexity_levels.count)
            
            # Determine construction type
            construction_type = self.infer_construction_type(all_elements)
            
            # Extract applicable codes
            applicable_codes = self.extract_applicable_codes(page_taxonomies)
            
            summary = StructuralSummary(
                building_type=building_type,
                structural_system=structural_system,
                total_floors=self.count_floors(page_taxonomies),
                total_area=self.estimate_total_area(page_taxonomies),
                structural_elements_count=dict(element_counts),
                complexity_assessment=overall_complexity,
                design_style=self.infer_design_style(page_taxonomies),
                construction_type=construction_type,
                applicable_codes=applicable_codes,
                compliance_notes=self.generate_compliance_notes(page_taxonomies)
            )
            
            logger.info("Structural summary generated successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Structural summary generation failed: {e}")
            
            # Return minimal summary
            return StructuralSummary(
                complexity_assessment="unknown",
                structural_elements_count={},
                applicable_codes=[],
                compliance_notes=[]
            )
    
    def infer_building_type(self, taxonomies: List[PageTaxonomy]) -> Optional[str]:
        """Infer building type from page analysis."""
        # Look for indicators in text content
        all_text = " ".join(t.textual_features.processed_text for t in taxonomies).lower()
        
        building_types = {
            "residential": ["house", "home", "residential", "apartment", "condo"],
            "commercial": ["office", "retail", "store", "commercial", "business"],
            "industrial": ["warehouse", "factory", "industrial", "manufacturing"],
            "institutional": ["school", "hospital", "church", "library", "government"]
        }
        
        for building_type, keywords in building_types.items():
            if any(keyword in all_text for keyword in keywords):
                return building_type
        
        # Infer from page types and elements
        floor_plans = sum(1 for t in taxonomies if t.page_type == "floor_plan")
        if floor_plans > 2:
            return "multi_level_building"
        
        return "unknown"
    
    def infer_structural_system(self, taxonomies: List[PageTaxonomy]) -> Optional[str]:
        """Infer structural system from element analysis."""
        # Count structural elements across all pages
        structural_elements = []
        for taxonomy in taxonomies:
            for element in taxonomy.structural_elements:
                if element.element_type.value in ["beam", "column", "wall", "foundation"]:
                    structural_elements.append(element.element_type.value)
        
        from collections import Counter
        element_counts = Counter(structural_elements)
        
        # Determine system based on predominant elements
        if element_counts.get("beam", 0) > 5 and element_counts.get("column", 0) > 3:
            return "frame_system"
        elif element_counts.get("wall", 0) > element_counts.get("beam", 0):
            return "bearing_wall_system"
        elif element_counts.get("foundation", 0) > 2:
            return "foundation_system"
        
        return "mixed_system"
    
    def count_floors(self, taxonomies: List[PageTaxonomy]) -> Optional[int]:
        """Count number of floors from floor plans."""
        floor_plan_count = sum(
            1 for t in taxonomies 
            if t.page_type == "floor_plan"
        )
        
        return floor_plan_count if floor_plan_count > 0 else None
    
    def estimate_total_area(self, taxonomies: List[PageTaxonomy]) -> Optional[float]:
        """Estimate total building area from measurements."""
        # Extract area measurements from text
        all_measurements = []
        for taxonomy in taxonomies:
            all_measurements.extend(taxonomy.textual_features.measurements)
        
        # Look for area measurements (simplified)
        area_indicators = ["sq ft", "sf", "square feet", "area"]
        for measurement in all_measurements:
            if any(indicator in measurement.lower() for indicator in area_indicators):
                # Try to extract numeric value (simplified)
                import re
                numbers = re.findall(r'\d+', measurement)
                if numbers:
                    return float(numbers[0])
        
        return None
    
    def infer_design_style(self, taxonomies: List[PageTaxonomy]) -> Optional[str]:
        """Infer architectural design style."""
        # Analyze visual characteristics
        avg_symmetry = sum(t.visual_features.symmetry_score for t in taxonomies) / len(taxonomies)
        avg_complexity = sum(
            1 if t.complexity_level == "high" else 0.5 if t.complexity_level == "medium" else 0
            for t in taxonomies
        ) / len(taxonomies)
        
        if avg_symmetry > 0.7 and avg_complexity < 0.4:
            return "traditional"
        elif avg_complexity > 0.7:
            return "modern_complex"
        elif avg_symmetry < 0.3:
            return "contemporary"
        
        return "mixed"
    
    def infer_construction_type(self, elements: List[Any]) -> Optional[str]:
        """Infer construction type from structural elements."""
        # Analyze material mentions and element types
        materials_mentioned = []
        
        for element in elements:
            if hasattr(element, 'material') and element.material:
                if hasattr(element.material, 'material_type'):
                    materials_mentioned.append(element.material.material_type)
        
        from collections import Counter
        material_counts = Counter(materials_mentioned)
        
        if material_counts.get("steel", 0) > material_counts.get("concrete", 0):
            return "steel_frame"
        elif material_counts.get("concrete", 0) > 0:
            return "concrete"
        elif material_counts.get("wood", 0) > 0:
            return "wood_frame"
        
        return "mixed_construction"
    
    def extract_applicable_codes(self, taxonomies: List[PageTaxonomy]) -> List[str]:
        """Extract applicable building codes from document."""
        codes = []
        
        # Look for code references in text
        code_keywords = ["ibc", "asce", "aisc", "aci", "nfpa", "local code", "building code"]
        
        for taxonomy in taxonomies:
            text = taxonomy.textual_features.processed_text.lower()
            for keyword in code_keywords:
                if keyword in text:
                    codes.append(keyword.upper())
        
        return list(set(codes))  # Remove duplicates
    
    def generate_compliance_notes(self, taxonomies: List[PageTaxonomy]) -> List[str]:
        """Generate compliance observation notes."""
        notes = []
        
        # Check for common compliance elements
        has_fire_exits = any(
            "exit" in t.textual_features.processed_text.lower()
            for t in taxonomies
        )
        
        has_accessibility = any(
            "ada" in t.textual_features.processed_text.lower() or "accessible" in t.textual_features.processed_text.lower()
            for t in taxonomies
        )
        
        has_structural_notes = any(
            len(t.textual_features.specifications) > 0
            for t in taxonomies
        )
        
        if has_fire_exits:
            notes.append("Fire egress provisions identified")
        if has_accessibility:
            notes.append("Accessibility compliance elements noted")
        if has_structural_notes:
            notes.append("Structural specifications provided")
        
        return notes
    
    async def synthesize_document_taxonomy(
        self,
        pdf_path: Path,
        page_taxonomies: List[PageTaxonomy],
        global_patterns: List[DocumentPattern],
        structural_summary: StructuralSummary,
        session: ProcessingSession
    ) -> DocumentTaxonomy:
        """Synthesize final document taxonomy."""
        
        # Calculate overall metrics
        overall_confidence = sum(t.analysis_confidence for t in page_taxonomies) / len(page_taxonomies)
        completeness_score = sum(t.completeness_score for t in page_taxonomies) / len(page_taxonomies)
        
        # Calculate consistency
        consistency_score = self.calculate_cross_page_consistency(page_taxonomies)
        
        # Generate key insights
        key_insights = self.generate_key_insights(page_taxonomies, global_patterns, structural_summary)
        
        # Create semantic graph
        semantic_graph = self.build_semantic_graph(page_taxonomies, global_patterns)
        
        # Performance metrics
        total_processing_time = session.end_time - session.start_time if session.end_time else 0
        pages_per_second = len(page_taxonomies) / total_processing_time if total_processing_time > 0 else 0
        
        document_taxonomy = DocumentTaxonomy(
            document_id=session.session_id,
            document_name=pdf_path.stem,
            document_metadata={
                "file_path": str(pdf_path),
                "file_size": pdf_path.stat().st_size,
                "processing_session": session.session_id
            },
            creation_timestamp=pdf_path.stat().st_mtime,
            analysis_timestamp=time.time(),
            
            page_taxonomies=page_taxonomies,
            total_pages=len(page_taxonomies),
            
            global_patterns=global_patterns,
            structural_summary=structural_summary,
            
            semantic_graph=semantic_graph,
            key_insights=key_insights,
            
            overall_confidence=round(overall_confidence, 3),
            completeness_score=round(completeness_score, 3),
            consistency_score=round(consistency_score, 3),
            
            total_processing_time=total_processing_time,
            pages_per_second=round(pages_per_second, 2),
            api_calls_made=sum(len(result.tools_used) for result in session.agent_results)
        )
        
        return document_taxonomy
    
    def calculate_cross_page_consistency(self, taxonomies: List[PageTaxonomy]) -> float:
        """Calculate consistency across pages."""
        if len(taxonomies) < 2:
            return 1.0
        
        # Check consistency in classification confidence
        confidences = [t.analysis_confidence for t in taxonomies]
        confidence_variance = sum((c - sum(confidences)/len(confidences))**2 for c in confidences) / len(confidences)
        confidence_consistency = max(0, 1 - confidence_variance)
        
        # Check consistency in element types
        element_types_per_page = []
        for taxonomy in taxonomies:
            page_element_types = set(e.element_type.value for e in taxonomy.structural_elements)
            element_types_per_page.append(page_element_types)
        
        # Calculate Jaccard similarity between consecutive pages
        similarities = []
        for i in range(len(element_types_per_page) - 1):
            set1 = element_types_per_page[i]
            set2 = element_types_per_page[i + 1]
            
            if set1 or set2:
                jaccard = len(set1.intersection(set2)) / len(set1.union(set2))
                similarities.append(jaccard)
        
        element_consistency = sum(similarities) / len(similarities) if similarities else 0.5
        
        # Combined consistency score
        return round((confidence_consistency * 0.6 + element_consistency * 0.4), 3)
    
    def generate_key_insights(
        self, 
        taxonomies: List[PageTaxonomy], 
        patterns: List[DocumentPattern], 
        summary: StructuralSummary
    ) -> List[str]:
        """Generate key insights from analysis."""
        insights = []
        
        # Document structure insights
        page_types = [t.page_type.value for t in taxonomies]
        unique_types = set(page_types)
        insights.append(f"Document contains {len(unique_types)} different page types: {', '.join(unique_types)}")
        
        # Complexity insights
        high_complexity_pages = sum(1 for t in taxonomies if t.complexity_level == "high")
        if high_complexity_pages > len(taxonomies) * 0.3:
            insights.append(f"High complexity document with {high_complexity_pages} detailed pages")
        
        # Pattern insights
        if patterns:
            insights.append(f"Identified {len(patterns)} structural patterns across pages")
        
        # Element insights
        total_elements = sum(len(t.structural_elements) for t in taxonomies)
        insights.append(f"Detected {total_elements} structural elements across {len(taxonomies)} pages")
        
        # Quality insights
        avg_confidence = sum(t.analysis_confidence for t in taxonomies) / len(taxonomies)
        if avg_confidence > 0.8:
            insights.append("High confidence analysis with reliable element detection")
        elif avg_confidence < 0.6:
            insights.append("Lower confidence analysis - document may require manual review")
        
        return insights
    
    def build_semantic_graph(
        self, 
        taxonomies: List[PageTaxonomy], 
        patterns: List[DocumentPattern]
    ) -> Dict[str, Any]:
        """Build semantic knowledge graph."""
        graph = {
            "nodes": [],
            "edges": [],
            "metadata": {
                "total_nodes": 0,
                "total_edges": 0,
                "graph_density": 0.0
            }
        }
        
        try:
            # Add page nodes
            for taxonomy in taxonomies:
                graph["nodes"].append({
                    "id": f"page_{taxonomy.page_number}",
                    "type": "page",
                    "properties": {
                        "page_type": taxonomy.page_type.value,
                        "complexity": taxonomy.complexity_level,
                        "confidence": taxonomy.analysis_confidence
                    }
                })
                
                # Add element nodes
                for element in taxonomy.structural_elements:
                    graph["nodes"].append({
                        "id": element.element_id,
                        "type": "structural_element",
                        "properties": {
                            "element_type": element.element_type.value,
                            "page_number": taxonomy.page_number,
                            "confidence": element.confidence
                        }
                    })
            
            # Add relationship edges
            for taxonomy in taxonomies:
                for relationship in taxonomy.spatial_relationships:
                    graph["edges"].append({
                        "source": relationship.element_1_id,
                        "target": relationship.element_2_id,
                        "type": relationship.relationship_type,
                        "properties": {
                            "distance": relationship.distance,
                            "confidence": relationship.confidence
                        }
                    })
            
            # Update metadata
            graph["metadata"]["total_nodes"] = len(graph["nodes"])
            graph["metadata"]["total_edges"] = len(graph["edges"])
            
            if graph["metadata"]["total_nodes"] > 1:
                max_edges = graph["metadata"]["total_nodes"] * (graph["metadata"]["total_nodes"] - 1) / 2
                graph["metadata"]["graph_density"] = graph["metadata"]["total_edges"] / max_edges
            
            return graph
            
        except Exception as e:
            logger.error(f"Semantic graph building failed: {e}")
            return graph
    
    def update_performance_metrics(self, session: ProcessingSession):
        """Update orchestrator performance metrics."""
        if session.duration:
            self.performance_metrics["total_pages_processed"] += session.total_pages_processed
            self.performance_metrics["total_processing_time"] += session.duration
            
            # Update averages
            if session.agent_results:
                avg_conf = sum(r.confidence for r in session.agent_results) / len(session.agent_results)
                self.performance_metrics["average_confidence"] = avg_conf
                
                success_count = sum(1 for r in session.agent_results if r.confidence > 0.5)
                self.performance_metrics["success_rate"] = success_count / len(session.agent_results)
        
        logger.info(f"Updated performance metrics: {self.performance_metrics}")


class StructuralBlueprintAgent:
    """
    Main agent interface for structural blueprint analysis.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.orchestrator = MasterOrchestrator(config)
        
    async def analyze_blueprint(self, pdf_path: Path) -> DocumentTaxonomy:
        """
        Main entry point for structural blueprint analysis.
        
        Args:
            pdf_path: Path to structural blueprint PDF
            
        Returns:
            Complete DocumentTaxonomy with intelligent analysis
        """
        logger.info(f"Starting structural blueprint analysis: {pdf_path}")
        
        try:
            # Validate input
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            if pdf_path.suffix.lower() != '.pdf':
                raise ValueError(f"File must be a PDF: {pdf_path}")
            
            # Execute analysis
            result = await self.orchestrator.analyze_structural_blueprint(pdf_path)
            
            logger.info("Structural blueprint analysis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Structural blueprint analysis failed: {e}")
            raise
