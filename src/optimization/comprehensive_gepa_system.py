"""
Comprehensive GEPA System for All Dynamic Prompts.

This module implements a complete GEPA system that evolves and optimizes
ALL dynamically generated prompts in the system, including:
- Discovery prompts
- Analysis prompts (general, sections, data_extraction)
- Classification prompts
- Question generation prompts
- Language optimization prompts
"""

import json
import time
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
import statistics

try:
    from ..services.gemini_client import GeminiClient
    from ..core.config import Config
    from ..utils.logging_config import get_logger
    from ..models.schemas import AnalysisType
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class PromptTestResult:
    """Result of prompt testing before optimization."""
    
    prompt_id: str
    prompt_text: str
    test_score: float
    effectiveness_metrics: Dict[str, float]
    execution_time: float
    error_rate: float
    confidence_average: float
    test_samples_used: int
    
    def is_optimization_worthy(self, min_score: float = 0.7) -> bool:
        """Determine if prompt needs optimization."""
        return self.test_score < min_score or self.error_rate > 0.2


@dataclass
class ComprehensiveGEPAResult:
    """Result of comprehensive GEPA optimization."""
    
    optimized_prompts: Dict[str, str]
    performance_improvements: Dict[str, float]
    test_results_before: Dict[str, PromptTestResult]
    test_results_after: Dict[str, PromptTestResult]
    evolution_generations: Dict[str, int]
    total_improvement: float
    optimization_time: float
    
    def get_summary(self) -> Dict[str, Any]:
        """Get optimization summary."""
        return {
            "prompts_optimized": len(self.optimized_prompts),
            "average_improvement": statistics.mean(self.performance_improvements.values()) if self.performance_improvements else 0.0,
            "best_improvement": max(self.performance_improvements.values()) if self.performance_improvements else 0.0,
            "total_improvement": self.total_improvement,
            "optimization_time_minutes": self.optimization_time / 60,
            "prompts_needing_optimization": len([t for t in self.test_results_before.values() if t.is_optimization_worthy()])
        }


class ComprehensiveGEPASystem:
    """
    Complete GEPA system for optimizing all dynamic prompts in the system.
    
    Covers:
    - Discovery prompts (pattern analysis, nomenclature parsing)
    - Analysis prompts (general, sections, data_extraction)
    - Classification prompts (element classification)
    - Question generation prompts
    - Language optimization prompts
    """
    
    def __init__(self, config: Config, gemini_client: Optional[GeminiClient] = None):
        self.config = config
        self.gemini_client = gemini_client or GeminiClient(config)
        
        # GEPA Configuration
        self.population_size = 8
        self.max_generations = 10
        self.mutation_rate = 0.25
        self.crossover_rate = 0.35
        self.elite_preservation = 3
        
        # Test configuration
        self.test_samples_per_prompt = 3
        self.optimization_threshold = 0.7
        self.improvement_target = 0.20
        
        # Prompt categories to optimize
        self.prompt_categories = [
            'discovery_exploration',
            'pattern_analysis', 
            'nomenclature_parsing',
            'element_classification',
            'general_analysis',
            'sections_analysis',
            'data_extraction',
            'question_generation',
            'language_optimization'
        ]
        
        # Performance tracking
        self.optimization_history = []
        self.prompt_performance_cache = {}
        
        logger.info("ComprehensiveGEPASystem initialized for full prompt optimization")
    
    async def optimize_all_dynamic_prompts(
        self,
        current_prompts: Dict[str, str],
        test_documents: List[Dict[str, Any]],
        target_domain: str = "construction_documents"
    ) -> ComprehensiveGEPAResult:
        """
        Optimize all dynamic prompts using comprehensive GEPA.
        
        Args:
            current_prompts: Dictionary of current prompts by category
            test_documents: Sample documents for testing
            target_domain: Target domain for optimization
            
        Returns:
            ComprehensiveGEPAResult with optimization results
        """
        start_time = time.time()
        logger.info(f"🧬 Starting comprehensive GEPA optimization for {len(current_prompts)} prompts")
        
        # Phase 1: Test current prompts
        logger.info("📊 Phase 1: Testing current prompts...")
        test_results_before = await self._test_all_prompts(current_prompts, test_documents)
        
        # Phase 2: Identify prompts needing optimization
        prompts_to_optimize = {
            prompt_id: prompt_text 
            for prompt_id, prompt_text in current_prompts.items()
            if test_results_before[prompt_id].is_optimization_worthy(self.optimization_threshold)
        }
        
        logger.info(f"🎯 {len(prompts_to_optimize)}/{len(current_prompts)} prompts need optimization")
        
        # Phase 3: Evolve prompts that need optimization
        optimized_prompts = {}
        performance_improvements = {}
        evolution_generations = {}
        
        for prompt_id, prompt_text in prompts_to_optimize.items():
            logger.info(f"🧬 Optimizing {prompt_id}...")
            
            evolved_prompt, improvement, generations = await self._evolve_single_prompt(
                prompt_id=prompt_id,
                base_prompt=prompt_text,
                target_domain=target_domain,
                test_documents=test_documents
            )
            
            optimized_prompts[prompt_id] = evolved_prompt
            performance_improvements[prompt_id] = improvement
            evolution_generations[prompt_id] = generations
            
            logger.info(f"✅ {prompt_id} optimized: {improvement:.3f} improvement in {generations} generations")
        
        # Phase 4: Test optimized prompts
        logger.info("📊 Phase 4: Testing optimized prompts...")
        final_prompts = {**current_prompts, **optimized_prompts}
        test_results_after = await self._test_all_prompts(final_prompts, test_documents)
        
        # Calculate total improvement
        total_improvement = self._calculate_total_improvement(test_results_before, test_results_after)
        
        optimization_time = time.time() - start_time
        
        result = ComprehensiveGEPAResult(
            optimized_prompts=optimized_prompts,
            performance_improvements=performance_improvements,
            test_results_before=test_results_before,
            test_results_after=test_results_after,
            evolution_generations=evolution_generations,
            total_improvement=total_improvement,
            optimization_time=optimization_time
        )
        
        # Save results
        await self._save_optimization_results(result, target_domain)
        
        logger.info(f"🎉 Comprehensive GEPA optimization complete:")
        logger.info(f"  - Total improvement: {total_improvement:.3f}")
        logger.info(f"  - Prompts optimized: {len(optimized_prompts)}")
        logger.info(f"  - Optimization time: {optimization_time/60:.1f} minutes")
        
        return result
    
    async def _test_all_prompts(
        self,
        prompts: Dict[str, str],
        test_documents: List[Dict[str, Any]]
    ) -> Dict[str, PromptTestResult]:
        """Test all prompts against sample documents."""
        
        test_results = {}
        
        for prompt_id, prompt_text in prompts.items():
            logger.debug(f"Testing prompt: {prompt_id}")
            
            test_result = await self._test_single_prompt(
                prompt_id=prompt_id,
                prompt_text=prompt_text,
                test_documents=test_documents[:self.test_samples_per_prompt]
            )
            
            test_results[prompt_id] = test_result
        
        return test_results
    
    async def _test_single_prompt(
        self,
        prompt_id: str,
        prompt_text: str,
        test_documents: List[Dict[str, Any]]
    ) -> PromptTestResult:
        """Test a single prompt against test documents."""
        
        start_time = time.time()
        scores = []
        errors = 0
        confidence_scores = []
        
        for doc in test_documents:
            try:
                # Create test evaluation prompt
                test_prompt = f"""
                PROMPT EFFECTIVENESS TEST
                
                Test this {prompt_id} prompt:
                "{prompt_text}"
                
                Against this document sample:
                {json.dumps(doc, indent=2)[:800]}
                
                Evaluate the prompt on:
                1. Clarity and specificity (0-1)
                2. Domain appropriateness (0-1)  
                3. Expected result quality (0-1)
                4. Instruction completeness (0-1)
                5. Overall effectiveness (0-1)
                
                RESPONSE FORMAT (JSON):
                {{
                    "clarity": 0.85,
                    "domain_appropriateness": 0.90,
                    "result_quality": 0.80,
                    "completeness": 0.85,
                    "overall_effectiveness": 0.85,
                    "confidence": 0.90,
                    "reasoning": "Brief explanation"
                }}
                """
                
                response = self.gemini_client.generate_text_only_content(
                    prompt=test_prompt,
                    response_schema=self._get_test_schema()
                )
                
                test_result = self._parse_test_response(response)
                if test_result:
                    effectiveness = (
                        test_result.get('clarity', 0.5) * 0.2 +
                        test_result.get('domain_appropriateness', 0.5) * 0.25 +
                        test_result.get('result_quality', 0.5) * 0.3 +
                        test_result.get('completeness', 0.5) * 0.15 +
                        test_result.get('overall_effectiveness', 0.5) * 0.1
                    )
                    scores.append(effectiveness)
                    confidence_scores.append(test_result.get('confidence', 0.5))
                else:
                    scores.append(0.5)
                    errors += 1
                    
            except Exception as e:
                logger.warning(f"Prompt test failed for {prompt_id}: {e}")
                scores.append(0.3)  # Low score for failed test
                errors += 1
        
        execution_time = time.time() - start_time
        
        return PromptTestResult(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            test_score=statistics.mean(scores) if scores else 0.5,
            effectiveness_metrics={
                'average_score': statistics.mean(scores) if scores else 0.5,
                'score_std': statistics.stdev(scores) if len(scores) > 1 else 0.0,
                'min_score': min(scores) if scores else 0.5,
                'max_score': max(scores) if scores else 0.5
            },
            execution_time=execution_time,
            error_rate=errors / len(test_documents) if test_documents else 0.0,
            confidence_average=statistics.mean(confidence_scores) if confidence_scores else 0.5,
            test_samples_used=len(test_documents)
        )
    
    async def _evolve_single_prompt(
        self,
        prompt_id: str,
        base_prompt: str,
        target_domain: str,
        test_documents: List[Dict[str, Any]]
    ) -> Tuple[str, float, int]:
        """Evolve a single prompt using genetic algorithm."""
        
        # Initialize population
        current_population = [base_prompt]
        
        # Generate initial population through mutations
        for _ in range(self.population_size - 1):
            mutated = await self._mutate_prompt(base_prompt, prompt_id, target_domain)
            current_population.append(mutated)
        
        best_performance = 0.0
        best_prompt = base_prompt
        generation = 0
        
        for generation in range(self.max_generations):
            # Evaluate population
            population_scores = []
            for prompt in current_population:
                test_result = await self._test_single_prompt(prompt_id, prompt, test_documents)
                population_scores.append((prompt, test_result.test_score))
            
            # Sort by performance
            population_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Track best performance
            current_best_score = population_scores[0][1]
            if current_best_score > best_performance:
                best_performance = current_best_score
                best_prompt = population_scores[0][0]
            
            # Check if target improvement reached
            improvement = best_performance - 0.5  # Baseline improvement
            if improvement >= self.improvement_target:
                logger.info(f"Target improvement reached for {prompt_id} in generation {generation + 1}")
                break
            
            # Create next generation
            current_population = await self._create_next_generation(
                population_scores, prompt_id, target_domain
            )
        
        final_improvement = best_performance - 0.5
        return best_prompt, final_improvement, generation + 1
    
    async def _mutate_prompt(self, prompt: str, prompt_type: str, domain: str) -> str:
        """Generate a mutated version of the prompt."""
        
        mutation_strategies = {
            'discovery_exploration': "Make more specific to document exploration and pattern discovery",
            'pattern_analysis': "Improve pattern recognition and structural analysis",
            'nomenclature_parsing': "Enhance code parsing and system understanding",
            'element_classification': "Refine classification accuracy and specificity",
            'general_analysis': "Improve document comprehension and insight generation",
            'sections_analysis': "Enhance section identification and analysis depth",
            'data_extraction': "Optimize data extraction completeness and accuracy",
            'question_generation': "Generate more relevant and insightful questions",
            'language_optimization': "Improve language detection and prompt adaptation"
        }
        
        strategy = mutation_strategies.get(prompt_type, "Improve overall effectiveness and clarity")
        
        mutation_prompt = f"""
        ADVANCED PROMPT EVOLUTION
        
        Original {prompt_type} prompt for {domain}:
        "{prompt}"
        
        Evolution objective: {strategy}
        
        Create an improved version that:
        1. Is more specific to {domain} documents
        2. Provides clearer instructions
        3. Improves expected output quality
        4. Reduces ambiguity and errors
        5. Enhances domain-specific effectiveness
        
        Advanced improvements:
        - Add specific examples or guidance
        - Include error prevention instructions
        - Optimize for {domain} terminology
        - Enhance structure and clarity
        - Add quality control measures
        
        Return only the improved prompt, no explanations.
        """
        
        try:
            evolved_prompt = self.gemini_client.generate_text_only_content(
                prompt=mutation_prompt,
                model="gemini-2.5-pro"
            )
            return evolved_prompt.strip()
        except Exception as e:
            logger.warning(f"Prompt mutation failed for {prompt_type}: {e}")
            return prompt
    
    async def _create_next_generation(
        self,
        population_scores: List[Tuple[str, float]],
        prompt_type: str,
        domain: str
    ) -> List[str]:
        """Create next generation using genetic operations."""
        
        next_generation = []
        
        # Elite preservation
        for i in range(min(self.elite_preservation, len(population_scores))):
            next_generation.append(population_scores[i][0])
        
        # Generate offspring
        while len(next_generation) < self.population_size:
            if len(population_scores) >= 2:
                # Select parents
                parent1 = self._tournament_selection(population_scores)
                parent2 = self._tournament_selection(population_scores)
                
                # Crossover
                if len(next_generation) < self.population_size - 1:
                    offspring = await self._crossover_prompts(parent1, parent2, prompt_type, domain)
                    next_generation.append(offspring)
                
                # Mutation
                if len(next_generation) < self.population_size:
                    mutated = await self._mutate_prompt(parent1, prompt_type, domain)
                    next_generation.append(mutated)
            else:
                # Fallback to mutation
                if population_scores:
                    mutated = await self._mutate_prompt(population_scores[0][0], prompt_type, domain)
                    next_generation.append(mutated)
                else:
                    break
        
        return next_generation[:self.population_size]
    
    def _tournament_selection(self, population_scores: List[Tuple[str, float]]) -> str:
        """Select parent using tournament selection."""
        import random
        
        tournament_size = min(3, len(population_scores))
        tournament = random.sample(population_scores, tournament_size)
        return max(tournament, key=lambda x: x[1])[0]
    
    async def _crossover_prompts(self, parent1: str, parent2: str, prompt_type: str, domain: str) -> str:
        """Create offspring through prompt crossover."""
        
        crossover_prompt = f"""
        ADVANCED PROMPT CROSSOVER
        
        Combine the best elements from these two {prompt_type} prompts for {domain}:
        
        PARENT 1:
        "{parent1}"
        
        PARENT 2:
        "{parent2}"
        
        Create a superior hybrid that:
        1. Takes the best instructional elements from both
        2. Combines effective domain-specific guidance
        3. Merges successful clarity improvements
        4. Maintains the most effective structural elements
        5. Optimizes for {domain} effectiveness
        
        Crossover strategies:
        - Combine the clearest instructions
        - Merge domain-specific improvements
        - Integrate best structural elements
        - Preserve effective examples or guidance
        - Optimize overall flow and clarity
        
        Return only the hybrid prompt, no explanations.
        """
        
        try:
            offspring = self.gemini_client.generate_text_only_content(
                prompt=crossover_prompt,
                model="gemini-2.5-pro"
            )
            return offspring.strip()
        except Exception as e:
            logger.warning(f"Crossover failed for {prompt_type}: {e}")
            return parent1
    
    def _get_test_schema(self) -> Dict[str, Any]:
        """Get JSON schema for prompt testing."""
        return {
            "type": "object",
            "properties": {
                "clarity": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Clarity and specificity of instructions"
                },
                "domain_appropriateness": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Appropriateness for target domain"
                },
                "result_quality": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Expected quality of results"
                },
                "completeness": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Completeness of instructions"
                },
                "overall_effectiveness": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Overall prompt effectiveness"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence in evaluation"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of evaluation"
                }
            },
            "required": ["clarity", "domain_appropriateness", "result_quality", "completeness", "overall_effectiveness"]
        }
    
    def _parse_test_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse test response from Gemini."""
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse test response: {e}")
        return None
    
    def _calculate_total_improvement(
        self,
        before_results: Dict[str, PromptTestResult],
        after_results: Dict[str, PromptTestResult]
    ) -> float:
        """Calculate total improvement across all prompts."""
        
        improvements = []
        
        for prompt_id in before_results:
            if prompt_id in after_results:
                before_score = before_results[prompt_id].test_score
                after_score = after_results[prompt_id].test_score
                improvement = after_score - before_score
                improvements.append(improvement)
        
        return statistics.mean(improvements) if improvements else 0.0
    
    async def _save_optimization_results(
        self,
        result: ComprehensiveGEPAResult,
        domain: str
    ) -> None:
        """Save comprehensive optimization results."""
        
        try:
            output_dir = Path(self.config.get_directories()["output"])
            results_file = output_dir / f"comprehensive_gepa_optimization_{domain}_{int(time.time())}.json"
            
            # Prepare serializable results
            optimization_data = {
                "optimization_results": {
                    "optimized_prompts": result.optimized_prompts,
                    "performance_improvements": result.performance_improvements,
                    "evolution_generations": result.evolution_generations,
                    "total_improvement": result.total_improvement,
                    "optimization_time": result.optimization_time
                },
                "test_results_before": {
                    prompt_id: {
                        "test_score": test.test_score,
                        "effectiveness_metrics": test.effectiveness_metrics,
                        "error_rate": test.error_rate,
                        "confidence_average": test.confidence_average,
                        "needs_optimization": test.is_optimization_worthy()
                    }
                    for prompt_id, test in result.test_results_before.items()
                },
                "test_results_after": {
                    prompt_id: {
                        "test_score": test.test_score,
                        "effectiveness_metrics": test.effectiveness_metrics,
                        "error_rate": test.error_rate,
                        "confidence_average": test.confidence_average
                    }
                    for prompt_id, test in result.test_results_after.items()
                },
                "domain": domain,
                "timestamp": time.time(),
                "gepa_config": {
                    "population_size": self.population_size,
                    "max_generations": self.max_generations,
                    "mutation_rate": self.mutation_rate,
                    "crossover_rate": self.crossover_rate,
                    "optimization_threshold": self.optimization_threshold
                },
                "summary": result.get_summary()
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(optimization_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Comprehensive GEPA results saved to: {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save optimization results: {e}")
    
    def get_optimized_prompt(self, prompt_type: str, domain: str) -> Optional[str]:
        """Get optimized prompt if available."""
        
        try:
            output_dir = Path(self.config.get_directories()["output"])
            pattern = f"comprehensive_gepa_optimization_{domain}_*.json"
            optimization_files = list(output_dir.glob(pattern))
            
            if not optimization_files:
                return None
            
            # Get most recent optimization
            latest_file = max(optimization_files, key=lambda p: p.stat().st_mtime)
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            optimized_prompts = data.get("optimization_results", {}).get("optimized_prompts", {})
            return optimized_prompts.get(prompt_type)
            
        except Exception as e:
            logger.warning(f"Failed to load optimized prompt for {prompt_type}: {e}")
        
        return None


# Integration functions for the main system
def create_comprehensive_gepa_system(config: Config, gemini_client: Optional[GeminiClient] = None) -> ComprehensiveGEPASystem:
    """Create comprehensive GEPA system instance."""
    return ComprehensiveGEPASystem(config, gemini_client)


async def optimize_system_prompts(
    config: Config,
    discovery_results: List[Dict[str, Any]],
    target_domain: str = "construction_documents"
) -> ComprehensiveGEPAResult:
    """
    Optimize all system prompts using comprehensive GEPA.
    
    Args:
        config: System configuration
        discovery_results: Historical discovery results for testing
        target_domain: Target domain for optimization
        
    Returns:
        ComprehensiveGEPAResult with optimization details
    """
    gepa_system = create_comprehensive_gepa_system(config)
    
    # Collect all current dynamic prompts from the system
    current_prompts = await _collect_all_dynamic_prompts(config, target_domain)
    
    return await gepa_system.optimize_all_dynamic_prompts(
        current_prompts=current_prompts,
        test_documents=discovery_results,
        target_domain=target_domain
    )


async def _collect_all_dynamic_prompts(config: Config, domain: str) -> Dict[str, str]:
    """Collect all dynamic prompts currently used in the system."""
    
    prompts = {}
    
    # Discovery prompts
    prompts['discovery_exploration'] = f"""
    You are analyzing a technical document with {{total_pages}} pages total.
    I'm providing you with {{sample_size}} strategically selected pages: {{sample_pages}}
    
    Your task is to discover WITHOUT ANY PRECONCEPTIONS:
    
    1. DOCUMENT TYPE & DOMAIN
       - What type of technical document is this?
       - What industry or domain does it belong to?
       - Is it engineering, architectural, electrical, process, civil, etc.?
    
    2. ORGANIZATION SYSTEM
       - How is the document organized?
       - What's the logic behind page ordering?
       - Are there sections, categories, or groupings?
    
    3. NOMENCLATURE & CODING
       - What naming conventions are used?
       - Are there codes like V-201, P-101, TAG numbers?
       - What do these codes mean?
       - Are there revision markers, sheet numbers?
    
    4. VISUAL PATTERNS
       - What types of drawings/diagrams are present?
       - What symbols appear repeatedly?
       - What line styles, colors, or conventions are used?
       - Are there standard symbols (electrical, P&ID, architectural)?
    
    5. RELATIONSHIPS & REFERENCES
       - How do pages reference each other?
       - Are there "see detail on sheet X" references?
       - What elements span multiple pages?
    
    6. UNIQUE ELEMENTS
       - What's specific to this particular document?
       - Any unusual or specialized elements?
       - Custom legends or symbol definitions?
    
    DO NOT assume this fits any standard category. Discover everything from direct observation.
    Be exhaustive in identifying ALL types of elements present across ALL the pages shown.
    
    Analyze the document holistically considering all pages together.
    """
    
    # Analysis prompts (adaptive versions)
    prompts['general_analysis'] = f"""
    Perform a comprehensive analysis of this {{document_type}} from the {{industry_domain}} domain. Provide:
    1. A clear and concise executive summary
    2. The main topics addressed in the document
    3. The most important and relevant insights
    4. The identified document type and its specific characteristics
    5. Your confidence level in the analysis
    
    Be precise, objective and structured in your response. Focus specifically on {{focus_areas}} aspects relevant to {{industry_domain}}.
    """
    
    prompts['sections_analysis'] = f"""
    Identify and analyze the main sections of this {{document_type}}.
    For each important section, provide:
    - Section title or heading
    - Content summary (maximum 500 characters)
    - Important data found
    - Questions or concerns arising from the content
    - Section type if identifiable
    
    Focus specifically on {{focus_areas}} aspects relevant to {{industry_domain}}. Pay attention to the specific organizational patterns discovered in this document.
    """
    
    prompts['data_extraction'] = f"""
    Extract specific structured data from this {{document_type}} in the {{industry_domain}} domain:
    - Entities mentioned (people, organizations, places, companies) - MAX 50 items
    - Relevant dates and important deadlines - MAX 30 items
    - Numbers, metrics and key statistics - MAX 40 items, each under 100 characters
    - References, citations and external sources - MAX 25 items
    - Technical terms and specialized vocabulary - MAX 30 items
    
    Focus on data types specific to {{industry_domain}}. Extract information that would be valuable for professionals in this field.
    """
    
    # Classification prompts
    prompts['element_classification'] = f"""
    You are an expert analyzing a technical drawing element.
    
    ELEMENT INFORMATION:
    - Text content: "{{text_content}}"
    - Visual features: {{visual_features}}
    - Location: {{location}}
    - Annotations: {{annotations}}
    
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
    
    # Question generation prompts
    prompts['question_generation'] = f"""
    Generate adaptive questions for {{document_type}} in {{industry_domain}} domain.
    
    Based on discovered patterns: {{discovered_patterns}}
    
    Create questions that:
    1. Are specific to the document type and domain
    2. Focus on the most important aspects for professionals
    3. Explore technical details and specifications
    4. Address compliance and safety requirements
    5. Investigate construction methods and systems
    
    Generate {{max_questions}} questions that would provide maximum value for analysis.
    """
    
    return prompts
