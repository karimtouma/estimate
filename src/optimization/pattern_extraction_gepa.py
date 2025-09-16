"""
GEPA Optimizer for Pattern Extraction and Prompt Specialization.

This module implements GEPA (Genetic Evolution Prompt Architecture) specifically
for evolving and specializing prompts used in pattern extraction and discovery.
"""

from collections import defaultdict

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from ..core.config import Config
    from ..services.gemini_client import GeminiClient
    from ..utils.logging_config import get_logger

    logger = get_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


@dataclass
class PromptEvolutionResult:
    """Result of prompt evolution using GEPA."""

    original_prompt: str
    evolved_prompt: str
    performance_improvement: float
    confidence_gain: float
    pattern_extraction_improvement: float
    specialization_score: float
    evolution_generations: int
    best_mutations: List[str]
    performance_history: List[float] = field(default_factory=list)

    def get_improvement_summary(self) -> Dict[str, Any]:
        """Get summary of improvements achieved."""
        return {
            "overall_improvement": self.performance_improvement,
            "confidence_improvement": self.confidence_gain,
            "pattern_extraction_improvement": self.pattern_extraction_improvement,
            "specialization_achieved": self.specialization_score,
            "generations_evolved": self.evolution_generations,
            "convergence_rate": (
                len([p for p in self.performance_history if p > 0.1])
                / len(self.performance_history)
                if self.performance_history
                else 0
            ),
        }


@dataclass
class PatternExtractionMetrics:
    """Metrics for evaluating pattern extraction quality."""

    patterns_discovered: int
    pattern_specificity: float
    nomenclature_accuracy: float
    structure_recognition: float
    confidence_average: float
    processing_time: float

    def calculate_overall_score(self) -> float:
        """Calculate overall pattern extraction score."""
        return (
            (self.patterns_discovered / 20) * 0.2  # Normalize to 20 patterns max
            + self.pattern_specificity * 0.25
            + self.nomenclature_accuracy * 0.25
            + self.structure_recognition * 0.2
            + self.confidence_average * 0.1
        )


class PatternExtractionGEPA:
    """
    GEPA optimizer specifically for pattern extraction and prompt specialization.

    Evolves prompts used in:
    - Pattern discovery
    - Nomenclature parsing
    - Structure recognition
    - Element classification
    """

    def __init__(self, config: Config, gemini_client: Optional[GeminiClient] = None):
        self.config = config
        self.gemini_client = gemini_client or GeminiClient(config)

        # GEPA Configuration
        self.population_size = 6  # Number of prompt variants per generation
        self.max_generations = 8  # Maximum evolution generations
        self.mutation_rate = 0.3  # Rate of prompt mutations
        self.crossover_rate = 0.4  # Rate of prompt crossover
        self.elite_preservation = 2  # Number of best prompts to preserve

        # Specialization targets
        self.specialization_domains = [
            "construction_documents",
            "architectural_plans",
            "structural_drawings",
            "mep_schematics",
            "site_plans",
        ]

        # Performance tracking
        self.evolution_history = []
        self.best_prompts = {}
        self.performance_cache = {}

        logger.info("PatternExtractionGEPA initialized for prompt evolution")

    async def evolve_pattern_extraction_prompts(
        self,
        base_prompts: Dict[str, str],
        target_domain: str,
        sample_documents: List[Dict[str, Any]],
        target_improvement: float = 0.15,
    ) -> Dict[str, PromptEvolutionResult]:
        """
        Evolve pattern extraction prompts using GEPA.

        Args:
            base_prompts: Dictionary of base prompts to evolve
            target_domain: Target domain for specialization
            sample_documents: Sample documents for evaluation
            target_improvement: Target improvement threshold

        Returns:
            Dictionary of evolved prompts with results
        """
        logger.info(
            f"Starting GEPA evolution for {len(base_prompts)} prompts in domain: {target_domain}"
        )

        evolved_results = {}

        for prompt_type, base_prompt in base_prompts.items():
            logger.info(f"Evolving {prompt_type} prompt...")

            evolution_result = await self._evolve_single_prompt(
                prompt_type=prompt_type,
                base_prompt=base_prompt,
                target_domain=target_domain,
                sample_documents=sample_documents,
                target_improvement=target_improvement,
            )

            evolved_results[prompt_type] = evolution_result

            logger.info(
                f"✅ {prompt_type} evolution complete: {evolution_result.performance_improvement:.3f} improvement"
            )

        # Save evolution results
        await self._save_evolution_results(evolved_results, target_domain)

        return evolved_results

    async def _evolve_single_prompt(
        self,
        prompt_type: str,
        base_prompt: str,
        target_domain: str,
        sample_documents: List[Dict[str, Any]],
        target_improvement: float,
    ) -> PromptEvolutionResult:
        """Evolve a single prompt using genetic algorithm."""

        # Initialize population with base prompt
        current_population = [base_prompt]

        # Generate initial population through mutations
        for _ in range(self.population_size - 1):
            mutated = await self._mutate_prompt(base_prompt, target_domain)
            current_population.append(mutated)

        best_performance = 0.0
        best_prompt = base_prompt
        performance_history = []
        generation = 0

        for generation in range(self.max_generations):
            logger.debug(
                f"GEPA Generation {generation + 1}/{self.max_generations} for {prompt_type}"
            )

            # Evaluate population
            population_scores = []
            for prompt in current_population:
                score = await self._evaluate_prompt_performance(
                    prompt, prompt_type, sample_documents, target_domain
                )
                population_scores.append((prompt, score))

            # Sort by performance
            population_scores.sort(key=lambda x: x[1], reverse=True)

            # Track best performance
            current_best_score = population_scores[0][1]
            if current_best_score > best_performance:
                best_performance = current_best_score
                best_prompt = population_scores[0][0]

            performance_history.append(current_best_score)

            # Check if target improvement reached
            improvement = (best_performance - 0.5) / 0.5  # Normalize from baseline 0.5
            if improvement >= target_improvement:
                logger.info(
                    f"Target improvement {target_improvement:.3f} reached in generation {generation + 1}"
                )
                break

            # Create next generation
            current_population = await self._create_next_generation(
                population_scores, target_domain
            )

        # Calculate final metrics
        final_improvement = (best_performance - 0.5) / 0.5
        confidence_gain = best_performance - 0.5

        return PromptEvolutionResult(
            original_prompt=base_prompt,
            evolved_prompt=best_prompt,
            performance_improvement=final_improvement,
            confidence_gain=confidence_gain,
            pattern_extraction_improvement=best_performance,
            specialization_score=await self._calculate_specialization_score(
                best_prompt, target_domain
            ),
            evolution_generations=generation + 1,
            best_mutations=self._extract_best_mutations(population_scores[:3]),
            performance_history=performance_history,
        )

    async def _mutate_prompt(self, prompt: str, domain: str) -> str:
        """Generate a mutated version of the prompt."""

        mutation_prompt = f"""
        PROMPT EVOLUTION MUTATION
        
        Original prompt for {domain} pattern extraction:
        "{prompt}"
        
        Create an improved version that:
        1. Is more specific to {domain} documents
        2. Extracts patterns more effectively
        3. Provides better structure recognition
        4. Improves nomenclature parsing
        5. Maintains the core functionality
        
        Mutation strategies:
        - Add domain-specific instructions
        - Improve pattern recognition guidance
        - Enhance structure analysis requests
        - Optimize for {domain} terminology
        - Refine output format requirements
        
        Return only the improved prompt, no explanations.
        """

        try:
            evolved_prompt = self.gemini_client.generate_text_only_content(
                prompt=mutation_prompt,
                model="gemini-2.5-pro",  # Use most capable model for evolution
            )
            return evolved_prompt.strip()
        except Exception as e:
            logger.warning(f"Prompt mutation failed: {e}")
            return prompt  # Return original if mutation fails

    async def _evaluate_prompt_performance(
        self, prompt: str, prompt_type: str, sample_documents: List[Dict[str, Any]], domain: str
    ) -> float:
        """Evaluate prompt performance on sample documents."""

        total_score = 0.0
        evaluations = 0

        # Test prompt on sample documents
        for doc_sample in sample_documents[:3]:  # Limit to 3 for efficiency
            try:
                # Create evaluation prompt
                eval_prompt = f"""
                PROMPT PERFORMANCE EVALUATION
                
                Test this {prompt_type} prompt on {domain} document sample:
                
                PROMPT TO TEST:
                "{prompt}"
                
                DOCUMENT SAMPLE:
                {json.dumps(doc_sample, indent=2)[:1000]}
                
                Evaluate the prompt on:
                1. Pattern extraction effectiveness (0-1)
                2. Specificity to {domain} (0-1)
                3. Structure recognition quality (0-1)
                4. Nomenclature parsing accuracy (0-1)
                5. Overall usefulness (0-1)
                
                RESPONSE FORMAT (JSON):
                {{
                    "pattern_extraction": 0.8,
                    "domain_specificity": 0.9,
                    "structure_recognition": 0.7,
                    "nomenclature_accuracy": 0.8,
                    "overall_usefulness": 0.85,
                    "reasoning": "Brief explanation"
                }}
                """

                response = self.gemini_client.generate_text_only_content(
                    prompt=eval_prompt, response_schema=self._get_evaluation_schema()
                )

                # Parse evaluation result
                eval_result = self._parse_evaluation_response(response)
                if eval_result:
                    score = (
                        eval_result.get("pattern_extraction", 0.5) * 0.3
                        + eval_result.get("domain_specificity", 0.5) * 0.25
                        + eval_result.get("structure_recognition", 0.5) * 0.2
                        + eval_result.get("nomenclature_accuracy", 0.5) * 0.15
                        + eval_result.get("overall_usefulness", 0.5) * 0.1
                    )
                    total_score += score
                    evaluations += 1

            except Exception as e:
                logger.warning(f"Prompt evaluation failed: {e}")
                total_score += 0.5  # Default score for failed evaluation
                evaluations += 1

        return total_score / max(1, evaluations)

    async def _create_next_generation(
        self, population_scores: List[Tuple[str, float]], domain: str
    ) -> List[str]:
        """Create next generation using genetic operations."""

        next_generation = []

        # Elite preservation - keep best prompts
        for i in range(self.elite_preservation):
            if i < len(population_scores):
                next_generation.append(population_scores[i][0])

        # Generate offspring through crossover and mutation
        while len(next_generation) < self.population_size:
            if len(population_scores) >= 2:
                # Select parents (weighted by performance)
                parent1 = self._select_parent(population_scores)
                parent2 = self._select_parent(population_scores)

                # Crossover
                if len(next_generation) < self.population_size - 1:
                    offspring = await self._crossover_prompts(parent1, parent2, domain)
                    next_generation.append(offspring)

                # Mutation
                if len(next_generation) < self.population_size:
                    mutated = await self._mutate_prompt(parent1, domain)
                    next_generation.append(mutated)
            else:
                # Fallback to mutation if insufficient population
                if population_scores:
                    mutated = await self._mutate_prompt(population_scores[0][0], domain)
                    next_generation.append(mutated)
                else:
                    break

        return next_generation[: self.population_size]

    def _select_parent(self, population_scores: List[Tuple[str, float]]) -> str:
        """Select parent using tournament selection."""
        import random

        # Tournament selection with size 3
        tournament_size = min(3, len(population_scores))
        tournament = random.sample(population_scores, tournament_size)

        # Return best from tournament
        return max(tournament, key=lambda x: x[1])[0]

    async def _crossover_prompts(self, parent1: str, parent2: str, domain: str) -> str:
        """Create offspring through prompt crossover."""

        crossover_prompt = f"""
        PROMPT GENETIC CROSSOVER
        
        Combine the best elements from these two {domain} pattern extraction prompts:
        
        PARENT 1:
        "{parent1}"
        
        PARENT 2:
        "{parent2}"
        
        Create a hybrid that:
        1. Takes the best pattern extraction techniques from both
        2. Combines effective structure recognition approaches
        3. Merges successful nomenclature parsing strategies
        4. Maintains domain specificity for {domain}
        5. Preserves the most effective instructions
        
        Return only the hybrid prompt, no explanations.
        """

        try:
            offspring = self.gemini_client.generate_text_only_content(
                prompt=crossover_prompt, model="gemini-2.5-pro"
            )
            return offspring.strip()
        except Exception as e:
            logger.warning(f"Prompt crossover failed: {e}")
            return parent1  # Return parent1 if crossover fails

    def _get_evaluation_schema(self) -> Dict[str, Any]:
        """Get JSON schema for prompt evaluation."""
        return {
            "type": "object",
            "properties": {
                "pattern_extraction": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Effectiveness at extracting patterns",
                },
                "domain_specificity": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Specificity to target domain",
                },
                "structure_recognition": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Quality of structure recognition",
                },
                "nomenclature_accuracy": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Accuracy in nomenclature parsing",
                },
                "overall_usefulness": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Overall prompt usefulness",
                },
                "reasoning": {"type": "string", "description": "Brief explanation of evaluation"},
            },
            "required": [
                "pattern_extraction",
                "domain_specificity",
                "structure_recognition",
                "nomenclature_accuracy",
                "overall_usefulness",
            ],
        }

    def _parse_evaluation_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse evaluation response from Gemini."""
        try:
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse evaluation response: {e}")
        return None

    async def _calculate_specialization_score(self, prompt: str, domain: str) -> float:
        """Calculate how specialized the prompt is for the domain."""

        specialization_prompt = f"""
        PROMPT SPECIALIZATION ANALYSIS
        
        Analyze how specialized this prompt is for {domain} documents:
        
        "{prompt}"
        
        Rate specialization on:
        1. Domain-specific terminology usage
        2. Relevant pattern recognition instructions
        3. Appropriate structure analysis guidance
        4. Effective nomenclature handling
        5. Overall domain adaptation
        
        RESPONSE FORMAT (JSON):
        {{
            "specialization_score": 0.85,
            "domain_terminology": 0.9,
            "pattern_relevance": 0.8,
            "structure_guidance": 0.85,
            "nomenclature_handling": 0.9,
            "reasoning": "Brief explanation"
        }}
        """

        try:
            response = self.gemini_client.generate_text_only_content(
                prompt=specialization_prompt, response_schema=self._get_specialization_schema()
            )

            result = self._parse_evaluation_response(response)
            if result:
                return result.get("specialization_score", 0.5)
        except Exception as e:
            logger.warning(f"Specialization calculation failed: {e}")

        return 0.5  # Default score

    def _get_specialization_schema(self) -> Dict[str, Any]:
        """Get JSON schema for specialization analysis."""
        return {
            "type": "object",
            "properties": {
                "specialization_score": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Overall specialization score",
                },
                "domain_terminology": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Use of domain-specific terminology",
                },
                "pattern_relevance": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Relevance of pattern recognition instructions",
                },
                "structure_guidance": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Quality of structure analysis guidance",
                },
                "nomenclature_handling": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Effectiveness of nomenclature handling",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explanation of specialization analysis",
                },
            },
            "required": ["specialization_score"],
        }

    def _extract_best_mutations(self, top_performers: List[Tuple[str, float]]) -> List[str]:
        """Extract the best mutations from top performers."""
        return [prompt for prompt, score in top_performers]

    async def _save_evolution_results(
        self, results: Dict[str, PromptEvolutionResult], domain: str
    ) -> None:
        """Save evolution results to file."""

        try:
            output_dir = Path(self.config.get_directories()["output"])
            results_file = output_dir / f"gepa_pattern_evolution_{domain}_{int(time.time())}.json"

            # Prepare serializable results
            serializable_results = {}
            for prompt_type, result in results.items():
                serializable_results[prompt_type] = {
                    "original_prompt": result.original_prompt,
                    "evolved_prompt": result.evolved_prompt,
                    "performance_improvement": result.performance_improvement,
                    "confidence_gain": result.confidence_gain,
                    "pattern_extraction_improvement": result.pattern_extraction_improvement,
                    "specialization_score": result.specialization_score,
                    "evolution_generations": result.evolution_generations,
                    "best_mutations": result.best_mutations,
                    "performance_history": result.performance_history,
                    "improvement_summary": result.get_improvement_summary(),
                }

            # Add metadata
            evolution_data = {
                "evolution_results": serializable_results,
                "domain": domain,
                "timestamp": time.time(),
                "gepa_config": {
                    "population_size": self.population_size,
                    "max_generations": self.max_generations,
                    "mutation_rate": self.mutation_rate,
                    "crossover_rate": self.crossover_rate,
                },
                "overall_summary": {
                    "prompts_evolved": len(results),
                    "average_improvement": sum(r.performance_improvement for r in results.values())
                    / len(results),
                    "best_improvement": max(r.performance_improvement for r in results.values()),
                    "total_generations": sum(r.evolution_generations for r in results.values()),
                },
            }

            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(evolution_data, f, indent=2, ensure_ascii=False)

            logger.info(f"GEPA evolution results saved to: {results_file}")

        except Exception as e:
            logger.error(f"Failed to save evolution results: {e}")

    def get_evolved_prompt(self, prompt_type: str, domain: str) -> Optional[str]:
        """Get the best evolved prompt for a specific type and domain."""

        try:
            output_dir = Path(self.config.get_directories()["output"])

            # Find most recent evolution file for domain
            pattern = f"gepa_pattern_evolution_{domain}_*.json"
            evolution_files = list(output_dir.glob(pattern))

            if not evolution_files:
                return None

            # Get most recent file
            latest_file = max(evolution_files, key=lambda p: p.stat().st_mtime)

            with open(latest_file, "r", encoding="utf-8") as f:
                evolution_data = json.load(f)

            # Extract evolved prompt
            results = evolution_data.get("evolution_results", {})
            if prompt_type in results:
                return results[prompt_type]["evolved_prompt"]

        except Exception as e:
            logger.warning(f"Failed to load evolved prompt: {e}")

        return None


# Convenience functions
def create_pattern_extraction_gepa(
    config: Config, gemini_client: Optional[GeminiClient] = None
) -> PatternExtractionGEPA:
    """Create a pattern extraction GEPA instance."""
    return PatternExtractionGEPA(config, gemini_client)


async def evolve_discovery_prompts(
    config: Config,
    discovery_results: List[Dict[str, Any]],
    target_domain: str = "construction_documents",
) -> Dict[str, PromptEvolutionResult]:
    """
    Convenience function to evolve discovery prompts using GEPA.

    Args:
        config: System configuration
        discovery_results: Results from previous discoveries for evaluation
        target_domain: Target domain for specialization

    Returns:
        Dictionary of evolved prompts
    """
    gepa = create_pattern_extraction_gepa(config)

    # Base prompts to evolve
    base_prompts = {
        "pattern_discovery": """
        Analyze these patterns found in a technical document and discover their structure and relationships.
        Focus on identifying consistent formats, naming conventions, and systematic organization.
        """,
        "nomenclature_parsing": """
        Analyze these codes found in a technical document and discover the nomenclature system.
        Identify patterns, hierarchies, and relationships without assuming standard conventions.
        """,
        "structure_recognition": """
        Analyze the structural organization of this document and identify how elements are arranged.
        Focus on discovering the logic behind organization and categorization.
        """,
        "element_classification": """
        Classify this element based on its characteristics and context within the document.
        Provide specific type classification with confidence and reasoning.
        """,
    }

    return await gepa.evolve_pattern_extraction_prompts(
        base_prompts=base_prompts,
        target_domain=target_domain,
        sample_documents=discovery_results,
        target_improvement=0.15,
    )
