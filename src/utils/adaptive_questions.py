"""
Adaptive Question Generator for Dynamic Analysis.

This module generates domain-specific questions based on discovery results,
replacing hardcoded questions with intelligent, adaptive questioning.
"""

import logging
from typing import Any, Dict, List, Optional

try:
    from ..utils.logging_config import get_logger

    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


class AdaptiveQuestionGenerator:
    """
    Generates questions dynamically based on discovered document characteristics.

    Replaces static, hardcoded questions with intelligent, context-aware questions
    that adapt to the specific document type and industry domain.
    """

    def __init__(self):
        # Base question templates by domain
        self.domain_question_templates = {
            "construction": {
                "document_analysis": [
                    "What type of {building_type} is shown in this {document_type}?",
                    "What is the scope and purpose of this {document_type}?",
                    "What construction phase does this {document_type} represent?",
                ],
                "technical_systems": [
                    "What are the main {system_types} visible in this document?",
                    "What {material_types} and specifications are indicated?",
                    "What construction methods and systems are being used?",
                ],
                "compliance": [
                    "What codes, standards, or regulations are referenced?",
                    "What safety or compliance requirements are specified?",
                    "Are there accessibility or energy efficiency requirements?",
                ],
                "measurements": [
                    "What are the key dimensions and measurements shown?",
                    "What structural loads, forces, or engineering details are specified?",
                    "Are there foundation details, footings, or underground elements shown?",
                ],
            },
            "process": {
                "system_analysis": [
                    "What type of process system is documented in this {document_type}?",
                    "What are the main process units and equipment shown?",
                    "What process flows and connections are illustrated?",
                ],
                "equipment": [
                    "What process equipment and instrumentation are specified?",
                    "What control systems and automation are indicated?",
                    "What safety systems and interlocks are shown?",
                ],
                "specifications": [
                    "What process conditions (pressure, temperature, flow) are specified?",
                    "What materials and specifications are indicated for process equipment?",
                    "What industry standards and codes are referenced?",
                ],
            },
            "electrical": {
                "system_design": [
                    "What type of electrical system is documented in this {document_type}?",
                    "What are the main electrical components and circuits shown?",
                    "What power distribution and control systems are illustrated?",
                ],
                "specifications": [
                    "What electrical specifications and ratings are indicated?",
                    "What protection and safety systems are specified?",
                    "What electrical codes and standards are referenced?",
                ],
            },
            "mechanical": {
                "design_analysis": [
                    "What type of mechanical system or assembly is shown in this {document_type}?",
                    "What are the main mechanical components and their relationships?",
                    "What manufacturing or assembly processes are indicated?",
                ],
                "specifications": [
                    "What materials, tolerances, and specifications are indicated?",
                    "What mechanical standards and codes are referenced?",
                    "What testing or quality requirements are specified?",
                ],
            },
            "naval": {
                "vessel_design": [
                    "What type of vessel or marine system is documented in this {document_type}?",
                    "What are the main hull structures and compartments shown?",
                    "What marine systems and equipment are illustrated?",
                ],
                "specifications": [
                    "What marine specifications and classifications are indicated?",
                    "What safety and regulatory requirements are specified?",
                    "What naval standards and codes are referenced?",
                ],
            },
            "aerospace": {
                "aircraft_design": [
                    "What type of aircraft or aerospace system is documented in this {document_type}?",
                    "What are the main structural and system components shown?",
                    "What flight systems and equipment are illustrated?",
                ],
                "specifications": [
                    "What aerospace specifications and certifications are indicated?",
                    "What safety and performance requirements are specified?",
                    "What aviation standards and regulations are referenced?",
                ],
            },
            "generic": {
                "document_analysis": [
                    "What type of document is this and what is its primary purpose?",
                    "What are the main topics and subjects covered?",
                    "What is the intended audience and use case for this document?",
                ],
                "content_analysis": [
                    "What are the key information elements and data presented?",
                    "What processes, systems, or methodologies are described?",
                    "What standards, references, or authorities are cited?",
                ],
                "technical_details": [
                    "What technical specifications or requirements are indicated?",
                    "What measurements, quantities, or performance criteria are specified?",
                    "What implementation or operational details are provided?",
                ],
            },
        }

        logger.info("Adaptive Question Generator initialized")

    def generate_questions(
        self, discovery_result: Dict[str, Any], max_questions: int = 8
    ) -> List[str]:
        """
        Generate adaptive questions based on discovery results.

        Args:
            discovery_result: Results from discovery analysis
            max_questions: Maximum number of questions to generate

        Returns:
            List of domain-specific questions
        """
        document_type = discovery_result.get("document_type", "unknown document")
        industry_domain = discovery_result.get("industry_domain", "unknown domain")
        discovered_patterns = discovery_result.get("discovered_patterns", {})

        logger.info(f"Generating adaptive questions for {document_type} in {industry_domain}")

        # Determine domain category
        domain_category = self._classify_domain(industry_domain, document_type)

        # Get question templates for domain
        templates = self.domain_question_templates.get(
            domain_category, self.domain_question_templates["generic"]
        )

        # Generate questions from templates
        questions = []

        # Extract context variables for template substitution
        context_vars = self._extract_context_variables(discovery_result)

        # Generate questions from each category
        for category, question_templates in templates.items():
            for template in question_templates:
                if len(questions) >= max_questions:
                    break

                try:
                    # Substitute variables in template
                    question = self._substitute_template_variables(template, context_vars)
                    questions.append(question)
                except Exception as e:
                    logger.warning(f"Failed to generate question from template '{template}': {e}")
                    # Add fallback question
                    questions.append(template)

            if len(questions) >= max_questions:
                break

        # If we don't have enough questions, add generic ones
        if len(questions) < max_questions:
            generic_questions = self._generate_generic_questions(discovery_result)
            questions.extend(generic_questions[: max_questions - len(questions)])

        # Limit to max_questions
        final_questions = questions[:max_questions]

        logger.info(
            f"Generated {len(final_questions)} adaptive questions for {domain_category} domain"
        )
        return final_questions

    def _classify_domain(self, industry_domain: str, document_type: str) -> str:
        """
        Classify the domain category for question generation.

        Args:
            industry_domain: Discovered industry domain
            document_type: Discovered document type

        Returns:
            Domain category for question templates
        """
        domain_lower = industry_domain.lower()
        doc_type_lower = document_type.lower()

        # Process engineering domain (check first - more specific)
        if any(
            keyword in domain_lower
            for keyword in ["process", "chemical", "petrochemical", "manufacturing"]
        ):
            return "process"
        elif any(keyword in doc_type_lower for keyword in ["p&id", "process", "piping"]):
            return "process"

        # Naval domain (check before general architecture)
        elif any(
            keyword in domain_lower for keyword in ["naval", "marine", "maritime", "shipbuilding"]
        ):
            return "naval"
        elif any(keyword in doc_type_lower for keyword in ["vessel", "ship", "marine"]):
            return "naval"

        # Aerospace domain (check before general engineering)
        elif any(keyword in domain_lower for keyword in ["aerospace", "aviation", "aircraft"]):
            return "aerospace"
        elif any(keyword in doc_type_lower for keyword in ["aircraft", "flight", "avionics"]):
            return "aerospace"

        # Construction/AEC domain
        elif any(keyword in domain_lower for keyword in ["construction", "aec", "architecture"]):
            return "construction"
        elif any(
            keyword in doc_type_lower for keyword in ["blueprint", "plan", "elevation", "section"]
        ):
            return "construction"
        elif "engineering" in domain_lower and "construction" in domain_lower:
            return "construction"

        # Electrical domain
        elif any(keyword in domain_lower for keyword in ["electrical", "electronics", "power"]):
            return "electrical"
        elif any(keyword in doc_type_lower for keyword in ["schematic", "circuit", "wiring"]):
            return "electrical"

        # Mechanical domain
        elif any(keyword in domain_lower for keyword in ["mechanical", "machinery", "automotive"]):
            return "mechanical"
        elif any(keyword in doc_type_lower for keyword in ["assembly", "part", "component"]):
            return "mechanical"

        # Default to generic
        else:
            return "generic"

    def _extract_context_variables(self, discovery_result: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract context variables for template substitution.

        Args:
            discovery_result: Discovery analysis results

        Returns:
            Dictionary of context variables for templates
        """
        variables = {
            "document_type": discovery_result.get("document_type", "document"),
            "industry_domain": discovery_result.get("industry_domain", "technical domain"),
            "building_type": "structure",  # Default
            "system_types": "systems",  # Default
            "material_types": "materials",  # Default
        }

        # Extract more specific variables from patterns
        discovered_patterns = discovery_result.get("discovered_patterns", {})

        # Determine building type from patterns
        if "patterns" in discovered_patterns:
            patterns = discovered_patterns["patterns"]
            if isinstance(patterns, list):
                for pattern in patterns:
                    pattern_str = str(pattern).lower()
                    if "residential" in pattern_str:
                        variables["building_type"] = "residential building"
                    elif "commercial" in pattern_str:
                        variables["building_type"] = "commercial building"
                    elif "industrial" in pattern_str:
                        variables["building_type"] = "industrial facility"
                    elif "institutional" in pattern_str:
                        variables["building_type"] = "institutional building"

        # Determine system types from patterns
        system_types = []
        if "patterns" in discovered_patterns:
            patterns = discovered_patterns["patterns"]
            if isinstance(patterns, list):
                for pattern in patterns:
                    pattern_str = str(pattern).lower()
                    if "structural" in pattern_str:
                        system_types.append("structural")
                    if "mechanical" in pattern_str or "hvac" in pattern_str:
                        system_types.append("mechanical")
                    if "electrical" in pattern_str:
                        system_types.append("electrical")
                    if "plumbing" in pattern_str:
                        system_types.append("plumbing")

        if system_types:
            variables["system_types"] = ", ".join(set(system_types)) + " systems"

        return variables

    def _substitute_template_variables(self, template: str, variables: Dict[str, str]) -> str:
        """
        Substitute variables in question template.

        Args:
            template: Question template with {variable} placeholders
            variables: Dictionary of variable values

        Returns:
            Question with variables substituted
        """
        try:
            return template.format(**variables)
        except KeyError as e:
            # If variable not found, use a generic replacement
            logger.warning(f"Template variable {e} not found, using generic replacement")
            # Replace missing variables with generic terms
            generic_replacements = {
                "building_type": "structure",
                "document_type": "document",
                "system_types": "systems",
                "material_types": "materials",
            }

            # Try again with generic replacements
            all_variables = {**variables, **generic_replacements}
            try:
                return template.format(**all_variables)
            except KeyError:
                # If still failing, replace any remaining {variable} with generic term
                import re

                result = re.sub(r"\{[^}]+\}", "element", template)
                return result

    def _generate_generic_questions(self, discovery_result: Dict[str, Any]) -> List[str]:
        """
        Generate generic questions when domain-specific templates are insufficient.

        Args:
            discovery_result: Discovery analysis results

        Returns:
            List of generic but relevant questions
        """
        document_type = discovery_result.get("document_type", "document")

        generic_questions = [
            f"What is the primary purpose and scope of this {document_type}?",
            f"What are the main components or elements described in this {document_type}?",
            f"What technical specifications or requirements are detailed?",
            f"What standards, codes, or references are cited in this {document_type}?",
            f"What implementation or operational details are provided?",
            f"What quality, safety, or compliance requirements are specified?",
            f"What relationships or dependencies between elements are shown?",
            f"What unique or specialized aspects are particular to this {document_type}?",
        ]

        return generic_questions


# Factory function for easy usage
def generate_adaptive_questions(
    discovery_result: Dict[str, Any], max_questions: int = 8
) -> List[str]:
    """
    Factory function to generate adaptive questions.

    Args:
        discovery_result: Discovery analysis results
        max_questions: Maximum number of questions to generate

    Returns:
        List of adaptive questions
    """
    generator = AdaptiveQuestionGenerator()
    return generator.generate_questions(discovery_result, max_questions)
