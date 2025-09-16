"""
Pattern Analyzer for discovering visual and structural patterns in documents.

Uses adaptive discovery without hardcoded assumptions.
"""

import re
from collections import Counter, defaultdict

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class PatternAnalyzer:
    """
    Analyzes patterns found in technical documents without preconceptions.

    This class uses adaptive discovery to identify patterns specific
    to each document, without imposing predefined categories.
    """

    def __init__(self, gemini_client=None):
        """
        Initialize with optional Gemini client for adaptive analysis.

        Args:
            gemini_client: Optional Gemini client for pattern inference
        """
        self.gemini_client = gemini_client

        # Everything is discovered, nothing hardcoded
        self.discovered_patterns = defaultdict(list)
        self.pattern_frequencies = Counter()
        self.learned_categories = {}  # Populated through discovery
        self.pattern_rules = []  # Learned rules from analysis

        # Model selection for different tasks (using latest Gemini 2.5 models)
        # Reference: https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini
        #
        # Gemini 2.5 Flash-Lite: Ultra-fast, lightweight model for simple tasks
        # - Best for: Binary decisions, simple pattern matching, quick validations
        # - Speed: Fastest response times
        # - Token limit: Optimized for short inputs/outputs
        #
        # Gemini 2.5 Flash: Fast, efficient model for general tasks
        # - Best for: Pattern analysis, categorization, standard processing
        # - Speed: Fast with good accuracy balance
        # - Token limit: Good for medium-sized contexts
        #
        # Gemini 2.5 Pro: Most advanced reasoning model
        # - Best for: Complex discovery, deep analysis, multi-step reasoning
        # - Speed: Slower but highest quality
        # - Token limit: 1,048,576 input / 65,535 output tokens

        self.FAST_MODEL = "gemini-2.5-flash-lite"  # Ultra-fast for simple pattern checks
        self.MEDIUM_MODEL = "gemini-2.5-flash"  # Fast for general analysis
        self.DEEP_MODEL = "gemini-2.5-pro"  # Advanced reasoning for complex discovery

    def _select_model_for_task(self, task_type: str) -> str:
        """
        Select the appropriate Gemini 2.5 model based on task complexity.

        Task types:
        - 'simple': Binary decisions, pattern checks (uses Flash-Lite)
        - 'medium': Pattern analysis, categorization (uses Flash)
        - 'complex': Deep discovery, reasoning (uses Pro)

        Returns:
            Model name string
        """
        model_map = {
            "simple": self.FAST_MODEL,  # gemini-2.5-flash-lite
            "medium": self.MEDIUM_MODEL,  # gemini-2.5-flash
            "complex": self.DEEP_MODEL,  # gemini-2.5-pro
        }
        return model_map.get(task_type, self.DEEP_MODEL)

    def _quick_gemini_call(self, prompt: str, response_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a quick Gemini call for simple pattern analysis.

        Uses Gemini 2.5 Flash-Lite for ultra-fast responses on simple tasks.

        Args:
            prompt: The prompt to send (keep it concise for speed)
            response_schema: Expected response structure

        Returns:
            Parsed response or empty dict on failure
        """
        if not self.gemini_client:
            return {}

        try:
            # TODO: When GeminiClient supports model switching, use self.FAST_MODEL (gemini-2.5-flash-lite)
            # For now, optimize by keeping prompts short and schemas simple
            response = self.gemini_client.generate_content(
                prompt=prompt, response_schema=response_schema
            )

            result = json.loads(response) if isinstance(response, str) else response
            return result
        except Exception as e:
            logger.debug(f"Quick Gemini call failed: {e}")
            return {}

    def analyze_patterns(self, patterns: List[str]) -> Dict[str, Any]:
        """
        Analyze a list of patterns to understand document structure.

        Args:
            patterns: List of pattern strings discovered in document

        Returns:
            Analysis of patterns including categories, frequencies, and relationships
        """
        logger.info(f"Analyzing {len(patterns)} patterns...")

        analysis = {
            "patterns": [],
            "categories": {},
            "frequencies": {},
            "visual_elements": [],
            "structural_elements": [],
            "relationships": [],
        }

        # Count pattern frequencies
        pattern_counter = Counter(patterns)

        # Identify unique patterns
        unique_patterns = set(patterns)
        analysis["patterns"] = list(unique_patterns)

        # Categorize patterns
        for pattern in unique_patterns:
            category = self._categorize_pattern(pattern)
            if category not in analysis["categories"]:
                analysis["categories"][category] = []
            analysis["categories"][category].append(pattern)

        # Analyze frequencies
        analysis["frequencies"] = dict(pattern_counter.most_common(20))

        # Identify visual vs structural elements
        for pattern in unique_patterns:
            if self._is_visual_element(pattern):
                analysis["visual_elements"].append(
                    {
                        "type": pattern,
                        "frequency": pattern_counter[pattern],
                        "category": self._categorize_pattern(pattern),
                    }
                )

            if self._is_structural_element(pattern):
                analysis["structural_elements"].append(
                    {
                        "type": pattern,
                        "frequency": pattern_counter[pattern],
                        "category": self._categorize_pattern(pattern),
                    }
                )

        # Identify pattern relationships
        analysis["relationships"] = self._identify_relationships(unique_patterns)

        # Store for future reference
        self.discovered_patterns["all"].extend(patterns)
        self.pattern_frequencies.update(pattern_counter)

        logger.info(f"Pattern analysis complete: {len(unique_patterns)} unique patterns found")

        return analysis

    def _categorize_pattern(self, pattern: str) -> str:
        """
        Categorize a pattern using adaptive discovery.

        Uses learned categories from the document itself, not hardcoded rules.
        """
        pattern_lower = pattern.lower()

        # First check if we've already learned this pattern's category
        if pattern in self.learned_categories:
            return self.learned_categories[pattern]

        # Use pattern structure to infer category (not content)
        category = self._infer_category_from_structure(pattern)

        # Store for future reference
        self.learned_categories[pattern] = category

        return category

    def _infer_category_from_structure(self, pattern: str) -> str:
        """
        Infer category from pattern structure alone, not content.

        This uses structural analysis, not semantic assumptions.
        """
        # Analyze pattern structure
        has_letters = bool(re.search(r"[A-Za-z]", pattern))
        has_numbers = bool(re.search(r"\d", pattern))
        has_separator = bool(re.search(r"[-_./]", pattern))
        has_special = bool(re.search(r"[^A-Za-z0-9\s\-_./]", pattern))

        # Create structural signature
        structure_sig = (has_letters, has_numbers, has_separator, has_special)

        # Map structure to generic category
        if structure_sig == (True, True, True, False):
            return "alphanumeric_code"  # Like P-101, V-201
        elif structure_sig == (False, True, True, False):
            return "numeric_sequence"  # Like 1.2.3, 100-200
        elif structure_sig == (True, False, False, False):
            return "text_label"  # Pure text
        elif structure_sig == (False, True, False, False):
            return "pure_number"  # Just numbers
        else:
            # Create unique category based on structure
            return f"pattern_type_{hash(structure_sig) % 1000}"

    def _is_visual_element(self, pattern: str) -> bool:
        """Determine if a pattern is primarily visual using fast AI or structural analysis."""
        result = self._quick_gemini_call(
            prompt=f"Is '{pattern}' a visual element in a technical document? Respond with only true or false.",
            response_schema={
                "type": "object",
                "properties": {"is_visual": {"type": "boolean"}, "confidence": {"type": "number"}},
            },
        )

        if result:
            return result.get("is_visual", False)

        # Fallback: check if pattern contains drawing-related characters
        return bool(re.search(r"[<>|\\/\-=]", pattern)) or "x" in pattern.lower()

    def _is_structural_element(self, pattern: str) -> bool:
        """Determine if a pattern represents a structural/functional element using fast AI or structural analysis."""
        result = self._quick_gemini_call(
            prompt=f"Is '{pattern}' a structural or functional element in a technical document? Respond with only true or false.",
            response_schema={
                "type": "object",
                "properties": {
                    "is_structural": {"type": "boolean"},
                    "element_type": {"type": "string"},
                },
            },
        )

        if result:
            return result.get("is_structural", False)

        # Fallback: check if pattern has alphanumeric codes (common in structural elements)
        return bool(re.search(r"[A-Z]-?\d+", pattern))

    def _identify_relationships(self, patterns: Set[str]) -> List[Dict[str, str]]:
        """Identify relationships between patterns."""
        relationships = []
        patterns_list = list(patterns)

        for i, pattern1 in enumerate(patterns_list):
            for pattern2 in patterns_list[i + 1 :]:
                relationship = self._detect_relationship(pattern1, pattern2)
                if relationship:
                    relationships.append(
                        {"pattern1": pattern1, "pattern2": pattern2, "type": relationship}
                    )

        return relationships

    def _detect_relationship(self, pattern1: str, pattern2: str) -> Optional[str]:
        """Detect if two patterns have a relationship."""
        p1_lower = pattern1.lower()
        p2_lower = pattern2.lower()

        # Check for hierarchical relationship
        if p1_lower in p2_lower or p2_lower in p1_lower:
            return "hierarchical"

        # Check for sequential relationship (e.g., P-101 and P-102)
        seq_match1 = re.search(r"([A-Z]-?)(\d+)", pattern1)
        seq_match2 = re.search(r"([A-Z]-?)(\d+)", pattern2)

        if seq_match1 and seq_match2:
            if seq_match1.group(1) == seq_match2.group(1):
                num1 = int(seq_match1.group(2))
                num2 = int(seq_match2.group(2))
                if abs(num1 - num2) == 1:
                    return "sequential"

        # Check for same category
        if self._categorize_pattern(pattern1) == self._categorize_pattern(pattern2):
            return "same_category"

        return None

    def identify_pattern_rules(self, patterns: List[str]) -> Dict[str, Any]:
        """
        Identify rules and conventions in patterns using adaptive discovery.

        Uses Gemini to understand patterns without hardcoded assumptions.
        """
        if not patterns:
            return {
                "naming_conventions": [],
                "numbering_systems": [],
                "hierarchies": [],
                "groupings": [],
            }

        # If we have Gemini client, use it for intelligent discovery
        if self.gemini_client:
            return self._discover_rules_with_ai(patterns)
        else:
            # Fallback to structural analysis only
            return self._discover_rules_structurally(patterns)

    def _discover_rules_with_ai(self, patterns: List[str]) -> Dict[str, Any]:
        """
        Use AI to discover rules without any hardcoded assumptions.

        Uses Gemini 2.5 Pro for complex pattern discovery and reasoning.
        """
        # Create discovery prompt that doesn't assume anything
        discovery_prompt = f"""
        Analyze these patterns found in a technical document:
        {json.dumps(patterns[:50], indent=2)}  # Limit to 50 for API
        
        WITHOUT making any assumptions about what these patterns mean:
        
        1. STRUCTURAL PATTERNS
           - What naming structures do you observe?
           - Are there consistent formats or templates?
           - What separators or delimiters are used?
        
        2. SEQUENCES & NUMBERING
           - Are there sequential patterns?
           - What numbering systems appear?
           - Are there hierarchical structures?
        
        3. GROUPINGS & RELATIONSHIPS
           - Which patterns seem related?
           - Are there parent-child relationships?
           - What natural groupings emerge?
        
        4. UNIQUE CONVENTIONS
           - What's unique about this document's system?
           - Are there custom codes or abbreviations?
           - What patterns don't fit standard categories?
        
        DO NOT assume these are engineering drawings or any specific type.
        Just analyze the patterns as they are.
        
        Return a structured analysis of discovered rules.
        """

        try:
            response = self.gemini_client.generate_content(
                prompt=discovery_prompt,
                response_schema={
                    "type": "object",
                    "properties": {
                        "naming_conventions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "pattern": {"type": "string"},
                                    "format": {"type": "string"},
                                    "examples": {"type": "array", "items": {"type": "string"}},
                                },
                            },
                        },
                        "numbering_systems": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string"},
                                    "pattern": {"type": "string"},
                                    "range": {"type": "string"},
                                },
                            },
                        },
                        "hierarchies": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "parent": {"type": "string"},
                                    "children": {"type": "array", "items": {"type": "string"}},
                                    "depth": {"type": "integer"},
                                },
                            },
                        },
                        "groupings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string"},
                                    "key": {"type": "string"},
                                    "members": {"type": "array", "items": {"type": "string"}},
                                },
                            },
                        },
                    },
                },
            )
            # Parse and return the response
            result = json.loads(response) if isinstance(response, str) else response
            return result
        except Exception as e:
            logger.warning(f"AI rule discovery failed: {e}, using structural analysis")
            return self._discover_rules_structurally(patterns)

    def _parse_ai_rules(self, response: str) -> Dict[str, Any]:
        """
        Parse AI response into structured rules.
        """
        # For now, return basic structure
        # In production, this would parse the AI response properly
        return {
            "naming_conventions": [],
            "numbering_systems": [],
            "hierarchies": [],
            "groupings": [],
        }

    def _discover_rules_structurally(self, patterns: List[str]) -> Dict[str, Any]:
        """
        Discover rules using only structural analysis, no semantic assumptions.
        """
        rules = {
            "naming_conventions": [],
            "numbering_systems": [],
            "hierarchies": [],
            "groupings": [],
        }

        # Analyze patterns structurally without semantic assumptions
        for pattern in patterns:
            structure = self._analyze_pattern_structure(pattern)
            if structure not in rules["naming_conventions"]:
                rules["naming_conventions"].append(structure)

        return rules

    def _analyze_pattern_structure(self, pattern: str) -> Dict[str, Any]:
        """
        Analyze the structure of a pattern without assuming meaning.
        """
        return {
            "pattern": pattern,
            "length": len(pattern),
            "has_numbers": bool(re.search(r"\d", pattern)),
            "has_letters": bool(re.search(r"[A-Za-z]", pattern)),
            "has_separators": bool(re.search(r"[-_./]", pattern)),
        }

    def _analyze_naming_conventions(self, tag_patterns: List[str]) -> List[Dict[str, Any]]:
        """Analyze naming conventions in tags."""
        conventions = []

        # Group by prefix
        prefix_groups = defaultdict(list)
        for tag in tag_patterns:
            match = re.search(r"([A-Z]+)-?(\d+)", tag)
            if match:
                prefix = match.group(1)
                number = match.group(2)
                prefix_groups[prefix].append(number)

        for prefix, numbers in prefix_groups.items():
            convention = {
                "prefix": prefix,
                "count": len(numbers),
                "range": f"{min(numbers)}-{max(numbers)}" if numbers else "",
                "pattern": f"{prefix}-XXX",
                "likely_meaning": self._guess_prefix_meaning(prefix),
            }
            conventions.append(convention)

        return conventions

    def _guess_prefix_meaning(self, prefix: str) -> str:
        """
        Infer prefix meaning using fast AI or return generic description.

        This learns from the document context, not hardcoded mappings.
        """
        # Get all patterns with this prefix for context
        prefix_patterns = [
            p for p in self.discovered_patterns.get("all", []) if p.startswith(prefix)
        ][:10]

        result = self._quick_gemini_call(
            prompt=f"""Given these patterns with prefix '{prefix}': {json.dumps(prefix_patterns)}
            What might this prefix represent? Be specific but don't assume document type.""",
            response_schema={
                "type": "object",
                "properties": {
                    "meaning": {"type": "string"},
                    "category": {"type": "string"},
                    "confidence": {"type": "number"},
                },
            },
        )

        if result:
            return result.get("meaning", f"Type-{prefix}")

        # Fallback: generic description
        return f"Type-{prefix}"

    def _analyze_numbering_systems(self, numbered_patterns: List[str]) -> List[Dict[str, str]]:
        """Analyze numbering systems used."""
        systems = []

        # Check for sequential numbering
        sequential_groups = self._find_sequential_groups(numbered_patterns)
        for group in sequential_groups:
            systems.append(
                {
                    "type": "sequential",
                    "examples": group[:5],  # First 5 examples
                    "pattern": self._extract_number_pattern(group),
                }
            )

        # Check for hierarchical numbering (e.g., 1.1, 1.2, 2.1)
        hierarchical = [p for p in numbered_patterns if re.search(r"\d+\.\d+", p)]
        if hierarchical:
            systems.append(
                {"type": "hierarchical", "examples": hierarchical[:5], "pattern": "X.Y format"}
            )

        return systems

    def _find_sequential_groups(self, patterns: List[str]) -> List[List[str]]:
        """Find groups of sequential patterns."""
        groups = []

        # Extract patterns with numbers
        number_patterns = []
        for pattern in patterns:
            matches = re.findall(r"(\d+)", pattern)
            if matches:
                number_patterns.append((pattern, [int(m) for m in matches]))

        # Group sequential patterns
        # This is simplified; real implementation would be more sophisticated
        current_group = []
        for pattern, numbers in sorted(number_patterns, key=lambda x: x[1]):
            if not current_group or self._is_sequential(current_group[-1][1], numbers):
                current_group.append(pattern)
            else:
                if len(current_group) > 2:
                    groups.append([p for p, _ in current_group])
                current_group = [pattern]

        if len(current_group) > 2:
            groups.append([p for p in current_group])

        return groups

    def _is_sequential(self, nums1: List[int], nums2: List[int]) -> bool:
        """Check if two number lists are sequential."""
        if len(nums1) != len(nums2):
            return False

        # Check if any number differs by exactly 1
        for n1, n2 in zip(nums1, nums2):
            if abs(n1 - n2) == 1:
                return True

        return False

    def _extract_number_pattern(self, group: List[str]) -> str:
        """Extract the pattern from a group of numbered items."""
        if not group:
            return ""

        # Find common prefix/suffix
        first = group[0]

        # Replace numbers with placeholder
        pattern = re.sub(r"\d+", "XXX", first)

        return pattern

    def _identify_hierarchies(self, patterns: List[str]) -> List[Dict[str, Any]]:
        """Identify hierarchical relationships in patterns."""
        hierarchies = []

        # Look for parent-child relationships
        for i, pattern1 in enumerate(patterns):
            children = []
            for pattern2 in patterns[i + 1 :]:
                if pattern1.lower() in pattern2.lower() and pattern1 != pattern2:
                    children.append(pattern2)

            if children:
                hierarchies.append(
                    {
                        "parent": pattern1,
                        "children": children[:5],  # Limit to 5 for brevity
                        "depth": self._calculate_hierarchy_depth(pattern1, children),
                    }
                )

        return hierarchies

    def _calculate_hierarchy_depth(self, parent: str, children: List[str]) -> int:
        """Calculate the depth of a hierarchy."""
        max_depth = 1

        for child in children:
            # Simple heuristic: count separators
            separators = child.count("-") + child.count("_") + child.count(".")
            depth = separators + 1
            max_depth = max(max_depth, depth)

        return max_depth

    def _identify_groupings(self, patterns: List[str]) -> List[Dict[str, str]]:
        """Identify natural groupings in patterns."""
        groupings = []

        # Group by common prefixes
        prefix_groups = defaultdict(list)
        for pattern in patterns:
            # Extract first word or prefix
            prefix = re.split(r"[-_\s\d]", pattern)[0]
            if prefix:
                prefix_groups[prefix].append(pattern)

        for prefix, group in prefix_groups.items():
            if len(group) > 2:  # Only significant groups
                groupings.append(
                    {
                        "type": "prefix",
                        "key": prefix,
                        "members": group[:10],  # Limit for brevity
                        "count": len(group),
                    }
                )

        return groupings

    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get a summary of all discovered patterns."""
        return {
            "total_patterns": len(self.discovered_patterns["all"]),
            "unique_patterns": len(set(self.discovered_patterns["all"])),
            "top_patterns": dict(self.pattern_frequencies.most_common(10)),
            "categories": {
                cat: len(patterns) for cat, patterns in self.discovered_patterns.items()
            },
        }
