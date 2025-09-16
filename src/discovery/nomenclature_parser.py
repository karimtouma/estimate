"""
Nomenclature Parser for understanding document-specific naming conventions.

Uses adaptive discovery without hardcoded assumptions.
"""

import re
from collections import Counter, defaultdict

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class NomenclatureParser:
    """
    Parses and understands nomenclature systems specific to each document.

    Instead of assuming standard conventions, this class discovers
    and decodes the unique naming system used in each document.
    """

    def __init__(self, gemini_client=None):
        """
        Initialize with optional Gemini client for adaptive analysis.

        Args:
            gemini_client: Optional Gemini client for intelligent discovery
        """
        self.gemini_client = gemini_client
        self.discovered_patterns = {}
        self.decoded_meanings = {}
        self.nomenclature_rules = []

    async def parse_nomenclature(self, codes: List[str]) -> Dict[str, Any]:
        """
        Parse nomenclature codes using adaptive discovery.

        Args:
            codes: List of codes found in the document (e.g., V-201, P-101-A)

        Returns:
            Analysis of the nomenclature system
        """
        logger.info(f"Parsing nomenclature from {len(codes)} codes...")

        analysis = {
            "patterns": {},
            "components": {},
            "rules": [],
            "hierarchy": {},
            "statistics": {},
        }

        if not codes:
            return analysis

        # Use AI discovery if available
        if self.gemini_client and len(codes) > 5:
            try:
                ai_analysis = self._discover_nomenclature_with_ai(codes)
                analysis.update(ai_analysis)
            except Exception as e:
                logger.warning(f"AI nomenclature discovery failed: {e}, using structural analysis")

        # Identify patterns in codes
        patterns = self._identify_patterns(codes)
        analysis["patterns"] = patterns

        # Break down code components
        components = self._analyze_components(codes)
        analysis["components"] = components

        # Derive rules
        rules = self._derive_rules(codes, patterns)
        analysis["rules"] = rules

        # Identify hierarchy
        hierarchy = self._identify_hierarchy(codes)
        analysis["hierarchy"] = hierarchy

        # Calculate statistics
        statistics = self._calculate_statistics(codes)
        analysis["statistics"] = statistics

        # Store for future reference
        self.discovered_patterns.update(patterns)
        self.nomenclature_rules.extend(rules)

        logger.info(f"Nomenclature parsing complete: {len(patterns)} patterns found")

        return analysis

    def _discover_nomenclature_with_ai(self, codes: List[str]) -> Dict[str, Any]:
        """
        Use AI to discover nomenclature system without assumptions.
        """
        # Create discovery prompt that makes NO assumptions
        discovery_prompt = f"""
        Analyze these codes found in a technical document:
        {json.dumps(codes[:100], indent=2)}  # Limit for API
        
        WITHOUT assuming these are engineering codes or any specific domain:
        
        1. PATTERN DISCOVERY
           - What consistent patterns do you observe?
           - What are the structural components (letters, numbers, separators)?
           - Are there templates or formats being followed?
        
        2. SYSTEM ANALYSIS
           - Is there a systematic naming convention?
           - What rules govern the code formation?
           - Are there hierarchies or categories?
        
        3. RELATIONSHIPS
           - Do some codes seem related to others?
           - Are there parent-child relationships?
           - Do you see groupings or families?
        
        4. UNIQUE CHARACTERISTICS
           - What's unique about this coding system?
           - Are there special markers or indicators?
           - What patterns are document-specific?
        
        DO NOT assume meanings. Just analyze the structure and patterns.
        DO NOT reference standard conventions. Discover from the data alone.
        
        Return a structured analysis of the nomenclature system.
        """

        try:
            # Use generate_content instead of analyze_text
            response = self.gemini_client.generate_content(
                prompt=discovery_prompt,
                response_schema={
                    "type": "object",
                    "properties": {
                        "nomenclature_system": {
                            "type": "object",
                            "properties": {
                                "patterns": {"type": "array", "items": {"type": "string"}},
                                "meanings": {"type": "object"},
                                "rules": {"type": "array", "items": {"type": "string"}},
                            },
                        }
                    },
                },
            )
            # Parse the response into structured format
            import json

            result = json.loads(response) if isinstance(response, str) else response
            return result.get("nomenclature_system", {})
        except Exception as e:
            logger.error(f"AI nomenclature discovery failed: {e}")
            return {}

    def _parse_ai_nomenclature_analysis(self, response: str) -> Dict[str, Any]:
        """
        Parse AI response into structured nomenclature analysis.
        """
        # This would parse the AI response into the expected format
        # For now, return empty dict as fallback
        return {}

    def _identify_patterns(self, codes: List[str]) -> Dict[str, Any]:
        """Identify patterns in nomenclature codes."""
        patterns = {}

        # Common patterns to look for
        pattern_types = {
            "prefix_number": r"^([A-Z]+)-?(\d+)$",  # P-101, V201
            "prefix_number_suffix": r"^([A-Z]+)-?(\d+)-?([A-Z]+)$",  # P-101-A
            "hierarchical": r"^(\d+)\.(\d+)(?:\.(\d+))?$",  # 1.2.3
            "alphanumeric": r"^([A-Z]+)(\d+)([A-Z]+)(\d+)$",  # AB12CD34
            "loop_number": r"^(\d{4})-?([A-Z]+)?$",  # 1001, 1001-A
            "sheet_reference": r"^[Ss]heet\s*(\d+)$",  # Sheet 12
            "revision": r"^[Rr]ev\.?\s*([A-Z0-9]+)$",  # Rev.3, Rev A
            "custom": r".*",  # Catch-all for unique patterns
        }

        for pattern_name, pattern_regex in pattern_types.items():
            matching_codes = []
            for code in codes:
                if re.match(pattern_regex, code):
                    matching_codes.append(code)

            if matching_codes:
                patterns[pattern_name] = {
                    "regex": pattern_regex,
                    "examples": matching_codes[:5],
                    "count": len(matching_codes),
                    "format": self._derive_format(pattern_name, matching_codes),
                }

        return patterns

    def _derive_format(self, pattern_name: str, examples: List[str]) -> str:
        """Derive the format string for a pattern."""
        if not examples:
            return ""

        # Analyze first example to create format string
        example = examples[0]

        if pattern_name == "prefix_number":
            match = re.match(r"^([A-Z]+)-?(\d+)$", example)
            if match:
                prefix_len = len(match.group(1))
                number_len = len(match.group(2))
                separator = "-" if "-" in example else ""
                return f"{'X' * prefix_len}{separator}{'#' * number_len}"

        elif pattern_name == "prefix_number_suffix":
            match = re.match(r"^([A-Z]+)-?(\d+)-?([A-Z]+)$", example)
            if match:
                return f"{match.group(1)}-###-{match.group(3)}"

        elif pattern_name == "hierarchical":
            parts = example.split(".")
            return ".".join(["#" * len(part) for part in parts])

        elif pattern_name == "loop_number":
            if "-" in example:
                return "####-X"
            else:
                return "####"

        # Default format
        format_str = re.sub(r"[A-Z]+", "X", example)
        format_str = re.sub(r"\d+", "#", format_str)

        return format_str

    def _analyze_components(self, codes: List[str]) -> Dict[str, Any]:
        """Analyze the components that make up codes."""
        components = {
            "prefixes": Counter(),
            "suffixes": Counter(),
            "numbers": [],
            "separators": Counter(),
            "lengths": Counter(),
        }

        for code in codes:
            # Extract prefix (leading letters)
            prefix_match = re.match(r"^([A-Z]+)", code)
            if prefix_match:
                components["prefixes"][prefix_match.group(1)] += 1

            # Extract suffix (trailing letters after numbers)
            suffix_match = re.search(r"\d+([A-Z]+)$", code)
            if suffix_match:
                components["suffixes"][suffix_match.group(1)] += 1

            # Extract numbers
            numbers = re.findall(r"\d+", code)
            components["numbers"].extend([int(n) for n in numbers])

            # Identify separators
            for sep in ["-", "_", ".", "/", "\\"]:
                if sep in code:
                    components["separators"][sep] += 1

            # Track code lengths
            components["lengths"][len(code)] += 1

        # Convert counters to sorted lists for JSON serialization
        return {
            "prefixes": dict(components["prefixes"].most_common(10)),
            "suffixes": dict(components["suffixes"].most_common(10)),
            "number_range": {
                "min": min(components["numbers"]) if components["numbers"] else 0,
                "max": max(components["numbers"]) if components["numbers"] else 0,
                "average": (
                    sum(components["numbers"]) / len(components["numbers"])
                    if components["numbers"]
                    else 0
                ),
            },
            "separators": dict(components["separators"]),
            "length_distribution": dict(components["lengths"]),
        }

    def _derive_rules(self, codes: List[str], patterns: Dict[str, Any]) -> List[Dict[str, str]]:
        """Derive rules that govern the nomenclature system."""
        rules = []

        # Rule 1: Prefix meanings
        prefix_meanings = self._infer_prefix_meanings(codes)
        for prefix, meaning in prefix_meanings.items():
            rules.append(
                {
                    "type": "prefix_meaning",
                    "pattern": f"{prefix}-XXX",
                    "rule": f"Prefix '{prefix}' likely means {meaning}",
                    "confidence": (
                        "high" if len([c for c in codes if c.startswith(prefix)]) > 3 else "medium"
                    ),
                }
            )

        # Rule 2: Number sequences
        sequence_rules = self._identify_sequences(codes)
        for rule in sequence_rules:
            rules.append(rule)

        # Rule 3: Suffix patterns
        suffix_rules = self._analyze_suffixes(codes)
        for rule in suffix_rules:
            rules.append(rule)

        # Rule 4: Hierarchical relationships
        if "hierarchical" in patterns:
            rules.append(
                {
                    "type": "hierarchy",
                    "pattern": patterns["hierarchical"]["format"],
                    "rule": "Codes follow hierarchical numbering (parent.child.subchild)",
                    "confidence": "high",
                }
            )

        return rules

    def _infer_prefix_meanings(self, codes: List[str]) -> Dict[str, str]:
        """
        Infer prefix meanings from patterns alone, no hardcoded assumptions.

        This method discovers meaning from context and structure.
        """
        prefix_groups = defaultdict(list)

        # Group codes by prefix
        for code in codes:
            prefix_match = re.match(r"^([A-Z]+)", code)
            if prefix_match:
                prefix = prefix_match.group(1)
                prefix_groups[prefix].append(code)

        meanings = {}

        for prefix, codes_list in prefix_groups.items():
            # Analyze the distribution and context to infer meaning
            # Don't assume - just describe what we observe

            # Count instances and number ranges
            numbers = []
            for code in codes_list:
                num_match = re.search(r"(\d+)", code)
                if num_match:
                    numbers.append(int(num_match.group(1)))

            if numbers:
                # Describe based on observed patterns
                num_range = f"{min(numbers)}-{max(numbers)}"
                count = len(codes_list)

                # Create descriptive label based on observations
                if count > 10:
                    meanings[prefix] = (
                        f"{prefix}-Series (frequent, {count} instances, range {num_range})"
                    )
                elif count > 5:
                    meanings[prefix] = f"{prefix}-Type (common, {count} instances)"
                else:
                    meanings[prefix] = f"{prefix}-Element ({count} instances)"
            else:
                meanings[prefix] = f"{prefix}-Category"

        return meanings

    def _identify_sequences(self, codes: List[str]) -> List[Dict[str, str]]:
        """Identify sequential numbering rules."""
        rules = []

        # Extract codes with numbers
        numbered_codes = []
        for code in codes:
            match = re.search(r"([A-Z]+)-?(\d+)", code)
            if match:
                prefix = match.group(1)
                number = int(match.group(2))
                numbered_codes.append((prefix, number, code))

        # Group by prefix
        prefix_groups = defaultdict(list)
        for prefix, number, code in numbered_codes:
            prefix_groups[prefix].append(number)

        # Analyze each group
        for prefix, numbers in prefix_groups.items():
            numbers_sorted = sorted(numbers)

            # Check for sequential pattern
            if len(numbers_sorted) > 2:
                gaps = [
                    numbers_sorted[i + 1] - numbers_sorted[i]
                    for i in range(len(numbers_sorted) - 1)
                ]

                if all(gap == 1 for gap in gaps):
                    rules.append(
                        {
                            "type": "sequence",
                            "pattern": f"{prefix}-XXX",
                            "rule": f"{prefix} codes are sequentially numbered",
                            "confidence": "high",
                        }
                    )
                elif all(gap == gaps[0] for gap in gaps):
                    rules.append(
                        {
                            "type": "sequence",
                            "pattern": f"{prefix}-XXX",
                            "rule": f"{prefix} codes increment by {gaps[0]}",
                            "confidence": "medium",
                        }
                    )

                # Check for series (100s, 200s, etc.)
                series = defaultdict(list)
                for num in numbers_sorted:
                    series[num // 100].append(num)

                if len(series) > 1:
                    rules.append(
                        {
                            "type": "series",
                            "pattern": f"{prefix}-XXX",
                            "rule": f"{prefix} codes are organized in {len(series)} series: {', '.join(f'{s*100}s' for s in sorted(series.keys()))}",
                            "confidence": "high",
                        }
                    )

        return rules

    def _analyze_suffixes(self, codes: List[str]) -> List[Dict[str, str]]:
        """Analyze suffix patterns without assuming meanings."""
        rules = []

        # Extract codes with suffixes
        suffix_codes = []
        for code in codes:
            match = re.match(r"^([A-Z]+-?\d+)-?([A-Z]+)$", code)
            if match:
                base = match.group(1)
                suffix = match.group(2)
                suffix_codes.append((base, suffix))

        if not suffix_codes:
            return rules

        # Group by base code
        base_groups = defaultdict(list)
        for base, suffix in suffix_codes:
            base_groups[base].append(suffix)

        # Analyze suffix patterns
        all_suffixes = [suffix for _, suffix in suffix_codes]
        suffix_counter = Counter(all_suffixes)

        # Discover suffix patterns without assuming meaning
        for suffix, count in suffix_counter.most_common():
            if count > 1:
                # Describe pattern, don't assume meaning
                rules.append(
                    {
                        "type": "suffix_pattern",
                        "pattern": f"XXX-{suffix}",
                        "rule": f"Suffix '{suffix}' appears {count} times",
                        "confidence": "observed",
                    }
                )

        # Check for paired patterns (like A/B)
        for base, suffixes in base_groups.items():
            suffix_set = set(suffixes)
            if len(suffix_set) == 2:
                rules.append(
                    {
                        "type": "paired_variants",
                        "pattern": f"{base}-{'/'.join(sorted(suffix_set))}",
                        "rule": f"{base} has variants: {', '.join(sorted(suffix_set))}",
                        "confidence": "observed",
                    }
                )
            elif len(suffix_set) > 2:
                rules.append(
                    {
                        "type": "multiple_variants",
                        "pattern": f"{base}-[{'/'.join(sorted(suffix_set))}]",
                        "rule": f"{base} has {len(suffix_set)} variants",
                        "confidence": "observed",
                    }
                )

        return rules

    def _identify_hierarchy(self, codes: List[str]) -> Dict[str, Any]:
        """Identify hierarchical relationships in codes."""
        hierarchy = {"levels": [], "relationships": [], "tree": {}}

        # Look for hierarchical patterns
        hierarchical_codes = []
        for code in codes:
            # Check for dot notation (1.2.3)
            if "." in code:
                parts = code.split(".")
                hierarchical_codes.append((code, parts))
            # Check for nested prefixes (P-101-A-1)
            elif code.count("-") > 1:
                parts = code.split("-")
                hierarchical_codes.append((code, parts))

        if not hierarchical_codes:
            return hierarchy

        # Build hierarchy tree
        for code, parts in hierarchical_codes:
            current_level = hierarchy["tree"]
            for i, part in enumerate(parts):
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]

                # Track levels
                level_name = f"Level {i+1}"
                if level_name not in hierarchy["levels"]:
                    hierarchy["levels"].append(level_name)

        # Identify parent-child relationships
        for code, parts in hierarchical_codes:
            if len(parts) > 1:
                parent = ".".join(parts[:-1]) if "." in code else "-".join(parts[:-1])
                hierarchy["relationships"].append(
                    {"parent": parent, "child": code, "level": len(parts)}
                )

        return hierarchy

    def _calculate_statistics(self, codes: List[str]) -> Dict[str, Any]:
        """Calculate statistics about the nomenclature system."""
        stats = {
            "total_codes": len(codes),
            "unique_codes": len(set(codes)),
            "average_length": sum(len(c) for c in codes) / len(codes) if codes else 0,
            "max_length": max(len(c) for c in codes) if codes else 0,
            "min_length": min(len(c) for c in codes) if codes else 0,
            "has_separators": any("-" in c or "_" in c or "." in c for c in codes),
            "has_suffixes": any(re.match(r"^[A-Z]+-?\d+-?[A-Z]+$", c) for c in codes),
            "has_hierarchy": any("." in c for c in codes),
        }

        return stats

    def decode_code(self, code: str) -> Dict[str, str]:
        """
        Decode a specific code based on learned patterns.

        Args:
            code: The code to decode

        Returns:
            Dictionary with decoded components and meaning
        """
        decoded = {
            "original": code,
            "components": {},
            "meaning": "",
            "pattern": "",
            "confidence": "low",
        }

        # Try to match against discovered patterns
        for pattern_name, pattern_info in self.discovered_patterns.items():
            if re.match(pattern_info["regex"], code):
                decoded["pattern"] = pattern_name
                decoded["confidence"] = "high" if pattern_info["count"] > 5 else "medium"
                break

        # Extract components
        # Prefix
        prefix_match = re.match(r"^([A-Z]+)", code)
        if prefix_match:
            prefix = prefix_match.group(1)
            decoded["components"]["prefix"] = prefix

            # Get meaning if known
            if prefix in self.decoded_meanings:
                decoded["components"]["prefix_meaning"] = self.decoded_meanings[prefix]

        # Number
        number_match = re.search(r"(\d+)", code)
        if number_match:
            decoded["components"]["number"] = number_match.group(1)

        # Suffix
        suffix_match = re.search(r"-([A-Z]+)$", code)
        if suffix_match:
            decoded["components"]["suffix"] = suffix_match.group(1)

        # Build meaning from observed patterns only
        meaning_parts = []
        if "prefix_meaning" in decoded["components"]:
            meaning_parts.append(decoded["components"]["prefix_meaning"])
        if "number" in decoded["components"]:
            meaning_parts.append(f"#{decoded['components']['number']}")
        if "suffix" in decoded["components"]:
            suffix = decoded["components"]["suffix"]
            # Don't assume meaning, just describe
            meaning_parts.append(f"variant-{suffix}")

        decoded["meaning"] = " ".join(meaning_parts) if meaning_parts else "Unknown"

        return decoded

    def get_nomenclature_summary(self) -> Dict[str, Any]:
        """Get a summary of the discovered nomenclature system."""
        return {
            "patterns_discovered": len(self.discovered_patterns),
            "rules_identified": len(self.nomenclature_rules),
            "pattern_types": list(self.discovered_patterns.keys()),
            "decoded_prefixes": len(self.decoded_meanings),
            "has_hierarchy": any("hierarchy" in rule["type"] for rule in self.nomenclature_rules),
            "has_sequences": any("sequence" in rule["type"] for rule in self.nomenclature_rules),
        }
