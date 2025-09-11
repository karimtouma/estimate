"""
Dynamic Discovery System for Adaptive Document Analysis.

This module implements a discovery system that learns the structure
of documents without imposing predefined taxonomies.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from PIL import Image
import fitz  # PyMuPDF

from ..core.config import Config
from ..services.gemini_client import GeminiClient
from ..utils.logging_config import get_logger
from .pattern_analyzer import PatternAnalyzer
from .nomenclature_parser import NomenclatureParser

logger = get_logger(__name__)


@dataclass
class DiscoveryResult:
    """Results from the discovery process."""
    document_type: str = "unknown"
    industry_domain: str = "unknown"
    discovered_patterns: Dict[str, Any] = field(default_factory=dict)
    nomenclature_system: Dict[str, Any] = field(default_factory=dict)
    page_organization: Dict[str, Any] = field(default_factory=dict)
    cross_references: List[Dict[str, Any]] = field(default_factory=list)
    element_types: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    discovery_metadata: Dict[str, Any] = field(default_factory=dict)


class DynamicPlanoDiscovery:
    """
    System that discovers document structure without preconceptions.
    
    This is the core of FASE 1: replacing fixed taxonomies with
    adaptive discovery that learns from the document itself.
    """
    
    def __init__(self, config: Config, pdf_path: Path):
        self.config = config
        self.pdf_path = pdf_path
        self.gemini_client = GeminiClient(config)
        # Pass Gemini client to analyzers for adaptive discovery
        self.pattern_analyzer = PatternAnalyzer(gemini_client=self.gemini_client)
        self.nomenclature_parser = NomenclatureParser(gemini_client=self.gemini_client)
        
        # Discovery state
        self.discovered_ontology = {}
        self.page_fingerprints = []
        self.emergent_patterns = {}
        
        # INTELLIGENT CACHING SYSTEM
        self.page_cache = {}  # Cache for page text and metadata
        self.complexity_cache = {}  # Cache for complexity scores
        self.visual_cache = {}  # Cache for visual elements
        
        # Open PDF for analysis
        self.pdf_document = None
        self._open_pdf()
        
        # PRE-CACHE critical pages for faster access
        self._initialize_smart_cache()
        
        logger.info(f"DynamicPlanoDiscovery initialized for: {pdf_path}")
    
    def _open_pdf(self):
        """Open the PDF document for analysis."""
        try:
            # Suppress MuPDF warnings for corrupted PDFs
            import os
            os.environ['MUPDF_QUIET'] = '1'
            fitz.TOOLS.mupdf_display_errors(False)
            
            self.pdf_document = fitz.open(str(self.pdf_path))
            self.total_pages = len(self.pdf_document)
            logger.info(f"PDF opened: {self.total_pages} pages")
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")
            raise
    
    def strategic_sampling(self, total_pages: int, sample_size: int = 10) -> List[int]:
        """
        EXHAUSTIVE strategic sampling for maximum coverage.
        
        NEW STRATEGY:
        - Adaptive sampling based on document size
        - Higher coverage for smaller documents
        - Smart distribution for larger documents
        - Include complexity-based selection
        """
        # ADAPTIVE SAMPLING: More aggressive for better analysis
        if total_pages <= 20:
            # Small documents: analyze ALL pages
            logger.info(f"📚 Small document ({total_pages} pages): analyzing ALL pages")
            return list(range(total_pages))
        elif total_pages <= 50:
            # Medium documents: analyze 60% of pages
            adaptive_sample_size = max(int(total_pages * 0.6), sample_size)
            logger.info(f"📖 Medium document ({total_pages} pages): analyzing {adaptive_sample_size} pages (60%)")
        else:
            # Large documents: analyze at least 30% but with smart distribution
            adaptive_sample_size = max(int(total_pages * 0.3), sample_size * 2)
            logger.info(f"📚 Large document ({total_pages} pages): analyzing {adaptive_sample_size} pages (30%)")
        
        # Strategic selection with higher coverage
        samples = set()
        
        # MANDATORY pages
        samples.add(0)  # First page
        samples.add(total_pages - 1)  # Last page
        if total_pages > 2:
            samples.add(total_pages // 2)  # Middle page
        
        # REGULAR DISTRIBUTION with higher density
        if adaptive_sample_size > 3:
            interval = max(1, total_pages // (adaptive_sample_size - 3))
            for i in range(1, adaptive_sample_size - 2):
                page_idx = min(i * interval, total_pages - 1)
                samples.add(page_idx)
        
        # HIGH-COMPLEXITY pages (visual richness indicators)
        complexity_samples = self._find_high_complexity_pages(limit=min(5, adaptive_sample_size // 3))
        samples.update(complexity_samples)
        
        # QUARTILE sampling for even distribution
        quartiles = [
            total_pages // 4,
            total_pages // 2, 
            (3 * total_pages) // 4
        ]
        samples.update([q for q in quartiles if 0 <= q < total_pages])
        
        # Ensure we don't exceed our target but get good coverage
        samples_list = sorted(list(samples))[:adaptive_sample_size]
        
        coverage_percent = (len(samples_list) / total_pages) * 100
        logger.info(f"🎯 EXHAUSTIVE sampling: {len(samples_list)} pages ({coverage_percent:.1f}% coverage)")
        logger.info(f"📋 Selected pages: {samples_list[:10]}{'...' if len(samples_list) > 10 else ''}")
        
        return samples_list
    
    def _find_high_complexity_pages(self, limit: int = 3) -> List[int]:
        """Find pages with high visual complexity."""
        complexity_scores = []
        
        for page_num in range(min(self.total_pages, 20)):  # Quick scan of first 20 pages
            page = self.pdf_document[page_num]
            
            # Simple complexity metric based on number of paths/drawings
            drawings = page.get_drawings()
            text_blocks = page.get_text("blocks")
            
            complexity = len(drawings) + len(text_blocks)
            complexity_scores.append((page_num, complexity))
        
        # Sort by complexity and return top pages
        complexity_scores.sort(key=lambda x: x[1], reverse=True)
        return [page_num for page_num, _ in complexity_scores[:limit]]
    
    async def initial_exploration(self, sample_size: int = 10, pdf_uri: str = None) -> DiscoveryResult:
        """
        First pass: understand what this document is about.
        
        OPTIMIZED VERSION: Uses batch analysis to reduce API calls from N to 1.
        """
        logger.info("Starting initial exploration phase...")
        
        # Select strategic pages
        sample_pages = self.strategic_sampling(self.total_pages, sample_size)
        
        # Build comprehensive exploration prompt for ALL pages at once
        exploration_prompt = f"""
        You are analyzing a technical document with {self.total_pages} pages total.
        I'm providing you with {len(sample_pages)} strategically selected pages: {sample_pages}
        
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
        
        # Single optimized discovery call
        logger.info(f"Performing batch analysis of {len(sample_pages)} pages...")
        
        if pdf_uri:
            logger.info("🚀 Using OPTIMIZED batch analysis with single API call")
            # Use the uploaded PDF directly - much more efficient
            discovery = self._analyze_batch_discovery(pdf_uri, exploration_prompt, sample_pages)
            # Process batch discovery result
            result = await self._process_batch_discovery(discovery, sample_pages)
        else:
            logger.warning("⚠️ No PDF URI provided - using PARALLEL page analysis")
            # Use parallel processing instead of sequential
            result = await self._analyze_pages_parallel(sample_pages, exploration_prompt)
        
        logger.info(f"Initial exploration complete. Document type: {result.document_type}")
        
        return result
    
    def _render_page(self, page_num: int) -> bytes:
        """Render a PDF page as an image with error handling."""
        try:
            page = self.pdf_document[page_num]
            
            # Render at high resolution for better analysis
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Convert to PIL Image then to bytes
            img_data = pix.tobytes("png")
            return img_data
        except Exception as e:
            logger.warning(f"Failed to render page {page_num}: {e}")
            # Return a blank image as fallback
            import io
            from PIL import Image
            img = Image.new('RGB', (850, 1100), color='white')
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            return buffer.getvalue()
    
    def _extract_page_text(self, page_num: int) -> str:
        """Extract text from a PDF page with intelligent caching."""
        # Check cache first
        if page_num in self.page_cache:
            return self.page_cache[page_num]['text']
        
        # Extract and cache
        page = self.pdf_document[page_num]
        text = page.get_text()
        
        # Cache with metadata
        self.page_cache[page_num] = {
            'text': text,
            'text_length': len(text),
            'extracted_at': time.time()
        }
        
        return text
    
    def _calculate_visual_complexity(self, page_num: int) -> float:
        """Calculate visual complexity score for a page with caching."""
        # Check cache first
        if page_num in self.complexity_cache:
            return self.complexity_cache[page_num]
        
        page = self.pdf_document[page_num]
        
        # Metrics for complexity
        drawings = page.get_drawings()
        text_blocks = page.get_text("blocks")
        images = page.get_images()
        
        # Enhanced complexity calculation
        complexity = (
            len(drawings) * 2.0 +  # Drawings are most complex
            len(text_blocks) * 1.0 +  # Text adds moderate complexity
            len(images) * 1.5 +  # Images add complexity
            (len(page.get_text()) / 1000) * 0.5  # Text density factor
        )
        
        # Normalize to 0-1 range
        normalized = min(complexity / 100.0, 1.0)
        
        # Cache the result
        self.complexity_cache[page_num] = normalized
        
        return normalized
    
    def _analyze_page_discovery(
        self, 
        page_image: bytes, 
        page_text: str, 
        prompt: str,
        pdf_uri: str = None
    ) -> Dict[str, Any]:
        """Analyze a page for discovery using Gemini - optimized version."""
        try:
            # If we have a PDF URI, use it (more efficient)
            if pdf_uri:
                response = self.gemini_client.generate_content(
                    file_uri=pdf_uri,
                    prompt=prompt + f"\n\nFocus on page content: {page_text[:500]}",
                    response_schema={
                        "type": "object",
                        "properties": {
                        "document_indicators": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "domain": {"type": "string"},
                                "confidence": {"type": "number"}
                            }
                        },
                        "organization": {
                            "type": "object",
                            "properties": {
                                "structure": {"type": "string"},
                                "sections": {"type": "array", "items": {"type": "string"}},
                                "hierarchy": {"type": "string"}
                            }
                        },
                        "nomenclature": {
                            "type": "object",
                            "properties": {
                                "codes_found": {"type": "array", "items": {"type": "string"}},
                                "naming_pattern": {"type": "string"},
                                "meanings": {"type": "string"}
                            }
                        },
                        "visual_patterns": {
                            "type": "object",
                            "properties": {
                                "drawing_types": {"type": "array", "items": {"type": "string"}},
                                "symbols": {"type": "array", "items": {"type": "string"}},
                                "conventions": {"type": "array", "items": {"type": "string"}}
                            }
                        },
                        "relationships": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string"},
                                    "reference": {"type": "string"},
                                    "target": {"type": "string"}
                                }
                            }
                        },
                        "unique_elements": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                }
                )
            else:
                # Fallback: analyze with just text if no PDF URI
                logger.warning("No PDF URI provided, using text-only analysis")
                return {
                    "document_indicators": {"type": "unknown", "domain": "unknown"},
                    "organization": {},
                    "nomenclature": {"codes_found": [], "naming_pattern": "", "meanings": ""},
                    "visual_patterns": {"drawing_types": [], "symbols": [], "conventions": []},
                    "relationships": [],
                    "unique_elements": []
                }
            
            # Parse the JSON response
            import json
            try:
                result = json.loads(response)
                return result
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response, using defaults")
                return {
                    "document_indicators": {"type": "unknown", "domain": "unknown"},
                    "organization": {},
                    "nomenclature": {},
                    "visual_patterns": {},
                    "relationships": [],
                    "unique_elements": []
                }
            
        except Exception as e:
            logger.error(f"Page discovery analysis failed: {e}")
            return {
                "document_indicators": {"type": "unknown", "domain": "unknown"},
                "organization": {},
                "nomenclature": {},
                "visual_patterns": {},
                "relationships": [],
                "unique_elements": []
            }
    
    async def _synthesize_discoveries(self, discoveries: List[Dict[str, Any]]) -> DiscoveryResult:
        """Synthesize individual page discoveries into document understanding."""
        logger.info("Synthesizing discoveries into document understanding...")
        
        # Aggregate findings
        all_document_types = []
        all_domains = []
        all_nomenclatures = []
        all_patterns = []
        all_relationships = []
        all_unique_elements = []
        
        for discovery in discoveries:
            findings = discovery['findings']
            
            # Collect document indicators
            if 'document_indicators' in findings:
                all_document_types.append(findings['document_indicators'].get('type', ''))
                all_domains.append(findings['document_indicators'].get('domain', ''))
            
            # Collect nomenclature
            if 'nomenclature' in findings:
                all_nomenclatures.extend(findings['nomenclature'].get('codes_found', []))
            
            # Collect patterns
            if 'visual_patterns' in findings:
                all_patterns.extend(findings['visual_patterns'].get('drawing_types', []))
            
            # Collect relationships
            if 'relationships' in findings:
                all_relationships.extend(findings['relationships'])
            
            # Collect unique elements
            if 'unique_elements' in findings:
                all_unique_elements.extend(findings['unique_elements'])
        
        # Determine document type and domain
        document_type = self._determine_consensus(all_document_types)
        industry_domain = self._determine_consensus(all_domains)
        
        # Parse nomenclature system (now async)
        nomenclature_system = await self.nomenclature_parser.parse_nomenclature(all_nomenclatures)
        
        # Analyze patterns
        pattern_analysis = self.pattern_analyzer.analyze_patterns(all_patterns)
        
        # Build result
        result = DiscoveryResult(
            document_type=document_type,
            industry_domain=industry_domain,
            discovered_patterns=pattern_analysis,
            nomenclature_system=nomenclature_system,
            page_organization=self._analyze_organization(discoveries),
            cross_references=all_relationships,
            element_types=list(set(all_unique_elements)),
            confidence_score=self._calculate_confidence(discoveries),
            discovery_metadata={
                "pages_analyzed": len(discoveries),
                "total_pages": self.total_pages,
                "unique_patterns": len(set(all_patterns)),
                "nomenclature_codes": len(set(all_nomenclatures))
            }
        )
        
        return result
    
    def _determine_consensus(self, items: List[str]) -> str:
        """Determine consensus from a list of items."""
        if not items:
            return "unknown"
        
        # Count occurrences
        from collections import Counter
        counts = Counter(items)
        
        # Filter out empty and unknown
        filtered_items = [item for item in items if item and item != "unknown"]
        
        if not filtered_items:
            return "unknown"
        
        # Create new counter with filtered items
        filtered_counts = Counter(filtered_items)
        
        # Return most common
        return filtered_counts.most_common(1)[0][0]
    
    def _analyze_organization(self, discoveries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze document organization from discoveries."""
        organization = {
            "structure": "unknown",
            "sections": [],
            "page_groups": [],
            "hierarchy": "flat"
        }
        
        # Analyze page progression
        page_numbers = [d['page'] for d in discoveries]
        
        # Look for patterns in organization
        for discovery in discoveries:
            findings = discovery['findings']
            if 'organization' in findings:
                org = findings['organization']
                if org.get('structure'):
                    organization['structure'] = org['structure']
                if org.get('sections'):
                    organization['sections'].extend(org['sections'])
                if org.get('hierarchy'):
                    organization['hierarchy'] = org['hierarchy']
        
        # Deduplicate sections
        organization['sections'] = list(set(organization['sections']))
        
        return organization
    
    def _calculate_confidence(self, discoveries: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score."""
        confidences = []
        
        for discovery in discoveries:
            findings = discovery['findings']
            if 'document_indicators' in findings:
                conf = findings['document_indicators'].get('confidence', 0.5)
                confidences.append(conf)
        
        if not confidences:
            return 0.5
        
        # Average confidence
        return sum(confidences) / len(confidences)
    
    async def deep_pattern_analysis(self, discovery_result: DiscoveryResult) -> Dict[str, Any]:
        """
        Perform deep analysis of discovered patterns.
        
        This goes beyond initial discovery to understand the deep
        structure and relationships in the document.
        """
        logger.info("Performing deep pattern analysis...")
        
        # Select pages that exemplify discovered patterns
        pattern_pages = self._select_pattern_exemplars(discovery_result)
        
        deep_analysis = {
            "pattern_definitions": {},
            "relationship_graph": {},
            "nomenclature_decoding": {},
            "visual_language": {}
        }
        
        # Analyze each pattern in detail
        for pattern_type, pages in pattern_pages.items():
            logger.info(f"Deep analysis of pattern: {pattern_type}")
            
            pattern_detail = await self._analyze_pattern_detail(
                pattern_type, 
                pages,
                discovery_result
            )
            
            deep_analysis["pattern_definitions"][pattern_type] = pattern_detail
        
        # Build relationship graph
        deep_analysis["relationship_graph"] = self._build_relationship_graph(
            discovery_result.cross_references
        )
        
        # Decode nomenclature system
        deep_analysis["nomenclature_decoding"] = await self._decode_nomenclature(
            discovery_result.nomenclature_system
        )
        
        # Understand visual language
        deep_analysis["visual_language"] = await self._understand_visual_language(
            discovery_result.discovered_patterns
        )
        
        return deep_analysis
    
    def _select_pattern_exemplars(self, discovery_result: DiscoveryResult) -> Dict[str, List[int]]:
        """Select pages that best exemplify each discovered pattern."""
        exemplars = {}
        
        # For each discovered pattern, find best example pages
        for pattern in discovery_result.discovered_patterns.get("patterns", []):
            # Simple heuristic: select pages mentioned in discovery
            # In production, this would be more sophisticated
            exemplars[pattern] = list(range(min(3, self.total_pages)))
        
        return exemplars
    
    async def _analyze_pattern_detail(
        self, 
        pattern_type: str, 
        pages: List[int],
        discovery_result: DiscoveryResult
    ) -> Dict[str, Any]:
        """Analyze a specific pattern in detail."""
        pattern_detail = {
            "type": pattern_type,
            "characteristics": [],
            "rules": [],
            "examples": []
        }
        
        # Analyze each example page
        for page_num in pages[:2]:  # Limit to avoid too many API calls
            page_image = self._render_page(page_num)
            
            # Pattern-specific analysis prompt
            prompt = f"""
            Analyze this page focusing on the pattern: {pattern_type}
            
            Based on initial discovery:
            - Document type: {discovery_result.document_type}
            - Domain: {discovery_result.industry_domain}
            
            Identify:
            1. Specific characteristics of this pattern
            2. Rules or conventions being followed
            3. How this pattern relates to others
            4. Any variations or exceptions
            
            Be very specific and technical.
            """
            
            analysis = await self.gemini_client.analyze_image(page_image, prompt)
            
            pattern_detail["examples"].append({
                "page": page_num,
                "analysis": analysis
            })
        
        return pattern_detail
    
    def _build_relationship_graph(self, cross_references: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a graph of relationships between document elements."""
        graph = {
            "nodes": {},
            "edges": []
        }
        
        for ref in cross_references:
            # Add nodes
            source = ref.get("reference", "")
            target = ref.get("target", "")
            
            if source:
                graph["nodes"][source] = {"type": "reference"}
            if target:
                graph["nodes"][target] = {"type": "target"}
            
            # Add edge
            if source and target:
                graph["edges"].append({
                    "from": source,
                    "to": target,
                    "type": ref.get("type", "references")
                })
        
        return graph
    
    async def _decode_nomenclature(self, nomenclature_system: Dict[str, Any]) -> Dict[str, Any]:
        """Decode the meaning of the nomenclature system."""
        decoded = {
            "patterns": nomenclature_system.get("patterns", {}),
            "meanings": {},
            "hierarchy": {}
        }
        
        # Decode each pattern
        for pattern_name, pattern_info in nomenclature_system.get("patterns", {}).items():
            decoded["meanings"][pattern_name] = {
                "format": pattern_info.get("format", ""),
                "examples": pattern_info.get("examples", []),
                "interpretation": pattern_info.get("meaning", "")
            }
        
        return decoded
    
    async def _understand_visual_language(self, discovered_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Understand the visual language used in the document."""
        visual_language = {
            "symbols": {},
            "line_styles": {},
            "color_meanings": {},
            "conventions": []
        }
        
        # Extract visual elements from patterns
        if "visual_elements" in discovered_patterns:
            for element in discovered_patterns["visual_elements"]:
                element_type = element.get("type", "unknown")
                visual_language["symbols"][element_type] = element.get("meaning", "")
        
        return visual_language
    
    def _initialize_smart_cache(self):
        """
        PRE-CACHE critical pages and metadata for instant access.
        This dramatically speeds up the sampling and analysis process.
        """
        logger.info(f"🚀 Initializing smart cache for {self.total_pages} pages...")
        
        start_time = time.time()
        
        # Pre-cache essential pages (first, last, middle)
        critical_pages = [0, self.total_pages - 1]
        if self.total_pages > 2:
            critical_pages.append(self.total_pages // 2)
        
        # Pre-cache text and complexity for critical pages
        for page_num in critical_pages:
            self._extract_page_text(page_num)  # This will cache automatically
            self._calculate_visual_complexity(page_num)  # This will cache automatically
        
        # Pre-calculate complexity for complexity-based sampling
        # Do this in batches to avoid memory issues
        batch_size = min(20, self.total_pages)
        for i in range(0, min(batch_size, self.total_pages)):
            self._calculate_visual_complexity(i)
        
        cache_time = time.time() - start_time
        logger.info(f"✅ Smart cache initialized in {cache_time:.2f}s - {len(self.page_cache)} pages cached")
    
    async def _analyze_pages_parallel(self, sample_pages: List[int], exploration_prompt: str) -> DiscoveryResult:
        """
        Analyze pages in parallel when PDF URI is not available.
        Uses asyncio with rate limiting to respect Gemini API limits.
        """
        import asyncio
        from asyncio import Semaphore
        
        logger.info(f"🔄 Starting PARALLEL analysis of {len(sample_pages)} pages with intelligent batching")
        
        # INTELLIGENT BATCHING: Group pages into batches for efficiency
        batch_size = 3  # Process 3 pages per API call for optimal balance
        batches = [sample_pages[i:i + batch_size] for i in range(0, len(sample_pages), batch_size)]
        
        logger.info(f"📦 Created {len(batches)} batches of ~{batch_size} pages each")
        
        # Rate limiting: max 2 concurrent batch requests
        semaphore = Semaphore(2)
        
        async def analyze_batch(batch_pages: List[int]) -> List[Dict[str, Any]]:
            async with semaphore:
                try:
                    batch_info = f"pages {[p+1 for p in batch_pages]}"
                    logger.info(f"📦 Analyzing batch: {batch_info}")
                    
                    batch_discoveries = []
                    
                    # Process each page in the batch (using cached data)
                    for page_num in batch_pages:
                        # Use cached data for speed
                        page_text = self._extract_page_text(page_num)
                        complexity = self._calculate_visual_complexity(page_num)
                        
                        # Simplified analysis for parallel processing
                        discovery = await asyncio.get_event_loop().run_in_executor(
                            None, 
                            self._analyze_page_discovery,
                            None, page_text, exploration_prompt, None
                        )
                        
                        batch_discoveries.append({
                            'page': page_num,
                            'findings': discovery,
                            'text': page_text,
                            'visual_complexity': complexity
                        })
                    
                    logger.info(f"✅ Batch {batch_info} completed")
                    return batch_discoveries
                    
                except Exception as e:
                    logger.error(f"Error analyzing batch {batch_pages}: {e}")
                    # Return default discoveries for failed batch
                    return [{
                        'page': page_num,
                        'findings': self._get_default_discovery(),
                        'text': "",
                        'visual_complexity': 0.0
                    } for page_num in batch_pages]
        
        # Execute all batches in parallel
        start_time = time.time()
        batch_results = await asyncio.gather(*[
            analyze_batch(batch) for batch in batches
        ])
        
        # Flatten batch results into discoveries
        discoveries = []
        for batch_result in batch_results:
            discoveries.extend(batch_result)
        
        parallel_time = time.time() - start_time
        
        logger.info(f"✅ Parallel analysis completed in {parallel_time:.1f}s for {len(sample_pages)} pages")
        
        # Synthesize discoveries
        result = await self._synthesize_discoveries(discoveries)
        return result
    
    def _analyze_batch_discovery(self, pdf_uri: str, prompt: str, sample_pages: List[int]) -> Dict[str, Any]:
        """
        Analyze multiple pages in a single API call for efficiency.
        
        This reduces API calls from N (one per page) to 1 (batch analysis).
        """
        try:
            # Add page focus instruction
            pages_info = ", ".join([str(p+1) for p in sample_pages])
            enhanced_prompt = f"{prompt}\n\nFocus particularly on pages: {pages_info}"
            
            response = self.gemini_client.generate_content(
                file_uri=pdf_uri,
                prompt=enhanced_prompt,
                response_schema={
                    "type": "object",
                    "properties": {
                        "document_indicators": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "domain": {"type": "string"},
                                "confidence": {"type": "number"}
                            }
                        },
                        "organization": {
                            "type": "object",
                            "properties": {
                                "structure": {"type": "string"},
                                "sections": {"type": "array", "items": {"type": "string"}},
                                "hierarchy": {"type": "string"}
                            }
                        },
                        "nomenclature": {
                            "type": "object",
                            "properties": {
                                "codes_found": {"type": "array", "items": {"type": "string"}},
                                "naming_pattern": {"type": "string"},
                                "meanings": {"type": "string"}
                            }
                        },
                        "visual_patterns": {
                            "type": "object",
                            "properties": {
                                "drawing_types": {"type": "array", "items": {"type": "string"}},
                                "symbols": {"type": "array", "items": {"type": "string"}},
                                "conventions": {"type": "array", "items": {"type": "string"}}
                            }
                        },
                        "relationships": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string"},
                                    "reference": {"type": "string"},
                                    "target": {"type": "string"}
                                }
                            }
                        },
                        "unique_elements": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                }
            )
            
            # Parse response
            import json
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                logger.warning("Failed to parse batch discovery response")
                return self._get_default_discovery()
                
        except Exception as e:
            logger.error(f"Batch discovery analysis failed: {e}")
            return self._get_default_discovery()
    
    async def _process_batch_discovery(self, discovery: Dict[str, Any], sample_pages: List[int]) -> DiscoveryResult:
        """Process batch discovery results into DiscoveryResult format."""
        
        # Extract nomenclature codes for parsing
        nomenclature_codes = discovery.get("nomenclature", {}).get("codes_found", [])
        
        # Parse nomenclature system (async)
        nomenclature_system = await self.nomenclature_parser.parse_nomenclature(nomenclature_codes)
        
        # Extract visual patterns for analysis
        visual_patterns = discovery.get("visual_patterns", {}).get("drawing_types", [])
        pattern_analysis = self.pattern_analyzer.analyze_patterns(visual_patterns)
        
        # Build result
        result = DiscoveryResult(
            document_type=discovery.get("document_indicators", {}).get("type", "unknown"),
            industry_domain=discovery.get("document_indicators", {}).get("domain", "unknown"),
            discovered_patterns=pattern_analysis,
            nomenclature_system=nomenclature_system,
            page_organization=discovery.get("organization", {}),
            cross_references=discovery.get("relationships", []),
            element_types=discovery.get("unique_elements", []),
            confidence_score=discovery.get("document_indicators", {}).get("confidence", 0.5),
            discovery_metadata={
                "pages_analyzed": len(sample_pages),
                "total_pages": self.total_pages,
                "unique_patterns": len(set(visual_patterns)),
                "nomenclature_codes": len(set(nomenclature_codes)),
                "batch_analysis": True  # Flag to indicate this was batch processed
            }
        )
        
        return result
    
    def _get_default_discovery(self) -> Dict[str, Any]:
        """Return default discovery structure when analysis fails."""
        return {
            "document_indicators": {"type": "unknown", "domain": "unknown", "confidence": 0.0},
            "organization": {"structure": "unknown", "sections": [], "hierarchy": "flat"},
            "nomenclature": {"codes_found": [], "naming_pattern": "", "meanings": ""},
            "visual_patterns": {"drawing_types": [], "symbols": [], "conventions": []},
            "relationships": [],
            "unique_elements": []
        }
    
    def close(self):
        """Close the PDF document."""
        try:
            if hasattr(self, 'pdf_document') and self.pdf_document is not None:
                # Check if document is not already closed
                try:
                    _ = len(self.pdf_document)  # This will raise if closed
                    self.pdf_document.close()
                    logger.info("PDF document closed")
                except:
                    pass  # Already closed
                finally:
                    self.pdf_document = None
        except Exception as e:
            logger.debug(f"Error during PDF cleanup: {e}")
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during cleanup
