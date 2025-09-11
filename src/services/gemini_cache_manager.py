"""
Gemini Cache Manager for optimized file handling and context caching.

This module implements best practices for Gemini API file management:
- Single file upload with caching
- Context caching for repeated queries
- Efficient token usage
"""

import time
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiCacheManager:
    """
    Manages Gemini file uploads and context caching for optimal performance.
    
    Following best practices from:
    - https://ai.google.dev/gemini-api/docs/caching
    - https://ai.google.dev/gemini-api/docs/files
    """
    
    def __init__(self, client: genai.Client, model: str = "models/gemini-2.5-pro"):
        """
        Initialize the cache manager.
        
        Args:
            client: Gemini client instance
            model: Model to use (must support caching)
        """
        self.client = client
        self.model = model
        self.uploaded_files = {}
        self.active_caches = {}
        
        logger.info(f"GeminiCacheManager initialized with model: {model}")
    
    def upload_pdf_once(self, pdf_path: Path, display_name: str = None) -> str:
        """
        Upload a PDF file only once and return its URI.
        
        Args:
            pdf_path: Path to the PDF file
            display_name: Optional display name
            
        Returns:
            URI of the uploaded file
        """
        # Check if already uploaded
        path_str = str(pdf_path)
        if path_str in self.uploaded_files:
            logger.info(f"File already uploaded: {path_str}")
            return self.uploaded_files[path_str].uri
        
        # Upload the file
        logger.info(f"Uploading PDF: {pdf_path}")
        display_name = display_name or pdf_path.stem
        
        with open(pdf_path, 'rb') as file:
            uploaded_file = self.client.files.upload(
                file=file,
                config=types.UploadFileConfig(
                    mime_type='application/pdf',
                    display_name=display_name
                )
            )
        
        # Wait for processing
        while uploaded_file.state.name == 'PROCESSING':
            logger.debug('Waiting for file to be processed...')
            time.sleep(2)
            uploaded_file = self.client.files.get(name=uploaded_file.name)
        
        logger.info(f"File uploaded successfully: {uploaded_file.uri}")
        
        # Store for reuse
        self.uploaded_files[path_str] = uploaded_file
        
        return uploaded_file.uri
    
    def create_cache_for_discovery(
        self, 
        pdf_file_uri: str,
        system_instruction: str = None,
        ttl: str = "600s"  # 10 minutes default
    ) -> str:
        """
        Create a cached context for discovery analysis.
        
        Args:
            pdf_file_uri: URI of the uploaded PDF
            system_instruction: System instructions for analysis
            ttl: Time to live for the cache
            
        Returns:
            Cache name for reuse
        """
        cache_key = f"discovery_{pdf_file_uri}"
        
        # Check if cache already exists
        if cache_key in self.active_caches:
            logger.info(f"Using existing cache: {cache_key}")
            return self.active_caches[cache_key].name
        
        # Default system instruction for discovery
        if not system_instruction:
            system_instruction = """
            You are an expert document analyzer specializing in technical blueprints and construction documents.
            Your task is to discover and understand the document's structure, patterns, and content without 
            making assumptions about predefined categories. Focus on:
            
            1. Document type and purpose
            2. Naming conventions and codes
            3. Visual patterns and symbols
            4. Page organization
            5. Technical elements and systems
            
            Be thorough and adaptive in your analysis.
            """
        
        # Create the cache
        logger.info(f"Creating cache for discovery with TTL: {ttl}")
        
        # Get the file reference
        file_ref = None
        for uploaded_file in self.uploaded_files.values():
            if uploaded_file.uri == pdf_file_uri:
                file_ref = uploaded_file
                break
        
        if not file_ref:
            # If not in our records, create a reference
            file_ref = types.File(uri=pdf_file_uri)
        
        cache = self.client.caches.create(
            model=self.model,
            config=types.CreateCachedContentConfig(
                display_name='document_discovery',
                system_instruction=system_instruction,
                contents=[file_ref],
                ttl=ttl,
            )
        )
        
        logger.info(f"Cache created: {cache.name}")
        self.active_caches[cache_key] = cache
        
        return cache.name
    
    def analyze_with_cache(
        self,
        cache_name: str,
        prompt: str,
        response_schema: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Analyze using cached content for efficiency.
        
        Args:
            cache_name: Name of the cache to use
            prompt: Analysis prompt
            response_schema: Optional structured response schema
            
        Returns:
            Analysis response
        """
        config = types.GenerateContentConfig(
            cached_content=cache_name
        )
        
        if response_schema:
            config.response_schema = response_schema
            config.response_mime_type = "application/json"
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config
        )
        
        # Log token usage for monitoring
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            logger.info(f"Token usage - Cached: {getattr(usage, 'cached_content_token_count', 0)}, "
                       f"Prompt: {getattr(usage, 'prompt_token_count', 0)}, "
                       f"Output: {getattr(usage, 'candidates_token_count', 0)}")
        
        return response.text
    
    def cleanup(self):
        """Clean up uploaded files and caches."""
        # Delete caches
        for cache_key, cache in self.active_caches.items():
            try:
                self.client.caches.delete(cache.name)
                logger.info(f"Deleted cache: {cache_key}")
            except Exception as e:
                logger.warning(f"Failed to delete cache {cache_key}: {e}")
        
        # Clear references
        self.active_caches.clear()
        self.uploaded_files.clear()
        
        logger.info("Cache manager cleanup completed")
