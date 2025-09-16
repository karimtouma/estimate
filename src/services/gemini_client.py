"""
Google Gemini API client with robust error handling and retry logic.

This module provides a clean interface to the Google Gemini API with
proper error handling, rate limiting, and retry mechanisms.
"""

import logging
import time
from google import genai
from google.genai import types
from pathlib import Path
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from typing import Any, Dict, List, Optional

from ..core.config import Config
from ..models.schemas import AnalysisType

logger = logging.getLogger(__name__)


class GeminiAPIError(Exception):
    """Custom exception for Gemini API errors."""

    pass


class FileUploadError(GeminiAPIError):
    """Raised when file upload fails."""

    pass


class ProcessingTimeoutError(GeminiAPIError):
    """Raised when file processing times out."""

    pass


class GeminiClient:
    """
    Google Gemini API client with enterprise-grade error handling.

    Provides a clean, robust interface to Google's Gemini API with
    automatic retries, rate limiting, and comprehensive error handling.
    """

    # Class-level statistics tracking
    _api_stats = {
        "total_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "calls_by_type": {},
        "processing_times": [],
        "errors": 0,
    }

    def __init__(self, config: Config):
        """
        Initialize the Gemini client.

        Args:
            config: Application configuration instance

        Raises:
            GeminiAPIError: If API key is not configured
        """
        self.config = config

        if not config.api.gemini_api_key:
            raise GeminiAPIError("Gemini API key not configured")

        self.client = genai.Client(api_key=config.api.gemini_api_key)
        self.uploaded_files: Dict[str, Any] = {}

        # PARALLEL PROCESSING: Global rate limiting for concurrent requests
        import threading

        if not hasattr(GeminiClient, "_global_semaphore"):
            # Limit concurrent API calls globally across all instances
            max_concurrent = getattr(config.api, "max_concurrent_requests", 3)
            GeminiClient._global_semaphore = threading.Semaphore(max_concurrent)
            logger.info(
                f"🚀 Global rate limiting initialized: {max_concurrent} concurrent requests"
            )

        self.global_semaphore = GeminiClient._global_semaphore

        logger.info(f"Gemini client initialized with model: {config.api.default_model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((FileUploadError, ConnectionError)),
    )
    def upload_file(self, file_path: Path, display_name: Optional[str] = None) -> str:
        """
        Upload a file to Google GenAI with retry logic.

        Args:
            file_path: Path to the file to upload
            display_name: Optional display name for the file

        Returns:
            URI of the uploaded file

        Raises:
            FileUploadError: If upload fails after retries
            ProcessingTimeoutError: If file processing times out
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Validate file size
        file_size = file_path.stat().st_size
        max_size = self.config.processing.max_pdf_size_mb * 1024 * 1024

        if file_size > max_size:
            raise FileUploadError(
                f"File too large: {file_size / (1024*1024):.1f}MB. "
                f"Maximum: {self.config.processing.max_pdf_size_mb}MB"
            )

        # Validate file type
        if self.config.security.validate_file_types:
            if file_path.suffix.lower() not in self.config.security.allowed_extensions:
                raise FileUploadError(
                    f"File type not allowed: {file_path.suffix}. "
                    f"Allowed: {self.config.security.allowed_extensions}"
                )

        display_name = display_name or file_path.stem

        logger.info(f"Uploading file: {file_path} ({file_size / (1024*1024):.1f}MB)")

        try:
            # Upload file using the correct API
            with open(file_path, "rb") as file:
                uploaded_file = self.client.files.upload(
                    file=file,
                    config=types.UploadFileConfig(
                        mime_type="application/pdf", display_name=display_name
                    ),
                )

            # Wait for processing
            self._wait_for_processing(uploaded_file.name)

            # Cache the uploaded file
            self.uploaded_files[display_name] = uploaded_file

            logger.info(f"File uploaded successfully: {uploaded_file.name}")
            return uploaded_file.uri

        except Exception as e:
            logger.error(f"File upload failed: {e}")
            raise FileUploadError(f"Failed to upload file: {e}") from e

    def _wait_for_processing(self, file_name: str, max_wait: Optional[int] = None) -> None:
        """
        Wait for file processing to complete.

        Args:
            file_name: Name of the uploaded file
            max_wait: Maximum wait time in seconds

        Raises:
            ProcessingTimeoutError: If processing times out
        """
        max_wait = max_wait or self.config.api.request_timeout
        start_time = time.time()

        logger.info(f"Waiting for file processing: {file_name}")

        while time.time() - start_time < max_wait:
            try:
                file_info = self.client.files.get(name=file_name)

                if file_info.state == "ACTIVE":
                    logger.info(f"File processing completed: {file_name}")
                    return
                elif file_info.state == "FAILED":
                    raise ProcessingTimeoutError(f"File processing failed: {file_name}")

                logger.debug(f"File state: {file_info.state}. Waiting...")
                time.sleep(2)

            except Exception as e:
                logger.warning(f"Error checking file state: {e}")
                time.sleep(2)

        raise ProcessingTimeoutError(f"File processing timeout after {max_wait}s: {file_name}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    )
    def generate_content(
        self,
        file_uri: str,
        prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        Generate content using Gemini API with structured response.

        Args:
            file_uri: URI of the uploaded file
            prompt: Text prompt for analysis
            response_schema: Optional JSON schema for structured response
            model: Optional model override

        Returns:
            Generated content as text

        Raises:
            GeminiAPIError: If content generation fails
        """
        model = model or self.config.api.default_model

        logger.debug(f"Generating content with model: {model}")

        try:
            # Prepare content parts
            content_parts = [
                types.Part.from_uri(file_uri=file_uri, mime_type="application/pdf"),
                types.Part.from_text(text=prompt),
            ]

            # Configure generation
            generation_config = types.GenerateContentConfig()

            if response_schema:
                generation_config.response_mime_type = "application/json"
                generation_config.response_schema = response_schema

            # PARALLEL PROCESSING: Use global semaphore for intelligent rate limiting
            with self.global_semaphore:
                # Track API call start time
                start_time = time.time()

                # Add safety settings to prevent hallucinations
                generation_config.temperature = (
                    0.3  # Lower temperature for more deterministic output
                )
                generation_config.top_p = 0.8  # Reduce randomness
                generation_config.top_k = 40  # Limit token selection
                generation_config.max_output_tokens = 8192  # Reasonable limit

                # Generate content
                response = self.client.models.generate_content(
                    model=model,
                    contents=[types.Content(role="user", parts=content_parts)],
                    config=generation_config,
                )

                # Track statistics
                processing_time = time.time() - start_time

                # Log response structure for debugging
                logger.debug(f"Response type: {type(response)}")
                if hasattr(response, "__dict__"):
                    logger.debug(f"Response attributes: {list(response.__dict__.keys())}")

                self._track_api_call(response, "generate_content", processing_time)

                if not response.text:
                    raise GeminiAPIError("Empty response from Gemini API")

                logger.debug(f"Content generated successfully ({len(response.text)} chars)")
                return response.text

        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            raise GeminiAPIError(f"Failed to generate content: {e}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    )
    def generate_text_only_content(
        self,
        prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        Generate content using Gemini API with text-only input (no file).

        Args:
            prompt: Text prompt for analysis
            response_schema: Optional JSON schema for structured response
            model: Optional model override

        Returns:
            Generated content as text

        Raises:
            GeminiAPIError: If content generation fails
        """
        model = model or self.config.api.default_model

        logger.debug(f"Generating text-only content with model: {model}")

        try:
            # Prepare content parts (text only)
            content_parts = [types.Part.from_text(text=prompt)]

            # Configure generation
            generation_config = types.GenerateContentConfig()

            if response_schema:
                generation_config.response_mime_type = "application/json"
                generation_config.response_schema = response_schema

            # PARALLEL PROCESSING: Use global semaphore for intelligent rate limiting
            with self.global_semaphore:
                # Track API call start time
                start_time = time.time()

                # Add safety settings to prevent hallucinations
                generation_config.temperature = (
                    0.3  # Lower temperature for more deterministic output
                )
                generation_config.top_p = 0.8  # Reduce randomness
                generation_config.top_k = 40  # Limit token selection
                generation_config.max_output_tokens = 8192  # Reasonable limit

                # Generate content
                response = self.client.models.generate_content(
                    model=model,
                    contents=[types.Content(role="user", parts=content_parts)],
                    config=generation_config,
                )

                # Track statistics
                processing_time = time.time() - start_time

                # Log response structure for debugging
                logger.debug(f"Response type: {type(response)}")
                if hasattr(response, "__dict__"):
                    logger.debug(f"Response attributes: {list(response.__dict__.keys())}")

                self._track_api_call(response, "generate_text_only_content", processing_time)

                if not response.text:
                    raise GeminiAPIError("Empty response from Gemini API")

                logger.debug(
                    f"Text-only content generated successfully ({len(response.text)} chars)"
                )
                return response.text

        except Exception as e:
            logger.error(f"Text-only content generation failed: {e}")
            raise GeminiAPIError(f"Failed to generate text-only content: {e}") from e

    def generate_multi_turn_content(
        self, file_uri: str, questions: List[str], context_parts: Optional[List[types.Part]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate multi-turn conversation content.

        OPTIMIZED VERSION: Processes all questions in a single API call.
        This reduces API calls from N (one per question) to 1.

        Args:
            file_uri: URI of the uploaded file
            questions: List of questions to ask
            context_parts: Optional context from previous interactions

        Returns:
            List of question-answer pairs with metadata

        Raises:
            GeminiAPIError: If multi-turn generation fails
        """
        if not questions:
            return []

        logger.info(f"Processing {len(questions)} questions in batch...")

        try:
            # Build batch prompt for all questions
            enhanced_instruction = self._get_enhanced_question_instruction()

            # Create numbered question list
            questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])

            batch_prompt = f"""
            {enhanced_instruction}
            
            Please answer ALL of the following questions about this document. 
            For each question, provide a complete analysis with confidence score and sources.
            
            Questions to answer:
            {questions_text}
            
            Respond with detailed answers for each question in the order given.
            """

            # Define batch response schema
            response_schema = {
                "type": "object",
                "properties": {
                    "answers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {"type": "string"},
                                "answer": {"type": "string"},
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                "sources": {"type": "array", "items": {"type": "string"}},
                                "follow_up_questions": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["question", "answer", "confidence"],
                        },
                    }
                },
                "required": ["answers"],
            }

            # Generate batch response
            response_text = self.generate_content(
                file_uri=file_uri, prompt=batch_prompt, response_schema=response_schema
            )

            # Parse batch result
            import json

            batch_result = json.loads(response_text)

            results = []
            answers = batch_result.get("answers", [])

            # Process each answer and ensure we have all questions covered
            for i, question in enumerate(questions):
                if i < len(answers):
                    result = answers[i].copy()
                    result["question_index"] = i + 1
                    # Ensure the question matches (in case of reordering)
                    result["question"] = question
                    results.append(result)
                else:
                    # Fallback for missing answers
                    logger.warning(f"Missing answer for question {i+1}: {question}")
                    results.append(
                        {
                            "question": question,
                            "question_index": i + 1,
                            "answer": "Answer not provided in batch response.",
                            "confidence": 0.0,
                            "sources": [],
                            "follow_up_questions": [],
                        }
                    )

            logger.info(f"Batch processing completed: {len(results)} answers generated")
            return results

        except Exception as e:
            logger.error(f"Batch multi-turn processing failed: {e}")
            # Fallback to sequential processing for critical failures
            logger.info("Falling back to sequential question processing...")
            return self._generate_multi_turn_sequential(file_uri, questions, context_parts)

    def _generate_multi_turn_sequential(
        self, file_uri: str, questions: List[str], context_parts: Optional[List[types.Part]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fallback method for sequential question processing.
        Used when batch processing fails.
        """
        results = []

        for i, question in enumerate(questions):
            logger.debug(f"Processing question {i+1}/{len(questions)}: {question}")

            try:
                # Enhanced instruction
                enhanced_instruction = self._get_enhanced_question_instruction()

                # Define response schema
                response_schema = {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "sources": {"type": "array", "items": {"type": "string"}},
                        "follow_up_questions": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["question", "answer", "confidence"],
                }

                # Generate response
                response_text = self.generate_content(
                    file_uri=file_uri,
                    prompt=f"{enhanced_instruction}Question: {question}",
                    response_schema=response_schema,
                )

                # Parse and enhance result
                import json

                result = json.loads(response_text)
                result["question_index"] = i + 1
                results.append(result)

                # Add small delay to avoid rate limiting
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error processing question '{question}': {e}")
                results.append(
                    {
                        "question": question,
                        "question_index": i + 1,
                        "error": str(e),
                        "confidence": 0.0,
                    }
                )

        return results

    def _get_enhanced_question_instruction(self) -> str:
        """
        Get enhanced instruction for Q&A analysis with GEPA optimization.

        Returns:
            Enhanced instruction string
        """
        base_instruction = "IMPORTANT: Always respond in English. Analyze technical drawings, blueprints, and construction documents. "

        try:
            # Try to load GEPA-optimized Q&A instruction
            import json
            from pathlib import Path

            # Check if GEPA optimization is enabled
            enable_gepa = getattr(self.config.analysis, "enable_dspy_optimization", False)
            if not enable_gepa:
                return base_instruction

            # Load GEPA results if available
            output_dir = Path(self.config.get_directories()["output"])
            gepa_results_file = output_dir / "gepa_optimization_results.json"

            if gepa_results_file.exists():
                with open(gepa_results_file, "r", encoding="utf-8") as f:
                    gepa_data = json.load(f)

                # Get optimized Q&A instruction
                optimization_results = gepa_data.get("gepa_optimization_report", {})
                optimized_prompts = optimization_results.get("optimized_prompts", {})

                qa_instruction = optimized_prompts.get("qa_analysis_instruction")
                if qa_instruction:
                    logger.debug("Using GEPA-optimized Q&A instruction")
                    return f"{base_instruction}{qa_instruction} "

            return base_instruction

        except Exception as e:
            logger.warning(f"Failed to load GEPA Q&A optimization: {e}")
            return base_instruction

    def get_file_info(self, file_name: str) -> Dict[str, Any]:
        """
        Get information about an uploaded file.

        Args:
            file_name: Name of the uploaded file

        Returns:
            File information dictionary

        Raises:
            GeminiAPIError: If file info retrieval fails
        """
        try:
            file_info = self.client.files.get(name=file_name)
            return {
                "name": file_info.name,
                "display_name": file_info.display_name,
                "mime_type": file_info.mime_type,
                "size_bytes": getattr(file_info, "size_bytes", 0),
                "state": file_info.state,
                "uri": file_info.uri,
            }
        except Exception as e:
            raise GeminiAPIError(f"Failed to get file info: {e}") from e

    def _track_api_call(self, response: Any, call_type: str, processing_time: float) -> None:
        """
        Track API call statistics including token usage.

        Args:
            response: The API response object
            call_type: Type of API call (e.g., 'generate_content', 'batch_qa')
            processing_time: Time taken for the API call in seconds
        """
        try:
            # Update call counts
            GeminiClient._api_stats["total_calls"] += 1
            GeminiClient._api_stats["processing_times"].append(processing_time)

            # Track by call type
            if call_type not in GeminiClient._api_stats["calls_by_type"]:
                GeminiClient._api_stats["calls_by_type"][call_type] = 0
            GeminiClient._api_stats["calls_by_type"][call_type] += 1

            # Extract token usage from response if available
            tokens_found = False
            input_tokens = 0
            output_tokens = 0
            cached_tokens = 0

            # Method 1: Direct usage_metadata attribute
            if hasattr(response, "usage_metadata"):
                usage = response.usage_metadata
                if usage:
                    input_tokens = getattr(usage, "prompt_token_count", 0)
                    output_tokens = getattr(usage, "candidates_token_count", 0)
                    cached_tokens = getattr(usage, "cached_content_token_count", 0)
                    tokens_found = input_tokens > 0 or output_tokens > 0

            # Method 2: Through _result attribute
            if not tokens_found and hasattr(response, "_result"):
                result = response._result
                if hasattr(result, "usage_metadata"):
                    usage = result.usage_metadata
                    if usage:
                        input_tokens = getattr(usage, "prompt_token_count", 0)
                        output_tokens = getattr(usage, "candidates_token_count", 0)
                        cached_tokens = getattr(usage, "cached_content_token_count", 0)
                        tokens_found = input_tokens > 0 or output_tokens > 0

            # Method 3: Through candidates attribute
            if not tokens_found and hasattr(response, "candidates"):
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if hasattr(candidate, "token_count"):
                        output_tokens = candidate.token_count
                        tokens_found = True

            # Update statistics if tokens found
            if tokens_found:
                GeminiClient._api_stats["input_tokens"] += input_tokens
                GeminiClient._api_stats["output_tokens"] += output_tokens
                GeminiClient._api_stats["cached_input_tokens"] += cached_tokens

                # Log token usage for this call
                logger.info(
                    f"📊 API Call [{call_type}]: "
                    f"Input={input_tokens:,} tokens, "
                    f"Output={output_tokens:,} tokens, "
                    f"Cached={cached_tokens:,} tokens, "
                    f"Time={processing_time:.2f}s"
                )
            else:
                # Log API call even without token info
                logger.info(
                    f"📊 API Call [{call_type}]: Time={processing_time:.2f}s (token info not available)"
                )

        except Exception as e:
            logger.debug(f"Could not track API statistics: {e}")

    @classmethod
    def get_api_statistics(cls) -> Dict[str, Any]:
        """
        Get comprehensive API usage statistics.

        Returns:
            Dictionary with detailed API usage metrics
        """
        stats = cls._api_stats.copy()

        # Calculate averages
        if stats["processing_times"]:
            stats["avg_processing_time"] = sum(stats["processing_times"]) / len(
                stats["processing_times"]
            )
            stats["total_processing_time"] = sum(stats["processing_times"])
        else:
            stats["avg_processing_time"] = 0
            stats["total_processing_time"] = 0

        # Calculate token costs (approximate)
        # Gemini pricing: ~$0.00025 per 1K input tokens, ~$0.001 per 1K output tokens
        stats["estimated_cost"] = {
            "input_cost": (stats["input_tokens"] / 1000) * 0.00025,
            "output_cost": (stats["output_tokens"] / 1000) * 0.001,
            "total_cost": ((stats["input_tokens"] / 1000) * 0.00025)
            + ((stats["output_tokens"] / 1000) * 0.001),
        }

        # Token efficiency
        stats["total_tokens"] = stats["input_tokens"] + stats["output_tokens"]
        stats["cache_efficiency"] = (
            (stats["cached_input_tokens"] / stats["input_tokens"] * 100)
            if stats["input_tokens"] > 0
            else 0
        )

        return stats

    @classmethod
    def reset_statistics(cls) -> None:
        """Reset API statistics tracking."""
        cls._api_stats = {
            "total_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "calls_by_type": {},
            "processing_times": [],
            "errors": 0,
        }

    def cleanup_uploaded_files(self) -> None:
        """Clean up uploaded files from the API."""
        for display_name, file_obj in self.uploaded_files.items():
            try:
                self.client.files.delete(name=file_obj.name)
                logger.info(f"Deleted uploaded file: {display_name}")
            except Exception as e:
                logger.warning(f"Failed to delete file {display_name}: {e}")

        self.uploaded_files.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup_uploaded_files()
