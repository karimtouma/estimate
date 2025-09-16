"""
File management utilities with robust error handling.

This module provides utilities for managing files, directories, and
data persistence with proper error handling and validation.
"""

import gzip
import shutil

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..core.config import Config

logger = logging.getLogger(__name__)


class FileManagerError(Exception):
    """Base exception for file manager errors."""

    pass


class FileManager:
    """
    File management utility with enterprise-grade features.

    Handles file operations, directory management, compression,
    and backup functionality with proper error handling.
    """

    def __init__(self, config: Config):
        """
        Initialize file manager.

        Args:
            config: Application configuration
        """
        self.config = config
        self.directories = config.get_directories()

        # Ensure directories exist
        self._ensure_directories()

        logger.debug("FileManager initialized")

    def _ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        for name, path in self.directories.items():
            try:
                Path(path).mkdir(parents=True, exist_ok=True)
                logger.debug(f"Directory ensured: {name} -> {path}")
            except Exception as e:
                logger.warning(f"Failed to create directory {name} ({path}): {e}")

    def save_results(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path],
        create_backup: Optional[bool] = None,
    ) -> Path:
        """
        Save analysis results to file with optional compression and backup.

        Args:
            results: Results dictionary to save
            output_path: Output file path
            create_backup: Whether to create backup (uses config default if None)

        Returns:
            Path to the saved file

        Raises:
            FileManagerError: If saving fails
        """
        output_path = Path(output_path)

        # Resolve relative paths to output directory
        if not output_path.is_absolute():
            output_path = Path(self.directories["output"]) / output_path

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Add metadata if configured
        if self.config.output.include_metadata:
            results = self._add_metadata(results)

        try:
            # Create backup if requested
            create_backup = (
                create_backup if create_backup is not None else self.config.output.create_backup
            )
            if create_backup and output_path.exists():
                self._create_backup(output_path)

            # Save file
            self._save_json_file(results, output_path)

            # Compress if file is large
            if self._should_compress(output_path):
                compressed_path = self._compress_file(output_path)
                logger.info(f"File compressed: {compressed_path}")

            logger.info(f"Results saved successfully: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to save results to {output_path}: {e}")
            raise FileManagerError(f"Failed to save results: {e}") from e

    def _add_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Add metadata to results."""
        metadata = {
            "timestamp": time.time(),
            "processor_version": "2.0.0",
            "model_used": self.config.api.default_model,
            "config_file": str(self.config.config_path),
            "environment": "container" if self.config.is_container_environment() else "local",
        }

        # Create a copy to avoid modifying original
        results_with_metadata = results.copy()
        results_with_metadata["_metadata"] = metadata

        return results_with_metadata

    def _save_json_file(self, data: Dict[str, Any], file_path: Path) -> None:
        """Save data as JSON file."""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            raise FileManagerError(f"Failed to write JSON file: {e}") from e

    def _create_backup(self, file_path: Path) -> Path:
        """Create backup of existing file."""
        timestamp = int(time.time())
        backup_path = file_path.with_suffix(f".backup_{timestamp}{file_path.suffix}")

        try:
            shutil.copy2(file_path, backup_path)
            logger.info(f"Backup created: {backup_path}")
            return backup_path
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")
            # Don't fail the main operation for backup failure
            return file_path

    def _should_compress(self, file_path: Path) -> bool:
        """Check if file should be compressed."""
        if not self.config.output.compress_large_files:
            return False

        try:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            threshold_mb = self.config.output.compression_threshold_mb
            return file_size_mb > threshold_mb
        except Exception:
            return False

    def _compress_file(self, file_path: Path) -> Path:
        """Compress file using gzip."""
        compressed_path = file_path.with_suffix(f"{file_path.suffix}.gz")

        try:
            with open(file_path, "rb") as f_in:
                with gzip.open(compressed_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Remove original file after successful compression
            file_path.unlink()

            return compressed_path
        except Exception as e:
            # Clean up partial compressed file
            if compressed_path.exists():
                compressed_path.unlink()
            raise FileManagerError(f"Compression failed: {e}") from e

    def load_results(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load results from file with automatic decompression.

        Args:
            file_path: Path to results file

        Returns:
            Loaded results dictionary

        Raises:
            FileManagerError: If loading fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileManagerError(f"File not found: {file_path}")

        try:
            # Handle compressed files
            if file_path.suffix == ".gz":
                with gzip.open(file_path, "rt", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

            logger.debug(f"Results loaded from: {file_path}")
            return data

        except Exception as e:
            logger.error(f"Failed to load results from {file_path}: {e}")
            raise FileManagerError(f"Failed to load results: {e}") from e

    def list_results(self, pattern: str = "*.json*") -> list[Path]:
        """
        List result files in output directory.

        Args:
            pattern: File pattern to match

        Returns:
            List of result file paths
        """
        output_dir = Path(self.directories["output"])

        try:
            files = list(output_dir.glob(pattern))
            files.sort(key=lambda p: p.stat().st_mtime, reverse=True)  # Most recent first
            return files
        except Exception as e:
            logger.error(f"Failed to list results: {e}")
            return []

    def cleanup_old_files(self, max_age_days: int = 30) -> int:
        """
        Clean up old files from output directory.

        Args:
            max_age_days: Maximum age of files to keep

        Returns:
            Number of files cleaned up
        """
        if max_age_days <= 0:
            logger.warning("Invalid max_age_days, skipping cleanup")
            return 0

        output_dir = Path(self.directories["output"])
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60

        cleaned_count = 0

        try:
            for file_path in output_dir.glob("*"):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime

                    if file_age > max_age_seconds:
                        file_path.unlink()
                        logger.debug(f"Cleaned up old file: {file_path}")
                        cleaned_count += 1

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old files")

            return cleaned_count

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return cleaned_count

    def get_directory_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about managed directories.

        Returns:
            Dictionary with directory information
        """
        info = {}

        for name, path in self.directories.items():
            path_obj = Path(path)

            try:
                if path_obj.exists():
                    files = list(path_obj.glob("*"))
                    total_size = sum(f.stat().st_size for f in files if f.is_file())

                    info[name] = {
                        "path": str(path_obj),
                        "exists": True,
                        "is_writable": os.access(path_obj, os.W_OK),
                        "file_count": len([f for f in files if f.is_file()]),
                        "total_size_mb": total_size / (1024 * 1024),
                        "free_space_mb": shutil.disk_usage(path_obj).free / (1024 * 1024),
                    }
                else:
                    info[name] = {
                        "path": str(path_obj),
                        "exists": False,
                        "is_writable": False,
                        "file_count": 0,
                        "total_size_mb": 0,
                        "free_space_mb": 0,
                    }

            except Exception as e:
                info[name] = {"path": str(path_obj), "error": str(e)}

        return info

    def validate_permissions(self) -> Dict[str, bool]:
        """
        Validate read/write permissions for all directories.

        Returns:
            Dictionary with permission status for each directory
        """
        import os

        permissions = {}

        for name, path in self.directories.items():
            path_obj = Path(path)

            try:
                # Test read permission
                can_read = os.access(path_obj, os.R_OK) if path_obj.exists() else False

                # Test write permission
                can_write = os.access(path_obj, os.W_OK) if path_obj.exists() else False

                # Try to create directory if it doesn't exist
                if not path_obj.exists():
                    try:
                        path_obj.mkdir(parents=True, exist_ok=True)
                        can_write = True
                    except Exception:
                        can_write = False

                permissions[name] = {
                    "readable": can_read,
                    "writable": can_write,
                    "valid": can_read and can_write,
                }

            except Exception as e:
                permissions[name] = {
                    "readable": False,
                    "writable": False,
                    "valid": False,
                    "error": str(e),
                }

        return permissions
