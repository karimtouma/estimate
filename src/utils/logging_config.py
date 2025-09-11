"""
Logging configuration with structured logging and proper formatting.

This module provides centralized logging configuration with support for
both development and production environments.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
import json
import time

from ..core.config import Config


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': time.time(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, default=str)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Get color for level
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format message
        formatted = super().format(record)
        
        # Add color
        return f"{color}{formatted}{reset}"


def setup_logging(config: Config) -> None:
    """
    Setup logging configuration based on application config.
    
    Args:
        config: Application configuration instance
    """
    # Get log level
    log_level = getattr(logging, config.processing.log_level.upper(), logging.INFO)
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set root level
    root_logger.setLevel(log_level)
    
    # Setup formatters
    if config.is_container_environment():
        # Structured logging for containers
        formatter = JSONFormatter()
        console_formatter = JSONFormatter()
    else:
        # Human-readable logging for development
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = ColoredFormatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%H:%M:%S'
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for containers or if logs directory exists
    directories = config.get_directories()
    logs_dir = Path(directories['logs'])
    
    if config.is_container_environment() or logs_dir.exists():
        try:
            # Ensure logs directory exists
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            # Rotating file handler
            log_file = logs_dir / 'pdf_processor.log'
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            
            # Error file handler
            error_file = logs_dir / 'pdf_processor_errors.log'
            error_handler = logging.handlers.RotatingFileHandler(
                error_file,
                maxBytes=5 * 1024 * 1024,  # 5MB
                backupCount=3,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            root_logger.addHandler(error_handler)
            
        except Exception as e:
            # Fallback to console only if file logging fails
            console_handler.emit(logging.LogRecord(
                name='logging_config',
                level=logging.WARNING,
                pathname=__file__,
                lineno=0,
                msg=f"Failed to setup file logging: {e}",
                args=(),
                exc_info=None
            ))
    
    # Configure third-party loggers
    _configure_third_party_loggers(log_level)
    
    # Log setup completion
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {config.processing.log_level}, Container: {config.is_container_environment()}")


def _configure_third_party_loggers(log_level: int) -> None:
    """Configure third-party library loggers."""
    # Google libraries
    logging.getLogger('google').setLevel(logging.WARNING)
    logging.getLogger('google.auth').setLevel(logging.WARNING)
    logging.getLogger('google.genai').setLevel(logging.INFO)
    
    # HTTP libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # Other libraries
    logging.getLogger('pydantic').setLevel(logging.WARNING)
    
    # Set our application loggers to the configured level
    logging.getLogger('src').setLevel(log_level)


def get_logger(name: str, extra_fields: Optional[dict] = None) -> logging.Logger:
    """
    Get a logger with optional extra fields.
    
    Args:
        name: Logger name
        extra_fields: Optional extra fields to include in log records
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if extra_fields:
        # Create adapter to add extra fields
        logger = LoggerAdapter(logger, extra_fields)
    
    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds extra fields to log records."""
    
    def process(self, msg, kwargs):
        """Process log record to add extra fields."""
        # Add extra fields to record
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        kwargs['extra']['extra_fields'] = self.extra
        
        return msg, kwargs


def log_performance(func):
    """Decorator to log function performance."""
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            logger.debug(
                f"Function {func.__name__} completed in {execution_time:.3f}s",
                extra={'extra_fields': {
                    'function': func.__name__,
                    'execution_time': execution_time,
                    'status': 'success'
                }}
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            logger.error(
                f"Function {func.__name__} failed after {execution_time:.3f}s: {e}",
                extra={'extra_fields': {
                    'function': func.__name__,
                    'execution_time': execution_time,
                    'status': 'error',
                    'error': str(e)
                }},
                exc_info=True
            )
            
            raise
    
    return wrapper


def log_api_call(endpoint: str, method: str = 'GET'):
    """Decorator to log API calls."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            start_time = time.time()
            
            logger.info(
                f"API call started: {method} {endpoint}",
                extra={'extra_fields': {
                    'endpoint': endpoint,
                    'method': method,
                    'status': 'started'
                }}
            )
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                logger.info(
                    f"API call completed: {method} {endpoint} ({execution_time:.3f}s)",
                    extra={'extra_fields': {
                        'endpoint': endpoint,
                        'method': method,
                        'execution_time': execution_time,
                        'status': 'completed'
                    }}
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                logger.error(
                    f"API call failed: {method} {endpoint} ({execution_time:.3f}s): {e}",
                    extra={'extra_fields': {
                        'endpoint': endpoint,
                        'method': method,
                        'execution_time': execution_time,
                        'status': 'failed',
                        'error': str(e)
                    }},
                    exc_info=True
                )
                
                raise
        
        return wrapper
    return decorator
