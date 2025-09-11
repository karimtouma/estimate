"""
Configuration management system using TOML and environment variables.

This module provides a clean, type-safe configuration system that supports
both TOML files and environment variable overrides.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging
from dataclasses import dataclass, field

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Fallback for earlier versions
    except ImportError:
        raise ImportError(
            "TOML support required. Install with: pip install tomli"
        ) from None

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """API configuration settings."""
    gemini_api_key: str = ""
    default_model: str = "gemini-2.5-flash"
    request_timeout: int = 300
    max_retries: int = 3


@dataclass
class ProcessingConfig:
    """Processing configuration settings."""
    max_pdf_size_mb: int = 100
    temp_dir: str = "/tmp/pdf_processor"
    keep_temp_files: bool = False
    log_level: str = "INFO"


@dataclass
class ContainerConfig:
    """Container-specific configuration."""
    input_dir: str = "/app/input"
    output_dir: str = "/app/output"
    logs_dir: str = "/app/logs"
    auto_detect: bool = True


@dataclass
class AnalysisConfig:
    """Analysis configuration settings."""
    enabled_types: List[str] = field(default_factory=lambda: ["general", "sections", "data_extraction"])
    default_questions: List[str] = field(default_factory=list)
    save_conversation_history: bool = True
    max_history_items: int = 100


@dataclass
class OutputConfig:
    """Output configuration settings."""
    format: str = "json"
    include_metadata: bool = True
    create_backup: bool = False
    compress_large_files: bool = True
    compression_threshold_mb: int = 10


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    validate_file_types: bool = True
    allowed_extensions: List[str] = field(default_factory=lambda: [".pdf"])
    scan_for_malware: bool = False
    strip_metadata: bool = False


class ConfigurationError(Exception):
    """Raised when there's a configuration error."""
    pass


class Config:
    """
    Main configuration class that manages all application settings.
    
    Supports TOML files with environment variable overrides for flexibility
    in different deployment scenarios.
    """
    
    def __init__(self, config_path: Union[str, Path] = "config.toml"):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to the TOML configuration file
            
        Raises:
            ConfigurationError: If configuration cannot be loaded
        """
        self.config_path = Path(config_path)
        self._raw_data: Dict[str, Any] = {}
        
        # Configuration sections
        self.api: APIConfig = APIConfig()
        self.processing: ProcessingConfig = ProcessingConfig()
        self.container: ContainerConfig = ContainerConfig()
        self.analysis: AnalysisConfig = AnalysisConfig()
        self.output: OutputConfig = OutputConfig()
        self.security: SecurityConfig = SecurityConfig()
        
        self._load_configuration()
    
    def _load_configuration(self) -> None:
        """Load configuration from TOML file and environment variables."""
        try:
            self._load_toml_config()
            self._apply_environment_overrides()
            self._populate_config_objects()
            logger.info(f"Configuration loaded from: {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self._create_default_config()
    
    def _load_toml_config(self) -> None:
        """Load configuration from TOML file."""
        if not self.config_path.exists():
            logger.warning(f"Configuration file not found: {self.config_path}")
            return
        
        try:
            with open(self.config_path, "rb") as f:
                self._raw_data = tomllib.load(f)
        except Exception as e:
            raise ConfigurationError(f"Failed to parse TOML file: {e}") from e
    
    def _apply_environment_overrides(self) -> None:
        """Apply environment variable overrides."""
        # Special handling for API key
        if api_key := os.getenv('GEMINI_API_KEY'):
            if 'api' not in self._raw_data:
                self._raw_data['api'] = {}
            self._raw_data['api']['gemini_api_key'] = api_key
        
        # Container detection override
        if os.getenv('CONTAINER') == 'true':
            if 'container' not in self._raw_data:
                self._raw_data['container'] = {}
            self._raw_data['container']['detected'] = True
    
    def _populate_config_objects(self) -> None:
        """Populate typed configuration objects from raw data."""
        if 'api' in self._raw_data:
            api_data = self._raw_data['api']
            self.api = APIConfig(
                gemini_api_key=api_data.get('gemini_api_key', ''),
                default_model=api_data.get('default_model', 'gemini-2.5-flash'),
                request_timeout=api_data.get('request_timeout', 300),
                max_retries=api_data.get('max_retries', 3)
            )
        
        if 'processing' in self._raw_data:
            proc_data = self._raw_data['processing']
            self.processing = ProcessingConfig(
                max_pdf_size_mb=proc_data.get('max_pdf_size_mb', 100),
                temp_dir=proc_data.get('temp_dir', '/tmp/pdf_processor'),
                keep_temp_files=proc_data.get('keep_temp_files', False),
                log_level=proc_data.get('log_level', 'INFO')
            )
        
        if 'container' in self._raw_data:
            cont_data = self._raw_data['container']
            self.container = ContainerConfig(
                input_dir=cont_data.get('input_dir', '/app/input'),
                output_dir=cont_data.get('output_dir', '/app/output'),
                logs_dir=cont_data.get('logs_dir', '/app/logs'),
                auto_detect=cont_data.get('auto_detect', True)
            )
        
        if 'analysis' in self._raw_data:
            analysis_data = self._raw_data['analysis']
            self.analysis = AnalysisConfig(
                enabled_types=analysis_data.get('enabled_types', ["general", "sections", "data_extraction"]),
                default_questions=analysis_data.get('default_questions', []),
                save_conversation_history=analysis_data.get('save_conversation_history', True),
                max_history_items=analysis_data.get('max_history_items', 100)
            )
        
        if 'output' in self._raw_data:
            output_data = self._raw_data['output']
            self.output = OutputConfig(
                format=output_data.get('format', 'json'),
                include_metadata=output_data.get('include_metadata', True),
                create_backup=output_data.get('create_backup', False),
                compress_large_files=output_data.get('compress_large_files', True),
                compression_threshold_mb=output_data.get('compression_threshold_mb', 10)
            )
        
        if 'security' in self._raw_data:
            security_data = self._raw_data['security']
            self.security = SecurityConfig(
                validate_file_types=security_data.get('validate_file_types', True),
                allowed_extensions=security_data.get('allowed_extensions', ['.pdf']),
                scan_for_malware=security_data.get('scan_for_malware', False),
                strip_metadata=security_data.get('strip_metadata', False)
            )
    
    def _create_default_config(self) -> None:
        """Create a default configuration file."""
        default_toml = '''# PDF Processor Configuration
# Modern configuration for PDF processing with Google GenAI

[api]
# Google GenAI Configuration
gemini_api_key = "your_api_key_here"
default_model = "gemini-2.5-flash"
request_timeout = 300
max_retries = 3

[processing]
# Processing Configuration
max_pdf_size_mb = 100
temp_dir = "/tmp/pdf_processor"
keep_temp_files = false
log_level = "INFO"

[container]
# Container Configuration
input_dir = "/app/input"
output_dir = "/app/output"
logs_dir = "/app/logs"
auto_detect = true

[analysis]
# Analysis Configuration
enabled_types = ["general", "sections", "data_extraction"]
default_questions = [
    "¿Cuáles son los puntos más importantes de este documento?",
    "¿Hay alguna información crítica que requiera atención inmediata?",
    "¿Qué recomendaciones o acciones se sugieren?",
    "¿Hay fechas límite o plazos mencionados?"
]
save_conversation_history = true
max_history_items = 100

[output]
# Output Configuration
format = "json"
include_metadata = true
create_backup = false
compress_large_files = true
compression_threshold_mb = 10

[security]
# Security Configuration
validate_file_types = true
allowed_extensions = [".pdf"]
scan_for_malware = false
strip_metadata = false
'''
        
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                f.write(default_toml)
            
            # Load the default configuration
            self._load_configuration()
            
            logger.info(f"Default configuration created: {self.config_path}")
            print(f"📝 Default configuration created: {self.config_path}")
            print("💡 Edit this file to configure your API key and preferences")
            
        except Exception as e:
            logger.error(f"Failed to create default configuration: {e}")
            # Set minimal defaults in memory
            self.api.gemini_api_key = ""
            self.api.default_model = "gemini-2.5-flash"
    
    def is_container_environment(self) -> bool:
        """Detect if running in a container environment."""
        if not self.container.auto_detect:
            return False
        
        return (
            os.path.exists('/.dockerenv') or
            os.getenv('CONTAINER') == 'true' or
            os.getenv('KUBERNETES_SERVICE_HOST') is not None
        )
    
    def get_directories(self) -> Dict[str, str]:
        """Get working directories based on environment."""
        if self.is_container_environment():
            return {
                'input': self.container.input_dir,
                'output': self.container.output_dir,
                'logs': self.container.logs_dir,
                'temp': self.processing.temp_dir
            }
        else:
            base_dir = Path.cwd()
            return {
                'input': str(base_dir / 'input'),
                'output': str(base_dir / 'output'),
                'logs': str(base_dir / 'logs'),
                'temp': str(base_dir / 'temp')
            }
    
    def validate(self) -> bool:
        """
        Validate the configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        errors = []
        
        # Validate API key
        if not self.api.gemini_api_key or self.api.gemini_api_key == "your_api_key_here":
            errors.append("Gemini API key not configured")
        
        # Validate directories
        directories = self.get_directories()
        for name, path in directories.items():
            if name != 'temp':  # temp is created automatically
                try:
                    Path(path).mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create directory {name}: {path} - {e}")
        
        # Validate model
        if not self.api.default_model:
            errors.append("Default model not specified")
        
        # Validate numeric ranges
        if not 0 < self.processing.max_pdf_size_mb <= 1000:
            errors.append("max_pdf_size_mb must be between 1 and 1000")
        
        if not 0 < self.api.request_timeout <= 3600:
            errors.append("request_timeout must be between 1 and 3600 seconds")
        
        if errors:
            logger.error("Configuration validation errors:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
        
        return True
    
    def print_summary(self) -> None:
        """Print a summary of the current configuration."""
        print("\n🔧 Configuration Summary:")
        print("=" * 50)
        print(f"📄 Config File: {self.config_path}")
        print(f"🔑 API Key: {'✅ Configured' if self.api.gemini_api_key and self.api.gemini_api_key != 'your_api_key_here' else '❌ Not configured'}")
        print(f"🤖 Model: {self.api.default_model}")
        print(f"📊 Log Level: {self.processing.log_level}")
        print(f"📦 Container: {'✅ Yes' if self.is_container_environment() else '❌ No'}")
        
        directories = self.get_directories()
        print(f"📁 Directories:")
        for name, path in directories.items():
            exists = "✅" if Path(path).exists() else "❌"
            print(f"   {name}: {path} {exists}")
        
        print(f"🔍 Analysis Types: {', '.join(self.analysis.enabled_types)}")
        print(f"💾 Save History: {'✅' if self.analysis.save_conversation_history else '❌'}")


# Global configuration instance
_config: Optional[Config] = None


def get_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Get the global configuration instance.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Configuration instance
    """
    global _config
    
    if _config is None or config_path is not None:
        _config = Config(config_path or "config.toml")
    
    return _config


def reload_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Reload the global configuration.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Reloaded configuration instance
    """
    global _config
    _config = None
    return get_config(config_path)


if __name__ == "__main__":
    # Configuration validation script
    config = get_config()
    config.print_summary()
    
    if config.validate():
        print("\n✅ Configuration is valid")
        sys.exit(0)
    else:
        print("\n❌ Configuration has errors")
        sys.exit(1)
