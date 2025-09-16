"""
PDF Estimator - Sistema de procesamiento de PDFs con IA.

Un sistema moderno y escalable para el análisis inteligente de documentos PDF
utilizando Google Gemini API con arquitectura limpia y patrones de diseño sólidos.
"""

__version__ = "1.0.0"
__author__ = "PDF Estimator Team"
__email__ = "team@pdf-estimator.com"

from .core.config import Config, get_config
from .core.processor import PDFProcessor
from .models.schemas import DataExtraction, DocumentAnalysis, SectionAnalysis

__all__ = [
    "PDFProcessor",
    "get_config",
    "Config",
    "DocumentAnalysis",
    "SectionAnalysis",
    "DataExtraction",
]
