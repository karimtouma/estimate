"""
PDF Estimator - Sistema de procesamiento de PDFs con IA.

Un sistema moderno y escalable para el análisis inteligente de documentos PDF
utilizando Google Gemini API con arquitectura limpia y patrones de diseño sólidos.
"""

__version__ = "1.0.0"
__author__ = "PDF Estimator Team"
__email__ = "team@pdf-estimator.com"

from .core.processor import PDFProcessor
from .core.config import get_config, Config
from .models.schemas import DocumentAnalysis, SectionAnalysis, DataExtraction

__all__ = [
    "PDFProcessor",
    "get_config",
    "Config", 
    "DocumentAnalysis",
    "SectionAnalysis",
    "DataExtraction"
]
