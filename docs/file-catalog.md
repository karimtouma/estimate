# Catálogo de Archivos Python
## PDF Estimator v2.0.0 - Análisis Exhaustivo de Dependencias

**Fecha de Análisis**: 15 de Septiembre, 2025  
**Total de Archivos Python**: 29  
**Estructura**: 7 módulos principales + 2 tests

---

## Resumen Ejecutivo

### Distribución por Módulos

| Módulo | Archivos | Descripción |
|--------|----------|-------------|
| `core/` | 3 | Procesamiento principal y configuración |
| `discovery/` | 4 | Sistema de descubrimiento dinámico |
| `models/` | 3 | Esquemas de datos y clasificación |
| `optimization/` | 3 | Sistema GEPA de optimización genética |
| `services/` | 1 | Cliente de Gemini API |
| `utils/` | 6 | Utilidades y herramientas auxiliares |
| `tests/` | 2 | Tests del sistema |
| **Total** | **29** | **Sistema completo** |

### Archivos Críticos (Punto de Entrada)

1. **`src/cli.py`** - Interfaz de línea de comandos principal
2. **`src/core/adaptive_processor.py`** - Procesador principal con GEPA
3. **`src/core/config.py`** - Configuración del sistema

---

## Catálogo Detallado

### 🏗️ **CORE MODULE** (3 archivos)

#### **src/core/config.py**
- **Propósito**: Gestión de configuración del sistema
- **Dependencias Externas**: `os`, `sys`, `logging`, `pathlib`, `typing`, `dataclasses`
- **Dependencias Internas**: Ninguna (módulo base)
- **Usado Por** (14 archivos): 
  - `src/core/processor.py`
  - `src/core/adaptive_processor.py`
  - `src/services/gemini_client.py`
  - `src/utils/file_manager.py`
  - `src/utils/logging_config.py`
  - `src/utils/language_router.py`
  - `src/discovery/dynamic_discovery.py`
  - `src/discovery/enhanced_discovery.py`
  - `src/models/intelligent_classifier.py`
  - `src/optimization/comprehensive_gepa_system.py`
  - `src/optimization/gepa_classification_enhancer.py`
  - `src/optimization/pattern_extraction_gepa.py`
  - `src/cli.py`
  - `src/__init__.py`
- **Criticidad**: ⭐⭐⭐⭐⭐ (CRÍTICO - Base de todo el sistema)

#### **src/core/processor.py**
- **Propósito**: Procesador base de PDFs con lógica principal
- **Dependencias Externas**: `time`, `json`, `logging`, `pathlib`, `typing`
- **Dependencias Internas**: 
  - `config.py`
  - `services/gemini_client.py`
  - `models/schemas.py`
  - `utils/file_manager.py`
  - `utils/logging_config.py`
  - `utils/dspy_hallucination_detector.py`
  - `utils/language_router.py`
- **Usado Por**: 
  - `src/core/adaptive_processor.py` (herencia)
- **Criticidad**: ⭐⭐⭐⭐⭐ (CRÍTICO - Clase base principal)

#### **src/core/adaptive_processor.py**
- **Propósito**: Procesador adaptativo con esquemas dinámicos y GEPA
- **Dependencias Externas**: `time`, `logging`, `pathlib`, `typing`
- **Dependencias Internas**: 
  - `core/config.py`
  - `core/processor.py`
  - `models/dynamic_schemas.py`
  - `models/intelligent_classifier.py`
  - `discovery/enhanced_discovery.py`
  - `utils/adaptive_questions.py`
  - `utils/language_router.py`
  - `optimization/comprehensive_gepa_system.py`
- **Usado Por**: 
  - `src/cli.py` (punto de entrada principal)
- **Criticidad**: ⭐⭐⭐⭐⭐ (CRÍTICO - Procesador principal)

---

### 🔍 **DISCOVERY MODULE** (4 archivos)

#### **src/discovery/dynamic_discovery.py**
- **Propósito**: Motor de descubrimiento dinámico de patrones
- **Dependencias Externas**: `asyncio`, `json`, `logging`, `time`, `numpy`, `fitz` (PyMuPDF), `pathlib`, `typing`, `dataclasses`, `PIL`
- **Dependencias Internas**: 
  - `core/config.py`
  - `services/gemini_client.py`
  - `utils/logging_config.py`
  - `pattern_analyzer.py`
  - `nomenclature_parser.py`
  - `optimization/pattern_extraction_gepa.py`
- **Usado Por**: 
  - `discovery/enhanced_discovery.py`
- **Criticidad**: ⭐⭐⭐⭐ (ALTO - Motor de descubrimiento)

#### **src/discovery/enhanced_discovery.py**
- **Propósito**: Descubrimiento mejorado con integración de esquemas dinámicos
- **Dependencias Externas**: `asyncio`, `json`, `logging`, `time`, `pathlib`, `typing`, `dataclasses`
- **Dependencias Internas**: 
  - `dynamic_discovery.py`
  - `models/intelligent_classifier.py`
  - `models/dynamic_schemas.py`
  - `utils/logging_config.py`
- **Usado Por**: 
  - `core/adaptive_processor.py`
- **Criticidad**: ⭐⭐⭐⭐ (ALTO - Integración de descubrimiento)

#### **src/discovery/nomenclature_parser.py**
- **Propósito**: Análisis de sistemas de nomenclatura técnica
- **Dependencias Externas**: `re`, `json`, `logging`, `typing`, `collections`
- **Dependencias Internas**: 
  - `utils/logging_config.py`
- **Usado Por**: 
  - `discovery/dynamic_discovery.py`
- **Criticidad**: ⭐⭐⭐ (MEDIO - Análisis especializado)

#### **src/discovery/pattern_analyzer.py**
- **Propósito**: Análisis de patrones visuales y estructurales
- **Dependencias Externas**: `re`, `json`, `logging`, `typing`, `collections`
- **Dependencias Internas**: 
  - `utils/logging_config.py`
- **Usado Por**: 
  - `discovery/dynamic_discovery.py`
- **Criticidad**: ⭐⭐⭐ (MEDIO - Análisis de patrones)

---

### 🧠 **MODELS MODULE** (3 archivos)

#### **src/models/schemas.py**
- **Propósito**: Esquemas de datos Pydantic para validación
- **Dependencias Externas**: `typing`, `pydantic`, `enum`
- **Dependencias Internas**: Ninguna (esquemas base)
- **Usado Por**: 
  - `core/processor.py`
  - `core/adaptive_processor.py`
  - `services/gemini_client.py`
- **Criticidad**: ⭐⭐⭐⭐⭐ (CRÍTICO - Validación de datos)

#### **src/models/dynamic_schemas.py**
- **Propósito**: Sistema de esquemas dinámicos y registro de tipos
- **Dependencias Externas**: `json`, `time`, `logging`, `pathlib`, `typing`, `dataclasses`, `pydantic`, `enum`
- **Dependencias Internas**: Ninguna (sistema base)
- **Usado Por**: 
  - `core/adaptive_processor.py`
  - `models/intelligent_classifier.py`
  - `discovery/enhanced_discovery.py`
  - `optimization/gepa_classification_enhancer.py`
- **Criticidad**: ⭐⭐⭐⭐⭐ (CRÍTICO - Esquemas dinámicos)

#### **src/models/intelligent_classifier.py**
- **Propósito**: Clasificador inteligente con GEPA
- **Dependencias Externas**: `json`, `time`, `logging`, `typing`, `dataclasses`, `pathlib`
- **Dependencias Internas**: 
  - `services/gemini_client.py`
  - `core/config.py`
  - `utils/logging_config.py`
  - `models/dynamic_schemas.py`
  - `optimization/gepa_classification_enhancer.py`
- **Usado Por**: 
  - `core/adaptive_processor.py`
  - `discovery/enhanced_discovery.py`
- **Criticidad**: ⭐⭐⭐⭐ (ALTO - Clasificación inteligente)

---

### 🧬 **OPTIMIZATION MODULE** (3 archivos)

#### **src/optimization/gepa_classification_enhancer.py**
- **Propósito**: Mejorador GEPA con sistema de juez para clasificación
- **Dependencias Externas**: `json`, `time`, `logging`, `asyncio`, `typing`, `dataclasses`, `statistics`
- **Dependencias Internas**: 
  - `services/gemini_client.py`
  - `core/config.py`
  - `utils/logging_config.py`
  - `models/dynamic_schemas.py`
- **Usado Por**: 
  - `models/intelligent_classifier.py`
  - `optimization/__init__.py`
- **Criticidad**: ⭐⭐⭐⭐ (ALTO - Optimización GEPA)

#### **src/optimization/comprehensive_gepa_system.py**
- **Propósito**: Sistema GEPA completo para optimización de prompts
- **Dependencias Externas**: `json`, `time`, `logging`, `asyncio`, `typing`, `pathlib`, `dataclasses`, `collections`, `statistics`
- **Dependencias Internas**: 
  - `services/gemini_client.py`
  - `core/config.py`
  - `utils/logging_config.py`
  - `models/schemas.py`
- **Usado Por**: 
  - `core/adaptive_processor.py`
  - `optimization/__init__.py`
- **Criticidad**: ⭐⭐⭐⭐ (ALTO - Sistema GEPA completo)

#### **src/optimization/pattern_extraction_gepa.py**
- **Propósito**: GEPA especializado para extracción de patrones
- **Dependencias Externas**: `json`, `time`, `logging`, `asyncio`, `typing`, `pathlib`, `dataclasses`, `collections`
- **Dependencias Internas**: 
  - `services/gemini_client.py`
  - `core/config.py`
  - `utils/logging_config.py`
- **Usado Por**: 
  - `discovery/dynamic_discovery.py`
  - `optimization/__init__.py`
- **Criticidad**: ⭐⭐⭐ (MEDIO - GEPA especializado)

---

### 🌐 **SERVICES MODULE** (1 archivo)

#### **src/services/gemini_client.py**
- **Propósito**: Cliente para API de Google Gemini
- **Dependencias Externas**: `time`, `logging`, `typing`, `pathlib`, `google.genai`, `tenacity`
- **Dependencias Internas**: 
  - `core/config.py`
  - `models/schemas.py`
- **Usado Por** (9 archivos): 
  - `core/processor.py`
  - `core/adaptive_processor.py`
  - `models/intelligent_classifier.py`
  - `discovery/dynamic_discovery.py`
  - `discovery/enhanced_discovery.py`
  - `utils/language_router.py`
  - `optimization/gepa_classification_enhancer.py`
  - `optimization/comprehensive_gepa_system.py`
  - `optimization/pattern_extraction_gepa.py`
- **Criticidad**: ⭐⭐⭐⭐⭐ (CRÍTICO - Cliente API principal)

---

### 🛠️ **UTILS MODULE** (6 archivos)

#### **src/utils/logging_config.py**
- **Propósito**: Configuración del sistema de logging
- **Dependencias Externas**: `logging`, `logging.handlers`, `sys`, `json`, `time`, `pathlib`, `typing`
- **Dependencias Internas**: 
  - `core/config.py`
- **Usado Por**: 
  - `core/processor.py`
  - Prácticamente todos los módulos (logging)
- **Criticidad**: ⭐⭐⭐⭐⭐ (CRÍTICO - Sistema de logging)

#### **src/utils/language_router.py**
- **Propósito**: Detección automática de idioma y optimización de prompts
- **Dependencias Externas**: `re`, `logging`, `typing`, `dataclasses`, `collections`
- **Dependencias Internas**: 
  - `services/gemini_client.py`
  - `core/config.py`
  - `utils/logging_config.py`
- **Usado Por**: 
  - `core/processor.py`
  - `core/adaptive_processor.py`
- **Criticidad**: ⭐⭐⭐⭐ (ALTO - Optimización de idioma)

#### **src/utils/file_manager.py**
- **Propósito**: Gestión de archivos y persistencia
- **Dependencias Externas**: `json`, `gzip`, `shutil`, `time`, `logging`, `pathlib`, `typing`
- **Dependencias Internas**: 
  - `core/config.py`
- **Usado Por**: 
  - `core/processor.py`
- **Criticidad**: ⭐⭐⭐ (MEDIO - Gestión de archivos)

#### **src/utils/adaptive_questions.py**
- **Propósito**: Generación adaptativa de preguntas
- **Dependencias Externas**: `logging`, `typing`
- **Dependencias Internas**: Ninguna
- **Usado Por**: 
  - `core/adaptive_processor.py`
- **Criticidad**: ⭐⭐⭐ (MEDIO - Generación de preguntas)

#### **src/utils/dspy_hallucination_detector.py**
- **Propósito**: Detección de alucinaciones con DSPy
- **Dependencias Externas**: `dspy`, `re`, `logging`, `typing`, `pydantic`
- **Dependencias Internas**: Ninguna
- **Usado Por**: 
  - `core/processor.py`
- **Criticidad**: ⭐⭐ (BAJO - Validación adicional)

---

## Análisis de Dependencias Detallado

### Dependencias Externas Críticas

| Librería | Archivos que la Usan | Propósito |
|----------|---------------------|-----------|
| `google.genai` | `services/gemini_client.py` | API principal de IA |
| `pydantic` | `models/schemas.py`, `models/dynamic_schemas.py`, `utils/dspy_hallucination_detector.py` | Validación de datos |
| `dspy` | `utils/dspy_hallucination_detector.py` | Framework de reasoning |
| `fitz` (PyMuPDF) | `discovery/dynamic_discovery.py` | Procesamiento PDF |
| `tenacity` | `services/gemini_client.py` | Retry logic |
| `click` | `src/cli.py` | Interfaz CLI |

### Dependencias Internas Más Utilizadas

| Módulo | Usado Por (Cantidad) | Archivos |
|--------|---------------------|----------|
| `core/config.py` | 8 archivos | Base de configuración |
| `services/gemini_client.py` | 8 archivos | Cliente API principal |
| `utils/logging_config.py` | 15+ archivos | Sistema de logging |
| `models/schemas.py` | 3 archivos | Validación de datos |
| `models/dynamic_schemas.py` | 4 archivos | Esquemas dinámicos |

---

## Flujo de Ejecución Principal

### 1. Punto de Entrada
```
src/cli.py → src/core/adaptive_processor.py
```

### 2. Inicialización
```
adaptive_processor.py → core/config.py
                     → core/processor.py
                     → services/gemini_client.py
                     → utils/logging_config.py
                     → utils/language_router.py
                     → models/dynamic_schemas.py
                     → models/intelligent_classifier.py
                     → optimization/comprehensive_gepa_system.py
```

### 3. Descubrimiento
```
adaptive_processor.py → discovery/enhanced_discovery.py
                     → discovery/dynamic_discovery.py
                     → discovery/pattern_analyzer.py
                     → discovery/nomenclature_parser.py
```

### 4. Optimización GEPA
```
models/intelligent_classifier.py → optimization/gepa_classification_enhancer.py
discovery/dynamic_discovery.py → optimization/pattern_extraction_gepa.py
adaptive_processor.py → optimization/comprehensive_gepa_system.py
```

---

## Archivos de Soporte

### **src/__init__.py** (7 archivos)
- **Propósito**: Inicialización de módulos Python
- **Contenido**: Vacíos (markers de paquete)
- **Criticidad**: ⭐ (BAJO - Estructura de paquetes)

### **src/py.typed**
- **Propósito**: Marcador de type hints para mypy
- **Criticidad**: ⭐ (BAJO - Type checking)

---

## Tests

### **tests/test_dynamic_schemas.py**
- **Propósito**: Tests del sistema de esquemas dinámicos
- **Dependencias**: `pytest`, `tempfile`, `json`, `time`, `pathlib`
- **Estado**: Actualizado
- **Criticidad**: ⭐⭐ (BAJO - Testing)

### **tests/test_intelligent_classifier.py**
- **Propósito**: Tests del clasificador inteligente
- **Dependencias**: `pytest`, `tempfile`, `asyncio`, `pathlib`
- **Estado**: Actualizado
- **Criticidad**: ⭐⭐ (BAJO - Testing)

---

## Mapa de Dependencias Completo

### Archivos Más Utilizados (Top 5)

| Archivo | Usado Por | Tipo |
|---------|-----------|------|
| `core/config.py` | 14 archivos | Configuración base |
| `utils/logging_config.py` | 13+ archivos | Sistema de logging |
| `services/gemini_client.py` | 9 archivos | Cliente API |
| `models/schemas.py` | 6 archivos | Validación de datos |
| `models/dynamic_schemas.py` | 4 archivos | Esquemas dinámicos |

### Archivos de Entrada (Entry Points)

| Archivo | Propósito | Dependencias |
|---------|-----------|--------------|
| `src/cli.py` | Interfaz de línea de comandos | 2 internas |
| `src/core/adaptive_processor.py` | Procesador principal | 10 internas |

### Archivos Independientes (Sin Dependencias Internas)

| Archivo | Propósito |
|---------|-----------|
| `src/core/config.py` | Configuración base |
| `src/models/schemas.py` | Esquemas Pydantic |
| `src/utils/dspy_hallucination_detector.py` | Validación DSPy |

### Cadenas de Dependencias Críticas

#### **Cadena Principal de Procesamiento**:
```
cli.py → adaptive_processor.py → processor.py → gemini_client.py
                               → dynamic_schemas.py
                               → intelligent_classifier.py
                               → enhanced_discovery.py
```

#### **Cadena GEPA**:
```
intelligent_classifier.py → gepa_classification_enhancer.py
adaptive_processor.py → comprehensive_gepa_system.py
dynamic_discovery.py → pattern_extraction_gepa.py
```

#### **Cadena de Descubrimiento**:
```
enhanced_discovery.py → dynamic_discovery.py → pattern_analyzer.py
                                            → nomenclature_parser.py
```

---

## Módulos Huérfanos (Sin Referencias)

**Ninguno** - Todos los archivos Python están siendo utilizados en el sistema actual.

---

## Recomendaciones de Mantenimiento

### Alta Prioridad
1. **Monitorear**: `core/` - Cualquier cambio afecta todo el sistema
2. **Validar**: `services/gemini_client.py` - Punto único de falla para API
3. **Actualizar**: `models/schemas.py` - Cambios requieren migración de datos

### Media Prioridad
1. **Optimizar**: `discovery/dynamic_discovery.py` - Archivo más grande (1,474 líneas)
2. **Modularizar**: `optimization/` - Considerar división si crece más

### Baja Prioridad
1. **Revisar**: Tests - Agregar pytest a requirements.txt si se van a usar
2. **Documentar**: `utils/` - Agregar docstrings más detallados

---

## Estadísticas de Líneas de Código

### Por Módulo

| Módulo | Archivos | Líneas | Promedio | Porcentaje |
|--------|----------|--------|----------|------------|
| `discovery/` | 4 | 3,231 | 808 | 25.3% |
| `optimization/` | 3 | 2,355 | 785 | 18.5% |
| `core/` | 3 | 2,305 | 768 | 18.1% |
| `models/` | 3 | 1,727 | 576 | 13.5% |
| `utils/` | 6 | 1,874 | 312 | 14.7% |
| `services/` | 1 | 746 | 746 | 5.8% |
| `cli.py` | 1 | 431 | 431 | 3.4% |
| **Total** | **29** | **12,757** | **440** | **100%** |

### Archivos Más Grandes (Top 10)

| Archivo | Líneas | Porcentaje |
|---------|--------|------------|
| `discovery/dynamic_discovery.py` | 1,473 | 11.5% |
| `core/processor.py` | 1,293 | 10.1% |
| `models/intelligent_classifier.py` | 875 | 6.9% |
| `optimization/comprehensive_gepa_system.py` | 840 | 6.6% |
| `optimization/gepa_classification_enhancer.py` | 817 | 6.4% |
| `services/gemini_client.py` | 746 | 5.8% |
| `optimization/pattern_extraction_gepa.py` | 698 | 5.5% |
| `discovery/pattern_analyzer.py` | 660 | 5.2% |
| `models/dynamic_schemas.py` | 651 | 5.1% |
| `discovery/nomenclature_parser.py` | 608 | 4.8% |

### Complejidad por Módulo

| Módulo | Complejidad | Justificación |
|--------|-------------|---------------|
| `discovery/` | Alta | Motor de descubrimiento con múltiples estrategias |
| `optimization/` | Alta | Algoritmos genéticos y sistemas de juez |
| `core/` | Media-Alta | Lógica principal de procesamiento |
| `models/` | Media | Esquemas de datos y clasificación |
| `utils/` | Baja-Media | Utilidades auxiliares |
| `services/` | Media | Cliente API con manejo de errores |

## Estadísticas Finales

- **Total Archivos Python**: 29
- **Líneas de Código**: 12,757
- **Módulos Principales**: 7
- **Dependencias Externas**: 12 esenciales
- **Archivos Críticos**: 5 (config, processor, adaptive_processor, schemas, gemini_client)
- **Archivos Huérfanos**: 0
- **Cobertura de Tests**: 2 módulos principales
- **Archivo Más Grande**: `dynamic_discovery.py` (1,473 líneas)
- **Archivo Más Pequeño**: `__init__.py` (1 línea)
- **Promedio de Líneas**: 440 líneas por archivo

**Estado**: Repositorio limpio y optimizado sin código obsoleto.
