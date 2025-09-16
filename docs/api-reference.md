# API Reference
## PDF Estimator v2.0.0

---

## 🏗️ **Arquitectura de Clases**

### **AdaptiveProcessor**
**Archivo**: `src/core/adaptive_processor.py`

#### **Métodos Principales**:

##### `comprehensive_analysis_adaptive()`
```python
def comprehensive_analysis_adaptive(
    self,
    pdf_path: Union[str, Path],
    questions: Optional[List[str]] = None,
    enable_discovery: bool = True
) -> ComprehensiveAnalysisResult
```

**Descripción**: Ejecuta análisis completo con esquemas dinámicos, optimización GEPA y detección automática de idioma

**Parámetros**:
- `pdf_path`: Ruta al archivo PDF
- `questions`: Lista opcional de preguntas (se generan adaptativas si no se proveen)
- `enable_discovery`: Habilitar descubrimiento dinámico

**Retorna**: `ComprehensiveAnalysisResult` con análisis completo, métricas GEPA y detección de idioma

**Fases Ejecutadas**:
1. Upload del PDF a Gemini
2. Enhanced Discovery con esquemas dinámicos
3. Language Detection y optimización de prompts
4. Clasificación GEPA con múltiples candidatos y juez inteligente
5. Análisis general, secciones y extracción de datos con prompts optimizados
6. Análisis multi-turn Q&A adaptativo
7. Generación de page map completo
8. Optimización GEPA en background para mejora continua
5. Mapeo completo de páginas

---

### **DynamicElementRegistry**
**Archivo**: `src/models/dynamic_schemas.py`

#### **Métodos de Gestión de Tipos**:

##### `register_discovered_type()`
```python
def register_discovered_type(
    self,
    type_name: str,
    category: CoreElementCategory,
    confidence: float,
    evidence: Optional[Dict[str, Any]] = None,
    domain_context: Optional[str] = None
) -> bool
```

**Descripción**: Registra un nuevo tipo descubierto

**Parámetros**:
- `type_name`: Nombre específico del tipo
- `category`: Categoría base (structural, architectural, mep, annotation, specialized)
- `confidence`: Nivel de confianza (0.0-1.0)
- `evidence`: Evidencia que soporta el descubrimiento
- `domain_context`: Contexto del dominio

**Retorna**: `True` si se registró exitosamente

##### `evolve_type_definition()`
```python
def evolve_type_definition(
    self,
    type_name: str,
    new_evidence: Dict[str, Any]
) -> bool
```

**Descripción**: Evoluciona la definición de un tipo existente

---

### **IntelligentTypeClassifier**
**Archivo**: `src/models/intelligent_classifier.py`

#### **Método Principal**:

##### `classify_element()`
```python
async def classify_element(
    self,
    element_data: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None
) -> ClassificationResult
```

**Descripción**: Clasifica un elemento usando múltiples estrategias

**Estrategias Aplicadas** (en orden):
1. Registry lookup
2. Pattern matching  
3. Nomenclature analysis
4. AI reasoning con Gemini

**Retorna**: `ClassificationResult` con tipo clasificado y confianza

---

### **GeminiClient**
**Archivo**: `src/services/gemini_client.py`

#### **Métodos de Generación**:

##### `generate_content()`
```python
def generate_content(
    self,
    file_uri: str,
    prompt: str,
    response_schema: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None
) -> str
```

**Descripción**: Genera contenido con archivo PDF

##### `generate_text_only_content()` *(NUEVO)*
```python
def generate_text_only_content(
    self,
    prompt: str,
    response_schema: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None
) -> str
```

**Descripción**: Genera contenido solo con texto (sin archivo)

##### `generate_multi_turn_content()`
```python
def generate_multi_turn_content(
    self,
    file_uri: str,
    questions: List[str],
    context_parts: Optional[List[types.Part]] = None
) -> List[Dict[str, Any]]
```

**Descripción**: Procesa múltiples preguntas en un solo lote

---

## 📊 **Modelos de Datos**

### **ComprehensiveAnalysisResult**
```python
class ComprehensiveAnalysisResult(BaseModel):
    file_info: dict
    general_analysis: Optional[DocumentAnalysis] = None
    sections_analysis: Optional[List[SectionAnalysis]] = None
    data_extraction: Optional[DataExtraction] = None
    qa_analysis: Optional[List[QuestionAnswer]] = None
    discovery_analysis: Optional[dict] = None
    dynamic_schema_results: Optional[dict] = None  # ← NUEVO
    page_map: Optional[DocumentPageMap] = None
    metadata: Optional[ProcessingMetadata] = None
```

### **ClassificationResult**
```python
@dataclass
class ClassificationResult:
    classified_type: str
    base_category: CoreElementCategory
    confidence: float
    alternative_types: List[Tuple[str, float]] = None
    discovery_method: DiscoveryMethod = DiscoveryMethod.HYBRID_ANALYSIS
    reasoning: Optional[str] = None
    evidence_used: List[str] = None
    domain_context: Optional[str] = None
    industry_context: Optional[str] = None
    is_new_discovery: bool = False
    requires_validation: bool = False
```

### **AdaptiveElementType**
```python
@dataclass
class AdaptiveElementType:
    type_name: str
    base_category: CoreElementCategory
    specific_attributes: Dict[str, Any]
    discovery_confidence: float
    occurrence_count: int = 1
    last_seen: float = field(default_factory=time.time)
    reliability_score: float = 0.0
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)
    domain_contexts: Set[str] = field(default_factory=set)
    evidence_samples: List[Dict[str, Any]] = field(default_factory=list)
```

---

## 🔧 **Configuración Avanzada**

### **config.toml - Secciones Principales**:

#### **[analysis]**
```toml
[analysis]
enable_dynamic_schemas = true
enabled_types = ["general", "sections", "data_extraction"]

# Umbrales de clasificación
auto_register_confidence_threshold = 0.85
validation_threshold = 0.7
new_discovery_threshold = 0.6

# Preguntas por defecto
default_questions = [
    "¿Qué tipo de estructura se muestra en este documento?",
    "¿Cuál es el alcance y propósito de este documento?",
    "¿Qué fase de construcción representa este documento?",
    "¿Cuáles son los principales sistemas visibles en este documento?",
    "¿Qué materiales y especificaciones se indican?",
    "¿Qué métodos y sistemas de construcción se están utilizando?",
    "¿Qué códigos, estándares o regulaciones se referencian?",
    "¿Qué requisitos de seguridad o cumplimiento se especifican?"
]
```

#### **[api]**
```toml
[api]
gemini_api_key = "${GEMINI_API_KEY}"
default_model = "gemini-2.5-pro"
max_concurrent_requests = 3
output_language = "spanish"
force_spanish_output = true

# Configuración de generación
temperature = 0.3
top_p = 0.8
top_k = 40
max_output_tokens = 8192
```

#### **[processing]**
```toml
[processing]
max_pdf_size_mb = 50
log_level = "INFO"
enable_complete_page_mapping = true
page_classification_batch_size = 5
max_parallel_page_batches = 2
```

---

## 🔍 **Estructura de Salida JSON**

### **Sección: dynamic_schema_results** *(NUEVO)*
```json
{
  "dynamic_schema_results": {
    "discovered_element_types": [
      {
        "specific_type": "general_note",
        "base_category": "annotation",
        "discovery_confidence": 0.95,
        "is_dynamically_discovered": true,
        "domain_context": "commercial"
      }
    ],
    "auto_registered_types": [],
    "registry_stats": {
      "total_types": 5,
      "category_counts": {
        "annotation": 5,
        "specialized": 0
      },
      "total_discoveries": 5,
      "discovery_rate": 0.83
    },
    "classification_performance": {
      "total_classifications": 6,
      "discoveries_made": 5,
      "average_accuracy": 0.0,
      "registry_size": 5
    }
  }
}
```

### **Sección: page_map**
```json
{
  "page_map": {
    "total_pages": 51,
    "pages": [
      {
        "page_number": 1,
        "primary_category": "Cover Sheets",
        "secondary_categories": ["Schedules"],
        "content_summary": "Project cover sheet with project information...",
        "key_elements": ["Project Title", "Sheet Index"],
        "complexity_score": 1.0,
        "confidence": 1.0
      }
    ],
    "category_distribution": {
      "Detail Drawings": [4, 7, 16, 17, ...],
      "Floor Plans": [9, 10, 14, 27, ...]
    },
    "coverage_analysis": {
      "Detail Drawings": {
        "total_pages": 17,
        "coverage_percentage": 43.1,
        "avg_confidence": 0.98
      }
    }
  }
}
```

---

## ⚙️ **Variables de Entorno**

### **Archivo**: `.env`
```bash
# Requeridas
GEMINI_API_KEY=tu_clave_api_aqui

# Opcionales
CONTAINER=true
LOG_LEVEL=INFO
DEBUG=false
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
```

---

## 🔄 **Ciclo de Vida del Análisis**

### **1. Inicialización**
```python
# Carga configuración
config = get_config()

# Inicializa procesador adaptativo
processor = AdaptiveProcessor(config)

# Inicializa registry dinámico
registry = get_dynamic_registry()
```

### **2. Descubrimiento**
```python
# Muestreo estratégico
sample_pages = discovery.strategic_sampling(total_pages, sample_size=20)

# Análisis por lotes
discovery_result = discovery.initial_exploration(sample_pages)

# Clasificación inteligente
enhanced_result = enhanced_discovery.enhanced_initial_exploration()
```

### **3. Clasificación**
```python
# Para cada elemento único
for element in unique_elements:
    classification = await classifier.classify_element(element, context)
    
    # Auto-registro si confianza alta
    if classification.confidence >= 0.85:
        registry.register_discovered_type(classification)
```

### **4. Análisis Principal**
```python
# Análisis general adaptativo
general_analysis = processor.analyze_document(file_uri, "general", discovery_context)

# Análisis de secciones
sections_analysis = processor.analyze_document(file_uri, "sections", discovery_context)

# Extracción de datos
data_extraction = processor.analyze_document(file_uri, "data_extraction", discovery_context)
```

### **5. Q&A Adaptativo**
```python
# Generación de preguntas adaptativas
if not questions:
    questions = generate_adaptive_questions(discovery_result, max_questions=8)

# Procesamiento por lotes
qa_results = processor.multi_turn_analysis(file_uri, questions)
```

### **6. Mapeo de Páginas**
```python
# Mapeo completo de todas las páginas
page_map = discovery.create_complete_page_map()

# Procesamiento en lotes de 5 páginas
for batch in page_batches:
    classifications = discovery.classify_page_batch(batch)
```

---

## 🎯 **Patrones de Uso**

### **Análisis Básico**:
```python
from src.core.adaptive_processor import AdaptiveProcessor
from src.core.config import get_config

config = get_config()
with AdaptiveProcessor(config) as processor:
    result = processor.comprehensive_analysis_adaptive("input/file.pdf")
    print(f"Análisis completado: {len(result.qa_analysis)} preguntas respondidas")
```

### **Análisis con Preguntas Personalizadas**:
```python
custom_questions = [
    "¿Cuál es el tipo de cimentación especificado?",
    "¿Qué sistemas MEP están incluidos?"
]

result = processor.comprehensive_analysis_adaptive(
    pdf_path="input/file.pdf",
    questions=custom_questions,
    enable_discovery=True
)
```

### **Acceso a Registry Dinámico**:
```python
from src.models.dynamic_schemas import get_dynamic_registry

registry = get_dynamic_registry()
stats = registry.get_statistics()
print(f"Tipos registrados: {stats['total_types']}")

# Ver tipos más confiables
reliable_types = stats['most_reliable_types']
for type_info in reliable_types:
    print(f"{type_info['type_name']}: {type_info['reliability_score']:.3f}")
```

---

## 📊 **Métricas y Estadísticas**

### **API Statistics**:
```json
{
  "api_statistics": {
    "total_api_calls": 21,
    "calls_by_type": {
      "generate_content": 16,
      "generate_text_only_content": 5
    },
    "performance": {
      "total_api_time_seconds": 550.35,
      "average_time_per_call": 25.02
    },
    "token_usage": {
      "input_tokens": 216151,
      "output_tokens": 15880,
      "cached_tokens": 117713,
      "total_tokens": 232031,
      "cache_efficiency_percent": 54.5
    },
    "estimated_cost_usd": {
      "input_cost": 0.054,
      "output_cost": 0.0159,
      "total_cost": 0.0699
    }
  }
}
```

### **Registry Statistics**:
```json
{
  "registry_stats": {
    "total_types": 5,
    "category_counts": {
      "structural": 0,
      "architectural": 0,
      "mep": 0,
      "annotation": 5,
      "specialized": 0
    },
    "total_discoveries": 5,
    "discovery_rate": 0.83,
    "most_reliable_types": [
      {
        "type_name": "cross_reference_note",
        "reliability_score": 0.716,
        "occurrence_count": 1,
        "category": "annotation"
      }
    ]
  }
}
```

---

## 🔧 **Extensibilidad**

### **Agregar Nueva Estrategia de Clasificación**:
```python
class CustomClassificationStrategy:
    def __init__(self, classifier):
        self.classifier = classifier
    
    async def classify(self, element_data, context):
        # Tu lógica de clasificación personalizada
        return ClassificationResult(...)

# Registrar estrategia
classifier.strategies.append(custom_strategy.classify)
```

### **Agregar Nuevo Tipo de Análisis**:
```python
# En config.toml
[analysis]
enabled_types = ["general", "sections", "data_extraction", "custom_analysis"]

# Implementar en processor
def analyze_custom(self, file_uri, discovery_context):
    # Tu lógica de análisis personalizada
    return custom_result
```

---

## 🛡️ **Manejo de Errores**

### **Excepciones Principales**:

```python
class ProcessorError(Exception):
    """Error base del procesador"""
    pass

class ValidationError(ProcessorError):
    """Error de validación de entrada"""
    pass

class GeminiAPIError(Exception):
    """Error de API de Gemini"""
    pass

class FileUploadError(GeminiAPIError):
    """Error de subida de archivo"""
    pass
```

### **Patrón de Manejo**:
```python
try:
    result = processor.comprehensive_analysis_adaptive(pdf_path)
except ValidationError as e:
    logger.error(f"Validation failed: {e}")
except GeminiAPIError as e:
    logger.error(f"API error: {e}")
except ProcessorError as e:
    logger.error(f"Processing failed: {e}")
except Exception as e:
    logger.error(f"Unexpected error: {e}")
```

---

## 🎯 **Mejores Prácticas**

### **1. Gestión de Recursos**:
```python
# Usar context manager para cleanup automático
with AdaptiveProcessor(config) as processor:
    result = processor.comprehensive_analysis_adaptive(pdf_path)
# Cleanup automático ejecutado
```

### **2. Configuración de Logging**:
```python
# Para debugging detallado
config.processing.log_level = "DEBUG"

# Para producción
config.processing.log_level = "INFO"
```

### **3. Optimización de Rendimiento**:
```python
# Habilitar caché para documentos similares
config.processing.enable_smart_cache = True

# Ajustar tamaño de lote para páginas
config.processing.page_classification_batch_size = 5
```

### **4. Manejo de Documentos Grandes**:
```python
# El sistema maneja automáticamente documentos grandes
# Muestreo estratégico se activa automáticamente para >15 páginas
# No se requiere configuración adicional
```

---

## 🔍 **Debugging y Diagnóstico**

### **Habilitar Logging Detallado**:
```python
import logging
logging.getLogger('src.models.intelligent_classifier').setLevel(logging.DEBUG)
logging.getLogger('src.discovery.dynamic_discovery').setLevel(logging.DEBUG)
```

### **Verificar Estado del Registry**:
```python
registry = get_dynamic_registry()
print(f"Tipos registrados: {len(registry.types)}")
for type_name, type_def in registry.types.items():
    print(f"  {type_name}: {type_def.reliability_score:.3f}")
```

### **Inspeccionar Clasificaciones**:
```python
# Ver detalles de clasificación
for element in discovered_elements:
    classification = await classifier.classify_element(element)
    print(f"Tipo: {classification.classified_type}")
    print(f"Confianza: {classification.confidence:.3f}")
    print(f"Método: {classification.discovery_method}")
    print(f"Razonamiento: {classification.reasoning}")
```

---

*Referencia API completa para PDF Estimator v2.0.0*
*Septiembre 2025*
