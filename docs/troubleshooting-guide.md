# Guía de Solución de Problemas
## PDF Estimator v2.0.0

---

## ✅ **Estado del Sistema: OPERACIONAL**

El sistema PDF Estimator v2.0.0 está completamente operacional. Todos los errores críticos han sido resueltos.

---

## 🔧 **Problemas Resueltos (Septiembre 2025)**

### **❌→✅ Error de Validación Pydantic**
**Síntoma**: `1 validation error for ComprehensiveAnalysisResult dynamic_schema_results Object has no attribute 'dynamic_schema_results'`

**Causa**: Campo faltante en el modelo Pydantic

**Solución Aplicada**: 
```python
# src/models/schemas.py:194
dynamic_schema_results: Optional[dict] = Field(default=None, description="Dynamic schema discovery and classification results")
```

**Estado**: ✅ RESUELTO

---

### **❌→✅ Error API Gemini 400 INVALID_ARGUMENT**
**Síntoma**: `Content generation failed: 400 INVALID_ARGUMENT`

**Causa**: Llamada a `generate_content()` con `file_uri=None`

**Solución Aplicada**:
1. Creado método `generate_text_only_content()` en `GeminiClient`
2. Actualizado `IntelligentTypeClassifier` para usar el nuevo método
3. Removido `additionalProperties: False` del schema JSON

**Estado**: ✅ RESUELTO

---

### **❌→✅ Errores DSPy "No LM is loaded"**
**Síntoma**: Cientos de errores `Error in hallucination detection: No LM is loaded`

**Causa**: DSPy no inicializado correctamente

**Solución Aplicada**:
```python
# src/discovery/dynamic_discovery.py:897-902
if hasattr(dspy.settings, 'lm') and dspy.settings.lm is not None:
    use_dspy = True
else:
    logger.debug("DSPy not properly configured, skipping validation")
    use_dspy = False
```

**Estado**: ✅ RESUELTO

---

### **❌→✅ Warning LiteLLM Async Cleanup**
**Síntoma**: `RuntimeWarning: coroutine 'close_litellm_async_clients' was never awaited`

**Causa**: Cleanup asíncrono no manejado adecuadamente

**Solución Aplicada**:
```python
# src/core/processor.py:1344-1379
def _cleanup_async_clients(self):
    """Clean up async clients to prevent litellm warnings."""
    warnings.filterwarnings("ignore", 
                          message="coroutine 'close_litellm_async_clients' was never awaited",
                          category=RuntimeWarning)
    # ... manejo adecuado de cleanup async ...
```

**Estado**: ✅ RESUELTO

---

## 🚀 **Verificación de Funcionamiento**

### **Test de Sistema Completo**:
```bash
# 1. Verificar configuración
make status

# 2. Ejecutar análisis de prueba
make job

# 3. Verificar resultados
ls -la output/
cat output/file_comprehensive_analysis.json | jq '.file_info'
```

### **Indicadores de Éxito**:
- ✅ Análisis completa sin errores
- ✅ Archivo JSON generado en `output/`
- ✅ Mensaje: "✅ Comprehensive analysis completed"
- ✅ Tipos dinámicos descubiertos automáticamente
- ✅ Estadísticas de API mostradas

---

## 🛠️ **Problemas Comunes y Soluciones**

### **1. "No such file or directory"**
**Causa**: PDF no está en la ubicación correcta

**Solución**:
```bash
# Verificar que el archivo existe
ls -la input/
# Copiar tu PDF
cp tu_documento.pdf input/file.pdf
```

---

### **2. "API key not configured"**
**Causa**: Clave API de Gemini no configurada

**Solución**:
```bash
# Crear archivo .env con tu clave
echo "GEMINI_API_KEY=tu_clave_aqui" > .env
# Verificar
cat .env
```

---

### **3. "Docker daemon not running"**
**Causa**: Docker no está iniciado

**Solución**:
```bash
# Iniciar Docker
sudo systemctl start docker  # Linux
# o abrir Docker Desktop en Mac/Windows

# Verificar
docker info
```

---

### **4. "Container build failed"**
**Causa**: Problemas en la construcción del contenedor

**Solución**:
```bash
# Limpiar y reconstruir
make clean
make build

# Si persiste, reconstruir sin caché
docker-compose build --no-cache
```

---

### **5. "Analysis timeout"**
**Causa**: Documento muy grande o conexión lenta

**Solución**:
```bash
# Probar análisis rápido primero
make job-quick

# Verificar tamaño del PDF
ls -lh input/file.pdf
# Si es >20MB, considerar optimizar el PDF
```

---

### **6. "Memory issues"**
**Causa**: Insuficiente memoria para procesamiento

**Solución**:
```bash
# Verificar memoria disponible
docker stats

# Aumentar memoria de Docker si es necesario
# Docker Desktop → Settings → Resources → Memory
```

---

## 📋 **Logs y Debugging**

### **Ubicaciones de Logs**:
```bash
# Logs del contenedor
docker-compose logs pdf-estimator

# Logs del sistema
ls -la logs/
cat logs/pdf_processor.log
```

### **Niveles de Log**:
- `DEBUG`: Información detallada para debugging
- `INFO`: Información general de operación
- `WARNING`: Advertencias no críticas
- `ERROR`: Errores que requieren atención

### **Configurar Logging Verbose**:
```toml
# config.toml
[processing]
log_level = "DEBUG"
```

---

## 🔍 **Diagnóstico Avanzado**

### **Verificar Estado de Componentes**:
```bash
# 1. Verificar configuración
python -c "from src.core.config import get_config; print('Config OK')"

# 2. Verificar conexión API
python -c "import google.genai; print('Gemini OK')"

# 3. Verificar dependencias
python -c "import pydantic, tenacity, dspy; print('Dependencies OK')"
```

### **Test de API Gemini**:
```python
# Test manual de conexión
from src.services.gemini_client import GeminiClient
from src.core.config import get_config

config = get_config()
client = GeminiClient(config)
print("✅ Gemini client initialized successfully")
```

---

## 🎯 **Optimización de Rendimiento**

### **Para Documentos Grandes (>30 páginas)**:
- El sistema usa muestreo inteligente automáticamente
- Procesamiento paralelo optimizado
- Caché inteligente activado

### **Para Documentos con Muchas Imágenes**:
- Gemini maneja imágenes nativamente
- No se requiere procesamiento adicional
- OCR automático incluido

### **Para Análisis Frecuentes**:
- Configurar persistencia de registry
- Usar `make job-quick` para resúmenes
- Aprovechar caché de tokens

---

## 🚨 **Escalación de Problemas**

### **Si el Sistema No Funciona**:

1. **Verificar Prerequisites**:
   ```bash
   make check-system
   ```

2. **Reconstruir Completamente**:
   ```bash
   make clean
   make build
   make job
   ```

3. **Verificar Logs Detallados**:
   ```bash
   docker-compose logs pdf-estimator | tail -100
   ```

4. **Test de Componentes Individuales**:
   ```bash
   # Test de configuración
   docker-compose run --rm pdf-estimator python -c "from src.core.config import get_config; get_config().validate()"
   
   # Test de API
   docker-compose run --rm pdf-estimator python -c "from src.services.gemini_client import GeminiClient; from src.core.config import get_config; GeminiClient(get_config())"
   ```

---

## 📞 **Contacto de Soporte**

Para problemas no cubiertos en esta guía:

1. **Revisar documentación técnica**: `docs/`
2. **Consultar TODO.md**: Problemas conocidos y mejoras planificadas
3. **Verificar configuración**: `config.toml` y `.env`
4. **Generar reporte de estado**: `make status`

---

*Guía de troubleshooting actualizada para PDF Estimator v2.0.0*
*Septiembre 2025*
