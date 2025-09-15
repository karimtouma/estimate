# Política de Seguridad
## PDF Estimator - Grupo DeAcero

---

## Versiones Soportadas

Proporcionamos actualizaciones de seguridad para las siguientes versiones:

| Versión | Soportada |
|---------|-----------|
| 2.0.x   | ✅ Sí     |
| 1.x.x   | ❌ No     |

---

## Reportar Vulnerabilidades

### Proceso de Reporte

Si descubres una vulnerabilidad de seguridad, por favor repórtala de manera responsable:

#### 📧 **Contacto Privado** (Preferido)
- **Email**: karim@deacero.com
- **Asunto**: `[SECURITY] PDF Estimator - Descripción breve`
- **Respuesta**: Dentro de 48 horas

#### 🔒 **Información a Incluir**
- Descripción detallada de la vulnerabilidad
- Pasos para reproducir el problema
- Impacto potencial y severidad
- Versión afectada
- Información del sistema (OS, Docker, etc.)

### Proceso de Manejo

#### 1. **Confirmación** (24-48 horas)
- Confirmamos la recepción del reporte
- Evaluación inicial de la vulnerabilidad

#### 2. **Investigación** (1-7 días)
- Análisis técnico de la vulnerabilidad
- Determinación del impacto y severidad
- Desarrollo de la corrección

#### 3. **Corrección** (1-14 días según severidad)
- Implementación de la solución
- Testing exhaustivo
- Preparación del release de seguridad

#### 4. **Divulgación** (Después de la corrección)
- Publicación del advisory de seguridad
- Actualización de documentación
- Reconocimiento del reportador (si lo desea)

---

## Clasificación de Severidad

### 🔴 **Crítica** (Corrección en 24-48 horas)
- Ejecución remota de código
- Acceso no autorizado a datos sensibles
- Escalación de privilegios

### 🟠 **Alta** (Corrección en 3-7 días)
- Denegación de servicio
- Exposición de información sensible
- Bypass de autenticación

### 🟡 **Media** (Corrección en 7-14 días)
- Inyección de datos
- Validación insuficiente
- Exposición de información no crítica

### 🟢 **Baja** (Corrección en siguiente release)
- Problemas de configuración
- Divulgación de información menor
- Vulnerabilidades que requieren acceso local

---

## Consideraciones de Seguridad

### 🔐 **Datos Sensibles**
- **API Keys**: Nunca incluir claves en código o logs
- **Documentos**: Los PDFs se procesan temporalmente y se eliminan
- **Logs**: No registrar información sensible de documentos

### 🐳 **Seguridad de Contenedores**
- **Usuario No-Root**: El contenedor ejecuta como `pdfuser` (UID 1000)
- **Privilegios Mínimos**: Solo permisos necesarios
- **Red Aislada**: Contenedor en red propia

### 🌐 **API External**
- **Rate Limiting**: Control de concurrencia implementado
- **Timeout**: Timeouts configurados para evitar hanging
- **Retry Logic**: Manejo robusto de errores de red

### 📁 **Manejo de Archivos**
- **Validación**: Solo archivos PDF permitidos
- **Tamaño**: Límite de 100MB por archivo
- **Cleanup**: Eliminación automática de archivos temporales

---

## Buenas Prácticas para Usuarios

### 🔑 **Gestión de API Keys**
```bash
# ✅ Correcto - usar archivo .env
echo "GEMINI_API_KEY=tu_clave" > .env

# ❌ Incorrecto - nunca en código
GEMINI_API_KEY = "sk-..." # NO HACER ESTO
```

### 📄 **Manejo de Documentos**
- **Datos Sensibles**: Revisa documentos antes de procesar
- **Backup**: Mantén copias de seguridad de documentos importantes
- **Limpieza**: El sistema elimina archivos automáticamente

### 🐳 **Uso de Docker**
```bash
# ✅ Correcto - usar usuario no-root
docker-compose run --rm pdf-estimator

# ❌ Evitar - no ejecutar como root
docker run --user root ...
```

---

## Dependencias de Seguridad

### Monitoreo Automático
- **Dependabot**: Configurado para actualizaciones automáticas
- **Security Advisories**: Notificaciones de vulnerabilidades
- **Audit Regular**: Revisión mensual de dependencias

### Dependencias Críticas
- `google-genai`: Cliente API principal
- `pydantic`: Validación de datos
- `tenacity`: Retry logic
- `httpx`: Cliente HTTP

---

## Historial de Seguridad

### v2.0.0 (Septiembre 2025)
- ✅ **Contenedorización Segura**: Usuario no-root implementado
- ✅ **Validación de Entrada**: Tipos de archivo validados
- ✅ **Rate Limiting**: Control de concurrencia en API
- ✅ **Cleanup Automático**: Eliminación de archivos temporales
- ✅ **Error Handling**: Manejo robusto de excepciones

### Vulnerabilidades Conocidas
**Ninguna** - No hay vulnerabilidades conocidas en la versión actual.

---

## Agradecimientos

Agradecemos a todos los investigadores de seguridad que reportan vulnerabilidades de manera responsable. Su trabajo ayuda a mantener seguro el proyecto para toda la comunidad.

---

**Última Actualización**: 15 de Septiembre, 2025  
**Política Versión**: 1.0  
**Contacto**: karim@deacero.com
