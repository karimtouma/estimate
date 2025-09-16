# Guía de Contribución
## PDF Estimator

¡Gracias por tu interés en contribuir a PDF Estimator! Este documento te guiará sobre cómo participar en este proyecto opensource.

---

## Código de Conducta

Al participar en este proyecto, te comprometes a mantener un ambiente respetuoso y colaborativo. Consulta nuestro [Código de Conducta](CODE_OF_CONDUCT.md) para más detalles.

---

## Formas de Contribuir

### 🐛 Reportar Errores
- Usa el [sistema de issues](https://github.com/karimtouma/estimate/issues)
- Incluye información del sistema y pasos para reproducir
- Adjunta logs relevantes si es posible

### 💡 Sugerir Mejoras
- Abre un issue con la etiqueta "enhancement"
- Describe claramente el problema que resuelve
- Proporciona ejemplos de uso si es posible

### 🔧 Contribuir Código
- Fork el repositorio
- Crea una rama para tu feature: `git checkout -b feature/nueva-funcionalidad`
- Sigue las convenciones de código existentes
- Incluye tests para nuevas funcionalidades
- Actualiza la documentación si es necesario

---

## Configuración del Entorno de Desarrollo

### Requisitos Previos
- Docker y Docker Compose
- Git
- Clave API de Google Gemini

### Instalación
```bash
# Clonar el repositorio
git clone https://github.com/karimtouma/estimate.git
cd estimate

# Configurar entorno
make setup
echo "GEMINI_API_KEY=tu_clave_api" > .env

# Verificar instalación
make status
```

### Ejecutar Tests
```bash
# Tests unitarios
docker-compose run --rm pdf-estimator python -m pytest tests/

# Test de integración
cp tu_documento_test.pdf input/file.pdf
make job
```

---

## Estándares de Código

### Estilo de Código
- **PEP 8**: Seguir convenciones de Python
- **Type Hints**: Usar type hints en funciones públicas
- **Docstrings**: Documentar clases y métodos públicos
- **Nombres**: Usar nombres descriptivos en inglés

### Estructura de Commits
```
tipo(scope): descripción breve

Descripción más detallada si es necesaria.

- Cambio específico 1
- Cambio específico 2
```

**Tipos válidos**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### Ejemplo de Commit
```
feat(gepa): implementar sistema de múltiples candidatos

Agrega generación de 5 candidatos por clasificación con evaluación
por juez inteligente para mejorar precisión.

- Generación de múltiples candidatos
- Sistema de juez con criterios técnicos
- Análisis de consenso entre candidatos
- Métricas de calidad mejoradas
```

---

## Proceso de Pull Request

### Antes de Enviar
1. **Sincronizar**: `git pull origin main`
2. **Tests**: Verificar que todos los tests pasan
3. **Linting**: Código sin errores de linting
4. **Documentación**: Actualizar docs si es necesario

### Descripción del PR
- **Título claro**: Describe qué hace el cambio
- **Descripción detallada**: Explica por qué es necesario
- **Tests**: Lista los tests agregados o modificados
- **Breaking Changes**: Menciona cambios incompatibles

### Revisión
- Los PRs requieren revisión antes de merge
- Responde a comentarios de manera constructiva
- Realiza cambios solicitados por los revisores

---

## Áreas de Contribución

### 🎯 Alta Prioridad
- **Optimización GEPA**: Mejoras en algoritmos genéticos
- **Nuevos Tipos de Documentos**: Soporte para más dominios
- **Performance**: Optimizaciones de velocidad y memoria
- **Tests**: Cobertura de tests adicional

### 🔧 Media Prioridad
- **Interfaz Web**: Dashboard para análisis
- **API REST**: Endpoint HTTP para integración
- **Exportación**: Formatos adicionales (Excel, Word)
- **Monitoreo**: Métricas y observabilidad

### 📚 Baja Prioridad
- **Documentación**: Tutoriales y ejemplos
- **Localización**: Soporte para más idiomas
- **Integración**: Conectores para otras plataformas
- **UI/UX**: Mejoras en interfaz de usuario

---

## Recursos de Desarrollo

### Arquitectura del Sistema
- **[Esquemas Dinámicos](docs/dynamic-schemas-architecture.md)** - Sistema adaptativo
- **[Sistema GEPA](docs/gepa-system-architecture.md)** - Optimización genética
- **[Catálogo de Archivos](docs/file-catalog.md)** - Mapa de dependencias

### Herramientas Útiles
- **Debugging**: `make job` con logs detallados
- **Análisis**: `docs/file-catalog.md` para entender dependencias
- **Testing**: `tests/` con ejemplos reales

---

## Contacto

### Maintainers
- **Karim Touma** - [@karimtouma](https://github.com/karimtouma)
- **Contributors** - Ver [contributors](https://github.com/karimtouma/estimate/graphs/contributors)

### Comunidad
- **Issues**: [GitHub Issues](https://github.com/karimtouma/estimate/issues)
- **Discussions**: [GitHub Discussions](https://github.com/karimtouma/estimate/discussions)

---

## Reconocimientos

Todas las contribuciones son valoradas y reconocidas. Los contribuyentes serán listados en el README y en los releases.

**¡Gracias por ayudar a mejorar PDF Estimator!**
