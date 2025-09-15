# PDF Estimator
## Sistema Autónomo de Análisis de Documentos Técnicos

[![Licencia: BSD-2-Clause](https://img.shields.io/badge/Licencia-BSD--2--Clause-blue.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![Gemini AI](https://img.shields.io/badge/IA-Gemini%202.5%20Pro-orange.svg)](https://ai.google.dev/)
[![Versión](https://img.shields.io/badge/versión-v2.0.0-blue.svg)](https://github.com/karimtouma/estimate)

PDF Estimator es un sistema autónomo para análisis inteligente de documentos técnicos que utiliza esquemas dinámicos y optimización genética (GEPA) para adaptarse automáticamente a cualquier tipo de documento sin configuración previa.

---

## Características Principales

### Autonomía Completa
- **Esquemas Dinámicos**: Adaptación automática a cualquier tipo de documento técnico
- **Descubrimiento Inteligente**: Identificación de patrones sin configuración previa
- **Clasificación Adaptativa**: Registro automático de nuevos tipos de elementos
- **Operación Sin Configuración**: Funcionamiento inmediato sin taxonomías predefinidas

### GEPA Optimization System
- **Múltiples Candidatos**: 5 opciones de clasificación por elemento
- **Juez Inteligente**: Evaluación técnica (99.7% judge score)
- **Consenso Automático**: Análisis de acuerdo entre candidatos (97.5%)
- **Evolución Genética**: Mejora continua mediante algoritmos evolutivos

### Language Router
- **Detección Automática**: Identificación del idioma principal
- **Optimización Adaptativa**: Ajuste de prompts por idioma
- **Soporte Multiidioma**: Documentos técnicos en múltiples idiomas
- **Configuración Flexible**: Idioma de salida configurable

### Rendimiento Empresarial
- **Tiempo**: 11-13 minutos (documentos de 51 páginas)
- **Costo**: $0.08 USD por análisis completo
- **Precisión**: 95-100% de elementos identificados
- **Eficiencia**: 46-54% reutilización de tokens

---

## Inicio Rápido

```bash
# Instalación
git clone https://github.com/karimtouma/estimate.git
cd estimate
make setup

# Configuración
echo "GEMINI_API_KEY=tu_clave_api" > .env

# Análisis
cp tu_documento.pdf input/file.pdf
make job

# Resultados
cat output/file_comprehensive_analysis.json | jq '.dynamic_schema_results'
```

### Comandos Disponibles

| Comando | Descripción | Tiempo |
|---------|-------------|--------|
| `make job` | Análisis completo autónomo | 11-13 min |
| `make job-quick` | Análisis rápido | 2-3 min |
| `make status` | Verificar configuración | <1 seg |
| `make results` | Ver últimos resultados | <1 seg |

---

## Arquitectura del Sistema

### Componentes Principales

**Discovery Engine**: Análisis estratégico de muestras documentales (30% de cobertura) para identificación de patrones estructurales.

**GEPA Classification**: Optimización genética que genera múltiples candidatos por elemento con evaluación por juez inteligente.

**Language Router**: Detección automática de idioma con optimización adaptativa de prompts.

**Intelligent Classifier**: Cuatro estrategias complementarias usando exclusivamente reasoning de IA.

**Auto-Registry**: Registro automático con evolución continua basada en evidencia.

### Flujo de Procesamiento

1. **Upload**: Subida del PDF a Gemini API
2. **Discovery**: Análisis de muestras con esquemas dinámicos
3. **Language Detection**: Identificación de idioma y optimización
4. **GEPA Classification**: Múltiples candidatos y evaluación por juez
5. **Core Analysis**: Análisis general, secciones y datos
6. **Q&A Adaptativo**: Preguntas contextuales automáticas
7. **Page Mapping**: Clasificación completa de páginas
8. **Results**: Compilación con métricas detalladas

---

## Casos de Uso

### Documentos de Construcción
- Planos arquitectónicos y estructurales
- Especificaciones técnicas MEP
- Análisis de códigos y cumplimiento normativo

### Documentos de Ingeniería
- Diagramas técnicos y esquemáticos
- Especificaciones de equipos
- Manuales técnicos y reportes

### Análisis Multiidioma
- Documentos en español, inglés o mixtos
- Optimización automática por idioma
- Preservación de terminología técnica

---

## Configuración

### config.toml Básico

```toml
[api]
output_language = "auto"
fallback_language = "spanish"

[analysis]
enable_dynamic_schemas = true
auto_register_confidence_threshold = 0.85

# GEPA optimization
enable_gepa_evolution = true
gepa_always_enhance = true
gepa_num_candidates = 5
```

### Variables de Entorno

```bash
GEMINI_API_KEY=tu_clave_api_gemini  # Requerido
LOG_LEVEL=INFO                      # Opcional
```

---

## Métricas de Rendimiento

| Métrica | Valor | Descripción |
|---------|--------|-------------|
| Tiempo | 11-13 min | Documentos de 51 páginas |
| Costo | $0.08 USD | Análisis completo |
| Precisión | 95-100% | Elementos identificados |
| Judge Score | 99.7% | Calidad GEPA |
| Consenso | 97.5% | Acuerdo entre candidatos |
| Cache Efficiency | 46-54% | Reutilización de tokens |

---

## Documentación Técnica

- **[Arquitectura de Esquemas Dinámicos](docs/dynamic-schemas-architecture.md)** - Sistema adaptativo
- **[Sistema GEPA](docs/gepa-system-architecture.md)** - Optimización genética
- **[API Reference](docs/api-reference.md)** - Métodos y configuración
- **[Troubleshooting](docs/troubleshooting-guide.md)** - Resolución de problemas

---

## Información del Proyecto

### Contribución Corporativa

PDF Estimator es una contribución de **Grupo DeAcero** a la comunidad de Inteligencia Artificial de México y Latinoamérica, con el objetivo de democratizar el acceso a tecnologías avanzadas de análisis de documentos técnicos.

### Licencia

Distribuido bajo licencia BSD-2-Clause. Permite uso comercial y modificación con atribución apropiada.

### Soporte

- **Repositorio**: [github.com/karimtouma/estimate](https://github.com/karimtouma/estimate)
- **Issues**: [GitHub Issues](https://github.com/karimtouma/estimate/issues)
- **Documentación**: Directorio `docs/` para referencia técnica
