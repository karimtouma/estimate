# Guía de Uso y Ejemplos

## Tabla de contenidos
- [Casos de uso comunes](#casos-de-uso-comunes)
- [Ejemplos de configuración](#ejemplos-de-configuración)
- [Análisis de resultados](#análisis-de-resultados)
- [Integración con pipelines](#integración-con-pipelines)
- [Optimización de rendimiento](#optimización-de-rendimiento)
- [Troubleshooting avanzado](#troubleshooting-avanzado)

---

## Casos de uso comunes

### 1. Análisis de planos arquitectónicos

```bash
# Configuración optimizada para planos
cat > config_planos.toml << EOF
[analysis]
enable_dynamic_schemas = true
auto_register_confidence_threshold = 0.90
enable_gepa_evolution = true
gepa_num_candidates = 7  # Más candidatos para mayor precisión

[discovery]
sample_percentage = 0.40  # Mayor cobertura para planos complejos
EOF

# Ejecutar con configuración específica
docker run --rm \
  --env-file .env \
  -v "$PWD/input:/app/input" \
  -v "$PWD/output:/app/output" \
  -v "$PWD/config_planos.toml:/app/config.toml:ro" \
  pdf-estimator:local make job
```

### 2. Procesamiento batch de múltiples documentos

```bash
#!/bin/bash
# batch_process.sh

for pdf in input/*.pdf; do
  filename=$(basename "$pdf")
  echo "Procesando: $filename"
  
  # Copiar a file.pdf (entrada esperada)
  cp "$pdf" input/file.pdf
  
  # Ejecutar análisis
  make job
  
  # Renombrar salida
  mv output/file_comprehensive_analysis.json \
     "output/${filename%.pdf}_analysis.json"
done
```

### 3. Análisis rápido para preview

```bash
# Configuración para análisis rápido
cat > config_quick.toml << EOF
[analysis]
enable_dynamic_schemas = false  # Usar esquemas base
enable_gepa_evolution = false    # Sin optimización GEPA
max_pages_to_process = 10        # Limitar páginas

[discovery]
sample_percentage = 0.15  # Muestreo mínimo
EOF

make job-quick
```

---

## Ejemplos de configuración

### Configuración para documentos en inglés

```toml
[api]
output_language = "english"
fallback_language = "english"

[language]
force_language = "english"  # Forzar idioma sin detección
optimize_prompts = true
```

### Configuración para máxima precisión

```toml
[analysis]
enable_dynamic_schemas = true
auto_register_confidence_threshold = 0.95  # Umbral alto
enable_gepa_evolution = true
gepa_always_enhance = true
gepa_num_candidates = 10  # Máximo de candidatos

[discovery]
sample_percentage = 0.50  # Cobertura extensa
enable_pattern_analysis = true
enable_nomenclature_parsing = true

[validation]
require_consensus = true
min_consensus_score = 0.90
```

### Configuración para minimizar costos

```toml
[analysis]
enable_dynamic_schemas = false
enable_gepa_evolution = false
gepa_num_candidates = 3  # Mínimo de candidatos

[discovery]
sample_percentage = 0.20  # Muestreo reducido

[cache]
enable_caching = true
cache_ttl_hours = 24
```

---

## Análisis de resultados

### Extracción de métricas clave con jq

```bash
# Crear reporte resumido
jq -r '
  "=== REPORTE DE ANÁLISIS ===\n" +
  "Documento: \(.metadata.filename)\n" +
  "Páginas: \(.metadata.total_pages)\n" +
  "Tiempo: \(.metrics.total_processing_time)s\n" +
  "Costo: $\(.api_statistics.estimated_cost_usd)\n" +
  "\n=== TIPOS DESCUBIERTOS ===\n" +
  (.dynamic_schema_results.discovered_element_types | 
   map("- \(.type_name): \(.confidence)%") | 
   join("\n")) +
  "\n\n=== MÉTRICAS GEPA ===\n" +
  "Judge Score: \(.dynamic_schema_results.gepa_statistics.average_judge_score)%\n" +
  "Consenso: \(.dynamic_schema_results.gepa_statistics.average_consensus)%\n" +
  "Mejoras: \(.dynamic_schema_results.gepa_statistics.total_enhancements)"
' output/file_comprehensive_analysis.json
```

### Conversión a CSV para análisis

```bash
# Exportar elementos a CSV
jq -r '
  ["Página", "Tipo", "Confianza", "Texto"] as $headers |
  $headers,
  (.page_mapping[] | 
   .page_number as $page |
   .elements[] |
   [$page, .element_type, .confidence, .text_content[:50]] |
   @csv)
' output/file_comprehensive_analysis.json > elementos.csv
```

### Generación de reporte HTML

```python
#!/usr/bin/env python3
# generate_report.py

import json
from pathlib import Path

def generate_html_report(json_path, output_path):
    with open(json_path) as f:
        data = json.load(f)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Análisis PDF - {data['metadata']['filename']}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .metric {{ background: #f0f0f0; padding: 10px; margin: 10px 0; }}
            .element {{ border-left: 3px solid #007bff; padding-left: 10px; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>Reporte de Análisis</h1>
        <div class="metric">
            <h2>Métricas Generales</h2>
            <p>Páginas: {data['metadata']['total_pages']}</p>
            <p>Tiempo: {data['metrics']['total_processing_time']}s</p>
            <p>Precisión GEPA: {data['dynamic_schema_results']['gepa_statistics']['average_judge_score']}%</p>
        </div>
        
        <h2>Elementos Descubiertos</h2>
        {"".join(f'<div class="element"><strong>{e["type_name"]}</strong>: {e["confidence"]}%</div>' 
                 for e in data['dynamic_schema_results']['discovered_element_types'])}
    </body>
    </html>
    """
    
    Path(output_path).write_text(html)
    print(f"Reporte generado: {output_path}")

if __name__ == "__main__":
    generate_html_report(
        "output/file_comprehensive_analysis.json",
        "output/report.html"
    )
```

---

## Integración con pipelines

### GitHub Actions workflow

```yaml
# .github/workflows/document-analysis.yml
name: Análisis de Documentos

on:
  workflow_dispatch:
    inputs:
      document_url:
        description: 'URL del documento PDF'
        required: true

jobs:
  analyze:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Download PDF
      run: |
        wget -O input/file.pdf "${{ github.event.inputs.document_url }}"
    
    - name: Run analysis
      env:
        GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
      run: |
        docker build -t pdf-estimator .
        docker run --rm \
          -e GEMINI_API_KEY \
          -v "$PWD/input:/app/input" \
          -v "$PWD/output:/app/output" \
          pdf-estimator make job
    
    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: analysis-results
        path: output/
```

### Script de integración con S3

```bash
#!/bin/bash
# s3_integration.sh

# Descargar desde S3
aws s3 cp s3://bucket/documents/input.pdf input/file.pdf

# Procesar
make job

# Subir resultados
aws s3 cp output/file_comprehensive_analysis.json \
          s3://bucket/results/$(date +%Y%m%d_%H%M%S)_analysis.json

# Notificar vía SNS
aws sns publish \
  --topic-arn arn:aws:sns:region:account:topic \
  --message "Análisis completado: $(date)"
```

---

## Optimización de rendimiento

### 1. Procesamiento paralelo

```python
# parallel_processor.py
import asyncio
from pathlib import Path
import subprocess

async def process_document(pdf_path):
    """Procesa un documento de forma asíncrona"""
    proc = await asyncio.create_subprocess_exec(
        'docker', 'run', '--rm',
        '--env-file', '.env',
        '-v', f'{pdf_path}:/app/input/file.pdf',
        '-v', f'{Path.cwd()}/output:/app/output',
        'pdf-estimator:local', 'make', 'job',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    return pdf_path, proc.returncode, stdout

async def main():
    pdfs = list(Path('input').glob('*.pdf'))
    tasks = [process_document(pdf) for pdf in pdfs[:3]]  # Limitar concurrencia
    results = await asyncio.gather(*tasks)
    
    for pdf_path, returncode, output in results:
        print(f"{pdf_path}: {'OK' if returncode == 0 else 'ERROR'}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Caché de resultados

```python
# cache_manager.py
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta

class ResultCache:
    def __init__(self, cache_dir="cache", ttl_hours=24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
    
    def get_cache_key(self, pdf_path):
        """Genera key único basado en contenido del PDF"""
        with open(pdf_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def get(self, pdf_path):
        """Obtiene resultado cacheado si existe y es válido"""
        key = self.get_cache_key(pdf_path)
        cache_file = self.cache_dir / f"{key}.json"
        
        if cache_file.exists():
            data = json.loads(cache_file.read_text())
            cached_time = datetime.fromisoformat(data['cached_at'])
            
            if datetime.now() - cached_time < self.ttl:
                return data['result']
        
        return None
    
    def set(self, pdf_path, result):
        """Guarda resultado en caché"""
        key = self.get_cache_key(pdf_path)
        cache_file = self.cache_dir / f"{key}.json"
        
        cache_file.write_text(json.dumps({
            'cached_at': datetime.now().isoformat(),
            'result': result
        }))
```

---

## Troubleshooting avanzado

### Diagnóstico de errores comunes

```bash
# Script de diagnóstico
#!/bin/bash

echo "=== DIAGNÓSTICO PDF ESTIMATOR ==="

# 1. Verificar API key
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ GEMINI_API_KEY no configurada"
else
    echo "✅ GEMINI_API_KEY configurada"
fi

# 2. Verificar Docker
if docker --version > /dev/null 2>&1; then
    echo "✅ Docker instalado: $(docker --version)"
else
    echo "❌ Docker no encontrado"
fi

# 3. Verificar imagen
if docker images | grep -q pdf-estimator; then
    echo "✅ Imagen pdf-estimator encontrada"
else
    echo "❌ Imagen no construida. Ejecutar: docker build -t pdf-estimator ."
fi

# 4. Verificar archivos de entrada
if [ -f "input/file.pdf" ]; then
    echo "✅ Archivo de entrada presente"
    echo "   Tamaño: $(du -h input/file.pdf | cut -f1)"
else
    echo "❌ No hay archivo en input/file.pdf"
fi

# 5. Verificar permisos
if [ -w "output/" ]; then
    echo "✅ Directorio output/ escribible"
else
    echo "❌ Sin permisos de escritura en output/"
fi

# 6. Test rápido de API
echo "Probando conexión con Gemini API..."
python3 -c "
import os
import google.generativeai as genai
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
try:
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content('Test')
    print('✅ API funcionando correctamente')
except Exception as e:
    print(f'❌ Error de API: {e}')
"
```

### Logs detallados para debugging

```bash
# Habilitar logs verbosos
export LOG_LEVEL=DEBUG

# Ejecutar con logs completos
docker run --rm \
  --env-file .env \
  -e LOG_LEVEL=DEBUG \
  -v "$PWD/input:/app/input" \
  -v "$PWD/output:/app/output" \
  -v "$PWD/logs:/app/logs" \
  pdf-estimator:local make job 2>&1 | tee debug.log

# Analizar logs de error
grep -E "ERROR|CRITICAL|Exception" debug.log
```

### Monitoreo de recursos

```bash
# Monitor de uso durante ejecución
docker stats --no-stream pdf-estimator

# Límites de recursos
docker run --rm \
  --memory="2g" \
  --cpus="2" \
  --env-file .env \
  -v "$PWD/input:/app/input" \
  -v "$PWD/output:/app/output" \
  pdf-estimator:local make job
```

---

Ver también:
- [API Reference](api-reference.md) - Documentación completa de la API
- [Architecture](dynamic-schemas-architecture.md) - Arquitectura del sistema
- [Troubleshooting Guide](troubleshooting-guide.md) - Guía de resolución de problemas
