#!/usr/bin/env python3
"""
Execute structural analysis job from YAML configuration.
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config import get_config
from src.core.job_manager import JobExecutor, JobConfigurationLoader


async def run_job():
    """Execute the structural analysis job."""
    print("🏗️ EJECUTANDO JOB DE ANÁLISIS ESTRUCTURAL")
    print("=" * 60)
    
    try:
        # Load configuration
        config = get_config()
        print("✅ Configuración cargada")
        
        # Initialize job system
        job_loader = JobConfigurationLoader(config)
        job_executor = JobExecutor(config)
        print("✅ Sistema de jobs inicializado")
        
        # Find available jobs
        available_jobs = job_loader.list_available_jobs()
        print(f"📋 Jobs encontrados: {len(available_jobs)}")
        
        if not available_jobs:
            print("❌ No se encontraron archivos de configuración de jobs en /jobs")
            return False
        
        # Use first available job
        job_file = available_jobs[0]
        print(f"🚀 Ejecutando: {job_file.name}")
        
        # Load and validate job configuration
        job_config = job_loader.load_job_config(job_file)
        print(f"📄 PDF: {job_config.pdf_path}")
        print(f"📊 Modo: {job_config.analysis_mode}")
        print(f"🔧 Workers: {job_config.parallel_workers}")
        print(f"📋 Análisis: {', '.join(job_config.enabled_analyses)}")
        
        print("\n🎯 INICIANDO ANÁLISIS COMPLETO...")
        print("⏱️  Esto puede tomar varios minutos...")
        
        # Execute the job
        result = await job_executor.execute_job_from_file(job_file)
        
        print(f"\n✅ ANÁLISIS COMPLETADO!")
        print(f"📁 Resultados guardados en: {job_config.result_path or 'output/'}")
        print(f"🎉 Job ID: {job_config.job_id}")
        
        return True
        
    except FileNotFoundError as e:
        print(f"❌ Archivo no encontrado: {e}")
        print("💡 Asegúrate de que el archivo PDF esté en la carpeta input/")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(run_job())
    sys.exit(0 if success else 1)
