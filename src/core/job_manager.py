"""
Job management system for structural blueprint analysis.

This module provides comprehensive job configuration, scheduling, and execution
management for the advanced structural blueprint analysis system.
"""

import yaml
import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from ..models.blueprint_schemas import DocumentTaxonomy, ProcessingSession
from ..core.config import Config
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class JobStatus(str, Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobConfiguration:
    """Job configuration loaded from YAML."""
    job_id: str
    job_name: str
    description: str
    
    # Input configuration
    pdf_path: str
    pdf_type: str
    expected_pages: str
    
    # Analysis configuration
    analysis_mode: str
    enable_multi_agent: bool
    parallel_workers: int
    enabled_analyses: List[str]
    
    # Thresholds
    page_classification_threshold: float
    element_detection_threshold: float
    relationship_mapping_threshold: float
    
    # DSPy configuration
    enable_dspy_optimization: bool
    reasoning_depth: int
    chain_types: List[str]
    
    # Context configuration
    enable_cross_page: bool
    context_window_size: int
    memory_retention: int
    
    # Output configuration
    base_filename: str
    save_intermediate_results: bool
    export_formats: List[str]
    organize_by_date: bool
    create_summary_report: bool
    
    # Performance configuration
    max_processing_time: int
    memory_limit: str
    enable_caching: bool
    cache_duration: int
    
    # Notification configuration
    enable_progress_updates: bool
    enable_completion_notification: bool
    log_level: str
    
    # Runtime properties
    status: JobStatus = field(default=JobStatus.PENDING)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = field(default=None)
    completed_at: Optional[float] = field(default=None)
    error_message: Optional[str] = field(default=None)
    result_path: Optional[str] = field(default=None)


class JobConfigurationLoader:
    """
    Load and validate job configurations from YAML files.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.jobs_dir = Path("jobs")
        
    def load_job_config(self, job_file: Path) -> JobConfiguration:
        """
        Load job configuration from YAML file.
        
        Args:
            job_file: Path to job configuration YAML file
            
        Returns:
            JobConfiguration object
            
        Raises:
            ValueError: If job configuration is invalid
        """
        logger.info(f"Loading job configuration: {job_file}")
        
        try:
            with open(job_file, 'r', encoding='utf-8') as f:
                # Load YAML (may contain multiple documents)
                job_docs = list(yaml.safe_load_all(f))
            
            if not job_docs:
                raise ValueError("No job configuration found in file")
            
            # Use first job configuration
            job_data = job_docs[0].get("job_config", {})
            
            # Validate required fields
            self.validate_job_config(job_data)
            
            # Create JobConfiguration object
            job_config = JobConfiguration(
                job_id=job_data["job_id"],
                job_name=job_data["job_name"], 
                description=job_data["description"],
                
                # Input configuration
                pdf_path=job_data["input"]["pdf_path"],
                pdf_type=job_data["input"]["pdf_type"],
                expected_pages=job_data["input"].get("expected_pages", "1-100"),
                
                # Analysis configuration
                analysis_mode=job_data["analysis"]["mode"],
                enable_multi_agent=job_data["analysis"]["enable_multi_agent"],
                parallel_workers=job_data["analysis"]["parallel_workers"],
                enabled_analyses=job_data["analysis"]["enabled_analyses"],
                
                # Thresholds
                page_classification_threshold=job_data["analysis"]["thresholds"]["page_classification"],
                element_detection_threshold=job_data["analysis"]["thresholds"]["element_detection"],
                relationship_mapping_threshold=job_data["analysis"]["thresholds"]["relationship_mapping"],
                
                # DSPy configuration
                enable_dspy_optimization=job_data["analysis"]["dspy"]["enable_optimization"],
                reasoning_depth=job_data["analysis"]["dspy"]["reasoning_depth"],
                chain_types=job_data["analysis"]["dspy"]["chain_types"],
                
                # Context configuration
                enable_cross_page=job_data["analysis"]["context"]["enable_cross_page"],
                context_window_size=job_data["analysis"]["context"]["context_window_size"],
                memory_retention=job_data["analysis"]["context"]["memory_retention"],
                
                # Output configuration
                base_filename=job_data["output"]["base_filename"],
                save_intermediate_results=job_data["output"]["save_intermediate_results"],
                export_formats=job_data["output"]["export_formats"],
                organize_by_date=job_data["output"]["organize_by_date"],
                create_summary_report=job_data["output"]["create_summary_report"],
                
                # Performance configuration
                max_processing_time=job_data["performance"]["max_processing_time"],
                memory_limit=job_data["performance"]["memory_limit"],
                enable_caching=job_data["performance"]["enable_caching"],
                cache_duration=job_data["performance"]["cache_duration"],
                
                # Notification configuration
                enable_progress_updates=job_data["notifications"]["enable_progress_updates"],
                enable_completion_notification=job_data["notifications"]["enable_completion_notification"],
                log_level=job_data["notifications"]["log_level"]
            )
            
            logger.info(f"Job configuration loaded successfully: {job_config.job_id}")
            return job_config
            
        except Exception as e:
            logger.error(f"Failed to load job configuration: {e}")
            raise ValueError(f"Invalid job configuration: {e}") from e
    
    def validate_job_config(self, job_data: Dict[str, Any]) -> None:
        """Validate job configuration data."""
        required_fields = [
            "job_id", "job_name", "description", "input", "analysis", 
            "output", "performance", "notifications"
        ]
        
        for field in required_fields:
            if field not in job_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate input section
        input_config = job_data["input"]
        if "pdf_path" not in input_config:
            raise ValueError("Missing pdf_path in input configuration")
        
        # Validate PDF path exists
        pdf_path = Path(input_config["pdf_path"])
        if not pdf_path.exists() and not pdf_path.is_absolute():
            # Try relative to input directory
            input_dir = Path(self.config.get_directories()["input"])
            alt_path = input_dir / pdf_path.name
            if not alt_path.exists():
                raise ValueError(f"PDF file not found: {pdf_path}")
        
        # Validate analysis configuration
        analysis_config = job_data["analysis"]
        if "enabled_analyses" not in analysis_config:
            raise ValueError("Missing enabled_analyses in analysis configuration")
        
        # Validate parallel workers
        workers = analysis_config.get("parallel_workers", 1)
        if not 1 <= workers <= 16:
            raise ValueError("parallel_workers must be between 1 and 16")
    
    def list_available_jobs(self) -> List[Path]:
        """List available job configuration files."""
        if not self.jobs_dir.exists():
            return []
        
        return list(self.jobs_dir.glob("*.yml")) + list(self.jobs_dir.glob("*.yaml"))
    
    def create_default_job_config(self, pdf_path: str, job_name: str = "Default Analysis") -> JobConfiguration:
        """Create a default job configuration for quick analysis."""
        timestamp = int(time.time())
        
        return JobConfiguration(
            job_id=f"job_{timestamp}",
            job_name=job_name,
            description=f"Default analysis job for {Path(pdf_path).name}",
            
            # Input
            pdf_path=pdf_path,
            pdf_type="structural_blueprint",
            expected_pages="1-50",
            
            # Analysis
            analysis_mode="comprehensive",
            enable_multi_agent=True,
            parallel_workers=4,
            enabled_analyses=["page_taxonomy", "element_detection", "spatial_analysis"],
            
            # Thresholds
            page_classification_threshold=0.7,
            element_detection_threshold=0.6,
            relationship_mapping_threshold=0.5,
            
            # DSPy
            enable_dspy_optimization=True,
            reasoning_depth=3,
            chain_types=["chain_of_thought", "react"],
            
            # Context
            enable_cross_page=True,
            context_window_size=10,
            memory_retention=100,
            
            # Output
            base_filename=Path(pdf_path).stem + "_analysis",
            save_intermediate_results=True,
            export_formats=["json", "html"],
            organize_by_date=True,
            create_summary_report=True,
            
            # Performance
            max_processing_time=1800,
            memory_limit="4GB",
            enable_caching=True,
            cache_duration=3600,
            
            # Notifications
            enable_progress_updates=True,
            enable_completion_notification=True,
            log_level="INFO"
        )


class JobExecutor:
    """
    Execute analysis jobs based on configuration.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.job_loader = JobConfigurationLoader(config)
        self.active_jobs: Dict[str, JobConfiguration] = {}
        
    async def execute_job_from_file(self, job_file: Path) -> DocumentTaxonomy:
        """
        Execute analysis job from configuration file.
        
        Args:
            job_file: Path to job configuration YAML file
            
        Returns:
            DocumentTaxonomy result
        """
        logger.info(f"Executing job from file: {job_file}")
        
        # Load job configuration
        job_config = self.job_loader.load_job_config(job_file)
        
        return await self.execute_job(job_config)
    
    async def execute_job(self, job_config: JobConfiguration) -> DocumentTaxonomy:
        """
        Execute analysis job with given configuration.
        
        Args:
            job_config: Job configuration object
            
        Returns:
            DocumentTaxonomy result
        """
        logger.info(f"Executing job: {job_config.job_id}")
        
        # Update job status
        job_config.status = JobStatus.RUNNING
        job_config.started_at = time.time()
        self.active_jobs[job_config.job_id] = job_config
        
        try:
            # Validate PDF path
            pdf_path = Path(job_config.pdf_path)
            if not pdf_path.exists():
                # Try relative to input directory
                input_dir = Path(self.config.get_directories()["input"])
                pdf_path = input_dir / pdf_path.name
                
                if not pdf_path.exists():
                    raise FileNotFoundError(f"PDF file not found: {job_config.pdf_path}")
            
            # Configure analysis based on job settings
            analysis_config = self.create_analysis_config(job_config)
            
            # Import and initialize agent
            from ..agents.orchestrator import StructuralBlueprintAgent
            agent = StructuralBlueprintAgent(self.config)
            
            # Execute analysis
            logger.info(f"Starting analysis for: {pdf_path}")
            result = await agent.analyze_blueprint(pdf_path)
            
            # Process results according to job configuration
            result = await self.process_job_results(result, job_config)
            
            # Update job status
            job_config.status = JobStatus.COMPLETED
            job_config.completed_at = time.time()
            job_config.result_path = str(self.save_job_results(result, job_config))
            
            logger.info(f"Job completed successfully: {job_config.job_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Job execution failed: {e}")
            
            # Update job status
            job_config.status = JobStatus.FAILED
            job_config.error_message = str(e)
            job_config.completed_at = time.time()
            
            raise
        
        finally:
            # Clean up active jobs
            if job_config.job_id in self.active_jobs:
                del self.active_jobs[job_config.job_id]
    
    def create_analysis_config(self, job_config: JobConfiguration) -> Dict[str, Any]:
        """Create analysis configuration from job config."""
        return {
            "parallel_workers": job_config.parallel_workers,
            "enable_multi_agent": job_config.enable_multi_agent,
            "enabled_analyses": job_config.enabled_analyses,
            "thresholds": {
                "page_classification": job_config.page_classification_threshold,
                "element_detection": job_config.element_detection_threshold,
                "relationship_mapping": job_config.relationship_mapping_threshold
            },
            "dspy": {
                "enable_optimization": job_config.enable_dspy_optimization,
                "reasoning_depth": job_config.reasoning_depth,
                "chain_types": job_config.chain_types
            },
            "context": {
                "enable_cross_page": job_config.enable_cross_page,
                "window_size": job_config.context_window_size,
                "memory_retention": job_config.memory_retention
            },
            "performance": {
                "max_time": job_config.max_processing_time,
                "memory_limit": job_config.memory_limit,
                "enable_caching": job_config.enable_caching
            }
        }
    
    async def process_job_results(
        self, 
        result: DocumentTaxonomy, 
        job_config: JobConfiguration
    ) -> DocumentTaxonomy:
        """Process results according to job configuration."""
        
        # Add job metadata to result
        result.document_metadata.update({
            "job_id": job_config.job_id,
            "job_name": job_config.job_name,
            "analysis_mode": job_config.analysis_mode,
            "job_configuration": job_config.__dict__
        })
        
        # Apply job-specific processing
        if job_config.organize_by_date:
            result.document_metadata["processing_date"] = time.strftime("%Y-%m-%d")
        
        # Filter results based on thresholds
        result = self.apply_confidence_thresholds(result, job_config)
        
        return result
    
    def apply_confidence_thresholds(
        self, 
        result: DocumentTaxonomy, 
        job_config: JobConfiguration
    ) -> DocumentTaxonomy:
        """Apply confidence thresholds to filter results."""
        
        # Filter page taxonomies by confidence
        filtered_pages = []
        for page in result.page_taxonomies:
            if page.analysis_confidence >= job_config.page_classification_threshold:
                # Filter elements by confidence
                filtered_elements = [
                    elem for elem in page.structural_elements
                    if elem.confidence >= job_config.element_detection_threshold
                ]
                
                # Update page with filtered elements
                page.structural_elements = filtered_elements
                filtered_pages.append(page)
            else:
                logger.warning(f"Page {page.page_number} filtered due to low confidence: {page.analysis_confidence}")
        
        result.page_taxonomies = filtered_pages
        
        # Recalculate overall metrics
        if filtered_pages:
            result.overall_confidence = sum(p.analysis_confidence for p in filtered_pages) / len(filtered_pages)
            result.total_pages = len(filtered_pages)
        
        return result
    
    def save_job_results(self, result: DocumentTaxonomy, job_config: JobConfiguration) -> Path:
        """Save job results according to configuration."""
        output_dir = Path(self.config.get_directories()["output"])
        
        # Create date-organized directory if requested
        if job_config.organize_by_date:
            date_dir = output_dir / time.strftime("%Y-%m-%d")
            date_dir.mkdir(exist_ok=True)
            output_dir = date_dir
        
        # Create job-specific directory
        job_dir = output_dir / job_config.job_id
        job_dir.mkdir(exist_ok=True)
        
        # Save main results
        main_result_path = job_dir / f"{job_config.base_filename}.json"
        
        with open(main_result_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(result.model_dump(), f, indent=2, ensure_ascii=False, default=str)
        
        # Save intermediate results if requested
        if job_config.save_intermediate_results:
            pages_dir = job_dir / "pages"
            pages_dir.mkdir(exist_ok=True)
            
            for page in result.page_taxonomies:
                page_file = pages_dir / f"page_{page.page_number:03d}_analysis.json"
                with open(page_file, 'w', encoding='utf-8') as f:
                    json.dump(page.model_dump(), f, indent=2, ensure_ascii=False, default=str)
        
        # Export in additional formats
        for export_format in job_config.export_formats:
            if export_format == "html":
                self.export_html_report(result, job_dir / f"{job_config.base_filename}.html")
            elif export_format == "markdown":
                self.export_markdown_report(result, job_dir / f"{job_config.base_filename}.md")
        
        # Create summary report if requested
        if job_config.create_summary_report:
            self.create_summary_report(result, job_config, job_dir / "summary_report.txt")
        
        logger.info(f"Job results saved to: {job_dir}")
        return main_result_path
    
    def export_html_report(self, result: DocumentTaxonomy, output_path: Path):
        """Export results as HTML report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Structural Blueprint Analysis - {result.document_name}</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
                .metric {{ display: inline-block; margin: 10px 20px 10px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #007bff; }}
                .page-analysis {{ border: 1px solid #dee2e6; margin: 20px 0; padding: 20px; border-radius: 8px; background: #ffffff; }}
                .element {{ background: #e9ecef; margin: 8px 0; padding: 12px; border-radius: 6px; border-left: 3px solid #28a745; }}
                .confidence {{ font-weight: bold; color: #28a745; }}
                .page-header {{ color: #495057; border-bottom: 2px solid #dee2e6; padding-bottom: 10px; margin-bottom: 15px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🏗️ Structural Blueprint Analysis Report</h1>
                    <p><strong>Document:</strong> {result.document_name}</p>
                    <p><strong>Analysis Date:</strong> {time.ctime(result.analysis_timestamp)}</p>
                </div>
                
                <div class="metrics-section">
                    <div class="metric">
                        <h3>📊 Total Pages</h3>
                        <p style="font-size: 2em; margin: 0; color: #007bff;">{result.total_pages}</p>
                    </div>
                    <div class="metric">
                        <h3>⏱️ Processing Time</h3>
                        <p style="font-size: 1.5em; margin: 0; color: #28a745;">{result.total_processing_time:.1f}s</p>
                    </div>
                    <div class="metric">
                        <h3>🎯 Confidence</h3>
                        <p style="font-size: 1.5em; margin: 0; color: #ffc107;">{result.overall_confidence:.2f}</p>
                    </div>
                    <div class="metric">
                        <h3>🏗️ Elements</h3>
                        <p style="font-size: 1.5em; margin: 0; color: #dc3545;">{sum(result.elements_by_type.values())}</p>
                    </div>
                </div>
                
                <h2>📋 Page Analysis Results</h2>
        """
        
        for page in result.page_taxonomies:
            html_content += f"""
                <div class="page-analysis">
                    <h3 class="page-header">Page {page.page_number}: {page.page_type.value.replace('_', ' ').title()}</h3>
                    <p><strong>Classification:</strong> {page.primary_category}</p>
                    <p><strong>Confidence:</strong> <span class="confidence">{page.analysis_confidence:.2f}</span></p>
                    <p><strong>Complexity:</strong> {page.complexity_level}</p>
                    <p><strong>Elements Detected:</strong> {len(page.structural_elements)}</p>
                    
                    <h4>Structural Elements:</h4>
            """
            
            for element in page.structural_elements[:10]:  # Show top 10
                html_content += f"""
                    <div class="element">
                        <strong>{element.element_type.value.replace('_', ' ').title()}:</strong> {element.description}
                        <span style="float: right; color: #6c757d;">Confidence: {element.confidence:.2f}</span>
                    </div>
                """
            
            html_content += "</div>"
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report exported: {output_path}")
    
    def export_markdown_report(self, result: DocumentTaxonomy, output_path: Path):
        """Export results as Markdown report."""
        md_content = f"""# 🏗️ Structural Blueprint Analysis Report

**Document:** {result.document_name}
**Analysis Date:** {time.ctime(result.analysis_timestamp)}
**Job ID:** {result.document_metadata.get('job_id', 'N/A')}

## 📊 Summary Metrics

| Metric | Value |
|--------|-------|
| **Total Pages** | {result.total_pages} |
| **Processing Time** | {result.total_processing_time:.2f} seconds |
| **Overall Confidence** | {result.overall_confidence:.2f} |
| **Completeness Score** | {result.completeness_score:.2f} |
| **Consistency Score** | {result.consistency_score:.2f} |
| **Total Elements** | {sum(result.elements_by_type.values())} |

## 📋 Page Types Distribution

"""
        
        for page_type, count in result.pages_by_type.items():
            percentage = (count / result.total_pages) * 100
            md_content += f"- **{page_type.replace('_', ' ').title()}:** {count} pages ({percentage:.1f}%)\n"
        
        md_content += "\n## 🏗️ Structural Elements Summary\n\n"
        
        for element_type, count in sorted(result.elements_by_type.items(), key=lambda x: x[1], reverse=True)[:15]:
            md_content += f"- **{element_type.replace('_', ' ').title()}:** {count} detected\n"
        
        if result.key_insights:
            md_content += "\n## 💡 Key Insights\n\n"
            for insight in result.key_insights:
                md_content += f"- {insight}\n"
        
        md_content += "\n## 📄 Page-by-Page Analysis\n\n"
        
        for page in result.page_taxonomies:
            md_content += f"""### Page {page.page_number}: {page.page_type.value.replace('_', ' ').title()}

- **Classification:** {page.primary_category}
- **Confidence:** {page.analysis_confidence:.2f}
- **Complexity:** {page.complexity_level}
- **Technical Level:** {page.technical_level}
- **Elements Detected:** {len(page.structural_elements)}

"""
            
            if page.structural_elements:
                md_content += "**Structural Elements:**\n"
                for element in page.structural_elements[:8]:
                    md_content += f"- {element.element_type.value}: {element.description} (conf: {element.confidence:.2f})\n"
                md_content += "\n"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"Markdown report exported: {output_path}")
    
    def create_summary_report(self, result: DocumentTaxonomy, job_config: JobConfiguration, output_path: Path):
        """Create executive summary report."""
        summary = f"""
STRUCTURAL BLUEPRINT ANALYSIS SUMMARY
=====================================

Job: {job_config.job_name} ({job_config.job_id})
Document: {result.document_name}
Analysis Date: {time.ctime(result.analysis_timestamp)}

EXECUTIVE SUMMARY
-----------------
• Total Pages Analyzed: {result.total_pages}
• Processing Time: {result.total_processing_time:.2f} seconds
• Analysis Speed: {result.pages_per_second:.2f} pages/second
• Overall Confidence: {result.overall_confidence:.2f}
• Analysis Completeness: {result.completeness_score:.2f}

DOCUMENT CHARACTERISTICS
------------------------
"""
        
        if result.structural_summary:
            summary += f"""• Building Type: {result.structural_summary.building_type or 'Not determined'}
• Structural System: {result.structural_summary.structural_system or 'Not determined'}
• Construction Type: {result.structural_summary.construction_type or 'Not determined'}
• Design Style: {result.structural_summary.design_style or 'Not determined'}
• Complexity Assessment: {result.structural_summary.complexity_assessment}
"""
        
        summary += f"""
PAGE TYPE BREAKDOWN
-------------------
"""
        
        for page_type, count in result.pages_by_type.items():
            percentage = (count / result.total_pages) * 100
            summary += f"• {page_type.replace('_', ' ').title()}: {count} pages ({percentage:.1f}%)\n"
        
        summary += f"""
STRUCTURAL ELEMENTS DETECTED
-----------------------------
Total Elements: {sum(result.elements_by_type.values())}

Top Element Types:
"""
        
        for element_type, count in sorted(result.elements_by_type.items(), key=lambda x: x[1], reverse=True)[:10]:
            summary += f"• {element_type.replace('_', ' ').title()}: {count}\n"
        
        if result.key_insights:
            summary += f"""
KEY INSIGHTS
------------
"""
            for insight in result.key_insights:
                summary += f"• {insight}\n"
        
        summary += f"""
TECHNICAL DETAILS
-----------------
• API Calls Made: {result.api_calls_made}
• Global Patterns Identified: {len(result.global_patterns)}
• Cross-page Consistency: {result.consistency_score:.2f}

ANALYSIS CONFIGURATION
----------------------
• Analysis Mode: {job_config.analysis_mode}
• Parallel Workers: {job_config.parallel_workers}
• DSPy Optimization: {'Enabled' if job_config.enable_dspy_optimization else 'Disabled'}
• Cross-page Analysis: {'Enabled' if job_config.enable_cross_page else 'Disabled'}

Generated by Structural Blueprint Agent v2.0.0
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        logger.info(f"Summary report created: {output_path}")
    
    def get_job_status(self, job_id: str) -> Optional[JobConfiguration]:
        """Get status of active or completed job."""
        return self.active_jobs.get(job_id)
    
    def list_active_jobs(self) -> List[JobConfiguration]:
        """List all active jobs."""
        return list(self.active_jobs.values())
    
    async def execute_default_job(self, pdf_path: str) -> DocumentTaxonomy:
        """Execute a default job configuration for quick analysis."""
        logger.info(f"Executing default job for: {pdf_path}")
        
        # Create default job configuration
        job_config = self.job_loader.create_default_job_config(pdf_path)
        
        # Execute job
        return await self.execute_job(job_config)
