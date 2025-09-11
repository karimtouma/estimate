"""
Advanced CLI for Structural Blueprint Agent.

Modern command-line interface with rich formatting and comprehensive
functionality for intelligent structural blueprint analysis.
"""

import asyncio
import sys
import click
from pathlib import Path
from typing import Optional
import json
import time
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.tree import Tree

from .core.config import get_config, Config
from .agents.orchestrator import StructuralBlueprintAgent
from .models.blueprint_schemas import DocumentTaxonomy, PageTaxonomy

console = Console()


def print_banner():
    """Print application banner with rich formatting."""
    banner = Panel.fit(
        """[bold blue]🏗️  STRUCTURAL BLUEPRINT AGENT v2.0.0[/bold blue]
[cyan]Advanced Multi-Agent Analysis with Gemini + DSPy + ADK[/cyan]
        
[dim]Intelligent taxonomy generation for structural blueprints[/dim]""",
        border_style="blue"
    )
    console.print(banner)


@click.group()
@click.version_option(version="2.0.0", prog_name="Structural Blueprint Agent")
@click.option(
    "--config", 
    "-c", 
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx: click.Context, config: Optional[Path], verbose: bool):
    """
    Structural Blueprint Agent - Advanced AI-powered blueprint analysis.
    
    Intelligent multi-agent system for comprehensive structural blueprint
    analysis using Gemini 2.5 Flash, DSPy reasoning, and Google ADK tools.
    """
    # Print banner
    print_banner()
    
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Load configuration
    try:
        if config:
            ctx.obj['config'] = Config(config)
        else:
            ctx.obj['config'] = get_config()
        
        # Set verbose logging if requested
        if verbose:
            ctx.obj['config'].processing.log_level = "DEBUG"
        
        ctx.obj['verbose'] = verbose
        
    except Exception as e:
        console.print(f"[red]❌ Configuration error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument("blueprint_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", 
    "-o", 
    type=click.Path(path_type=Path),
    help="Output file path (default: auto-generated)"
)
@click.option(
    "--parallel-workers",
    "-w",
    type=int,
    default=4,
    help="Number of parallel workers (default: 4)"
)
@click.option(
    "--save-intermediate",
    is_flag=True,
    help="Save intermediate results for each page"
)
@click.pass_context
def analyze_blueprint(
    ctx: click.Context,
    blueprint_path: Path,
    output: Optional[Path],
    parallel_workers: int,
    save_intermediate: bool
):
    """
    Analyze a structural blueprint with advanced AI agents.
    
    Performs comprehensive multi-modal analysis of structural blueprints
    including page-by-page taxonomy generation, element detection,
    and spatial relationship mapping.
    """
    config: Config = ctx.obj['config']
    verbose: bool = ctx.obj['verbose']
    
    console.print(f"\n[bold green]🔍 Analyzing Structural Blueprint[/bold green]")
    console.print(f"📄 File: [cyan]{blueprint_path}[/cyan]")
    console.print(f"⚙️  Workers: [yellow]{parallel_workers}[/yellow]")
    
    try:
        # Initialize agent
        agent = StructuralBlueprintAgent(config)
        
        # Create progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            expand=True
        ) as progress:
            
            # Main analysis task
            main_task = progress.add_task("🏗️ Analyzing blueprint...", total=100)
            
            # Execute analysis
            start_time = time.time()
            
            # Run async analysis
            result = asyncio.run(agent.analyze_blueprint(blueprint_path))
            
            progress.update(main_task, completed=100)
            
            duration = time.time() - start_time
        
        # Display results summary
        display_analysis_summary(result, duration)
        
        # Save results
        if not output:
            output = blueprint_path.stem + "_structural_analysis.json"
        
        save_path = Path(config.get_directories()['output']) / output
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(result.model_dump(), f, indent=2, ensure_ascii=False, default=str)
        
        console.print(f"\n[green]✅ Analysis completed successfully![/green]")
        console.print(f"💾 Results saved to: [cyan]{save_path}[/cyan]")
        
        # Save intermediate results if requested
        if save_intermediate:
            save_intermediate_results(result, save_path.parent)
            console.print(f"📁 Intermediate results saved to: [cyan]{save_path.parent / 'pages'}[/cyan]")
    
    except Exception as e:
        console.print(f"[red]❌ Analysis failed: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.argument("results_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--page-details",
    is_flag=True,
    help="Show detailed page-by-page analysis"
)
@click.option(
    "--export-format",
    type=click.Choice(["json", "html", "markdown"]),
    default="json",
    help="Export format for detailed report"
)
def view_results(results_path: Path, page_details: bool, export_format: str):
    """
    View and analyze structural blueprint analysis results.
    
    Provides comprehensive visualization and analysis of previously
    generated structural blueprint taxonomies.
    """
    console.print(f"\n[bold blue]📊 Viewing Analysis Results[/bold blue]")
    console.print(f"📄 Results file: [cyan]{results_path}[/cyan]")
    
    try:
        # Load results
        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Parse into DocumentTaxonomy
        result = DocumentTaxonomy(**data)
        
        # Display summary
        display_analysis_summary(result, result.total_processing_time)
        
        # Show page details if requested
        if page_details:
            display_page_details(result)
        
        # Export detailed report if requested
        if export_format != "json":
            export_detailed_report(result, results_path.parent, export_format)
    
    except Exception as e:
        console.print(f"[red]❌ Failed to view results: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument("results_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--page-number",
    "-p",
    type=int,
    help="Specific page to analyze (default: all pages)"
)
@click.option(
    "--element-type",
    "-e",
    help="Filter by element type"
)
def query_elements(results_path: Path, page_number: Optional[int], element_type: Optional[str]):
    """
    Query and filter structural elements from analysis results.
    
    Provides advanced querying capabilities for exploring detected
    structural elements and their relationships.
    """
    console.print(f"\n[bold green]🔍 Querying Structural Elements[/bold green]")
    
    try:
        # Load results
        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        result = DocumentTaxonomy(**data)
        
        # Filter pages
        pages_to_analyze = result.page_taxonomies
        if page_number:
            pages_to_analyze = [p for p in pages_to_analyze if p.page_number == page_number]
        
        # Collect elements
        all_elements = []
        for page in pages_to_analyze:
            for element in page.structural_elements:
                if not element_type or element.element_type.value == element_type:
                    all_elements.append((page.page_number, element))
        
        # Display results
        if all_elements:
            table = Table(title="Structural Elements Found")
            table.add_column("Page", style="cyan")
            table.add_column("Element Type", style="green")
            table.add_column("Description", style="white")
            table.add_column("Confidence", style="yellow")
            
            for page_num, element in all_elements:
                table.add_row(
                    str(page_num),
                    element.element_type.value,
                    element.description[:50] + "..." if len(element.description) > 50 else element.description,
                    f"{element.confidence:.2f}"
                )
            
            console.print(table)
            console.print(f"\n[green]Found {len(all_elements)} matching elements[/green]")
        else:
            console.print("[yellow]No matching elements found[/yellow]")
    
    except Exception as e:
        console.print(f"[red]❌ Query failed: {e}[/red]")
        sys.exit(1)


def display_analysis_summary(result: DocumentTaxonomy, duration: float):
    """Display comprehensive analysis summary."""
    
    # Main summary panel
    summary_text = f"""[bold]Document:[/bold] {result.document_name}
[bold]Pages Analyzed:[/bold] {result.total_pages}
[bold]Processing Time:[/bold] {duration:.2f} seconds ({result.pages_per_second:.1f} pages/sec)
[bold]Overall Confidence:[/bold] {result.overall_confidence:.2f}
[bold]Completeness Score:[/bold] {result.completeness_score:.2f}
[bold]Consistency Score:[/bold] {result.consistency_score:.2f}"""
    
    summary_panel = Panel(summary_text, title="📊 Analysis Summary", border_style="green")
    console.print(summary_panel)
    
    # Page types distribution
    page_types = result.pages_by_type
    if page_types:
        table = Table(title="📋 Page Types Distribution")
        table.add_column("Page Type", style="cyan")
        table.add_column("Count", style="green")
        table.add_column("Percentage", style="yellow")
        
        for page_type, count in page_types.items():
            percentage = (count / result.total_pages) * 100
            table.add_row(page_type.replace("_", " ").title(), str(count), f"{percentage:.1f}%")
        
        console.print(table)
    
    # Structural elements summary
    elements_by_type = result.elements_by_type
    if elements_by_type:
        table = Table(title="🏗️ Structural Elements Detected")
        table.add_column("Element Type", style="cyan")
        table.add_column("Total Count", style="green")
        table.add_column("Avg per Page", style="yellow")
        
        for element_type, count in sorted(elements_by_type.items(), key=lambda x: x[1], reverse=True)[:10]:
            avg_per_page = count / result.total_pages
            table.add_row(
                element_type.replace("_", " ").title(),
                str(count),
                f"{avg_per_page:.1f}"
            )
        
        console.print(table)
    
    # Key insights
    if result.key_insights:
        insights_text = "\n".join(f"• {insight}" for insight in result.key_insights[:5])
        insights_panel = Panel(insights_text, title="💡 Key Insights", border_style="yellow")
        console.print(insights_panel)
    
    # Structural summary
    if result.structural_summary:
        summary = result.structural_summary
        summary_items = []
        
        if summary.building_type:
            summary_items.append(f"[bold]Building Type:[/bold] {summary.building_type}")
        if summary.structural_system:
            summary_items.append(f"[bold]Structural System:[/bold] {summary.structural_system}")
        if summary.total_floors:
            summary_items.append(f"[bold]Floors:[/bold] {summary.total_floors}")
        if summary.construction_type:
            summary_items.append(f"[bold]Construction:[/bold] {summary.construction_type}")
        
        if summary_items:
            summary_text = "\n".join(summary_items)
            summary_panel = Panel(summary_text, title="🏢 Structural Summary", border_style="blue")
            console.print(summary_panel)


def display_page_details(result: DocumentTaxonomy):
    """Display detailed page-by-page analysis."""
    console.print(f"\n[bold blue]📄 Page-by-Page Analysis Details[/bold blue]")
    
    for page in result.page_taxonomies:
        # Page header
        page_title = f"Page {page.page_number}: {page.page_type.value.replace('_', ' ').title()}"
        
        # Page details
        details = []
        details.append(f"[bold]Classification:[/bold] {page.primary_category}")
        if page.secondary_category:
            details.append(f"[bold]Subtype:[/bold] {page.secondary_category}")
        
        details.append(f"[bold]Complexity:[/bold] {page.complexity_level}")
        details.append(f"[bold]Technical Level:[/bold] {page.technical_level}")
        details.append(f"[bold]Confidence:[/bold] {page.analysis_confidence:.2f}")
        details.append(f"[bold]Elements Detected:[/bold] {len(page.structural_elements)}")
        
        if page.purpose:
            details.append(f"[bold]Purpose:[/bold] {page.purpose}")
        
        details_text = "\n".join(details)
        
        # Create expandable panel
        page_panel = Panel(details_text, title=page_title, border_style="cyan", expand=False)
        console.print(page_panel)
        
        # Show elements if any
        if page.structural_elements and len(page.structural_elements) <= 10:
            element_tree = Tree(f"🏗️ Structural Elements ({len(page.structural_elements)})")
            
            for element in page.structural_elements:
                element_info = f"{element.element_type.value} (conf: {element.confidence:.2f})"
                if element.description:
                    element_info += f" - {element.description[:30]}"
                element_tree.add(element_info)
            
            console.print(element_tree)
        
        console.print()  # Add spacing


def save_intermediate_results(result: DocumentTaxonomy, output_dir: Path):
    """Save intermediate results for each page."""
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(exist_ok=True)
    
    for page in result.page_taxonomies:
        page_file = pages_dir / f"page_{page.page_number:03d}_analysis.json"
        
        with open(page_file, 'w', encoding='utf-8') as f:
            json.dump(page.model_dump(), f, indent=2, ensure_ascii=False, default=str)


def export_detailed_report(result: DocumentTaxonomy, output_dir: Path, format_type: str):
    """Export detailed report in specified format."""
    timestamp = int(time.time())
    
    if format_type == "html":
        report_file = output_dir / f"structural_report_{timestamp}.html"
        generate_html_report(result, report_file)
    elif format_type == "markdown":
        report_file = output_dir / f"structural_report_{timestamp}.md"
        generate_markdown_report(result, report_file)
    
    console.print(f"[green]📄 Detailed report exported: {report_file}[/green]")


def generate_html_report(result: DocumentTaxonomy, output_path: Path):
    """Generate HTML report."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Structural Blueprint Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background: #f0f8ff; padding: 20px; border-radius: 8px; }}
            .section {{ margin: 20px 0; }}
            .page-analysis {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; }}
            .element {{ background: #f9f9f9; margin: 5px 0; padding: 8px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🏗️ Structural Blueprint Analysis Report</h1>
            <p><strong>Document:</strong> {result.document_name}</p>
            <p><strong>Analysis Date:</strong> {time.ctime(result.analysis_timestamp)}</p>
            <p><strong>Total Pages:</strong> {result.total_pages}</p>
            <p><strong>Overall Confidence:</strong> {result.overall_confidence:.2f}</p>
        </div>
        
        <div class="section">
            <h2>📊 Summary Statistics</h2>
            <p><strong>Processing Time:</strong> {result.total_processing_time:.2f} seconds</p>
            <p><strong>Pages per Second:</strong> {result.pages_per_second:.2f}</p>
            <p><strong>Total Elements Detected:</strong> {sum(result.elements_by_type.values())}</p>
        </div>
        
        <div class="section">
            <h2>📋 Page Analysis</h2>
    """
    
    for page in result.page_taxonomies:
        html_content += f"""
            <div class="page-analysis">
                <h3>Page {page.page_number}: {page.page_type.value.replace('_', ' ').title()}</h3>
                <p><strong>Classification:</strong> {page.primary_category}</p>
                <p><strong>Confidence:</strong> {page.analysis_confidence:.2f}</p>
                <p><strong>Elements:</strong> {len(page.structural_elements)}</p>
                
                <h4>Detected Elements:</h4>
        """
        
        for element in page.structural_elements[:5]:  # Show first 5 elements
            html_content += f"""
                <div class="element">
                    <strong>{element.element_type.value}:</strong> {element.description}
                    (Confidence: {element.confidence:.2f})
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


def generate_markdown_report(result: DocumentTaxonomy, output_path: Path):
    """Generate Markdown report."""
    md_content = f"""# 🏗️ Structural Blueprint Analysis Report

**Document:** {result.document_name}
**Analysis Date:** {time.ctime(result.analysis_timestamp)}
**Total Pages:** {result.total_pages}
**Overall Confidence:** {result.overall_confidence:.2f}

## 📊 Summary Statistics

- **Processing Time:** {result.total_processing_time:.2f} seconds
- **Pages per Second:** {result.pages_per_second:.2f}
- **Total Elements Detected:** {sum(result.elements_by_type.values())}
- **Completeness Score:** {result.completeness_score:.2f}
- **Consistency Score:** {result.consistency_score:.2f}

## 📋 Page Type Distribution

"""
    
    for page_type, count in result.pages_by_type.items():
        percentage = (count / result.total_pages) * 100
        md_content += f"- **{page_type.replace('_', ' ').title()}:** {count} pages ({percentage:.1f}%)\n"
    
    md_content += "\n## 🏗️ Structural Elements Summary\n\n"
    
    for element_type, count in sorted(result.elements_by_type.items(), key=lambda x: x[1], reverse=True)[:10]:
        md_content += f"- **{element_type.replace('_', ' ').title()}:** {count} detected\n"
    
    md_content += "\n## 💡 Key Insights\n\n"
    
    for insight in result.key_insights:
        md_content += f"- {insight}\n"
    
    md_content += "\n## 📄 Page-by-Page Analysis\n\n"
    
    for page in result.page_taxonomies:
        md_content += f"""### Page {page.page_number}: {page.page_type.value.replace('_', ' ').title()}

- **Classification:** {page.primary_category}
- **Confidence:** {page.analysis_confidence:.2f}
- **Complexity:** {page.complexity_level}
- **Elements Detected:** {len(page.structural_elements)}

"""
        
        if page.structural_elements:
            md_content += "**Structural Elements:**\n"
            for element in page.structural_elements[:5]:
                md_content += f"- {element.element_type.value}: {element.description} (conf: {element.confidence:.2f})\n"
            md_content += "\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)


@cli.command()
@click.pass_context
def system_status(ctx: click.Context):
    """
    Show system status and configuration.
    
    Displays comprehensive system information including configuration,
    available tools, and performance metrics.
    """
    config: Config = ctx.obj['config']
    
    console.print(f"\n[bold blue]🔧 System Status[/bold blue]")
    
    # Configuration status
    config_table = Table(title="Configuration Status")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")
    config_table.add_column("Status", style="yellow")
    
    # Check API key
    api_key_status = "✅ Configured" if config.api.gemini_api_key and config.api.gemini_api_key != "your_api_key_here" else "❌ Not configured"
    config_table.add_row("Gemini API Key", "***", api_key_status)
    
    config_table.add_row("Model", config.api.default_model, "✅ Set")
    config_table.add_row("Log Level", config.processing.log_level, "✅ Set")
    config_table.add_row("Container Mode", "Yes" if config.is_container_environment() else "No", "✅ Detected")
    
    console.print(config_table)
    
    # Directory status
    directories = config.get_directories()
    dir_table = Table(title="Directory Status")
    dir_table.add_column("Directory", style="cyan")
    dir_table.add_column("Path", style="white")
    dir_table.add_column("Status", style="green")
    
    for name, path in directories.items():
        status = "✅ Exists" if Path(path).exists() else "❌ Missing"
        dir_table.add_row(name.title(), path, status)
    
    console.print(dir_table)
    
    # Tool availability
    tool_table = Table(title="Advanced Tools Status")
    tool_table.add_column("Tool", style="cyan")
    tool_table.add_column("Status", style="green")
    tool_table.add_column("Version", style="yellow")
    
    # Check tool availability
    tools_status = check_tool_availability()
    
    for tool, status in tools_status.items():
        tool_table.add_row(tool, status["status"], status.get("version", "N/A"))
    
    console.print(tool_table)


def check_tool_availability() -> Dict[str, Dict[str, str]]:
    """Check availability of advanced tools."""
    tools = {}
    
    # Check DSPy
    try:
        import dspy
        tools["DSPy Framework"] = {"status": "✅ Available", "version": getattr(dspy, '__version__', 'unknown')}
    except ImportError:
        tools["DSPy Framework"] = {"status": "❌ Not installed", "version": "N/A"}
    
    # Check PyMuPDF
    try:
        import fitz
        tools["PyMuPDF"] = {"status": "✅ Available", "version": fitz.version[0]}
    except ImportError:
        tools["PyMuPDF"] = {"status": "❌ Not installed", "version": "N/A"}
    
    # Check OpenCV
    try:
        import cv2
        tools["OpenCV"] = {"status": "✅ Available", "version": cv2.__version__}
    except ImportError:
        tools["OpenCV"] = {"status": "❌ Not installed", "version": "N/A"}
    
    # Check ChromaDB
    try:
        import chromadb
        tools["ChromaDB"] = {"status": "✅ Available", "version": getattr(chromadb, '__version__', 'unknown')}
    except ImportError:
        tools["ChromaDB"] = {"status": "❌ Not installed", "version": "N/A"}
    
    return tools


def main():
    """Main entry point for the advanced CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]❌ Unexpected error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
