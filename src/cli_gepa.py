"""
GEPA optimization CLI for structural blueprint analysis.

Command-line interface for running GEPA prompt optimization and
managing optimized analysis workflows.
"""

import asyncio
import click
import sys
from pathlib import Path
from typing import Optional
import json
import time

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import get_config, Config
from src.optimization.gepa_optimizer import GEPAPromptOptimizer, run_gepa_optimization_pipeline
from src.agents.orchestrator import StructuralBlueprintAgent
from src.agents.taxonomy_engine import IntelligentTaxonomyEngine

console = Console()


@click.group()
@click.version_option(version="2.0.0", prog_name="GEPA Blueprint Optimizer")
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
    GEPA Blueprint Optimizer - Advanced prompt optimization for structural analysis.
    
    Optimize prompts using GEPA (Genetic-Pareto) reflective text evolution
    for enhanced structural blueprint analysis performance.
    """
    # Print banner
    banner = Panel.fit(
        """[bold blue]🧬 GEPA BLUEPRINT OPTIMIZER v2.0.0[/bold blue]
[cyan]Reflective Text Evolution for Structural Analysis[/cyan]

[green]🎯 Optimize prompts with GEPA + DSPy for enhanced accuracy[/green]""",
        border_style="blue"
    )
    console.print(banner)
    
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
@click.option(
    "--generations",
    "-g",
    type=int,
    default=10,
    help="Number of GEPA generations (default: 10)"
)
@click.option(
    "--population-size",
    "-p",
    type=int,
    default=8,
    help="Population size for evolution (default: 8)"
)
@click.option(
    "--save-results",
    is_flag=True,
    help="Save optimization results for future use"
)
@click.pass_context
def optimize_prompts(
    ctx: click.Context,
    generations: int,
    population_size: int,
    save_results: bool
):
    """
    Run GEPA optimization to evolve better prompts for structural blueprint analysis.
    
    Uses reflective text evolution to improve prompt accuracy and performance
    through genetic-pareto optimization with DSPy integration.
    """
    config: Config = ctx.obj['config']
    verbose: bool = ctx.obj['verbose']
    
    console.print(f"\n[bold green]🧬 GEPA Prompt Optimization[/bold green]")
    console.print("=" * 60)
    
    console.print(f"🔬 [cyan]Generations:[/cyan] {generations}")
    console.print(f"👥 [cyan]Population Size:[/cyan] {population_size}")
    console.print(f"🎯 [cyan]Target:[/cyan] Structural blueprint analysis prompts")
    
    try:
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
            
            # Main optimization task
            main_task = progress.add_task(
                "🧬 [bold]Running GEPA optimization...[/bold]", 
                total=100
            )
            
            # Subtasks
            init_task = progress.add_task("🔧 Initializing GEPA...", total=10)
            evolution_task = progress.add_task("🧠 Evolving prompts...", total=80)
            finalization_task = progress.add_task("✅ Finalizing results...", total=10)
            
            # Execute optimization
            start_time = time.time()
            
            # Initialize
            progress.update(init_task, completed=10)
            progress.update(main_task, completed=10)
            
            # Run GEPA optimization pipeline
            results = await run_gepa_optimization_pipeline(config)
            
            progress.update(evolution_task, completed=80)
            progress.update(main_task, completed=90)
            
            # Finalize
            if save_results:
                save_optimization_results(results, config)
            
            progress.update(finalization_task, completed=10)
            progress.update(main_task, completed=100)
        
        duration = time.time() - start_time
        
        # Display results
        display_optimization_results(results, duration)
        
        if save_results:
            output_dir = Path(config.get_directories()['output'])
            results_file = output_dir / "gepa_optimization_results.json"
            console.print(f"\n[green]💾 Results saved to: [cyan]{results_file}[/cyan][/green]")
    
    except Exception as e:
        console.print(f"[red]❌ GEPA optimization failed: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.argument("blueprint_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--use-optimization",
    "-o",
    type=click.Path(exists=True, path_type=Path),
    help="Path to GEPA optimization results file"
)
@click.option(
    "--compare-baseline",
    is_flag=True,
    help="Compare optimized vs baseline performance"
)
@click.pass_context
def analyze_with_gepa(
    ctx: click.Context,
    blueprint_path: Path,
    use_optimization: Optional[Path],
    compare_baseline: bool
):
    """
    Analyze structural blueprint using GEPA-optimized prompts.
    
    Performs analysis with enhanced prompts evolved through GEPA
    for improved accuracy and performance.
    """
    config: Config = ctx.obj['config']
    verbose: bool = ctx.obj['verbose']
    
    console.print(f"\n[bold blue]🏗️ GEPA-Enhanced Blueprint Analysis[/bold blue]")
    console.print("=" * 60)
    
    console.print(f"📄 Blueprint: [cyan]{blueprint_path}[/cyan]")
    
    if use_optimization:
        console.print(f"🧬 Optimization: [green]{use_optimization}[/green]")
    else:
        console.print("[yellow]⚠️ No optimization file specified - using baseline[/yellow]")
    
    try:
        # Load optimization results if provided
        optimization_results = None
        if use_optimization:
            with open(use_optimization, 'r', encoding='utf-8') as f:
                optimization_data = json.load(f)
                optimization_results = optimization_data.get("gepa_optimization_report")
        
        # Initialize enhanced agent
        agent = StructuralBlueprintAgent(config)
        
        # If we have optimization results, update the taxonomy engine
        if optimization_results:
            agent.orchestrator.taxonomy_engine = IntelligentTaxonomyEngine(
                config, optimization_results
            )
            console.print("[green]✅ GEPA-optimized prompts loaded[/green]")
        
        # Run analysis with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            analysis_task = progress.add_task(
                "🏗️ [bold]Analyzing with GEPA enhancement...[/bold]"
            )
            
            start_time = time.time()
            
            # Execute analysis
            result = await agent.analyze_blueprint(blueprint_path)
            
            duration = time.time() - start_time
        
        # Display enhanced results
        display_gepa_analysis_results(result, duration, optimization_results)
        
        # Compare with baseline if requested
        if compare_baseline and optimization_results:
            await compare_with_baseline(blueprint_path, result, config)
        
        # Save results
        output_dir = Path(config.get_directories()['output'])
        output_file = output_dir / f"{blueprint_path.stem}_gepa_analysis.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result.model_dump(), f, indent=2, ensure_ascii=False, default=str)
        
        console.print(f"\n[green]💾 Enhanced analysis saved to: [cyan]{output_file}[/cyan][/green]")
    
    except Exception as e:
        console.print(f"[red]❌ GEPA-enhanced analysis failed: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.option(
    "--optimization-file",
    "-o",
    type=click.Path(exists=True, path_type=Path),
    help="Path to GEPA optimization results"
)
@click.pass_context
def show_optimization_status(ctx: click.Context, optimization_file: Optional[Path]):
    """
    Show GEPA optimization status and performance improvements.
    
    Displays detailed information about GEPA optimization results
    and performance enhancements achieved.
    """
    config: Config = ctx.obj['config']
    
    console.print(f"\n[bold magenta]🧬 GEPA Optimization Status[/bold magenta]")
    console.print("=" * 60)
    
    try:
        if optimization_file:
            # Load specific optimization file
            with open(optimization_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            optimization_results = data.get("gepa_optimization_report", {})
            
        else:
            # Look for default optimization results
            output_dir = Path(config.get_directories()['output'])
            default_file = output_dir / "gepa_optimization_results.json"
            
            if default_file.exists():
                with open(default_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                optimization_results = data.get("gepa_optimization_report", {})
            else:
                console.print("[yellow]📭 No GEPA optimization results found[/yellow]")
                console.print("Run 'gepa optimize-prompts' first to create optimization results")
                return
        
        # Display optimization status
        status_table = Table(title="GEPA Optimization Status")
        status_table.add_column("Metric", style="cyan")
        status_table.add_column("Value", style="green")
        status_table.add_column("Status", style="yellow")
        
        # Basic metrics
        opt_time = optimization_results.get("optimization_time", 0)
        generations = optimization_results.get("generations_completed", 0)
        improvement = optimization_results.get("performance_improvement", 0)
        
        status_table.add_row(
            "Optimization Time", 
            f"{opt_time:.2f} seconds", 
            "✅ Completed"
        )
        status_table.add_row(
            "Generations", 
            str(generations), 
            "✅ Evolved"
        )
        status_table.add_row(
            "Performance Improvement", 
            f"{improvement:.1%}", 
            "🚀 Enhanced" if improvement > 0.1 else "📈 Improved"
        )
        
        console.print(status_table)
        
        # Performance metrics
        final_metrics = optimization_results.get("final_metrics", {})
        if final_metrics:
            metrics_table = Table(title="Performance Metrics")
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Score", style="green")
            metrics_table.add_column("Improvement", style="yellow")
            
            for metric, score in final_metrics.items():
                metrics_table.add_row(
                    metric.replace("_", " ").title(),
                    f"{score:.3f}",
                    f"+{improvement:.1%}" if improvement > 0 else "baseline"
                )
            
            console.print(metrics_table)
        
        # Deployment recommendation
        deployment_ready = data.get("deployment_ready", False)
        
        if deployment_ready:
            recommendation = Panel(
                "[bold green]✅ READY FOR PRODUCTION DEPLOYMENT[/bold green]\n\n"
                "The GEPA-optimized prompts show significant improvement and are\n"
                "recommended for production use. Enhanced accuracy and performance\n"
                "will improve structural blueprint analysis quality.",
                title="🚀 Deployment Recommendation",
                border_style="green"
            )
        else:
            recommendation = Panel(
                "[bold yellow]⚠️ OPTIMIZATION IN PROGRESS[/bold yellow]\n\n"
                "Current optimization shows modest improvement. Consider running\n"
                "additional generations or adjusting optimization parameters\n"
                "for better performance gains.",
                title="📈 Optimization Status",
                border_style="yellow"
            )
        
        console.print(recommendation)
        
    except Exception as e:
        console.print(f"[red]❌ Failed to show optimization status: {e}[/red]")


async def compare_with_baseline(
    blueprint_path: Path, 
    optimized_result: Any, 
    config: Config
):
    """Compare GEPA-optimized analysis with baseline."""
    console.print(f"\n[bold yellow]📊 BASELINE COMPARISON[/bold yellow]")
    
    try:
        # Run baseline analysis
        console.print("🔄 Running baseline analysis for comparison...")
        
        baseline_agent = StructuralBlueprintAgent(config)
        baseline_result = await baseline_agent.analyze_blueprint(blueprint_path)
        
        # Create comparison table
        comparison_table = Table(title="GEPA vs Baseline Performance")
        comparison_table.add_column("Metric", style="cyan")
        comparison_table.add_column("Baseline", style="white")
        comparison_table.add_column("GEPA-Optimized", style="green")
        comparison_table.add_column("Improvement", style="yellow")
        
        # Compare key metrics
        metrics = [
            ("Overall Confidence", baseline_result.overall_confidence, optimized_result.overall_confidence),
            ("Completeness Score", baseline_result.completeness_score, optimized_result.completeness_score),
            ("Consistency Score", baseline_result.consistency_score, optimized_result.consistency_score),
            ("Processing Time", baseline_result.total_processing_time, optimized_result.total_processing_time),
            ("Elements Detected", sum(baseline_result.elements_by_type.values()), sum(optimized_result.elements_by_type.values()))
        ]
        
        for metric_name, baseline_val, optimized_val in metrics:
            if isinstance(baseline_val, float) and isinstance(optimized_val, float):
                improvement = ((optimized_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
                improvement_str = f"{improvement:+.1f}%"
                
                # Color code improvement
                if improvement > 5:
                    improvement_str = f"[green]{improvement_str}[/green]"
                elif improvement < -5:
                    improvement_str = f"[red]{improvement_str}[/red]"
                else:
                    improvement_str = f"[yellow]{improvement_str}[/yellow]"
                
                comparison_table.add_row(
                    metric_name,
                    f"{baseline_val:.3f}" if isinstance(baseline_val, float) else str(baseline_val),
                    f"{optimized_val:.3f}" if isinstance(optimized_val, float) else str(optimized_val),
                    improvement_str
                )
        
        console.print(comparison_table)
        
        # Summary assessment
        avg_improvement = sum(
            ((opt - base) / base * 100) if base > 0 else 0
            for _, base, opt in metrics[:3]  # Focus on quality metrics
        ) / 3
        
        if avg_improvement > 10:
            summary = "[bold green]🚀 SIGNIFICANT IMPROVEMENT[/bold green]\nGEPA optimization provides substantial performance gains"
        elif avg_improvement > 5:
            summary = "[bold yellow]📈 MODERATE IMPROVEMENT[/bold yellow]\nGEPA optimization shows measurable benefits"
        elif avg_improvement > 0:
            summary = "[bold cyan]📊 MINOR IMPROVEMENT[/bold cyan]\nGEPA optimization provides slight enhancements"
        else:
            summary = "[bold red]⚠️ NO SIGNIFICANT IMPROVEMENT[/bold red]\nConsider adjusting optimization parameters"
        
        summary_panel = Panel(summary, title="Performance Assessment", border_style="blue")
        console.print(summary_panel)
        
    except Exception as e:
        console.print(f"[red]❌ Baseline comparison failed: {e}[/red]")


def display_optimization_results(results: Dict[str, Any], duration: float):
    """Display GEPA optimization results."""
    
    console.print(f"\n[bold green]🎉 GEPA OPTIMIZATION COMPLETED[/bold green]")
    console.print("=" * 60)
    
    # Main results panel
    results_text = f"""[bold]⏱️ Optimization Time:[/bold] {duration:.2f} seconds
[bold]🧬 Generations:[/bold] {results.get('generations_completed', 0)}
[bold]🎯 Best Score:[/bold] {results.get('best_score', 0.0):.3f}
[bold]📈 Improvement:[/bold] {results.get('improvement', 0.0):.1%}
[bold]✅ Success Rate:[/bold] {results.get('final_metrics', {}).get('accuracy', 0.0):.1%}"""
    
    results_panel = Panel(results_text, title="🧬 GEPA Optimization Results", border_style="green")
    console.print(results_panel)
    
    # Performance improvements
    final_metrics = results.get("final_metrics", {})
    if final_metrics:
        metrics_table = Table(title="Performance Improvements")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Score", style="green")
        metrics_table.add_column("Quality", style="yellow")
        
        for metric, score in final_metrics.items():
            quality = "Excellent" if score > 0.9 else "Good" if score > 0.8 else "Fair" if score > 0.7 else "Needs Improvement"
            metrics_table.add_row(
                metric.replace("_", " ").title(),
                f"{score:.3f}",
                quality
            )
        
        console.print(metrics_table)
    
    # Optimization insights
    improvement = results.get("improvement", 0.0)
    
    if improvement > 0.15:
        insight = "[bold green]🚀 EXCEPTIONAL OPTIMIZATION[/bold green]\nGEPA achieved outstanding prompt improvements"
    elif improvement > 0.10:
        insight = "[bold blue]📈 SIGNIFICANT OPTIMIZATION[/bold blue]\nGEPA delivered substantial performance gains"
    elif improvement > 0.05:
        insight = "[bold yellow]📊 MODERATE OPTIMIZATION[/bold yellow]\nGEPA provided measurable improvements"
    else:
        insight = "[bold orange]⚠️ LIMITED OPTIMIZATION[/bold orange]\nConsider adjusting parameters or adding more training data"
    
    insight_panel = Panel(insight, title="💡 Optimization Insight", border_style="blue")
    console.print(insight_panel)


def save_optimization_results(results: Dict[str, Any], config: Config):
    """Save GEPA optimization results."""
    try:
        output_dir = Path(config.get_directories()['output'])
        results_file = output_dir / "gepa_optimization_results.json"
        
        # Add metadata
        save_data = {
            "gepa_optimization_report": results,
            "deployment_ready": results.get("improvement", 0.0) > 0.1,
            "created_at": time.time(),
            "framework_versions": {
                "gepa": "0.0.12",
                "dspy": "3.0.3",
                "structural_agent": "2.0.0"
            },
            "optimization_config": {
                "target_system": "structural_blueprint_analysis",
                "optimization_method": "reflective_text_evolution",
                "performance_metric": "multi_modal_accuracy"
            }
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"GEPA optimization results saved: {results_file}")
        
    except Exception as e:
        logger.error(f"Failed to save optimization results: {e}")


def display_gepa_analysis_results(
    result: Any, 
    duration: float, 
    optimization_results: Optional[Dict[str, Any]]
):
    """Display analysis results with GEPA enhancement indicators."""
    
    console.print(f"\n[bold green]🎉 GEPA-ENHANCED ANALYSIS COMPLETED[/bold green]")
    console.print("=" * 70)
    
    # Enhancement status
    if optimization_results:
        improvement = optimization_results.get("performance_improvement", 0.0)
        enhancement_text = f"""[bold]🧬 GEPA Enhancement:[/bold] Active
[bold]📈 Performance Boost:[/bold] {improvement:.1%}
[bold]🎯 Optimization Quality:[/bold] {optimization_results.get('best_score_achieved', 0.0):.3f}
[bold]⚡ Enhanced Accuracy:[/bold] Prompts evolved through reflective text evolution"""
        
        enhancement_panel = Panel(enhancement_text, title="🧬 GEPA Enhancement Status", border_style="magenta")
        console.print(enhancement_panel)
    
    # Standard results display
    metrics_text = f"""[bold]📄 Document:[/bold] {result.document_name}
[bold]📊 Pages:[/bold] {result.total_pages}
[bold]⏱️ Time:[/bold] {duration:.2f} seconds
[bold]🎯 Confidence:[/bold] {result.overall_confidence:.3f}
[bold]✅ Completeness:[/bold] {result.completeness_score:.3f}
[bold]🔄 Consistency:[/bold] {result.consistency_score:.3f}"""
    
    metrics_panel = Panel(metrics_text, title="📊 Analysis Results", border_style="green")
    console.print(metrics_panel)
    
    # GEPA-specific insights
    if optimization_results:
        gepa_insights = [
            "✨ Prompts optimized through reflective text evolution",
            "🎯 Enhanced page classification accuracy",
            "🏗️ Improved structural element detection",
            "🧠 Better contextual understanding across pages",
            "📈 Measurable performance improvements over baseline"
        ]
        
        insights_text = "\n".join(f"• {insight}" for insight in gepa_insights)
        insights_panel = Panel(insights_text, title="🧬 GEPA Enhancements", border_style="blue")
        console.print(insights_panel)


def main():
    """Main entry point for GEPA CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 GEPA optimization interrupted[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]❌ Unexpected error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
