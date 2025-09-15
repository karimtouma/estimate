"""
Command-line interface for PDF Estimator.

Modern CLI with rich formatting and comprehensive functionality.
"""

import sys
import click
from pathlib import Path
from typing import Optional, List
import json

from .core.config import get_config, Config
from .core.processor import PDFProcessor
from .core.adaptive_processor import AdaptiveProcessor
from .core.processor import ProcessorError
from .models.schemas import AnalysisType


@click.group()
@click.version_option(version="2.0.0", prog_name="PDF Estimator")
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
    PDF Estimator - Modern PDF processing with Google GenAI.
    
    A powerful tool for analyzing PDF documents using artificial intelligence.
    """
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
        click.echo(f"❌ Configuration error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", 
    "-o", 
    type=click.Path(path_type=Path),
    help="Output file path (default: auto-generated)"
)
@click.option(
    "--analysis-type",
    "-t",
    type=click.Choice(["general", "sections", "data_extraction", "comprehensive"]),
    default="comprehensive",
    help="Type of analysis to perform"
)
@click.option(
    "--questions",
    "-q",
    multiple=True,
    help="Custom questions for analysis (can be used multiple times)"
)
@click.pass_context
def analyze(
    ctx: click.Context,
    pdf_path: Path,
    output: Optional[Path],
    analysis_type: str,
    questions: tuple[str, ...]
):
    """
    Analyze a PDF document.
    
    Performs comprehensive analysis of the specified PDF file using AI.
    """
    config: Config = ctx.obj['config']
    verbose: bool = ctx.obj['verbose']
    
    if verbose:
        click.echo(f"📄 Analyzing: {pdf_path}")
        click.echo(f"🔍 Analysis type: {analysis_type}")
    
    try:
        # Choose processor based on dynamic schemas configuration
        enable_dynamic_schemas = getattr(config, 'enable_dynamic_schemas', False)
        if hasattr(config, 'analysis') and hasattr(config.analysis, 'enable_dynamic_schemas'):
            enable_dynamic_schemas = config.analysis.enable_dynamic_schemas
        
        # Debug logging
        click.echo(f"🔍 Debug: enable_dynamic_schemas = {enable_dynamic_schemas}")
        click.echo(f"🔍 Debug: config has analysis = {hasattr(config, 'analysis')}")
        if hasattr(config, 'analysis'):
            click.echo(f"🔍 Debug: analysis.enable_dynamic_schemas = {getattr(config.analysis, 'enable_dynamic_schemas', 'NOT_FOUND')}")
        
        processor_class = AdaptiveProcessor if enable_dynamic_schemas else PDFProcessor
        processor_name = "Adaptive" if enable_dynamic_schemas else "Standard"
        
        click.echo(f"🧠 Using {processor_name} Processor (dynamic schemas: {'enabled' if enable_dynamic_schemas else 'disabled'})")
        
        with processor_class(config) as processor:
            if analysis_type == "comprehensive":
                # Comprehensive analysis
                questions_list = list(questions) if questions else None
                
                if enable_dynamic_schemas:
                    # Use fully adaptive analysis
                    result = processor.comprehensive_analysis_adaptive(pdf_path, questions_list)
                else:
                    # Use standard analysis
                    result = processor.comprehensive_analysis(pdf_path, questions_list)
                
                # Generate output filename if not provided
                if not output:
                    output = pdf_path.stem + "_comprehensive_analysis.json"
                
                # Save results
                saved_path = processor.save_results(result, output)
                
                click.echo(f"✅ Comprehensive analysis completed")
                click.echo(f"💾 Results saved to: {saved_path}")
                
                # Show summary
                if verbose and result.general_analysis:
                    ga = result.general_analysis
                    click.echo(f"\n📊 Summary:")
                    click.echo(f"   Document Type: {ga.document_type}")
                    click.echo(f"   Confidence: {ga.confidence_score:.2f}")
                    click.echo(f"   Topics: {', '.join(ga.main_topics[:3])}")
                
            else:
                # Single analysis type
                file_uri = processor.upload_pdf(pdf_path)
                result = processor.analyze_document(file_uri, AnalysisType(analysis_type))
                
                # Generate output filename if not provided
                if not output:
                    output = pdf_path.stem + f"_{analysis_type}_analysis.json"
                
                # Save results
                saved_path = processor.save_results(result, output)
                
                click.echo(f"✅ {analysis_type.title()} analysis completed")
                click.echo(f"💾 Results saved to: {saved_path}")
    
    except ProcessorError as e:
        click.echo(f"❌ Processing error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file path (default: auto-generated)"
)
@click.pass_context
def chat(ctx: click.Context, pdf_path: Path, output: Optional[Path]):
    """
    Interactive chat with a PDF document.
    
    Start an interactive session to ask questions about the PDF.
    """
    config: Config = ctx.obj['config']
    verbose: bool = ctx.obj['verbose']
    
    if verbose:
        click.echo(f"📄 Loading PDF: {pdf_path}")
    
    try:
        with PDFProcessor(config) as processor:
            # Upload PDF
            file_uri = processor.upload_pdf(pdf_path)
            click.echo(f"✅ PDF loaded successfully")
            click.echo(f"💬 Interactive mode started. Type 'exit' to quit, 'help' for commands.")
            
            conversation_results = []
            
            while True:
                try:
                    question = click.prompt("\n❓ Your question", type=str)
                    
                    if question.lower() in ['exit', 'quit', 'q']:
                        break
                    elif question.lower() == 'help':
                        click.echo("\n📋 Available commands:")
                        click.echo("   exit, quit, q  - Exit chat")
                        click.echo("   help          - Show this help")
                        click.echo("   history       - Show conversation history")
                        click.echo("   clear         - Clear conversation history")
                        continue
                    elif question.lower() == 'history':
                        history = processor.get_conversation_history()
                        click.echo(f"\n📚 Conversation history ({len(history)} items):")
                        for i, entry in enumerate(history[-5:], 1):
                            entry_type = entry.get('type', 'unknown')
                            if entry_type == 'question':
                                click.echo(f"   {i}. Q: {entry.get('question', 'N/A')}")
                        continue
                    elif question.lower() == 'clear':
                        processor.clear_history()
                        conversation_results.clear()
                        click.echo("🧹 History cleared")
                        continue
                    elif not question.strip():
                        continue
                    
                    # Process question
                    with click.progressbar([1], label="🤔 Processing") as bar:
                        results = processor.multi_turn_analysis(file_uri, [question])
                        bar.update(1)
                    
                    if results and "error" not in results[0]:
                        result = results[0]
                        conversation_results.append(result)
                        
                        click.echo(f"\n💡 Answer:")
                        click.echo(f"   {result['answer']}")
                        click.echo(f"   Confidence: {result.get('confidence', 0):.2f}")
                        
                        # Show follow-up questions if available
                        if result.get('follow_up_questions'):
                            click.echo(f"\n🔍 Suggested follow-up questions:")
                            for fq in result['follow_up_questions'][:3]:
                                click.echo(f"   • {fq}")
                    else:
                        error_msg = results[0].get('error', 'Unknown error') if results else 'No response'
                        click.echo(f"❌ Error: {error_msg}")
                
                except KeyboardInterrupt:
                    click.echo("\n👋 Chat interrupted by user")
                    break
                except Exception as e:
                    click.echo(f"❌ Error processing question: {e}")
                    if verbose:
                        import traceback
                        traceback.print_exc()
            
            # Save conversation if there were interactions
            if conversation_results:
                if not output:
                    output = pdf_path.stem + "_chat_session.json"
                
                chat_data = {
                    "pdf_file": str(pdf_path),
                    "session_timestamp": processor.get_conversation_history()[0].get('timestamp') if processor.get_conversation_history() else None,
                    "questions_and_answers": conversation_results,
                    "total_interactions": len(conversation_results)
                }
                
                saved_path = processor.save_results(chat_data, output)
                click.echo(f"\n💾 Chat session saved to: {saved_path}")
            
            click.echo("\n👋 Chat session ended")
    
    except ProcessorError as e:
        click.echo(f"❌ Processing error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.pass_context
def config_check(ctx: click.Context):
    """
    Validate configuration and show system information.
    
    Checks configuration validity and displays system status.
    """
    config: Config = ctx.obj['config']
    
    click.echo("🔧 Configuration Check")
    click.echo("=" * 50)
    
    # Show configuration summary
    config.print_summary()
    
    # Validate configuration
    click.echo("\n🔍 Validation Results:")
    if config.validate():
        click.echo("✅ Configuration is valid")
    else:
        click.echo("❌ Configuration has errors")
        sys.exit(1)
    
    # Show system information
    try:
        with PDFProcessor(config) as processor:
            system_info = processor.get_system_info()
            
            click.echo(f"\n🖥️  System Information:")
            click.echo(f"   Processor Version: {system_info['processor_version']}")
            click.echo(f"   Python Version: {system_info['python_version']}")
            click.echo(f"   Platform: {system_info['platform']}")
            click.echo(f"   Container: {'Yes' if system_info['container_environment'] else 'No'}")
            click.echo(f"   Model: {system_info['model']}")
    
    except Exception as e:
        click.echo(f"⚠️  Could not get system information: {e}")


@cli.command()
@click.option(
    "--pattern",
    "-p",
    default="*.json*",
    help="File pattern to match (default: *.json*)"
)
@click.pass_context
def list_results(ctx: click.Context, pattern: str):
    """
    List analysis results in the output directory.
    
    Shows all result files matching the specified pattern.
    """
    config: Config = ctx.obj['config']
    
    try:
        from .utils.file_manager import FileManager
        
        file_manager = FileManager(config)
        results = file_manager.list_results(pattern)
        
        if not results:
            click.echo(f"📭 No results found matching pattern: {pattern}")
            return
        
        click.echo(f"📋 Found {len(results)} result files:")
        click.echo("=" * 50)
        
        for i, result_path in enumerate(results, 1):
            stat = result_path.stat()
            size_mb = stat.st_size / (1024 * 1024)
            
            click.echo(f"{i:2d}. {result_path.name}")
            click.echo(f"    Size: {size_mb:.2f}MB")
            click.echo(f"    Modified: {stat.st_mtime}")
            click.echo()
    
    except Exception as e:
        click.echo(f"❌ Error listing results: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--days",
    "-d",
    default=30,
    help="Maximum age of files to keep (default: 30 days)"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be deleted without actually deleting"
)
@click.pass_context
def cleanup(ctx: click.Context, days: int, dry_run: bool):
    """
    Clean up old result files.
    
    Removes result files older than the specified number of days.
    """
    config: Config = ctx.obj['config']
    
    if days <= 0:
        click.echo("❌ Days must be a positive number", err=True)
        sys.exit(1)
    
    try:
        from .utils.file_manager import FileManager
        
        file_manager = FileManager(config)
        
        if dry_run:
            click.echo(f"🔍 Dry run: Would clean files older than {days} days")
            # TODO: Implement dry run functionality
            click.echo("⚠️  Dry run not yet implemented")
        else:
            if click.confirm(f"⚠️  This will delete files older than {days} days. Continue?"):
                cleaned_count = file_manager.cleanup_old_files(days)
                click.echo(f"🧹 Cleaned up {cleaned_count} old files")
            else:
                click.echo("❌ Cleanup cancelled")
    
    except Exception as e:
        click.echo(f"❌ Error during cleanup: {e}", err=True)
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\n👋 Interrupted by user")
        sys.exit(130)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
