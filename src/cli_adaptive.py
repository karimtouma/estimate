"""
Adaptive CLI with Dynamic Schema Integration.

Command-line interface that uses the adaptive processor for
fully autonomous analysis without hardcoded assumptions.
"""

import click
import sys
from pathlib import Path
from typing import Optional, List

try:
    from .core.config import get_config
    from .core.adaptive_processor import create_adaptive_processor
    from .utils.logging_config import setup_logging
    from .utils.file_manager import FileManager
except ImportError:
    print("Error: Cannot import required modules. Make sure you're running from the project root.")
    sys.exit(1)


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, config, verbose):
    """
    Adaptive PDF Estimator with Dynamic Schema Integration.
    
    Fully autonomous analysis system that adapts to any document type
    without hardcoded assumptions or predefined taxonomies.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Load configuration
    try:
        if config:
            config_obj = get_config(config_path=Path(config))
        else:
            config_obj = get_config()
        
        ctx.obj['config'] = config_obj
        ctx.obj['verbose'] = verbose
        
        # Setup logging
        setup_logging(config_obj)
        
        if verbose:
            click.echo("🚀 Adaptive PDF Estimator initialized")
            click.echo(f"📁 Config: {config or 'default'}")
            click.echo(f"🧠 Dynamic Schemas: {'enabled' if getattr(config_obj, 'enable_dynamic_schemas', True) else 'disabled'}")
        
    except Exception as e:
        click.echo(f"❌ Initialization failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--questions', '-q', multiple=True, help='Custom questions to ask (if none provided, generates adaptive questions)')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--enable-discovery/--disable-discovery', default=True, help='Enable/disable discovery phase')
@click.pass_context
def analyze(ctx, pdf_path, questions, output, enable_discovery):
    """
    Perform adaptive comprehensive analysis of a PDF document.
    
    Uses enhanced discovery and dynamic schemas for fully autonomous analysis
    that adapts to any document type without hardcoded assumptions.
    """
    config = ctx.obj['config']
    verbose = ctx.obj['verbose']
    
    if verbose:
        click.echo(f"📄 Analyzing: {pdf_path}")
        click.echo(f"🔍 Discovery: {'enabled' if enable_discovery else 'disabled'}")
        click.echo(f"❓ Questions: {'custom' if questions else 'adaptive'}")
    
    try:
        # Create adaptive processor
        processor = create_adaptive_processor(config)
        
        # Convert questions tuple to list
        questions_list = list(questions) if questions else None
        
        # Perform adaptive analysis
        result = processor.comprehensive_analysis_adaptive(
            pdf_path=Path(pdf_path),
            questions=questions_list,
            enable_discovery=enable_discovery
        )
        
        # Generate output filename if not provided
        if not output:
            pdf_stem = Path(pdf_path).stem
            output = f"{pdf_stem}_adaptive_analysis.json"
        
        # Save results
        file_manager = FileManager(config)
        saved_path = file_manager.save_results(result, output)
        
        click.echo(f"✅ Adaptive analysis completed")
        click.echo(f"💾 Results saved to: {saved_path}")
        
        # Display adaptive analysis stats
        if verbose:
            stats = processor.get_adaptive_analysis_stats()
            click.echo("\n📊 Adaptive Analysis Statistics:")
            click.echo(f"  🧠 Dynamic schemas: {'enabled' if stats['dynamic_schemas_enabled'] else 'disabled'}")
            click.echo(f"  📝 Total discovered types: {stats['total_discovered_types']}")
            click.echo(f"  🎯 Classification accuracy: {stats['classification_accuracy']:.3f}")
            click.echo(f"  🔍 Discovery rate: {stats['discovery_rate']:.3f}")
        
    except Exception as e:
        click.echo(f"❌ Analysis failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def registry_stats(ctx):
    """Show dynamic schema registry statistics."""
    config = ctx.obj['config']
    
    try:
        from .models.dynamic_schemas import get_dynamic_registry
        registry = get_dynamic_registry()
        stats = registry.get_registry_stats()
        
        click.echo("📊 Dynamic Schema Registry Statistics")
        click.echo("=" * 40)
        click.echo(f"Total Types: {stats['total_types']}")
        click.echo(f"Total Discoveries: {stats['total_discoveries']}")
        click.echo("\nCategory Counts:")
        for category, count in stats['category_counts'].items():
            click.echo(f"  {category}: {count}")
        
        click.echo("\nMost Reliable Types:")
        for type_info in stats['most_reliable_types']:
            click.echo(f"  {type_info['type_name']}: {type_info['reliability_score']:.3f}")
        
        click.echo("\nRecent Discoveries:")
        for discovery in stats['recent_discoveries']:
            click.echo(f"  {discovery['type_name']}: {discovery['discovery_method']}")
        
    except Exception as e:
        click.echo(f"❌ Failed to get registry stats: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('type_name')
@click.pass_context
def type_info(ctx, type_name):
    """Show detailed information about a discovered element type."""
    try:
        from .models.dynamic_schemas import get_dynamic_registry
        registry = get_dynamic_registry()
        type_def = registry.get_type_definition(type_name)
        
        if not type_def:
            click.echo(f"❌ Type '{type_name}' not found in registry")
            return
        
        click.echo(f"📋 Type Information: {type_name}")
        click.echo("=" * 40)
        click.echo(f"Base Category: {type_def.base_category.value}")
        click.echo(f"Discovery Confidence: {type_def.discovery_confidence:.3f}")
        click.echo(f"Discovery Method: {type_def.discovery_method.value}")
        click.echo(f"Occurrence Count: {type_def.occurrence_count}")
        click.echo(f"Reliability Score: {type_def.reliability_score:.3f}")
        
        if type_def.domain_context:
            click.echo(f"Domain Context: {type_def.domain_context}")
        if type_def.industry_context:
            click.echo(f"Industry Context: {type_def.industry_context}")
        if type_def.description:
            click.echo(f"Description: {type_def.description}")
        
        if type_def.related_types:
            click.echo(f"Related Types: {', '.join(type_def.related_types)}")
        
        if type_def.typical_properties:
            click.echo("Typical Properties:")
            for prop, value in type_def.typical_properties.items():
                click.echo(f"  {prop}: {value}")
        
    except Exception as e:
        click.echo(f"❌ Failed to get type info: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--sample-size', '-s', default=10, help='Number of pages to sample for discovery')
@click.pass_context
def discover(ctx, pdf_path, sample_size):
    """
    Run enhanced discovery analysis only.
    
    Performs discovery with dynamic schema integration to understand
    document structure and register new element types.
    """
    config = ctx.obj['config']
    verbose = ctx.obj['verbose']
    
    try:
        from .discovery.enhanced_discovery import create_enhanced_discovery
        import asyncio
        
        if verbose:
            click.echo(f"🔍 Running enhanced discovery on: {pdf_path}")
            click.echo(f"📊 Sample size: {sample_size}")
        
        # Create enhanced discovery
        enhanced_discovery = create_enhanced_discovery(config, Path(pdf_path))
        
        # Run discovery
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                enhanced_discovery.enhanced_initial_exploration(sample_size=sample_size)
            )
        finally:
            loop.close()
        
        # Display results
        click.echo("✅ Enhanced Discovery Complete")
        click.echo(f"📄 Document Type: {result.document_type}")
        click.echo(f"🏭 Industry Domain: {result.industry_domain}")
        click.echo(f"🔍 Patterns Found: {len(result.discovered_patterns.get('patterns', []))}")
        click.echo(f"🧠 Element Types Discovered: {len(result.discovered_element_types)}")
        click.echo(f"📝 Auto-Registered Types: {len(result.auto_registered_types)}")
        click.echo(f"🎯 Confidence Score: {result.confidence_score:.3f}")
        
        if result.auto_registered_types and verbose:
            click.echo("\n📝 Auto-Registered Types:")
            for type_name in result.auto_registered_types:
                click.echo(f"  • {type_name}")
        
    except Exception as e:
        click.echo(f"❌ Discovery failed: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()
