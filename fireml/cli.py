#!/usr/bin/env python3
"""
Command Line Interface for FireAutoML.

This module provides a command-line interface for using FireAutoML
functionality without writing Python code.
"""
import sys
import argparse
import logging
from fireml.data_loader import load_data
from fireml.main_pipeline import run_full_pipeline
from fireml.utils.logging_config import setup_logging

# Setup logging for the CLI
setup_logging()
logger = logging.getLogger(__name__)

def setup_parser() -> argparse.ArgumentParser:
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='FireAutoML: Automated Machine Learning Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full AutoML pipeline with automatic target detection
  fireml analyze --input data.csv

  # Run with specific target column and save to a specific directory
  fireml analyze --input data.csv --target price --output ./my_analysis

  # Run the web API server (requires 'web' extra)
  fireml web --host 0.0.0.0 --port 8000
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute', required=True)
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Run full AutoML analysis pipeline')
    analyze_parser.add_argument('--input', '-i', required=True, help='Input data file (CSV, Excel, etc.)')
    analyze_parser.add_argument('--target', '-t', help='Target column for prediction (auto-detected if not specified)')
    analyze_parser.add_argument('--task', choices=['auto', 'classification', 'regression'], 
                              default='auto', help='ML task type (auto-detected if not specified)')
    analyze_parser.add_argument('--output', '-o', help='Directory to save results')
    analyze_parser.add_argument('--deep', action='store_true', help='Include deep learning models (requires [deep_learning] extra)')
    
    # Web command
    web_parser = subparsers.add_parser('web', help='Run the FireAutoML web API')
    web_parser.add_argument('--host', default='127.0.0.1', help='Host to run the web server on')
    web_parser.add_argument('--port', type=int, default=5000, help='Port to run the web server on')
    
    return parser

def main():
    """Main entry point for the CLI."""
    parser = setup_parser()
    args = parser.parse_args()
    
    if args.command == 'analyze':
        logger.info(f"Starting analysis of {args.input}")
        df, _ = load_data(args.input)
        if df.empty:
            logger.error("Failed to load data. Exiting.")
            sys.exit(1)
        
        run_full_pipeline(
            df=df,
            target_column=args.target,
            task_type=args.task,
            output_dir=args.output,
            run_deep_learning=args.deep
        )
        logger.info("Analysis complete.")

    elif args.command == 'web':
        try:
            from fireml.web.app import app
            logger.info(f"Starting FireAutoML web server on http://{args.host}:{args.port}")
            app.run(host=args.host, port=args.port)
        except ImportError:
            logger.error("Could not start web server. Please install the 'web' extra: pip install .[web]")
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
