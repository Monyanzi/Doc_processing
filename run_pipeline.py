#!/usr/bin/env python3
"""
=================================================
Document Intelligence Pipeline Orchestrator
=================================================

This script is the central entry point for the document intelligence pipeline.
It orchestrates the different stages of processing:
1.  Ingestion: Finding and extracting text from documents.
2.  Classification: Determining the type of each document.
3.  Extraction: Pulling structured data from the document.
4.  Validation: Checking the quality and correctness of the extracted data.
5.  Reporting: Generating summary reports of the pipeline run.

The pipeline is configured through the `config.yaml` file.
"""

import argparse
import logging
from pathlib import Path
import yaml
from datetime import datetime

from src.ingestion import DocumentIngester
from src.classification import DocumentClassifier
from src.extraction import DocumentExtractor
from src.validation import DataValidator
from src.reporting import Reporter

def setup_logging(log_level="INFO"):
    """Sets up the root logger for the application."""
    logging.basicConfig(
        level=log_level.upper(),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    """Main function to run the document intelligence pipeline."""
    parser = argparse.ArgumentParser(description="Document Intelligence Pipeline Orchestrator")
    parser.add_argument(
        "--config",
        type=Path,
        default="config.yaml",
        help="Path to the configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        action='append', # Allows specifying multiple input directories
        help="Specify an input directory. Overrides config file. Can be used multiple times."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the output directory specified in the config file."
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose DEBUG logging."
    )
    args = parser.parse_args()

    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: {args.config}")
        return
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration file: {e}")
        return

    # Setup logging from config, with CLI override
    log_level = "DEBUG" if args.verbose else config.get("log_level", "INFO")
    setup_logging(log_level)

    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting Document Intelligence Pipeline...")
    logger.info(f"✅ Configuration loaded from {args.config}")

    # --- Setup Directories ---
    # Use CLI arguments if provided, otherwise use the config file
    if args.input_dir:
        input_dirs = [Path(p) for p in args.input_dir]
    else:
        input_dirs = [Path(p) for p in config.get("input_directories", ["sample-documents/"])]

    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    else:
        base_output_dir = Path(config.get("output_directory", "output/"))

    # Create a timestamped directory for this specific run
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_output_dir = base_output_dir / run_timestamp
    run_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Outputs for this run will be saved in: {run_output_dir}")

    # --- Instantiate Pipeline Components ---
    ingester = DocumentIngester(config)
    classifier = DocumentClassifier(config)
    extractor = DocumentExtractor(config)
    validator = DataValidator(config)
    reporter = Reporter(config, run_output_dir)

    # --- Run Pipeline Stages ---
    all_documents = []
    try:
        # 1. Ingestion from all specified directories
        for in_dir in input_dirs:
            if not in_dir.exists():
                logger.error(f"Input directory does not exist: {in_dir}")
                continue
            all_documents.extend(ingester.ingest(in_dir))

        if not all_documents:
            logger.warning("No documents were successfully ingested. Pipeline will stop.")
            return

        # 2. Classification
        classified_docs = classifier.classify_documents(all_documents)

        # 3. Extraction
        extracted_docs = extractor.extract_data(classified_docs)

        # 4. Validation
        validated_docs = validator.validate_documents(extracted_docs)

        # 5. Reporting
        reporter.generate_reports(validated_docs)

    except Exception as e:
        logger.critical(f"A critical error occurred during pipeline execution: {e}", exc_info=True)
        logger.critical("Pipeline execution halted.")
        return

    logger.info("✅ Pipeline finished successfully.")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
