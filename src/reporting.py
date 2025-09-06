import logging
import csv
import json
from pathlib import Path
from typing import List, Dict, Any

from .validation import ValidatedDocument


class Reporter:
    """
    Generates summary reports for the pipeline run.
    """
    def __init__(self, config: Dict[str, Any], output_directory: Path):
        """
        Initializes the Reporter.

        Args:
            config: The configuration dictionary from config.yaml.
            output_directory: The root directory for all outputs for this run.
        """
        self.config = config
        self.output_directory = output_directory
        self.logger = logging.getLogger(__name__)

        # Ensure the output directory exists
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def generate_reports(self, documents: List[ValidatedDocument]):
        """
        Generates all configured reports for the processed documents.

        Args:
            documents: The final list of ValidatedDocument objects.
        """
        self.logger.info(f"Generating reports in: {self.output_directory}")

        if not documents:
            self.logger.warning("No documents were processed. Skipping report generation.")
            return

        # Generate a consolidated CSV summary
        self.generate_csv_summary(documents)

        # Generate individual JSON files for each document
        self.generate_json_outputs(documents)

        self.logger.info("Successfully generated reports.")

    def generate_csv_summary(self, documents: List[ValidatedDocument]):
        """
        Creates a single CSV file summarizing all processed documents.
        """
        csv_path = self.output_directory / "summary_report.csv"

        # Dynamically determine headers from all extracted data fields
        all_headers = set()
        for doc in documents:
            if doc.extracted_data:
                all_headers.update(doc.extracted_data.keys())

        # Define a fixed order for standard columns, then add dynamic ones
        fieldnames = [
            "source_file", "document_type", "classification_method",
            "validation_status", "validation_errors"
        ]

        # Add the dynamic headers, sorted for consistency
        dynamic_headers = sorted(list(all_headers))
        for header in dynamic_headers:
            if header not in fieldnames:
                fieldnames.append(header)

        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()

                for doc in documents:
                    row = {
                        "source_file": doc.path.name,
                        "document_type": doc.doc_type,
                        "classification_method": doc.classification_method,
                        "validation_status": "valid" if doc.validation_summary.get('is_valid') else "invalid",
                        "validation_errors": ", ".join(doc.validation_summary.get('errors', []))
                    }
                    if doc.extracted_data:
                        row.update(doc.extracted_data)
                    writer.writerow(row)

            self.logger.info(f"Successfully created CSV summary: {csv_path}")

        except Exception as e:
            self.logger.error(f"Failed to create CSV summary: {e}")

    def generate_json_outputs(self, documents: List[ValidatedDocument]):
        """
        Creates one JSON file for each processed document with all its details.
        """
        json_output_dir = self.output_directory / "json_details"
        json_output_dir.mkdir(exist_ok=True)

        for doc in documents:
            json_path = json_output_dir / f"{doc.path.stem}.json"

            output_data = {
                "source_file": str(doc.path),
                "ingestion_method": doc.method,
                "classification": {
                    "document_type": doc.doc_type,
                    "confidence": doc.confidence,
                    "method": doc.classification_method
                },
                "extraction": {
                    "data": doc.extracted_data
                },
                "validation": doc.validation_summary
            }

            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2)
            except Exception as e:
                self.logger.error(f"Failed to write JSON output for {doc.path.name}: {e}")

        self.logger.info(f"Successfully created {len(documents)} individual JSON reports in {json_output_dir}")
