#!/usr/bin/env python3
"""
04_Analyze - Business Intelligence Engine (V2.0 - Holistic Analysis)

This script performs Stage 4 of the pipeline. It has been upgraded to perform
a holistic analysis across multiple document types, not just invoices.

It includes:
1.  Comprehensive Data Quality checks for invoices, receipts, contracts, etc.
2.  Smarter "Net Spend" analysis that accounts for credit notes.
3.  Duplicate Payment and Contract Expiry detection as before.
"""

import argparse
import json
import logging
import csv
import re
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from collections import defaultdict

# --- SCRIPT-RELATIVE PATH SETUP ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_OUTPUT_FOLDER = SCRIPT_DIR / "output"
# --- END PATH SETUP ---

# --- USER INPUT (defaults) ---
DEFAULT_VERBOSE = False
CONTRACT_EXPIRY_DAYS = 90
# --- END USER INPUT ---


# =============================================================================
# 1. ANALYSIS LOGIC
# =============================================================================

class DocumentAnalyzer:
    """Encapsulates all the business intelligence logic."""
    def __init__(self, documents: List[Dict], contract_expiry_days: int):
        self.documents = documents
        self.logger = logging.getLogger('DocumentAnalyzer')
        self.contract_expiry_days = contract_expiry_days

    def find_duplicate_invoices(self) -> List[Dict]:
        """Identifies potential duplicate invoices based on a unique fingerprint."""
        self.logger.info("Running: Duplicate Invoice Detection...")
        invoices = [doc for doc in self.documents if doc['type'] == 'invoice' and doc['data'].get('extraction_status', '').startswith('success')]
        fingerprints = defaultdict(list)
        for doc in invoices:
            vendor = str(doc['data'].get('vendor_name', 'UNKNOWN')).strip().lower()
            inv_num = str(doc['data'].get('invoice_number', 'UNKNOWN')).strip().lower()
            amount = 0.0
            try: amount = float(re.sub(r'[$,]', '', str(doc['data'].get('total_amount', 0))))
            except (ValueError, TypeError): pass
            if inv_num != 'unknown' and vendor != 'unknown' and amount > 0:
                fingerprints[f"{vendor}|{inv_num}|{amount}"].append(doc['name'])
        duplicates = [{"fingerprint": fp, "documents": docs} for fp, docs in fingerprints.items() if len(docs) > 1]
        self.logger.info(f"Found {len(duplicates)} potential duplicate payment groups.")
        return duplicates

    def check_contract_expiries(self) -> List[Dict]:
        """Finds contracts that are expiring within the configured risk window."""
        self.logger.info("Running: Contract Expiry Risk Analysis...")
        contracts = [doc for doc in self.documents if doc['type'] == 'contract' and doc['data'].get('extraction_status', '').startswith('success')]
        expiring_soon, today = [], datetime.now()
        for doc in contracts:
            expiry_date_str = doc['data'].get('termination_date')
            if not expiry_date_str: continue
            try:
                expiry_date = datetime.fromisoformat(str(expiry_date_str).replace('Z', ''))
                days_to_expiry = (expiry_date - today).days
                if 0 <= days_to_expiry <= self.contract_expiry_days:
                    expiring_soon.append({"document_name": doc['name'], "termination_date": expiry_date_str, "days_to_expiry": days_to_expiry, "parties": doc['data'].get('contracting_parties', [])})
            except (ValueError, TypeError): self.logger.warning(f"Could not parse date '{expiry_date_str}' in contract '{doc['name']}'")
        expiring_soon.sort(key=lambda x: x['days_to_expiry'])
        self.logger.info(f"Found {len(expiring_soon)} contracts expiring within {self.contract_expiry_days} days.")
        return expiring_soon

    def analyze_vendor_spend(self) -> List[Dict]:
        """Aggregates total spend and document count per vendor, accounting for credit notes."""
        self.logger.info("Running: Net Vendor Spend Analysis...")
        docs_with_spend = [doc for doc in self.documents if doc['type'] in ['invoice', 'receipt', 'credit_note'] and doc['data'].get('extraction_status', '').startswith('success')]
        vendor_spend = defaultdict(lambda: {'total_spend': 0.0, 'doc_count': 0})

        for doc in docs_with_spend:
            vendor = doc['data'].get('vendor_name')
            raw_amount = doc['data'].get('total_amount')
            amount = 0.0
            if raw_amount:
                try: amount = float(re.sub(r'[$,]', '', str(raw_amount)))
                except (ValueError, TypeError): pass
            
            if vendor and amount > 0:
                # NEW: Subtract credit notes for a "Net Spend" figure
                if doc['type'] == 'credit_note':
                    vendor_spend[vendor]['total_spend'] -= amount
                else:
                    vendor_spend[vendor]['total_spend'] += amount
                vendor_spend[vendor]['doc_count'] += 1
        
        spend_summary = [{"vendor_name": vendor, "net_spend": round(data['total_spend'], 2), "doc_count": data['doc_count']} for vendor, data in vendor_spend.items()]
        spend_summary.sort(key=lambda x: x['net_spend'], reverse=True)
        self.logger.info(f"Aggregated net spend for {len(spend_summary)} unique vendors.")
        return spend_summary

    def check_data_quality(self) -> List[Dict]:
        """Performs basic validation checks on extracted data across ALL relevant document types."""
        self.logger.info("Running: Holistic Data Quality Guard...")
        quality_flags = []
        for doc in self.documents:
            if not doc['data'].get('extraction_status', '').startswith('success'):
                quality_flags.append({"document_name": doc['name'], "document_type": doc['type'], "issue": "Extraction Failed", "details": doc['data'].get('error', 'Unknown error')})
                continue

            doc_type, doc_data = doc['type'], doc['data']
            
            # --- NEW: Holistic, type-specific checks ---
            if doc_type == 'invoice':
                if not doc_data.get('invoice_number'): quality_flags.append({"document_name": doc['name'], "document_type": doc_type, "issue": "Missing Field", "details": "Invoice number is missing."})
                if not doc_data.get('total_amount') or float(doc_data.get('total_amount', 0)) <= 0: quality_flags.append({"document_name": doc['name'], "document_type": doc_type, "issue": "Invalid Value", "details": "Total amount is zero or missing."})
            elif doc_type == 'receipt':
                if not doc_data.get('transaction_date'): quality_flags.append({"document_name": doc['name'], "document_type": doc_type, "issue": "Missing Field", "details": "Transaction date is missing."})
                if not doc_data.get('total_amount') or float(doc_data.get('total_amount', 0)) <= 0: quality_flags.append({"document_name": doc['name'], "document_type": doc_type, "issue": "Invalid Value", "details": "Total amount is zero or missing."})
            elif doc_type == 'contract':
                if not doc_data.get('effective_date'): quality_flags.append({"document_name": doc['name'], "document_type": doc_type, "issue": "Missing Field", "details": "Effective date is missing."})
                if len(doc_data.get('contracting_parties', [])) < 2: quality_flags.append({"document_name": doc['name'], "document_type": doc_type, "issue": "Invalid Value", "details": "Fewer than two contracting parties were found."})
            # Add more checks for other document types here...

        self.logger.info(f"Found {len(quality_flags)} data quality issues across all document types.")
        return quality_flags

# =============================================================================
# 2. SCRIPT ORCHESTRATION
# =============================================================================

def setup_logging(verbose: bool):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if Path.cwd() != SCRIPT_DIR: logging.warning(f"Running from {Path.cwd()}. Outputs will be saved in {BASE_OUTPUT_FOLDER}")

def find_latest_run_folder(stage_dir: Path) -> Path | None:
    run_folders = [d for d in stage_dir.iterdir() if d.is_dir()]
    return max(run_folders, key=lambda d: d.name) if run_folders else None

def main():
    parser = argparse.ArgumentParser(description='Stage 4: Business Intelligence Engine (Holistic)')
    args = parser.parse_args()
    setup_logging(DEFAULT_VERBOSE)
    logger = logging.getLogger('analyze_runner')

    base_out_dir, run_ts = BASE_OUTPUT_FOLDER, datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = base_out_dir / "04_analysis" / run_ts; run_dir.mkdir(parents=True, exist_ok=True)
    
    classification_run_folder = find_latest_run_folder(base_out_dir / "02_classification")
    extraction_run_folder = find_latest_run_folder(base_out_dir / "03_extraction")
    if not classification_run_folder or not extraction_run_folder:
        logger.error("Could not find outputs from Stage 2 or 3. Please run them first."); return 1

    logger.info(f"Reading classifications from: {classification_run_folder.name}")
    logger.info(f"Reading extracted JSONs from: {extraction_run_folder.name}")

    documents = []
    classification_file = classification_run_folder / "classification_metrics.csv"
    json_dir = extraction_run_folder / "extracted_json"
    
    if not classification_file.exists():
        logger.error(f"Classification metrics not found: {classification_file}"); return 1

    with classification_file.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_name = row['document_name']
            json_path = json_dir / f"{doc_name}.json"
            if json_path.exists():
                with json_path.open('r', encoding='utf-8') as jf:
                    documents.append({"name": doc_name, "type": row['predicted_type'], "data": json.load(jf)})
            else:
                logger.warning(f"JSON file for '{doc_name}' not found. It may have been skipped during extraction.")

    if not documents:
        logger.error("No valid documents with extracted data found. Exiting."); return 1

    analyzer = DocumentAnalyzer(documents, contract_expiry_days=CONTRACT_EXPIRY_DAYS)
    analysis_results = {
        "run_timestamp": datetime.now().isoformat(),
        "source_run_folders": {"classification": str(classification_run_folder.name), "extraction": str(extraction_run_folder.name)},
        "duplicate_invoices": analyzer.find_duplicate_invoices(),
        "expiring_contracts": analyzer.check_contract_expiries(),
        "vendor_spend": analyzer.analyze_vendor_spend(),
        "data_quality_flags": analyzer.check_data_quality(),
    }

    output_path = run_dir / "analysis_results.json"
    output_path.write_text(json.dumps(analysis_results, indent=2), encoding='utf-8')
    logger.info(f"Successfully ran all analyses. Results saved to: {output_path}")

    print("\n" + "="*60 + "\nANALYSIS SUMMARY\n" + "="*60)
    print(f"  - Potential Duplicate Invoice Groups: {len(analysis_results['duplicate_invoices'])}")
    print(f"  - Contracts Expiring Soon: {len(analysis_results['expiring_contracts'])}")
    print(f"  - Unique Vendors Analyzed: {len(analysis_results['vendor_spend'])}")
    print(f"  - Data Quality Issues Flagged: {len(analysis_results['data_quality_flags'])}")
    print("="*60)

    return 0

if __name__ == '__main__':
    raise SystemExit(main())