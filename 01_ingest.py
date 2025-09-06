#!/usr/bin/env python3
"""
01_Ingestion - Document Text Extraction Runner

This script performs Stage 1 of the document intelligence pipeline. It finds all
supported documents in the specified input folders, extracts their raw text
content, and saves the output. It includes logic to reuse pre-extracted text
if found in subdirectories named `extracted_text`.

For each run, it creates a timestamped folder under `output/01_ingestion/`
containing:
- `extracted_text/`: A folder with one .txt file per successfully processed document.
- `ingestion_metrics.csv`: A detailed CSV report of the outcome for every file.
- `ingestion_summary.json`: A JSON summary of the entire run's statistics.
- `ingestion_run.log`: A simple text log of the run.

It also appends a new sheet to a master `stage_tracker.xlsx` in the root
output folder to track which files have completed this stage.
"""

import argparse
import json
import logging
import csv
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# You will need to create this file with your DocumentIngester class.
from ingest import DocumentIngester

# --- SCRIPT-RELATIVE PATH SETUP (Solves running from outside the project folder) ---
# Get the absolute path to the directory where this script is located
SCRIPT_DIR = Path(__file__).resolve().parent
# Define the base output folder relative to this script's location
BASE_OUTPUT_FOLDER = SCRIPT_DIR / "output"
# --- END PATH SETUP ---


# --- USER INPUT (can be overridden by config file or CLI flags) ---
INPUT_FOLDERS = [
    r"C:\\Users\\Monya\\Documents\\Visa\\Bank_Statement_Savings_Account",
    r"C:\\Users\\Monya\\Documents\\Visa\\Mexico",
    r"C:\\Users\\Monya\\Desktop\\Cursor\\AI_Accountant\\documents_sorted",
    r"C:\Users\Monya\Desktop\Cursor\Invoice Automation\sample-documents"
]

# Optional: Maximum file size to process (in MB)
MAX_FILE_SIZE_MB = 50

# Optional: Create detailed extraction log file
CREATE_EXTRACTION_LOG = True

# Optional: Reuse pre-extracted text under *extracted_text* folders if available
ALLOW_REUSE = True

# Optional: Path to tesseract executable on Windows
TESSERACT_PATH = None  # e.g., r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Optional: Default verbosity if no CLI flag provided
DEFAULT_VERBOSE = False
# --- END USER INPUT ---

# Expanded list of supported file extensions
SUPPORTED_EXTS = {
    '.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.txt',
    '.docx', '.doc', '.pptx', '.ppt',
    '.heic', '.webp',
    '.eml', '.msg',
}


def setup_logging(verbose: bool):
    """Configures the root logger for the script."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # --- NEW LOGIC: Add a warning if running from a different directory ---
    if Path.cwd() != SCRIPT_DIR:
        logging.warning(
            f"Running script from a different directory: {Path.cwd()}\n"
            f"Outputs will be saved relative to the script's location: {SCRIPT_DIR}"
        )


def find_documents_in_root(root: Path, max_size_mb: float | None) -> List[Path]:
    """Finds all supported documents in a single root path."""
    # This function is unchanged
    docs: List[Path] = []
    if not root.exists():
        logging.warning(f"Input directory does not exist, skipping: {root}")
        return docs
    
    logging.info(f"Searching for documents in: {root}")
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            if max_size_mb is not None:
                try:
                    if p.stat().st_size > max_size_mb * 1024 * 1024:
                        logging.debug(f"Skipping large file: {p.name}")
                        continue
                except Exception:
                    pass
            docs.append(p)
    return docs


def index_reused_text(root: Path) -> Dict[str, Path]:
    """Indexes pre-extracted text files under any `*extracted_text*` directory."""
    # This function is unchanged
    index: Dict[str, Path] = {}
    for d in root.rglob('*extracted_text*'):
        if not d.is_dir():
            continue
        for txt in d.rglob('*.txt'):
            stem = txt.stem.lower()
            index[stem] = txt
    return index


def process_document(doc_path: Path, ingester: DocumentIngester, text_output_dir: Path, reuse_index: Dict[str, Path]) -> Dict[str, Any]:
    """
    Extracts text from a single document, saves the text, and returns metrics.
    Checks for reusable text first before performing a full extraction.
    """
    # This function is unchanged
    logger = logging.getLogger('process_document')
    result = {
        'source_path': doc_path, 'method': 'failed', 'success': False,
        'text_length': 0, 'ocr_pages': 0, 'error': '', 'text_output_path': None
    }
    
    text_output_path = text_output_dir / f"{doc_path.stem}.txt"
    text = ''
    meta = {}

    try:
        # 1. Check for reusable text first
        reused_path = reuse_index.get(doc_path.stem.lower())
        if reused_path and reused_path.exists():
            text = reused_path.read_text(encoding='utf-8', errors='ignore')
            method = 'reused'
        else:
            # 2. If no reusable text, perform full extraction
            suffix = doc_path.suffix.lower()
            if suffix == '.txt':
                text, method, meta = ingester.extract_text_from_text_file(str(doc_path))
            elif suffix == '.pdf':
                text, method, meta = ingester.extract_text_from_pdf(str(doc_path))
            else: # Assumes all others are images for now
                text, method, meta = ingester.extract_text_from_image(str(doc_path))
        
        result['method'] = method
        if len(text.strip()) > 0:
            result['success'] = True
            result['text_length'] = len(text)
            if method == 'ocr':
                result['ocr_pages'] = len(meta.get('ocr_pages', []))
        else:
            result['error'] = 'No text extracted'

    except Exception as e:
        logger.error(f"Failed to process {doc_path.name}: {e}")
        result['error'] = str(e)
        result['success'] = False

    # 3. Save the extracted text if successful
    if result['success']:
        try:
            text_output_path.write_text(text, encoding='utf-8')
            result['text_output_path'] = str(text_output_path)
        except Exception as e:
            logger.error(f"Succeeded extraction but failed to write text for {doc_path.name}: {e}")
            result['success'] = False
            result['error'] = f"File write failed: {e}"

    return result


def write_reports(results: List[Dict[str, Any]], run_dir: Path, base_out_dir: Path, run_ts: str):
    """Writes all CSV, JSON, Excel, and log file outputs for the run."""
    # This function is unchanged
    logger = logging.getLogger('write_reports')
    
    csv_path = run_dir / 'ingestion_metrics.csv'
    csv_rows = [{'file_path': str(r['source_path']), 'method': r['method'], 'success': r['success'], 'text_length': r['text_length'], 'ocr_pages': r['ocr_pages'], 'error': r['error']} for r in results]
    try:
        if csv_rows:
            with csv_path.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
                writer.writeheader()
                writer.writerows(csv_rows)
            logger.info(f"Wrote per-file metrics: {csv_path}")
    except (IOError, IndexError) as e:
        logger.warning(f"Could not write CSV report: {e}")

    summary_path = run_dir / 'ingestion_summary.json'
    total = len(results)
    success = sum(1 for r in results if r['success'])
    methods = {m: sum(1 for r in results if r['method'] == m) for m in {r['method'] for r in results}}
    
    summary = {'run_timestamp': datetime.now().isoformat(), 'docs_total': total, 'success': success, 'fail': total - success, 'success_rate_pct': round((success / total * 100.0) if total else 0, 2), 'method_distribution': methods}
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    logger.info(f"Wrote summary: {summary_path}")

    tracker_path = base_out_dir / 'stage_tracker.xlsx'
    sheet_name = f'01_Ingestion_{run_ts}'
    excel_rows = [{'document_name': r['source_path'].name, 'source_folder': str(r['source_path'].parent), 'path': str(r['source_path']), 'extension': r['source_path'].suffix.lower(), 'method': r['method'].upper(), 'status': 'SUCCESS' if r['success'] else 'FAILED'} for r in results]
    
    try:
        import pandas as pd
        if excel_rows:
            df = pd.DataFrame(excel_rows)
            with pd.ExcelWriter(tracker_path, engine='openpyxl', mode='a' if tracker_path.exists() else 'w', if_sheet_exists='new' if tracker_path.exists() else None) as writer:
                df.to_excel(writer, index=False, sheet_name=sheet_name)
            logger.info(f"Updated master tracker: {tracker_path} [Sheet: {sheet_name}]")
    except ImportError:
        logger.warning("`pandas` or `openpyxl` not installed. Skipping Excel tracker.")
    except Exception as e:
        logger.error(f"Failed to write to Excel tracker: {e}")
        
    if CREATE_EXTRACTION_LOG:
        log_path = run_dir / 'ingestion_run.log'
        with log_path.open('w', encoding='utf-8') as f:
            for r in results:
                f.write(f"{r['source_path']} | {r['method']} | success={r['success']} | len={r['text_length']}\n")
        logger.info(f"Wrote ingestion log: {log_path}")

    print("\n" + "="*60 + "\nINGESTION SUMMARY\n" + "="*60)
    print(json.dumps(summary, indent=2))
    print("="*60)


def main():
    """Main script execution."""
    parser = argparse.ArgumentParser(description='Stage 1: Document Ingestion Runner')
    parser.add_argument('--input', '-i', type=str, help='Input directory (overrides INPUT_FOLDERS).')
    parser.add_argument('--output', '-o', type=str, help='Base output directory (overrides BASE_OUTPUT_FOLDER).')
    parser.add_argument('--max-docs', type=int, default=None, help='Maximum number of documents to process.')
    parser.add_argument('--no-reuse', action='store_true', help='Do not reuse pre-extracted text')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose DEBUG logging.')
    args = parser.parse_args()

    setup_logging(args.verbose or DEFAULT_VERBOSE)
    logger = logging.getLogger('ingest_runner')

    base_out_dir = Path(args.output) if args.output else Path(BASE_OUTPUT_FOLDER)
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = base_out_dir / "01_ingestion" / run_ts
    extracted_text_dir = run_dir / "extracted_text"
    run_dir.mkdir(parents=True, exist_ok=True)
    extracted_text_dir.mkdir(exist_ok=True)
    
    logger.info(f"Run ID: {run_ts}")
    logger.info(f"Output for this run will be in: {run_dir}")

    input_roots = [Path(args.input)] if args.input else [Path(p) for p in INPUT_FOLDERS]
    reuse_is_enabled = ALLOW_REUSE and not args.no_reuse
    ingester = DocumentIngester(tesseract_path=TESSERACT_PATH)
    all_results = []
    total_docs_found = 0

    for root in input_roots:
        docs_in_root = find_documents_in_root(root, MAX_FILE_SIZE_MB)
        if not docs_in_root: continue
        
        total_docs_found += len(docs_in_root)
        reuse_index = index_reused_text(root) if reuse_is_enabled else {}
        if reuse_is_enabled and reuse_index:
            logger.info(f"Indexed {len(reuse_index)} reusable text files under {root}")
        
        for doc in docs_in_root:
            all_results.append(process_document(doc, ingester, extracted_text_dir, reuse_index))
            if args.max_docs and len(all_results) >= args.max_docs: break
        if args.max_docs and len(all_results) >= args.max_docs: break
    
    logger.info(f"Found {total_docs_found} total documents to process across all folders.")
    if not all_results:
        logger.info("No documents were processed. Exiting.")
        return 0

    write_reports(all_results, run_dir, base_out_dir, run_ts)
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())