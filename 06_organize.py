#!/usr/bin/env python3
"""
06_Organize - Final File Organizer (V1.2 - Corrected Merge Logic)

This script performs the final stage of the pipeline. It has been upgraded to
robustly merge data from the tracker by normalizing document names (removing
file extensions) before joining the ingestion and classification sheets.
"""

import argparse
import logging
import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime
import re

# --- SCRIPT-RELATIVE PATH SETUP ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_OUTPUT_FOLDER = SCRIPT_DIR / "output"
# --- END PATH SETUP ---

# --- USER INPUT (defaults) ---
DEFAULT_VERBOSE = False
# --- END USER INPUT ---

def setup_logging(verbose: bool):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if Path.cwd() != SCRIPT_DIR: logging.warning(f"Running from {Path.cwd()}. Outputs will be saved in {BASE_OUTPUT_FOLDER}")

def find_latest_sheet_by_prefix(tracker_path: Path, prefix: str) -> str | None:
    """Finds the most recent sheet in the tracker with a given prefix."""
    if not tracker_path.exists():
        raise FileNotFoundError(f"Stage tracker not found: {tracker_path}")
    
    xls = pd.ExcelFile(tracker_path)
    relevant_sheets = [s for s in xls.sheet_names if s.startswith(prefix)]
    if not relevant_sheets:
        return None
    
    latest_sheet = max(relevant_sheets, key=lambda s: s.split('_')[-1])
    return latest_sheet

def sanitize_foldername(name: str) -> str:
    """Removes characters that are invalid for folder names."""
    return re.sub(r'[<>:"/\\|?*]', '_', name)

def main():
    parser = argparse.ArgumentParser(description='Stage 6: Final File Organizer')
    args = parser.parse_args()
    setup_logging(DEFAULT_VERBOSE)
    logger = logging.getLogger('organize_runner')

    base_out_dir, run_ts = BASE_OUTPUT_FOLDER, datetime.now().strftime('%Y%m%d_%H%M%S')
    
    organized_folder_name = f"Organized_Documents_{run_ts}"
    run_dir = base_out_dir / "06_organization" / organized_folder_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    tracker_path = base_out_dir / 'stage_tracker.xlsx'
    
    try:
        ingestion_sheet = find_latest_sheet_by_prefix(tracker_path, "01_Ingestion_")
        classification_sheet = find_latest_sheet_by_prefix(tracker_path, "02_Classification_")

        if not ingestion_sheet or not classification_sheet:
            logger.error("Could not find the latest Ingestion or Classification sheet in the tracker. Please run Stages 1 and 2.")
            return 1
            
        logger.info(f"Reading file paths from sheet: {ingestion_sheet}")
        ingestion_df = pd.read_excel(tracker_path, sheet_name=ingestion_sheet)
        
        logger.info(f"Reading classifications from sheet: {classification_sheet}")
        classification_df = pd.read_excel(tracker_path, sheet_name=classification_sheet)
        
        logger.info("Merging ingestion paths with classification results...")
        
        # --- FIX: Normalize the 'document_name' in the ingestion data ---
        # This removes the file extension (e.g., '.pdf') to match the classification data
        ingestion_df['document_name_stem'] = ingestion_df['document_name'].apply(lambda x: Path(x).stem)
        
        # Select only the essential columns before merging
        ingestion_data = ingestion_df[['document_name_stem', 'path']]
        classification_data = classification_df[['document_name', 'predicted_type']]
        
        # Merge on the common 'document_name_stem' and 'document_name' columns
        merged_df = pd.merge(ingestion_data, classification_data, left_on='document_name_stem', right_on='document_name', how='inner')

    except Exception as e:
        logger.error(f"Failed to read and merge data from the stage tracker: {e}"); return 1

    if merged_df.empty:
        logger.warning("No matching documents found between ingestion and classification. Nothing to organize.")
        return 0

    logger.info(f"Organizing {len(merged_df)} files into: {run_dir}")
    
    success_count, fail_count = 0, 0
    
    for index, row in merged_df.iterrows():
        try:
            source_path = Path(row['path'])
            doc_type = sanitize_foldername(str(row['predicted_type']))
            
            if not source_path.exists():
                logger.warning(f"Original file not found, skipping: {source_path}")
                fail_count += 1
                continue

            dest_folder = run_dir / doc_type
            dest_folder.mkdir(exist_ok=True)
            
            shutil.copy2(source_path, dest_folder)
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to copy file {row.get('path', 'N/A')}: {e}")
            fail_count += 1
            
    print("\n" + "="*60 + "\nFILE ORGANIZATION COMPLETE\n" + "="*60)
    print(f"  - Successfully copied: {success_count} files")
    print(f"  - Failed or skipped:   {fail_count} files")
    print(f"  - Your organized files are located in: {run_dir}")
    print("="*60)

    return 0

if __name__ == '__main__':
    raise SystemExit(main())