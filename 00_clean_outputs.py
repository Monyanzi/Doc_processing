#!/usr/bin/env python3
"""
clean_project.py - Comprehensive Project Cleaning Utility

This script safely cleans the entire project directory of all generated files
and folders, restoring it to a clean, source-only state.

It will remove:
- All `__pycache__` directories found within the project.
- The entire `output/` directory.
- The entire `logs/` directory.
- The entire `temp/` directory.

It includes a --dry-run mode for safety and a confirmation prompt before
deleting any files.
"""

import argparse
import logging
import shutil
from pathlib import Path

# --- SCRIPT-RELATIVE PATH SETUP ---
# This ensures the script always targets the directory it is located in,
# making it safe to run from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent
# --- END PATH SETUP ---

# --- CONFIGURATION: Directories to be cleaned ---
# Add any other generated folder names to this list.
DIRS_TO_DELETE = [
    "output",
    "logs",
    "temp",
]
# --- END CONFIGURATION ---

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    parser = argparse.ArgumentParser(description='Clean all generated files and directories from the project.')
    parser.add_argument('--dry-run', action='store_true', help="Show what would be deleted without actually deleting anything.")
    parser.add_argument('-y', '--yes', action='store_true', help="Bypass the confirmation prompt for automated use.")
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger('project_cleaner')

    print("="*60)
    print("Project Cleaning Utility")
    print(f"Targeting project root: {PROJECT_ROOT}")
    print("="*60)

    # --- Step 1: Find all __pycache__ directories ---
    pycache_dirs = list(PROJECT_ROOT.rglob('__pycache__'))

    # --- Step 2: Find all configured top-level directories ---
    top_level_dirs_to_delete = []
    for dir_name in DIRS_TO_DELETE:
        path = PROJECT_ROOT / dir_name
        if path.exists():
            top_level_dirs_to_delete.append(path)

    all_targets = pycache_dirs + top_level_dirs_to_delete

    if not all_targets:
        logger.info("Project is already clean. Nothing to do.")
        return 0

    # --- Step 3: Execute Dry Run or Deletion ---
    if args.dry_run:
        logger.info("--- DRY RUN MODE ---")
        if not all_targets:
            logger.info("Project is already clean.")
        for target in all_targets:
            logger.info(f"[WOULD DELETE] Directory: {target.relative_to(PROJECT_ROOT)}")
        logger.info("--- No files were harmed. ---")
        return 0

    # --- Step 4: Get Confirmation and Delete ---
    print("The following directories and all their contents will be PERMANENTLY DELETED:")
    for target in all_targets:
        print(f"  - {target.relative_to(PROJECT_ROOT)}")
    print("-" * 60)

    if not args.yes:
        confirm = input("Are you sure you want to continue? (yes/no): ")
        if confirm.lower() != 'yes':
            logger.info("Operation cancelled by user.")
            return 0

    deleted_count = 0
    for target in all_targets:
        try:
            shutil.rmtree(target)
            logger.info(f"Deleted: {target.relative_to(PROJECT_ROOT)}")
            deleted_count += 1
        except Exception as e:
            logger.error(f"Failed to delete {target.relative_to(PROJECT_ROOT)}: {e}")

    print("="*60)
    logger.info(f"Cleaning complete. {deleted_count} items removed.")
    print("="*60)

    return 0

if __name__ == '__main__':
    raise SystemExit(main())