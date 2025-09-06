#!/usr/bin/env python3
"""
05_Dashboard - Executive Dashboard Generator (V1.2 - Professional Prompt)

This script performs Stage 5 of the pipeline. It transforms the raw analytical
data from Stage 4 into a polished, human-readable deliverable.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import pandas as pd

# --- SCRIPT-RELATIVE PATH SETUP ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_OUTPUT_FOLDER = SCRIPT_DIR / "output"
# --- END PATH SETUP ---

# --- USER INPUT (defaults) ---
OLLAMA_MODEL = "gemma3:1b"
DEFAULT_VERBOSE = False
CONSULTANCY_NAME = "Insight Analytics" # Define your consultancy name here
# --- END USER INPUT ---


# =============================================================================
# 1. DASHBOARD GENERATION LOGIC
# =============================================================================

class DashboardGenerator:
    """Encapsulates all logic for creating the final executive dashboard."""
    def __init__(self, analysis_data: Dict, ollama_model: str):
        self.analysis_data = analysis_data
        self.ollama_model = ollama_model
        self.logger = logging.getLogger('DashboardGenerator')

    def generate_llm_summary(self) -> str:
        """Uses an LLM to generate a concise, non-jargon executive summary."""
        self.logger.info(f"Engaging LLM ({self.ollama_model}) to write executive summary...")
        
        summary_context = {
            "duplicate_alerts": len(self.analysis_data.get('duplicate_invoices', [])),
            "expiring_contracts": len(self.analysis_data.get('expiring_contracts', [])),
            "total_vendors": len(self.analysis_data.get('vendor_spend', [])),
            "data_quality_issues": len(self.analysis_data.get('data_quality_flags', [])),
            "top_vendor": self.analysis_data.get('vendor_spend', [{}])[0].get('vendor_name', 'N/A')
        }
        
        prompt = f"""You are a senior analyst at '{CONSULTANCY_NAME}', a document processing consultancy.
        Your task is to write a professional comprehensive summary for our client based on our analysis of their document batch. it should be comprehensive in detail butno fluff in terms of wrods. 

        **CRITICAL RULES:**
        - The tone must be professional, objective, and concise.
        - Start with an overall 1 -2 line introduction.
        - Then use bullet and sub-bullet points to convey the detail, include numbers. 
        - Refer to the client's data as "your documents" or "your data".
        - **DO NOT** include any conversational phrases like "Okay, here is..." or "I hope this helps".
        - **DO NOT** ask any questions.
        - **DO NOT** explain your own reasoning or the structure of the summary.
        - The output should be ONLY the professional summary text itself, ready to be copied directly into the client's report.

        DATA:
        {json.dumps(summary_context)}

        EXECUTIVE SUMMARY:
        """

        try:
            import ollama
            options = {"temperature": 0.15}
            response = ollama.chat(model=self.ollama_model, messages=[{'role': 'user', 'content': prompt}], options=options)
            summary_text = response['message']['content']
            return summary_text.strip()
        except Exception as e:
            self.logger.error(f"Failed to generate LLM summary: {e}")
            return "LLM summary generation failed. Please check your Ollama connection."

    def create_excel_dashboard(self, summary_text: str, output_path: Path):
        """Creates a multi-sheet Excel dashboard from the analysis data."""
        self.logger.info(f"Creating Excel dashboard at: {output_path}")
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: Dashboard Summary
            summary_df = pd.DataFrame({
                "Metric": ["🚨 CRITICAL ALERTS (Duplicate Payments)", "⏳ UPCOMING RISKS (Expiring Contracts)", "🚩 Data Quality Issues Flagged", "💰 Unique Vendors Analyzed"],
                "Value": [len(self.analysis_data['duplicate_invoices']), len(self.analysis_data['expiring_contracts']), len(self.analysis_data['data_quality_flags']), len(self.analysis_data['vendor_spend'])]
            })
            pd.DataFrame([summary_text]).to_excel(writer, sheet_name='Dashboard Summary', index=False, header=False, startrow=0)
            summary_df.to_excel(writer, sheet_name='Dashboard Summary', index=False, startrow=4)
            worksheet = writer.sheets['Dashboard Summary']
            worksheet.column_dimensions['A'].width = 50
            worksheet.column_dimensions['B'].width = 15

            # Sheet 2: Duplicate Payment Alerts
            if self.analysis_data['duplicate_invoices']:
                dup_rows = []
                for i, group in enumerate(self.analysis_data['duplicate_invoices']):
                    fp = group['fingerprint'].split('|')
                    for doc_name in group['documents']:
                        dup_rows.append({"ALERT GROUP": f"GROUP {i+1}", "Document Name": doc_name, "Vendor Name": fp[0], "Invoice Number": fp[1], "Amount": float(fp[2])})
                pd.DataFrame(dup_rows).to_excel(writer, sheet_name='Duplicate Payment Alerts', index=False)
                worksheet = writer.sheets['Duplicate Payment Alerts']
                worksheet.column_dimensions['A'].width = 15; worksheet.column_dimensions['B'].width = 35; worksheet.column_dimensions['C'].width = 25; worksheet.column_dimensions['D'].width = 25; worksheet.column_dimensions['E'].width = 15

            # Sheet 3: Contract Expiry Risks
            if self.analysis_data['expiring_contracts']:
                pd.DataFrame(self.analysis_data['expiring_contracts']).to_excel(writer, sheet_name='Contract Expiry Risks', index=False)
                worksheet = writer.sheets['Contract Expiry Risks']
                worksheet.column_dimensions['A'].width = 35; worksheet.column_dimensions['B'].width = 20; worksheet.column_dimensions['C'].width = 20; worksheet.column_dimensions['D'].width = 50

            # Sheet 4: Top Vendor Net Spend
            if self.analysis_data['vendor_spend']:
                pd.DataFrame(self.analysis_data['vendor_spend']).to_excel(writer, sheet_name='Top Vendor Net Spend', index=False)
                worksheet = writer.sheets['Top Vendor Net Spend']
                worksheet.column_dimensions['A'].width = 35; worksheet.column_dimensions['B'].width = 20; worksheet.column_dimensions['C'].width = 20

            # Sheet 5: Data Quality Flags
            if self.analysis_data['data_quality_flags']:
                pd.DataFrame(self.analysis_data['data_quality_flags']).to_excel(writer, sheet_name='Data Quality Flags', index=False)
                worksheet = writer.sheets['Data Quality Flags']
                worksheet.column_dimensions['A'].width = 35; worksheet.column_dimensions['B'].width = 20; worksheet.column_dimensions['C'].width = 25; worksheet.column_dimensions['D'].width = 50

        self.logger.info("Excel dashboard created successfully.")

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
    parser = argparse.ArgumentParser(description='Stage 5: Executive Dashboard Generator')
    args = parser.parse_args()
    setup_logging(DEFAULT_VERBOSE)
    logger = logging.getLogger('dashboard_runner')

    base_out_dir, run_ts = BASE_OUTPUT_FOLDER, datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = base_out_dir / "05_dashboard" / run_ts; run_dir.mkdir(parents=True, exist_ok=True)
    
    analysis_run_folder = find_latest_run_folder(base_out_dir / "04_analysis")
    if not analysis_run_folder:
        logger.error("Could not find outputs from Stage 4. Please run it first."); return 1

    logger.info(f"Reading analysis data from: {analysis_run_folder.name}")
    analysis_file = analysis_run_folder / "analysis_results.json"
    if not analysis_file.exists():
        logger.error(f"Analysis results file not found: {analysis_file}"); return 1

    with analysis_file.open('r', encoding='utf-8') as f:
        analysis_data = json.load(f)

    dashboard_generator = DashboardGenerator(analysis_data, ollama_model=OLLAMA_MODEL)
    summary_text = dashboard_generator.generate_llm_summary()
    dashboard_path = run_dir / f"Executive_Dashboard_{run_ts}.xlsx"
    dashboard_generator.create_excel_dashboard(summary_text, dashboard_path)
    
    print("\n" + "="*60 + "\nDASHBOARD CREATION COMPLETE\n" + "="*60)
    print(f"  - Executive Dashboard saved to: {dashboard_path}")
    print("\n--- EXECUTIVE SUMMARY ---")
    print(summary_text)
    print("="*60)

    return 0

if __name__ == '__main__':
    raise SystemExit(main())