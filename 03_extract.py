#!/usr/bin/env python3
"""
03_Extraction - Structured Data Extractor (V4.1 - Verbatim Extraction, Final)

This script performs Stage 3 using a cloud API. It has been upgraded to prevent
the LLM from autocorrecting data by using a strict prompt and setting the
temperature to 0.0.
"""

import argparse
import json
import logging
import csv
import re
import os
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import concurrent.futures

# --- SCRIPT-RELATIVE PATH SETUP ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_OUTPUT_FOLDER = SCRIPT_DIR / "output"
# --- END PATH SETUP ---

# --- USER INPUT & API CONFIG ---
from dotenv import load_dotenv
load_dotenv()
DEFAULT_VERBOSE = False
MAX_WORKERS = 8 # Can use more workers for APIs as they scale better
API_PROVIDER = "openai"
API_CONFIG = {"openai": {"url": "https://api.openai.com/v1/chat/completions", "model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")}}
# --- END CONFIG ---


# =============================================================================
# 1. EXTRACTION LOGIC
# =============================================================================
REGEX_RULES = {
    "invoice": {"invoice_number": r"invoice[\s_.-]*(?:number|no|num|#)\.?[\s\S]*?([A-Z0-9-]{5,})", "due_date": r"due[\s_.-]*date[\s\S]*?(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", "total_amount": r"(?:total|balance)[\s_.-]*(?:amount|due)?[\s\S]*?\$?([\d,]+\.\d{2})", "vat_number": r"vat[\s_.-]*(?:no|num|number|#|registration)?[\s\S]*?(\b4\d{9}\b)"},
    "receipt": {"transaction_date": r"date[\s\S]*?(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", "total_amount": r"total[\s\S]*?\$?([\d,]+\.\d{2})", "vat_number": r"vat[\s_.-]*(?:no|num|number|#|registration)?[\s\S]*?(\b4\d{9}\b)"}
}

class DocumentExtractor:
    def __init__(self):
        self.api_config = API_CONFIG.get(API_PROVIDER)
        if not self.api_config or not self.api_config.get("api_key"): raise ValueError(f"API configuration for '{API_PROVIDER}' missing or key not found in .env.")

    def _get_extraction_schema(self, doc_type: str) -> Dict[str, Any] | None:
        schemas = {"invoice": {"invoice_number": "string", "vendor_name": "string", "total_amount": "number", "due_date": "YYYY-MM-DD", "vat_number": "string"}, "receipt": {"vendor_name": "string", "transaction_date": "YYYY-MM-DD", "total_amount": "number", "payment_method": "string", "vat_number": "string"}, "contract": {"contract_title": "string", "effective_date": "YYYY-MM-DD", "termination_date": "YYYY-MM-DD", "contracting_parties": ["string"]}, "bank_statement": {"bank_name": "string", "account_holder": "string", "statement_period": "string", "ending_balance": "number"}}
        return schemas.get(doc_type)

    def _truncate_text(self, text: str, max_length: int = 8000) -> str:
        if len(text) <= max_length: return text
        return f"{text[:int(max_length * 0.7)]}\n...\n{text[-int(max_length * 0.3):]}"

    def _clean_json_response(self, response: str) -> str:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        return match.group(0) if match else "{}"

    def extract_with_rules(self, text: str, doc_type: str) -> Dict[str, Any]:
        extracted_data, type_regex_rules = {}, REGEX_RULES.get(doc_type, {})
        for field, pattern in type_regex_rules.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if "amount" in field or "balance" in field: value = float(value.replace(",", ""))
                extracted_data[field], extracted_data[f"{field}_method"] = value, "regex"
        return extracted_data

    def extract_with_llm(self, text: str, doc_type: str, already_extracted: Dict) -> Dict[str, Any]:
        import requests
        target_schema, missing_fields = self._get_extraction_schema(doc_type), [f for f in self._get_extraction_schema(doc_type) if f not in already_extracted]
        if not missing_fields: return already_extracted
        
        missing_schema = {field: target_schema[field] for field in missing_fields}
        prompt = f"""You are a verbatim data extraction engine. Your ONLY job is to find and copy text that matches the required fields. DO NOT alter, correct, infer, or change any information. Extract the data EXACTLY as it appears in the text. If a field is not present, use a null value. Respond with ONLY a valid JSON object matching this schema: {json.dumps(missing_schema)}"""
        
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_config['api_key']}"}
        payload = {"model": self.api_config['model'], "messages": [{"role": "user", "content": f"{prompt}\n\nDOCUMENT TEXT:\n\"{self._truncate_text(text)}\""}], "temperature": 0.0, "response_format": {"type": "json_object"}}

        try:
            response = requests.post(self.api_config['url'], headers=headers, json=payload, timeout=90)
            response.raise_for_status()
            raw_json = response.json()['choices'][0]['message']['content']
            llm_data = json.loads(self._clean_json_response(raw_json))
            for field, value in llm_data.items(): already_extracted[field], already_extracted[f"{field}_method"] = value, "llm_fallback"
            already_extracted['extraction_status'] = 'success_hybrid'
            return already_extracted
        except Exception as e:
            already_extracted.update({'extraction_status': 'failed_llm', 'error': str(e)})
            return already_extracted

# =============================================================================
# 2. SCRIPT ORCHESTRATION AND REPORTING (Identical to Ollama Version)
# =============================================================================
def setup_logging(verbose: bool):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if Path.cwd() != SCRIPT_DIR: logging.warning(f"Running from {Path.cwd()}. Outputs will be saved in {BASE_OUTPUT_FOLDER}")
def find_latest_run_folder(stage_dir: Path) -> Path | None:
    run_folders = [d for d in stage_dir.iterdir() if d.is_dir()]
    return max(run_folders, key=lambda d: d.name) if run_folders else None
def write_reports(results: List[Dict[str, Any]], run_dir: Path, base_out_dir: Path, run_ts: str):
    if not results: return
    flat_results, all_keys = [], set(['document_name'])
    for res in results:
        flat_res = {'document_name': res['document_name']}
        for key, value in res['data'].items():
            if not isinstance(value, (list, dict)): flat_res[key], all_keys.add(key) = value, None
        flat_results.append(flat_res)
    try:
        csv_path = run_dir / 'extraction_summary.csv'
        with csv_path.open('w', newline='', encoding='utf-8') as f: writer = csv.DictWriter(f, fieldnames=sorted(list(all_keys))); writer.writeheader(); writer.writerows(flat_results)
        logging.info(f"Wrote extraction summary: {csv_path}")
        import pandas as pd
        df = pd.DataFrame(flat_results)
        tracker_path = base_out_dir / 'stage_tracker.xlsx'
        sheet_name = f'03_Extraction_{run_ts}'
        with pd.ExcelWriter(tracker_path, engine='openpyxl', mode='a', if_sheet_exists='new') as writer: df.to_excel(writer, index=False, sheet_name=sheet_name)
        logging.info(f"Updated master tracker: {tracker_path} [Sheet: {sheet_name}]")
        summary = {'run_timestamp': datetime.now().isoformat(), 'docs_processed': len(results), 'successful_extractions': sum(1 for r in results if 'success' in r['data'].get('extraction_status', ''))}
        (run_dir / 'extraction_run_summary.json').write_text(json.dumps(summary, indent=2))
        print("\n" + "="*60 + "\nEXTRACTION SUMMARY\n" + "="*60); print(json.dumps(summary, indent=2)); print("="*60)
    except Exception as e: logging.error(f"Failed to write one or more reports: {e}")
def process_single_document_llm(doc: Dict, ingestion_folder: Path, json_output_dir: Path, extractor: DocumentExtractor) -> Dict:
    doc_name, doc_type, text_content, partial_data = doc['name'], doc['type'], doc['text'], doc['partial_data']
    extracted_data = extractor.extract_with_llm(text_content, doc_type, partial_data)
    (json_output_dir / f"{doc_name}.json").write_text(json.dumps(extracted_data, indent=2), encoding='utf-8')
    return {'document_name': doc_name, 'data': extracted_data}
def main():
    parser = argparse.ArgumentParser(description='Stage 3: Structured Data Extractor (Verbatim API, Final)')
    args = parser.parse_args()
    setup_logging(DEFAULT_VERBOSE)
    logger = logging.getLogger('extract_runner')
    base_out_dir, run_ts = BASE_OUTPUT_FOLDER, datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = base_out_dir / "03_extraction" / run_ts
    json_output_dir = run_dir / "extracted_json"; json_output_dir.mkdir(parents=True, exist_ok=True)
    ingestion_run_folder, classification_run_folder = find_latest_run_folder(base_out_dir / "01_ingestion"), find_latest_run_folder(base_out_dir / "02_classification")
    if not ingestion_run_folder or not classification_run_folder: logger.error("Could not find outputs from Stage 1 or 2. Please run them first."); return 1
    classification_file = classification_run_folder / "classification_metrics.csv"
    if not classification_file.exists(): logger.error(f"Classification metrics not found: {classification_file}"); return 1
    documents_to_process = [{'name': row['document_name'], 'type': row['predicted_type']} for row in csv.DictReader(classification_file.open('r', encoding='utf-8'))]
    extractor = DocumentExtractor()
    final_results, docs_for_llm = [], []
    logger.info(f"--- Starting Pass 1: Regex extraction for {len(documents_to_process)} documents ---")
    for doc in documents_to_process:
        doc_name, doc_type = doc['name'], doc['type']
        text_path = ingestion_run_folder / "extracted_text" / f"{doc_name}.txt"
        if not text_path.exists(): final_results.append({'document_name': doc_name, 'data': {'extraction_status': 'failed', 'error': 'source text file not found'}}); continue
        text_content, target_schema = text_path.read_text(encoding='utf-8'), extractor._get_extraction_schema(doc_type)
        if not target_schema: final_results.append({'document_name': doc_name, 'data': {'extraction_status': 'skipped', 'reason': f'No schema for {doc_type}'}}); continue
        partial_data = extractor.extract_with_rules(text_content, doc_type)
        if len(partial_data) == len(target_schema) * 2:
            partial_data['extraction_status'] = 'success_regex'
            (json_output_dir / f"{doc_name}.json").write_text(json.dumps(partial_data, indent=2), encoding='utf-8')
            final_results.append({'document_name': doc_name, 'data': partial_data})
        else: docs_for_llm.append({'name': doc_name, 'type': doc_type, 'text': text_content, 'partial_data': partial_data})
    logger.info(f"--- Pass 1 Complete. {len(docs_for_llm)} documents require LLM gap-filling. ---")
    if docs_for_llm:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_doc = {executor.submit(process_single_document_llm, doc, ingestion_run_folder, json_output_dir, extractor): doc for doc in docs_for_llm}
            for future in concurrent.futures.as_completed(future_to_doc): final_results.append(future.result())
    write_reports(final_results, run_dir, base_out_dir, run_ts)
    return 0

if __name__ == '__main__': raise SystemExit(main())