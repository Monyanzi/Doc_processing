#!/usr/bin/env python3
"""
02_Classification - Document Classifier Runner (V6 - Filename Analysis & Batching)

This single script performs the complete Stage 2 of the document intelligence
pipeline. It uses a highly efficient "Hierarchy of Evidence" approach:
1. Tier 1: Classifies based on keywords in the FILENAME for instant, high-confidence results.
2. Tier 2: If Tier 1 fails, classifies based on keywords in the document CONTENT.
3. Tier 3: If both tiers fail, batches the documents for a final classification
   using a local Ollama LLM.
"""

import argparse
import json
import logging
import csv
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
from enum import Enum

# --- SCRIPT-RELATIVE PATH SETUP ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_OUTPUT_FOLDER = SCRIPT_DIR / "output"
# --- END PATH SETUP ---

# --- USER INPUT (defaults) ---
CONFIDENCE_THRESHOLD = 0.5
USE_LLM_FALLBACK = True
OLLAMA_MODEL = "gemma3:1b"
DEFAULT_VERBOSE = False
# --- END USER INPUT ---


# =============================================================================
# 1. SCHEMAS AND CLASSIFICATION LOGIC
# =============================================================================

class DocumentType(Enum):
    INVOICE = "invoice"; RECEIPT = "receipt"; PURCHASE_ORDER = "purchase_order"; CREDIT_NOTE = "credit_note"
    QUOTATION = "quotation"; BANK_STATEMENT = "bank_statement"; BALANCE_SHEET = "balance_sheet"
    PROFIT_AND_LOSS = "profit_and_loss"; CONTRACT = "contract"; RESUME = "resume"
    BILL_OF_LADING = "bill_of_lading"; PACKING_SLIP = "packing_slip"; UTILITY_BILL = "utility_bill"
    OTHER = "other"

# NEW: Tier 1 Rules - Checks the filename for obvious clues. Fast and highly accurate.
FILENAME_RULES = {
    DocumentType.INVOICE: ["invoice", "inv-"],
    DocumentType.RECEIPT: ["receipt"],
    DocumentType.PURCHASE_ORDER: ["purchase-order", "po-"],
    DocumentType.CREDIT_NOTE: ["credit-note", "credit_note"],
    DocumentType.BANK_STATEMENT: ["bank_statement", "bank-statement"],
    DocumentType.BALANCE_SHEET: ["balance sheet", "balance_sheet"],
    DocumentType.PROFIT_AND_LOSS: ["profit and loss", "p&l"],
    DocumentType.CONTRACT: ["contract", "agreement"],
    DocumentType.QUOTATION: ["quote", "quotation"],
}

# Tier 2 Rules - Checks the document content.
CONTENT_RULES = {
    DocumentType.INVOICE: ["invoice", "bill to", "remittance", "invoice number", "amt due"],
    DocumentType.RECEIPT: ["receipt", "cash sale", "payment confirmation", "paid", "credit card"],
    DocumentType.PURCHASE_ORDER: ["purchase order", "po number", "ship to", "vendor"],
    DocumentType.CREDIT_NOTE: ["credit note", "credit memo", "adjustment note"],
    DocumentType.BANK_STATEMENT: ["bank statement", "account summary", "statement period", "ending balance"],
    DocumentType.BALANCE_SHEET: ["balance sheet", "assets", "liabilities", "equity", "total assets"],
    DocumentType.PROFIT_AND_LOSS: ["profit and loss", "income statement", "revenue", "net income", "cogs"],
    DocumentType.CONTRACT: ["agreement", "contract", "parties", "whereas", "indemnification"],
    DocumentType.RESUME: ["resume", "curriculum vitae", "experience", "education", "skills"],
    DocumentType.BILL_OF_LADING: ["bill of lading", "bol", "carrier", "shipper", "consignee"],
    DocumentType.PACKING_SLIP: ["packing slip", "packing list", "item description", "quantity shipped"],
    DocumentType.UTILITY_BILL: ["utility bill", "gas bill", "electric bill", "water bill", "usage"],
}

class FastDocumentClassifier:
    """A hybrid classifier using a tiered 'Hierarchy of Evidence' approach."""
    def __init__(self, use_llm_fallback=True, ollama_model="gemma3:1b"):
        self.use_llm_fallback = use_llm_fallback
        self.ollama_model = ollama_model

    def _truncate_text(self, text: str, max_length: int = 4000) -> str:
        if len(text) <= max_length: return text
        return f"{text[:int(max_length * 0.6)]}\n...\n{text[-int(max_length * 0.4):]}"

    def _clean_json_response(self, response: str) -> str:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        return match.group(0) if match else "{}"

    def classify_with_rules(self, text: str, filename: str, confidence_threshold: float) -> Tuple[Any, float, Dict] | None:
        """Attempts to classify using the Tier 1 (Filename) and Tier 2 (Content) rules."""
        # --- TIER 1: Filename Analysis ---
        fn_lower = filename.lower()
        for doc_type, keywords in FILENAME_RULES.items():
            if any(kw in fn_lower for kw in keywords):
                return doc_type, 0.98, {"classification_method": "rules_filename"}
        
        # --- TIER 2: Content Keyword Analysis ---
        text_lower = text.lower()
        scores = {doc_type: sum(1 for kw in kws if kw in text_lower) for doc_type, kws in CONTENT_RULES.items()}
        best_match, max_score = max(scores.items(), key=lambda item: item[1])
        
        if max_score >= 3:
            return best_match, 0.95, {"classification_method": "rules_content_high"}
        if max_score > 0 and (0.6 if max_score == 2 else 0.4) >= confidence_threshold:
            return best_match, (0.6 if max_score == 2 else 0.4), {"classification_method": "rules_content_low"}
        
        return None # Indicates failure, requires LLM fallback

    def classify_with_llm(self, text: str) -> (str, float, dict):
        """Tier 3: Classifies a single document using the generative LLM."""
        truncated_text = self._truncate_text(text)
        prompt = f"""You are an expert document analyst. Analyze the following text and return a concise document type. Examples: "Invoice", "Bank Statement", "Employment Contract". Respond with ONLY a valid JSON object in this exact format: {{"predicted_type": "The concise document type", "confidence": 0.85, "reasoning": "A brief explanation for the classification.", "key_indicators": ["list", "of", "key", "words"]}} TEXT: "{truncated_text}" """
        
        try:
            import ollama
            logging.info(f"Engaging local LLM ({self.ollama_model})...")
            response = ollama.chat(model=self.ollama_model, messages=[{'role': 'user', 'content': prompt}], format='json')
            raw_json = response['message']['content']
            data = json.loads(self._clean_json_response(raw_json))
            metadata = {"classification_method": "llm_generative", "reasoning": data.get("reasoning", ""), "key_indicators": data.get("key_indicators", [])}
            return data.get("predicted_type", "LLM_Parse_Error"), float(data.get("confidence", 0.5)), metadata
        except Exception as e:
            logging.error(f"Ollama LLM classification failed: {e}")
            return "LLM_Connection_Error", 0.0, {"classification_method": "llm_generative", "error": str(e)}

# =============================================================================
# 2. SCRIPT ORCHESTRATION AND REPORTING
# =============================================================================
def setup_logging(verbose: bool):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if Path.cwd() != SCRIPT_DIR: logging.warning(f"Running from {Path.cwd()}. Outputs will be saved in {BASE_OUTPUT_FOLDER}")

def find_latest_ingestion_run(ingestion_dir: Path) -> Path | None:
    run_folders = [d for d in ingestion_dir.iterdir() if d.is_dir() and (d / "extracted_text").exists()]
    return max(run_folders, key=lambda d: d.name) if run_folders else None

def write_reports(results: List[Dict[str, Any]], run_dir: Path, base_out_dir: Path, run_ts: str):
    if not results: return
    try:
        with (run_dir / 'classification_metrics.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys()); writer.writeheader(); writer.writerows(results)
        logging.info(f"Wrote per-file metrics to {run_dir / 'classification_metrics.csv'}")
        
        type_counts = {}; [type_counts.update({r['predicted_type']: type_counts.get(r['predicted_type'], 0) + 1}) for r in results]
        summary = {'run_timestamp': datetime.now().isoformat(), 'docs_total': len(results), 'llm_fallbacks_used': sum(1 for r in results if 'llm' in r['method']), 'type_distribution': type_counts}
        (run_dir / 'classification_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        logging.info(f"Wrote summary to {run_dir / 'classification_summary.json'}")
        
        import pandas as pd
        df = pd.DataFrame(results)[['document_name', 'predicted_type', 'confidence', 'method', 'reasoning']]
        tracker_path = base_out_dir / 'stage_tracker.xlsx'
        sheet_name = f'02_Classification_{run_ts}'
        with pd.ExcelWriter(tracker_path, engine='openpyxl', mode='a', if_sheet_exists='new') as writer: df.to_excel(writer, index=False, sheet_name=sheet_name)
        logging.info(f"Updated master tracker: {tracker_path} [Sheet: {sheet_name}]")
        
        print("\n" + "="*60 + "\nCLASSIFICATION SUMMARY\n" + "="*60); print(json.dumps(summary, indent=2)); print("="*60)
    except Exception as e:
        logging.error(f"Failed to write one or more reports: {e}")

def main():
    parser = argparse.ArgumentParser(description='Stage 2: Document Classifier Runner (V6 - Filename Analysis)')
    parser.add_argument('--run-folder', type=str, help='Path to a specific 01_ingestion run folder. If omitted, the latest is used.')
    args = parser.parse_args()
    setup_logging(DEFAULT_VERBOSE)
    logger = logging.getLogger('classify_runner')

    base_out_dir = BASE_OUTPUT_FOLDER; run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = base_out_dir / "02_classification" / run_ts; run_dir.mkdir(parents=True, exist_ok=True)
    
    source_run_folder = Path(args.run_folder) if args.run_folder else find_latest_ingestion_run(base_out_dir / "01_ingestion")
    if not source_run_folder:
        logger.error(f"Valid Stage 1 output folder not found. Please run Stage 1 first."); return 1

    logger.info(f"Processing text files from: {source_run_folder}")
    text_files = sorted((source_run_folder / "extracted_text").glob("*.txt"))
    if not text_files:
        logger.warning("No text files found in the source folder. Exiting."); return 0
    
    classifier = FastDocumentClassifier(use_llm_fallback=USE_LLM_FALLBACK, ollama_model=OLLAMA_MODEL)
    logger.info(f"Classifier initialized. LLM Fallback is {'ENABLED' if USE_LLM_FALLBACK else 'DISABLED'}.")

    final_results, docs_for_llm = [], []
    logger.info(f"--- Starting Pass 1: Rules-based classification for {len(text_files)} documents ---")
    for text_path in text_files:
        try:
            text_content = text_path.read_text(encoding='utf-8')
            if not text_content.strip(): raise ValueError("Empty text file")
            
            # Pass BOTH text and filename to the smarter classifier
            rules_result = classifier.classify_with_rules(text_content, text_path.name, CONFIDENCE_THRESHOLD)
            
            if rules_result:
                doc_type_obj, confidence, metadata = rules_result
                final_results.append({'document_name': text_path.stem, 'predicted_type': doc_type_obj.value, 'confidence': confidence, 'method': metadata['classification_method'], 'reasoning': 'Classified by keyword rules.', 'key_indicators': '[]', 'error': ''})
            else:
                docs_for_llm.append((text_path, text_content))
        except Exception as e:
            final_results.append({'document_name': text_path.stem, 'predicted_type': 'ERROR', 'confidence': 0.0, 'method': 'rules_error', 'reasoning': '', 'key_indicators': '[]', 'error': str(e)})

    logger.info(f"--- Pass 1 Complete. {len(docs_for_llm)} documents require LLM fallback. ---")
    if docs_for_llm and classifier.use_llm_fallback:
        for text_path, text_content in docs_for_llm:
            doc_type_str, confidence, metadata = classifier.classify_with_llm(text_content)
            final_results.append({'document_name': text_path.stem, 'predicted_type': doc_type_str.lower(), 'confidence': round(confidence, 4), 'method': metadata.get('classification_method', 'llm_generative'), 'reasoning': metadata.get('reasoning', ''), 'key_indicators': str(metadata.get('key_indicators', [])), 'error': metadata.get('error', '')})
    elif docs_for_llm:
        for text_path, _ in docs_for_llm:
            final_results.append({'document_name': text_path.stem, 'predicted_type': DocumentType.OTHER.value, 'confidence': 0.0, 'method': 'rules_failed', 'reasoning': 'Rules failed and LLM fallback is disabled.', 'key_indicators': '[]', 'error': ''})

    write_reports(final_results, run_dir, base_out_dir, run_ts)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())