#!/usr/bin/env python3
"""
=================================================
Unified Invoice Processing Pipeline with OCR (Steps 1, 2, 3)
=================================================

This single script performs the complete invoice processing workflow:
1.  **PDF Text & OCR Extraction:** Extracts text from native PDFs and uses OCR for scanned/image-based PDFs.
2.  **Document Classification:** Identifies and skips non-invoice documents like credit notes.
3.  **Data Parsing:** Uses a library of regex patterns to parse text into structured data.
4.  **Validation & Cleaning:** Validates, cleans, and formats the data.
5.  **Consolidation:** Creates final, consolidated CSV reports.

This script processes all data in memory to avoid creating intermediate files
and folders, saving only the final results.

Dependencies:
pip install pdfplumber pandas numpy python-dateutil pytesseract Pillow pdf2image

**External Dependency Note:**
This script requires Google's Tesseract-OCR engine and Poppler.
- Tesseract: https://github.com/tesseract-ocr/tesseract
- Poppler: https://github.com/oschwartz10612/poppler-windows/releases/

"""

import os
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from dateutil import parser as date_parser
from typing import Dict, List, Optional, Tuple, Any
import pdfplumber
import pytesseract
from PIL import Image

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None


# --- USER INPUT ---
# Define your folder paths here

# Input folder containing the original PDF invoice files
INPUT_FOLDER = r"C:\Users\Monya\Desktop\Cursor\Invoice Automation\sample-documents"

# A single, clean output folder for all final reports
OUTPUT_FOLDER = r"C:\Users\Monya\Desktop\Cursor\Invoice Automation\final_output"

# --- TESSERACT OCR CONFIGURATION ---
# If tesseract is not in your system's PATH, provide the path to the executable
# Example: TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
TESSERACT_CMD = None # Set to None if tesseract is in your PATH

# --- PROCESSING SETTINGS ---

# Step 1: PDF Extraction Settings
MAX_FILE_SIZE_MB = 50
OCR_RESOLUTION = 300 # DPI for OCR image conversion

# Step 2: Regex Parsing Settings
DEBUG_MODE = False  # Set to True for detailed console logs during parsing
MIN_CONFIDENCE_SCORE = 0.5 # Lowered slightly to accommodate OCR variations

# Step 3: Validation & Cleaning Settings
STRICT_VALIDATION = True  # Enable strict validation rules (e.g., must have invoice #)
REMOVE_DUPLICATES = True  # Remove duplicate line items within the same invoice
MIN_UNIT_PRICE = 0.01
MAX_UNIT_PRICE = 100000
MIN_QUANTITY = 0.001
MAX_QUANTITY = 10000
STANDARDIZE_DESCRIPTIONS = True
FIX_CALCULATION_ERRORS = True
CURRENCY_SYMBOL = "$"

# --- END USER INPUT ---

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


class InvoiceParser:
    """
    (Step 2 Logic)
    Extracts structured data from invoice text using a library of regex patterns.
    """
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        self.setup_regex_patterns()

    def setup_regex_patterns(self):
        """Define a comprehensive library of regex patterns to handle various invoice formats."""
        self.invoice_patterns = [
            r'invoice\s*(?:no\.?|number|#)\s*:?\s*([A-Z0-9\-_]+)',
            r'inv\.?\s*(?:no\.?|#)\s*:?\s*([A-Z0-9\-_]+)',
            r'bill\s*(?:no\.?|#)\s*:?\s*([A-Z0-9\-_]+)',
            r'reference\s*(?:no\.?|#)\s*:?\s*([A-Z0-9\-_]+)',
        ]
        self.date_patterns = [
            r'(?:invoice\s*)?date\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
            r'(\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{4})',
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})',
            r'(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})', # ISO format
        ]
        self.total_patterns = [
            r'(?:grand\s*)?total\s*(?:amount)?\s*:?\s*\$?\s*([0-9,]+\.\d{2})',
            r'amount\s*due\s*:?\s*\$?\s*([0-9,]+\.\d{2})',
            r'balance\s*due\s*:?\s*\$?\s*([0-9,]+\.\d{2})',
        ]
        self.line_item_patterns = [
            r'([A-Za-z][A-Za-z\s\-&\(\)]{5,50}?)\s{2,}(\d+(?:\.\d+)?)\s{2,}\$?\s*([0-9,]+\.?\d{0,2})\s{2,}\$?\s*([0-9,]+\.?\d{0,2})',
            r'([A-Za-z].*?)\s+(\d+)\s+@\s+\$?([\d,]+\.\d{2})\s+\$?([\d,]+\.\d{2})'
        ]
        self.vendor_patterns = [
            r'(?:from|bill\s*to|vendor|company)\s*:?\s*([A-Za-z][A-Za-z\s&\.,\-]{5,50})',
            r'^\s*([A-Z][A-Z\s&\.,\-]{10,50})\s*$',
        ]
        self.tax_patterns = [
            r'(?:sales\s*)?tax\s*(?:\([0-9.%]+\))?\s*:?\s*\$?\s*([0-9,]+\.?\d{0,2})',
            r'vat\s*(?:\([0-9.%]+\))?\s*:?\s*\$?\s*([0-9,]+\.?\d{0,2})'
        ]

    def parse_text(self, text_content: str) -> Dict:
        """Main parsing method to process raw text."""
        basic_fields = self._extract_basic_fields(text_content)
        line_items = self._extract_line_items(text_content)
        overall_confidence = self._calculate_overall_confidence(basic_fields, line_items)

        return {
            "overall_confidence": overall_confidence,
            "basic_fields": basic_fields,
            "line_items": line_items,
        }

    def _extract_field_with_patterns(self, text: str, patterns: List[str], field_name: str) -> Tuple[Optional[str], float]:
        """Extract a field using multiple regex patterns."""
        for i, pattern in enumerate(patterns):
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                value = match.group(1).strip()
                confidence = 1.0 - (i * 0.1)
                if self.debug_mode:
                    print(f"  ✓ {field_name}: '{value}' (conf: {confidence:.2f})")
                return value, confidence
        return None, 0.0

    def _extract_basic_fields(self, text: str) -> Dict:
        """Extract main invoice fields like number, date, and total."""
        extracted_data = {"confidence_scores": {}}

        val, conf = self._extract_field_with_patterns(text, self.invoice_patterns, "invoice_number")
        if val:
            extracted_data["invoice_number"] = val
            extracted_data["confidence_scores"]["invoice_number"] = conf

        val, conf = self._extract_field_with_patterns(text, self.date_patterns, "invoice_date")
        if val:
            try:
                extracted_data["invoice_date"] = date_parser.parse(val, fuzzy=True).strftime("%Y-%m-%d")
                extracted_data["confidence_scores"]["invoice_date"] = conf
            except (ValueError, TypeError): pass

        val, conf = self._extract_field_with_patterns(text, self.total_patterns, "total_amount")
        if val:
            try:
                extracted_data["total_amount"] = float(val.replace(",", ""))
                extracted_data["confidence_scores"]["total_amount"] = conf
            except ValueError: pass

        val, conf = self._extract_field_with_patterns(text, self.vendor_patterns, "vendor_name")
        if val:
            extracted_data["vendor_name"] = val.title()
            extracted_data["confidence_scores"]["vendor_name"] = conf
            
        return extracted_data

    def _extract_line_items(self, text: str) -> List[Dict]:
        """Extract all line items from the invoice text."""
        line_items = []
        for pattern_num, pattern in enumerate(self.line_item_patterns, 1):
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                try:
                    quantity = float(match[1])
                    unit_price = float(match[2].replace(",", ""))
                    total_price = float(match[3].replace(",", ""))
                    
                    if quantity > 0 and unit_price > 0 and total_price > 0:
                        line_items.append({
                            "description": match[0].strip().title(),
                            "quantity": quantity,
                            "unit_price": unit_price,
                            "total_price": total_price,
                            "confidence": 0.9 - (pattern_num - 1) * 0.1
                        })
                except (ValueError, IndexError): continue
        return line_items

    def _calculate_overall_confidence(self, basic_fields: Dict, line_items: List[Dict]) -> float:
        """Calculate a weighted confidence score for the entire extraction."""
        basic_conf = np.mean(list(basic_fields.get("confidence_scores", {}).values())) if basic_fields.get("confidence_scores") else 0
        line_item_conf = np.mean([item['confidence'] for item in line_items]) if line_items else 0
        
        if basic_conf > 0 and line_item_conf > 0:
            return (basic_conf * 0.4) + (line_item_conf * 0.6)
        return max(basic_conf, line_item_conf)


class DataValidator:
    """
    (Step 3 Logic)
    Validates and cleans the structured data extracted by the InvoiceParser.
    """
    def __init__(self, strict_mode=True):
        self.strict_mode = strict_mode
        self.errors = []
        self.warnings = []

    def validate_data(self, parsed_data: Dict) -> Tuple[Dict, bool]:
        """Main validation method."""
        self.errors, self.warnings = [], []
        
        basic_fields = self._validate_basic_fields(parsed_data.get("basic_fields", {}))
        line_items = self._validate_line_items(parsed_data.get("line_items", []))

        is_valid = not self.errors
        
        validated_data = {
            "basic_fields": basic_fields,
            "line_items": line_items,
            "validation_summary": {
                "passed": is_valid,
                "errors": self.errors,
                "warnings": self.warnings
            }
        }
        return validated_data, is_valid

    def _validate_basic_fields(self, fields: Dict) -> Dict:
        """Validate the main invoice fields."""
        if not fields.get("invoice_number") and self.strict_mode:
            self.errors.append("Missing invoice number")
        if not fields.get("invoice_date") and self.strict_mode:
            self.errors.append("Missing invoice date")
        if fields.get("total_amount", 0) < 0:
            self.errors.append("Negative total amount")
        return fields

    def _validate_line_items(self, items: List[Dict]) -> List[Dict]:
        """Validate each line item for correctness."""
        valid_items = []
        for item in items:
            is_item_valid = True
            if not (MIN_QUANTITY <= item.get("quantity", 0) <= MAX_QUANTITY):
                self.warnings.append(f"Invalid quantity for item '{item['description']}'")
                is_item_valid = False
            if not (MIN_UNIT_PRICE <= item.get("unit_price", 0) <= MAX_UNIT_PRICE):
                self.warnings.append(f"Invalid unit price for item '{item['description']}'")
                is_item_valid = False
            
            calculated_total = item['quantity'] * item['unit_price']
            if abs(item['total_price'] - calculated_total) > 0.02 and FIX_CALCULATION_ERRORS:
                item['total_price'] = round(calculated_total, 2)
                item['price_corrected'] = True

            if is_item_valid:
                if STANDARDIZE_DESCRIPTIONS:
                    item['description'] = " ".join(item['description'].split())
                valid_items.append(item)
        
        return valid_items


class InvoicePipeline:
    """
    Orchestrator for the entire PDF-to-CSV pipeline.
    """
    def __init__(self, input_dir: str, output_dir: str):
        self.input_path = Path(input_dir)
        self.output_path = Path(output_dir)
        self.parser = InvoiceParser(debug_mode=DEBUG_MODE)
        self.validator = DataValidator(strict_mode=STRICT_VALIDATION)
        self.all_results = []

    def run(self):
        """Execute the entire processing pipeline."""
        print("🚀 Starting Unified Invoice Processing Pipeline...")
        self._setup_output_folder()
        pdf_files = self._find_pdf_files()

        if not pdf_files:
            print("❌ No PDF files found in the input directory. Exiting.")
            return

        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_file.name}")
            
            text_content = self._extract_text_from_pdf(pdf_file)
            if not text_content:
                print(f"  ⚠️ Skipping file due to total text extraction failure.")
                continue

            # **NEW**: Classify document type before processing
            doc_type = self._classify_document(text_content)
            if doc_type != "invoice":
                print(f"  ⚠️ Skipping file: Classified as '{doc_type}'.")
                continue
            
            parsed_data = self.parser.parse_text(text_content)
            if parsed_data["overall_confidence"] < MIN_CONFIDENCE_SCORE:
                print(f"  ⚠️ Skipping file due to low confidence score ({parsed_data['overall_confidence']:.2f})")
                continue

            validated_data, is_valid = self.validator.validate_data(parsed_data)
            
            result = {"source_file": pdf_file.name, **validated_data}
            self.all_results.append(result)
            
            if is_valid:
                print("  ✅ Processing and validation successful.")
            else:
                print(f"  ❌ Validation failed: {validated_data['validation_summary']['errors']}")

        self._create_consolidated_datasets()
        print(f"\n✅ Pipeline finished. All reports saved to: {self.output_path}")

    def _setup_output_folder(self):
        """Create the main output folder and necessary subfolders."""
        print(f"📁 Setting up output directory: {self.output_path}")
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / "final_reports").mkdir(exist_ok=True)
        (self.output_path / "logs_and_reports").mkdir(exist_ok=True)

    def _find_pdf_files(self) -> List[Path]:
        """Find all PDF files in the input directory, filtering by size."""
        print(f"🔍 Searching for PDF files in: {self.input_path}")
        all_pdfs = list(self.input_path.rglob("*.pdf"))
        
        valid_files = [
            pdf for pdf in all_pdfs 
            if pdf.stat().st_size / (1024 * 1024) <= MAX_FILE_SIZE_MB
        ]
        print(f"📄 Found {len(valid_files)} PDF files to process.")
        return valid_files

    def _classify_document(self, text: str) -> str:
        """Identifies the document type based on keywords."""
        text_lower = text.lower()
        if "credit note" in text_lower or "credit memo" in text_lower:
            return "credit_note"
        if "remittance advice" in text_lower:
            return "remittance_advice"
        if "purchase order" in text_lower:
            return "purchase_order"
        if "invoice" in text_lower:
            return "invoice"
        # Default to invoice if no other keywords are found but it contains common invoice fields
        if any(re.search(p, text_lower) for p in self.parser.invoice_patterns):
            return "invoice"
        return "unknown"

    def _extract_text_from_pdf(self, pdf_path: Path) -> Optional[str]:
        """
        Extracts text from a PDF, first trying direct extraction, then falling back to OCR.
        """
        print("  - Extracting text...")
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            if len(text.strip()) < 150: # Increased threshold
                print("  - Low text detected, attempting OCR...")
                if convert_from_path is None:
                    print("  - ❌ CRITICAL: 'pdf2image' is not installed, and OCR on PDFs will fail.")
                    print("  - Please run 'pip install pdf2image' and install Poppler to process scanned documents.")
                    return text # Return whatever little text we got
                
                try:
                    images = convert_from_path(pdf_path, dpi=OCR_RESOLUTION)
                    ocr_text = ""
                    for img in images:
                        ocr_text += pytesseract.image_to_string(img) + "\n"
                    
                    if len(ocr_text) > len(text):
                        print("  - OCR provided more text. Using OCR result.")
                        return ocr_text
                except Exception as ocr_error:
                    print(f"  - ❌ OCR failed for {pdf_path.name}: {ocr_error}")
                    print("  - Ensure Poppler is installed and in your system's PATH.")
                    return text # Return original text on failure
            
            return text

        except Exception as e:
            print(f"  ❌ Error processing {pdf_path.name}: {e}")
            return None

    def _create_consolidated_datasets(self):
        """Creates and saves the final summary and line item CSVs."""
        print("\n📊 Creating consolidated final reports...")

        valid_results = [r for r in self.all_results if r['validation_summary']['passed']]
        if not valid_results:
            print("  ⚠️ No valid invoices were processed. No final reports will be generated.")
            return

        invoice_summaries = [
            {
                "source_file": res['source_file'],
                "invoice_number": res['basic_fields'].get('invoice_number'),
                "invoice_date": res['basic_fields'].get('invoice_date'),
                "vendor_name": res['basic_fields'].get('vendor_name'),
                "total_amount": res['basic_fields'].get('total_amount'),
                "line_items_count": len(res['line_items'])
            } for res in valid_results
        ]
        invoices_df = pd.DataFrame(invoice_summaries)
        invoices_path = self.output_path / "final_reports" / "consolidated_invoices.csv"
        invoices_df.to_csv(invoices_path, index=False)
        print(f"  - Saved consolidated invoices summary to: {invoices_path}")

        all_line_items = []
        for res in valid_results:
            invoice_num = res['basic_fields'].get('invoice_number', 'N/A')
            for item in res['line_items']:
                item['invoice_number'] = invoice_num
                all_line_items.append(item)
        
        line_items_df = pd.DataFrame(all_line_items)
        line_items_path = self.output_path / "final_reports" / "consolidated_line_items.csv"
        line_items_df.to_csv(line_items_path, index=False)
        print(f"  - Saved consolidated line items to: {line_items_path}")
        
        summary_log = {
            "processing_time": datetime.now().isoformat(),
            "total_files_found": len(list(self.input_path.rglob("*.pdf"))),
            "total_files_processed": len(self.all_results),
            "successful_invoices": len(valid_results),
            "failed_or_skipped_invoices": len(self.all_results) - len(valid_results),
        }
        log_path = self.output_path / "logs_and_reports" / f"summary_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_path, 'w') as f:
            json.dump(summary_log, f, indent=2)
        print(f"  - Saved processing summary log to: {log_path}")


def main():
    """Main function to run the pipeline."""
    if not INPUT_FOLDER or not OUTPUT_FOLDER:
        print("❌ Error: Please set INPUT_FOLDER and OUTPUT_FOLDER paths at the top of the script.")
        return
    
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Error: Input folder does not exist: {INPUT_FOLDER}")
        return
    
    pipeline = InvoicePipeline(input_dir=INPUT_FOLDER, output_dir=OUTPUT_FOLDER)
    pipeline.run()


if __name__ == "__main__":
    main()
