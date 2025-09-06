# Document Intelligence Pipeline (V2 local-first)

A comprehensive, LLM-powered document processing pipeline that automatically ingests, classifies, extracts, and validates information from various document types including invoices, receipts, contracts, reports, and purchase orders.

## 🚀 Features

- **Multi-format Support**: PDF, images (PNG, JPG, TIFF, BMP), and text files
- **Intelligent Classification**: LLM-based (Ollama local models) or rules-first classification with confidence scoring
- **Advanced Extraction**: Rules-first extraction with optional LLM fallback
- **Quality Assurance**: Comprehensive validation with arithmetic checks, date validation, and business logic rules
- **OCR Fallback**: Automatic OCR when PDF text extraction fails
- **Batch Processing**: Efficient processing of multiple documents
- **Rich Output**: JSON results per document + CSV summary with validation flags

## 📋 Supported Document Types

- **Invoices**: Vendor, invoice number, dates, amounts, line items, tax calculations
- **Receipts**: Vendor, transaction date, total amount, payment method, items
- **Contracts**: Parties, effective date, contract value, terms, signatures
- **Reports**: Title, author, date, abstract, keywords, sections
- **Purchase Orders**: PO number, vendor, order date, amounts, line items
- **Other**: Flexible field extraction for unrecognized document types

## 🏗️ Architecture

The pipeline follows a modular, 5-stage architecture:

1. **Document Ingestion** (`ingest.py`): Text extraction with OCR fallback
2. **Classification** (`classify.py`): LLM-based document type identification
3. **Field Extraction** (`extract.py`): Type-specific information extraction
4. **Quality Assurance** (`qa_guard.py`): Validation and issue flagging
5. **Pipeline Orchestration** (`pipeline.py`): Coordinates all stages

## 📦 Installation

### Prerequisites

- Python 3.8+
- Tesseract OCR (for image processing)
- Ollama installed and running (for local LLM) or OpenAI API key if using cloud

### Setup

1. **Clone and navigate to the project:**
   ```bash
   cd document-intelligence-pipeline
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Tesseract OCR:**
   - **Windows**: Download from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`

5. **Set up environment variables:**
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

## ⚙️ Configuration

The pipeline is configured via `config.yaml`.

Recommended local-first model configuration (Ollama):

```yaml
# LLM Model Configuration
model:
  provider: "ollama"
  name: "gemma3:1b"
  temperature: 0.0
  max_tokens: 2000

# Ollama Configuration
ollama:
  base_url: "http://localhost:11434"
  timeout: 120
  retry_attempts: 3
```

```yaml
# LLM Model Configuration
model:
  name: "gpt-4"  # or "gpt-3.5-turbo"
  temperature: 0.0
  max_tokens: 2000

# OCR Configuration
ocr:
  language: "eng"
  fallback: true
  confidence_threshold: 0.7

# Classification Configuration
classification:
  confidence_threshold: 0.8
  document_types: [invoice, receipt, contract, report, purchase_order, other]

# File Paths
paths:
  input: "../Invoice Automation/sample-documents"
  output: "./output"
  temp: "./temp"
  logs: "./logs"
```

## 🚀 Usage

### V2 Quick Start (Stage 01 Ingestion, multi-folder, Excel tracker)

Use the dedicated ingestion runner `01_ingest.py` to ingest one or more folders, write per-file metrics (CSV), a summary (JSON), and an Excel tracker (`output/stage_tracker.xlsx`).

User-config (at top of `01_ingest.py`):

```python
# --- USER INPUT ---
INPUT_FOLDERS = [
    r"C:\\Users\\Monya\\Documents\\Visa\\Bank_Statement_Savings_Account",
    r"C:\\Users\\Monya\\Documents\\Visa\\Mexico",
    r"C:\\Users\\Monya\\Desktop\\Cursor\\AI_Accountant\\documents_sorted",
]

OUTPUT_FOLDER = r"./output"          # Where CSV/JSON/Excel reports are written
MAX_FILE_SIZE_MB = 50                 # Optional cap; set None to disable
CREATE_EXTRACTION_LOG = True          # Writes a simple text log per run
ALLOW_REUSE = True                    # Reuse text under *extracted_text* if found
TESSERACT_PATH = None                 # Optional Windows override path to tesseract.exe
DEFAULT_VERBOSE = False
# --- END USER INPUT ---
```

Run options:

```bash
# Run with configured INPUT_FOLDERS
python 01_ingest.py --no-reuse

# Run a single folder (overrides INPUT_FOLDERS)
python 01_ingest.py --input "C:\Users\Monya\Documents\Visa\Mexico" --no-reuse

# Custom output directory and max size
python 01_ingest.py -i "C:\path\to\docs" -o "C:\path\to\out" --max-size-mb 100
```

Outputs:

- `output/ingestion_per_file_*.csv`: Per-file records of method (PDF_TEXT/OCR/REUSED), success, text length
- `output/ingestion_summary_*.json`: Run summary with success rates and method distribution
- `output/stage_tracker.xlsx`: Excel with sheet `01_Ingestion_YYYYMMDD_HHMMSS` and columns `[document_name, source_folder, path, extension, method]`

### V2 Quick Start (Stage 02 Classification, rules-first + optional LLM fallback)

Use the Stage 02 runner `02_classify.py` to classify documents found in the latest Stage 01 tracker sheet (default) or from a specific folder. It writes per-file metrics (CSV), a summary (JSON), and appends a classification sheet to `output/stage_tracker.xlsx`.

Basic runs:

```bash
# Classify using the latest 01_Ingestion sheet in output/stage_tracker.xlsx
python 02_classify.py

# Classify a specific folder (overrides tracker)
python 02_classify.py --input "C:\path\to\docs"

# Disable LLM fallback and raise the confidence threshold
python 02_classify.py --no-llm --confidence-threshold 0.8
```

Outputs:

- `output/classification_per_file_*.csv`: Per-file predicted type, confidence, method, error (if any)
- `output/classification_summary_*.json`: Summary with type counts, avg confidence, unknown_rate, fallback_rate
- `output/stage_tracker.xlsx`: Excel with sheet `02_Classification_YYYYMMDD_HHMMSS` and columns `[document_name, path, predicted_type, confidence, method]`

Notes:

- Rules/regex-first classification via `fast_classifier.py`; optional Ollama fallback if confidence is low (`--no-llm` to disable)
- Ensure `openpyxl` and `pandas` are installed for Excel output; see `requirements.txt`
- For OCR-backed text extraction, install Tesseract and optionally set `--tesseract-path` if not in PATH

### Basic Usage

```bash
# Process documents using default configuration
python run.py

# Process with verbose logging
python run.py --verbose

# Process from custom input directory
python run.py --input /path/to/documents

# Custom output directory
python run.py --output /path/to/results
```

### Advanced Options

```bash
# Show pipeline configuration
python run.py --show-config

# Dry run (show what would be processed)
python run.py --dry-run

# Process specific document types only
python run.py --types invoice receipt

# Limit number of documents
python run.py --max-docs 10

# Custom configuration file
python run.py --config custom_config.yaml
```

### Programmatic Usage

```python
from pipeline import DocumentIntelligencePipeline

# Initialize pipeline
pipeline = DocumentIntelligencePipeline()

# Process documents
summary = pipeline.process_documents("/path/to/documents")

# Access results
print(f"Processed {summary.total_documents} documents")
print(f"Success rate: {summary.successful_documents/summary.total_documents:.1%}")
```

## 📊 Output

### Individual Document Results

Each processed document generates a JSON file with:

```json
{
  "file_path": "/path/to/document.pdf",
  "file_name": "document.pdf",
  "processing_timestamp": "2024-01-15T10:30:00",
  "extraction": {
    "method": "pdf_text",
    "text_length": 1500,
    "metadata": {...}
  },
  "classification": {
    "document_type": "invoice",
    "confidence": 0.95,
    "metadata": {...}
  },
  "extraction_results": {
    "vendor": "ABC Company",
    "invoice_number": "INV-2024-001",
    "total_amount": 1250.00,
    ...
  },
  "validation": {
    "issues": [...],
    "issue_count": 0,
    "has_errors": false,
    "has_warnings": false
  },
  "status": "success"
}
```

### Summary CSV

A summary CSV file provides an overview:

| file_name | document_type | confidence | issue_count | has_errors | has_warnings | status | vendor | total_amount | currency |
|-----------|---------------|------------|-------------|------------|--------------|---------|---------|--------------|----------|
| invoice1.pdf | invoice | 0.95 | 0 | false | false | success | ABC Co | 1250.00 | USD |
| receipt1.pdf | receipt | 0.88 | 1 | false | true | success | XYZ Store | 45.99 | USD |

## 🔍 Quality Assurance

The pipeline includes comprehensive validation:

- **Required Fields**: Ensures essential fields are present
- **Format Validation**: Validates dates, amounts, currency codes
- **Arithmetic Checks**: Verifies calculations (e.g., subtotal + tax = total)
- **Date Logic**: Ensures due dates > invoice dates, no future dates
- **Business Rules**: Flags suspicious amounts, missing vendor info

### Validation Issue Types

- **ERROR**: Critical issues that prevent successful processing
- **WARNING**: Issues that may indicate problems but don't block processing
- **INFO**: Informational flags for review

## 🧪 Testing

### Test Individual Components

```bash
# Test document ingestion
python ingest.py

# Test classification
python classify.py

# Test field extraction
python extract.py

# Test QA validation
python qa_guard.py
```

### Test Complete Pipeline

```bash
# Dry run to see what would be processed
python run.py --dry-run

# Process a small subset
python run.py --max-docs 5
```

## 🔧 Customization

### Adding New Document Types

1. **Update schemas.py**: Add new document type enum and schema
2. **Update classify.py**: Add classification patterns
3. **Update extract.py**: Add extraction prompts
4. **Update qa_guard.py**: Add validation rules
5. **Update config.yaml**: Add to document types list

### Custom Validation Rules

```python
# In qa_guard.py
def _check_custom_business_logic(self, doc_type, extracted_fields):
    issues = []
    
    # Add your custom validation logic here
    if doc_type == DocumentType.INVOICE:
        # Custom invoice validation
        pass
    
    return issues
```

### Custom Extraction Prompts

```python
# In extract.py
def _get_custom_prompt(self) -> str:
    return """
    You are an expert at extracting custom information.
    Respond with ONLY a valid JSON object:
    {
        "custom_field": "value"
    }
    """
```

## 📈 Performance

### Optimization Tips

- **Batch Processing**: Process multiple documents together
- **Model Selection**: Use GPT-3.5-turbo for cost optimization
- **Text Truncation**: Pipeline automatically truncates long documents
- **Parallel Processing**: Future versions will support concurrent processing

### Expected Performance

- **PDF Text Extraction**: ~0.1-0.5 seconds per page
- **OCR Processing**: ~2-5 seconds per page
- **LLM Classification**: ~1-3 seconds per document
- **Field Extraction**: ~2-5 seconds per document
- **Validation**: ~0.1 seconds per document

## 🚨 Troubleshooting

### Common Issues

1. **Tesseract not found**: Install Tesseract OCR and ensure it's in PATH
2. **OpenAI API errors**: Check API key and rate limits
3. **Memory issues**: Process documents in smaller batches
4. **PDF extraction failures**: Ensure PDFs aren't password-protected

### Debug Mode

```bash
# Enable verbose logging
python run.py --verbose

# Check logs
tail -f pipeline.log
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **PyMuPDF**: PDF text extraction
- **Tesseract**: OCR capabilities
- **OpenAI**: LLM integration
- **Pydantic**: Data validation
- **Rich**: Terminal output formatting

## 📞 Support

For questions, issues, or contributions:

- Create an issue on GitHub
- Check the documentation
- Review the example configurations

---

**Ready to process your documents?** Start with:

```bash
python run.py --dry-run  # See what will be processed
python run.py --verbose   # Run with detailed logging
```
#   D o c _ p r o c e s s i n g 
 
 #   D o c _ p r o c e s s i n g 
 
 
