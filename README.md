# Document Intelligence Pipeline (V3 - Unified Architecture)

A comprehensive, configurable, and production-ready document processing pipeline. It automatically ingests, classifies, extracts, and validates information from various document types. This version has been refactored for robustness, scalability, and maintainability.

## 🚀 Features

- **Unified Pipeline**: A single, orchestrated pipeline (`run_pipeline.py`) manages all stages.
- **Modular Architecture**: Core logic is organized into modules (`ingestion`, `classification`, `extraction`, `validation`, `reporting`) within a `src` directory.
- **Highly Configurable**: All pipeline settings are managed through a central `config.yaml` file (paths, models, rules, schemas).
- **Hybrid Extraction Engine**: Combines the speed of regex with the power of LLMs for efficient and accurate data extraction.
- **Multi-format Support**: Handles PDF, images (PNG, JPG, etc.), and text files with a robust PDF text extraction and OCR fallback mechanism.
- **Flexible LLM Backend**: Easily switch between LLM providers (OpenAI, Ollama) via the configuration file.
- **Concurrent Processing**: Utilizes parallel execution for faster LLM-based extraction.
- **Comprehensive Reporting**: Generates a consolidated CSV summary and detailed JSON output for each document.

## 🏗️ Architecture

The pipeline follows a modular, 5-stage architecture orchestrated by `run_pipeline.py`:

1.  **Ingestion (`src/ingestion.py`)**: Finds documents, extracts raw text using direct PDF extraction with an OCR fallback.
2.  **Classification (`src/classification.py`)**: Identifies the document type using a tiered strategy (filename -> content rules -> LLM fallback).
3.  **Extraction (`src/extraction.py`)**: Pulls structured data using a hybrid approach (regex -> LLM gap-filling).
4.  **Validation (`src/validation.py`)**: Checks the extracted data for correctness and quality.
5.  **Reporting (`src/reporting.py`)**: Saves the results in CSV and JSON formats.

## 📦 Installation

### Prerequisites

- Python 3.8+
- Tesseract OCR (for image and scanned PDF processing)
- Poppler (for converting PDFs to images for OCR)
- An active OpenAI API key or a running Ollama instance.

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file by copying the `.env.example` file and add your OpenAI API key:
    ```bash
    cp .env.example .env
    # Now edit .env and add your key
    # OPENAI_API_KEY="your_openai_api_key_here"
    ```

## ⚙️ Configuration

The entire pipeline is configured via `config.yaml`. Here you can define input/output paths, select your LLM provider, set classification and extraction rules, and define schemas for new document types.

**Example: Switching to a local Ollama model**

```yaml
# In config.yaml
llm:
  active_provider: "ollama" # Change from "openai" to "ollama"

  ollama:
    model: "llama3:8b"
    # ... other ollama settings
```

## 🚀 Usage

The pipeline is run from a single entry point: `run_pipeline.py`.

### Basic Usage

This will run the pipeline using the settings in `config.yaml`.

```bash
python run_pipeline.py
```

### Advanced Options

You can override the input and output directories from the command line:

```bash
# Run with a custom input directory
python run_pipeline.py --input-dir /path/to/your/documents

# Run with a custom output directory and verbose logging
python run_pipeline.py --input-dir /path/to/your/documents --output-dir /path/to/your/results -v
```

## 📊 Output

For each run, the pipeline creates a new timestamped folder in your configured output directory. This folder contains:

-   `summary_report.csv`: A CSV file with a summary of all processed documents.
-   `json_details/`: A directory containing a detailed JSON file for each document, including extracted data and validation results.

## 🔧 Customization

### Adding a New Document Type (e.g., "Waybill")

1.  **`config.yaml`**:
    *   Add `"waybill"` to `classification.filename_rules` and `classification.content_rules`.
    *   Add a new `waybill` schema under `extraction.schemas`.
    *   Add any regex rules under `extraction.regex_rules.waybill`.
2.  **Run the pipeline!** No code changes are needed.

## 🤝 Contributing

1.  Fork the repository.
2.  Create a feature branch.
3.  Make your changes.
4.  Add or update tests as needed.
5.  Submit a pull request.
