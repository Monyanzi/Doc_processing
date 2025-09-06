# Document Intelligence Pipeline - Technical Guide

This document provides a technical overview of the pipeline's architecture, data flow, and customization options for developers.

## 🏗️ Architecture Overview

The pipeline is designed as a series of modular, sequential stages. Each stage is an independent script that consumes the output of the previous stage. This decoupled design allows for easier debugging, testing, and modification.

**The flow is as follows:**

`[Input Files]` -> **01_ingest** -> **02_classify** -> **03_extract** -> **04_analyze** -> **05_dashboard** -> `[Final Reports]`
`...` -> **06_organize** -> `[Organized Files]`

-   **`00_clean_outputs.py`**: A utility to reset the `output/` directory.
-   **`01_ingest.py`**: Ingests raw files (PDF, JPG, etc.), performs OCR, and saves raw text content.
-   **`02_classify.py`**: Classifies documents using a hybrid Filename/Content Regex + LLM fallback model.
-   **`03_extract.py`**: Extracts structured data (JSON) using a hybrid Regex + LLM fallback model.
-   **`04_analyze.py`**: A pure-logic script that runs business intelligence checks on the extracted JSON data.
-   **`05_dashboard.py`**: Generates a human-readable Excel dashboard, using an LLM for the executive summary.
-   **`06_organize.py`**: Copies original source files into a cleanly organized folder structure based on final classifications.

## 📊 Data Flow

The pipeline uses a "baton pass" system where each stage's output becomes the next stage's input. All outputs are stored in the `./output` directory.

-   **Stage 1 Output**: `output/01_ingestion/[run_id]/extracted_text/*.txt`
-   **Stage 2 Input**: Reads the `.txt` files from the latest `01_ingestion` run.
-   **Stage 2 Output**: `output/02_classification/[run_id]/classification_metrics.csv`
-   **Stage 3 Input**: Reads the `.csv` from Stage 2 and the `.txt` files from Stage 1.
-   **Stage 3 Output**: `output/03_extraction/[run_id]/extracted_json/*.json`
-   **Stage 4 Input**: Reads the `.json` files from Stage 3 and the `.csv` from Stage 2.
-   **Stage 4 Output**: `output/04_analysis/[run_id]/analysis_results.json`
-   **Stage 5 Input**: Reads the `.json` file from Stage 4.
-   **Stage 5 Output**: `output/05_dashboard/[run_id]/Executive_Dashboard_*.xlsx`
-   **`stage_tracker.xlsx`**: A master Excel file in `./output` that is appended to by stages 1, 2, and 3 to provide a cumulative record of processing.

## 🔧 Customization

The pipeline is designed to be easily extended.

### Adding a New Document Type

1.  **`02_classify.py`**:
    -   Add the new type to the `DocumentType` Enum.
    -   Add corresponding patterns to the `FILENAME_RULES` and `CONTENT_RULES` dictionaries.
2.  **`03_extract.py`**:
    -   Add a new schema to the `_get_extraction_schema` method.
    -   (Optional) Add new `REGEX_RULES` for the type to speed up extraction.
3.  **`04_analyze.py`**:
    -   (Optional) Add new, type-specific business logic to the `check_data_quality` method.

### Modifying the LLM

-   **Model:** The `OLLAMA_MODEL` variable at the top of scripts `02`, `03`, and `05` can be changed to any model available in your Ollama instance.
-   **Prompts:** Prompts are clearly defined within each script's `Classifier` or `Extractor` class and can be easily modified to change the LLM's behavior or output format.

## 🏃‍♂️ Running Stages Individually

For debugging, you can run each script sequentially from your terminal. The scripts are designed to automatically find the latest output from the previous stage.

```bash
# Run each stage in order
python 01_ingest.py
python 02_classify.py
python 03_extract.py
python 04_analyze.py
python 05_dashboard.py
python 06_organize.py
```