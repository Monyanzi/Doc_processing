# Document Intelligence Pipeline - User Guide

Welcome! This guide will walk you through using the pipeline to turn your messy digital documents into organized, actionable business intelligence.

## 🚀 Quick Start: 3 Simple Steps

1.  **Prepare Your Documents:** Place all the documents you want to process (PDFs, images, etc.) into a single folder on your computer.

2.  **Run the Pipeline:** Open your terminal or command prompt, navigate to the project folder, and run the main pipeline script:
    ```bash
    python run_pipeline.py --input "C:\Path\To\Your\Documents"
    ```
    *(Note: A `run_pipeline.py` script will be the final step to tie all stages together.)*

3.  **Get Your Insights:** Once the process is complete, look inside the `output/` folder. You will find your **`Executive_Dashboard_[Date].xlsx`**. This file contains all the key insights and data from your documents.

## 📁 Understanding the Output

The pipeline generates a powerful Excel dashboard with several sheets.

*   **Dashboard Summary:** This is your "at-a-glance" view. It contains a plain-English summary of the most critical findings, such as duplicate payment alerts and expiring contracts.
*   **Duplicate Payment Alerts:** If the system found any invoices that look like duplicates, the details will be on this sheet for your immediate review.
*   **Contract Expiry Risks:** This sheet lists any contracts that are set to expire soon, giving you time to renegotiate or act.
*   **Top Vendor Net Spend:** A summary of how much you've spent with each vendor, which can be useful for budgeting and negotiations.
*   **Data Quality Flags:** A list of any documents that had missing or illogical information (e.g., an invoice with no total amount), helping you improve your record-keeping.

Additionally, in the `output/06_organization/` folder, you will find a neatly organized copy of all your original files, sorted into folders by document type.

## ⚙️ Prerequisites & Setup

Before you begin, please ensure the following are installed:

1.  **Python:** Version 3.10 or higher.
2.  **Ollama:** The Ollama application must be running on your computer with the `gemma3:1b` model downloaded.
3.  **Required Libraries:** Open a terminal and run the following command once:
    ```bash
    pip install -r requirements.txt
    ```

## ❓ Troubleshooting

*   **Error: "Ollama connection failed"**: Make sure the Ollama application is running on your computer before starting the pipeline.
*   **Slow Performance:** The classification and extraction stages use an AI model, which can be resource-intensive. For best performance, close other demanding applications while the pipeline is running.
*   **No Output:** Ensure the `--input` path you provided points to a folder that contains supported document files.