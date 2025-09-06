# Usage Guide for Document Intelligence Pipeline

This guide shows you how to process documents from the two specified directories:
1. `C:\Users\Monya\Desktop\Cursor\Invoice Automation\sample-documents`
2. `C:\Users\Monya\Documents\Visa\Mexico`

## Quick Start

### Option 1: Use the Batch Processor (Recommended)
```bash
# Process both directories automatically
python batch_process.py
```

### Option 2: Use the CLI Runner
```bash
# Process both directories with CLI
python run.py --input "C:\Users\Monya\Desktop\Cursor\Invoice Automation\sample-documents" "C:\Users\Monya\Documents\Visa\Mexico"

# Process with verbose logging
python run.py --input "C:\Users\Monya\Desktop\Cursor\Invoice Automation\sample-documents" "C:\Users\Monya\Documents\Visa\Mexico" --verbose

# Dry run to see what would be processed
python run.py --input "C:\Users\Monya\Desktop\Cursor\Invoice Automation\sample-documents" "C:\Users\Monya\Documents\Visa\Mexico" --dry-run
```

### Option 3: Process One Directory at a Time
```bash
# Process only Invoice Automation documents
python run.py --input "C:\Users\Monya\Desktop\Cursor\Invoice Automation\sample-documents"

# Process only Visa documents
python run.py --input "C:\Users\Monya\Documents\Visa\Mexico"
```

## Before Running

### 1. Install Ollama
The pipeline now uses **Ollama** for local LLM processing instead of OpenAI API calls.

**Windows:**
- Download from [https://ollama.ai/download](https://ollama.ai/download)
- Install and start the Ollama service

**macOS/Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve
```

### 2. Pull the Required Model
```bash
# Pull the gemma3:1b model (configured in config.yaml)
ollama pull gemma3:1b

# Alternative models you can use:
# ollama pull llama3:8b-instruct    # Larger, more capable
# ollama pull mistral:7b-instruct   # Good balance of speed/quality
# ollama pull qwen2.5:3b            # Fast, good quality
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Test the Setup
```bash
# Test Ollama connection
python ollama_client.py

# Test complete pipeline
python test_pipeline.py
```

## What Happens During Processing

1. **Document Discovery**: Scans directories for supported files (PDF, images, text)
2. **Text Extraction**: Extracts text from documents (PDF text + OCR fallback)
3. **Classification**: Uses **Ollama gemma3:1b** to identify document types (invoice, receipt, contract, etc.)
4. **Field Extraction**: Extracts relevant information based on document type using Ollama
5. **Validation**: Performs quality checks and flags issues
6. **Output**: Saves results as JSON files + CSV summary

## Expected Output

- **Individual Results**: One JSON file per processed document
- **Summary CSV**: Overview of all processed documents
- **Batch Summary**: Overall processing statistics
- **Logs**: Detailed processing logs

## Monitoring Progress

- **Verbose Mode**: Use `--verbose` flag for detailed logging
- **Progress Tracking**: Shows current directory and document being processed
- **Real-time Updates**: Displays success/failure counts as processing continues

## Troubleshooting

### Common Issues

1. **Ollama Not Running:**
   ```
   Error: Cannot connect to Ollama server
   ```
   **Solution**: Start Ollama service (`ollama serve`)

2. **Model Not Available:**
   ```
   Error: Model gemma3:1b not found
   ```
   **Solution**: Pull the model (`ollama pull gemma3:1b`)

3. **Directory Not Found:**
   ```
   Error: Input directory does not exist
   ```
   **Solution**: Check the exact path and ensure it exists

4. **No Documents Found:**
   ```
   Warning: No supported documents found
   ```
   **Solution**: Ensure directories contain PDF, image, or text files

5. **Memory Issues:**
   ```
   Error: Out of memory
   ```
   **Solution**: Process fewer documents at once using `--max-docs 10`

### Performance Tips

- **Start Small**: Process a few documents first to test
- **Use Dry Run**: Check what will be processed before running
- **Monitor Memory**: Large documents may require more RAM
- **Model Selection**: gemma3:1b is fast but smaller models may be less accurate

## Example Commands

```bash
# Quick test with 5 documents
python run.py --input "C:\Users\Monya\Desktop\Cursor\Invoice Automation\sample-documents" --max-docs 5

# Process only invoices and receipts
python run.py --input "C:\Users\Monya\Desktop\Cursor\Invoice Automation\sample-documents" "C:\Users\Monya\Documents\Visa\Mexico" --types invoice receipt

# Custom output directory
python run.py --input "C:\Users\Monya\Desktop\Cursor\Invoice Automation\sample-documents" --output "./custom_output"

# Show current configuration
python run.py --show-config
```

## Output Structure

```
output/
├── 20241215_143022/
│   ├── document1_20241215_143022.json
│   ├── document2_20241215_143022.json
│   └── summary_20241215_143022.csv
├── batch_summary_20241215_143022.json
└── complete_results_20241215_143022.json
```

## Next Steps

After successful processing:

1. **Review Results**: Check individual JSON files for extracted data
2. **Analyze Issues**: Look at validation flags in the output
3. **Refine Prompts**: Adjust extraction prompts if needed
4. **Scale Up**: Process larger document collections
5. **Customize**: Add new document types or validation rules

## Support

If you encounter issues:

1. Check the logs in the `logs/` directory
2. Run with `--verbose` for detailed error information
3. Review the test output from `python test_pipeline.py`
4. Check that Ollama is running and the model is available
5. Verify all dependencies are properly installed

## Ollama Model Management

### Check Available Models
```bash
ollama list
```

### Switch Models
Edit `config.yaml` and change the model name:
```yaml
model:
  provider: "ollama"
  name: "llama3:8b-instruct"  # Change this line
```

### Model Performance Comparison
- **gemma3:1b**: Fastest, good for basic tasks, lower memory usage
- **llama3:8b-instruct**: Better quality, slower, higher memory usage
- **mistral:7b-instruct**: Good balance of speed and quality
- **qwen2.5:3b**: Fast, good quality, moderate memory usage

---

**Ready to process your documents?** Start with:

```bash
# Test Ollama connection
python ollama_client.py

# Dry run to see what will be processed
python run.py --dry-run

# Process with detailed logging
python run.py --verbose
```
