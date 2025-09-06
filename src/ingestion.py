import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pytesseract
from PIL import Image

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

class Document:
    """A simple data class to hold document information."""
    def __init__(self, path: Path, text: str, method: str):
        self.path = path
        self.text = text
        self.method = method  # e.g., "pdf_text", "ocr", "text_file"

    def __repr__(self):
        return f"Document(path={self.path.name}, text_length={len(self.text)}, method='{self.method}')"


class DocumentIngester:
    """
    Handles the discovery and text extraction of documents from various sources.
    It uses a robust strategy of trying direct text extraction from PDFs first,
    then falling back to OCR if needed.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DocumentIngester with the pipeline's configuration.

        Args:
            config: The configuration dictionary, typically loaded from config.yaml.
        """
        self.config = config.get('ingestion', {})
        self.logger = logging.getLogger(__name__)

        # Configure Tesseract if a path is provided
        tesseract_path = self.config.get('ocr', {}).get('tesseract_path')
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            self.logger.info(f"Set Tesseract command path to: {tesseract_path}")

    def ingest(self, input_directory: Path) -> List[Document]:
        """
        Finds all supported documents in the input directory and extracts their text.

        Args:
            input_directory: The path to the directory to scan for documents.

        Returns:
            A list of Document objects, each containing the path and extracted text.
        """
        self.logger.info(f"Starting ingestion process for directory: {input_directory}")
        document_paths = self._find_documents(input_directory)
        self.logger.info(f"Found {len(document_paths)} supported documents to process.")

        processed_documents = []
        for doc_path in document_paths:
            self.logger.debug(f"Processing file: {doc_path.name}")
            try:
                document = self._process_single_document(doc_path)
                if document:
                    processed_documents.append(document)
            except Exception as e:
                self.logger.error(f"Failed to process {doc_path.name}: {e}", exc_info=True)

        self.logger.info(f"Successfully ingested {len(processed_documents)} documents.")
        return processed_documents

    def _find_documents(self, input_directory: Path) -> List[Path]:
        """Finds all supported files in the given directory and its subdirectories."""
        supported_exts = self.config.get('supported_extensions', [])
        max_size_bytes = self.config.get('max_file_size_mb', 50) * 1024 * 1024

        docs = []
        for p in sorted(input_directory.rglob('*')):
            if p.is_file() and p.suffix.lower() in supported_exts:
                if p.stat().st_size > max_size_bytes:
                    self.logger.warning(f"Skipping large file: {p.name} (size > {self.config.get('max_file_size_mb')}MB)")
                    continue
                docs.append(p)
        return docs

    def _process_single_document(self, doc_path: Path) -> Optional[Document]:
        """
        Extracts text from a single document based on its file type.
        """
        suffix = doc_path.suffix.lower()
        text, method = "", "failed"

        if suffix == ".txt":
            text, method = self._extract_from_text_file(doc_path)
        elif suffix == ".pdf":
            text, method = self._extract_from_pdf(doc_path)
        elif suffix in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            text, method = self._extract_from_image(doc_path)
        else:
            self.logger.warning(f"Unsupported file type: {suffix}. Skipping {doc_path.name}.")
            return None

        if not text.strip():
            self.logger.warning(f"No text extracted from {doc_path.name} using method '{method}'.")
            return None

        return Document(path=doc_path, text=text, method=method)

    def _extract_from_text_file(self, doc_path: Path) -> (str, str):
        """Reads text directly from a .txt file."""
        try:
            return doc_path.read_text(encoding='utf-8', errors='ignore'), "text_file"
        except Exception as e:
            self.logger.error(f"Error reading text file {doc_path.name}: {e}")
            return "", "failed"

    def _extract_from_image(self, doc_path: Path) -> (str, str):
        """Extracts text from an image file using OCR."""
        try:
            with Image.open(doc_path) as img:
                text = pytesseract.image_to_string(img)
            return text, "ocr"
        except Exception as e:
            self.logger.error(f"OCR failed for image {doc_path.name}: {e}")
            return "", "failed"

    def _extract_from_pdf(self, doc_path: Path) -> (str, str):
        """
        Extracts text from a PDF, trying direct extraction first and falling back to OCR.
        """
        if not PyPDF2:
            self.logger.error("PyPDF2 is not installed. Cannot process PDFs.")
            return "", "failed"

        # Try direct text extraction first
        text = ""
        try:
            with open(doc_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""

            # If text is very short, it might be a scanned PDF.
            if len(text.strip()) > 150:
                self.logger.debug(f"Successfully extracted text from {doc_path.name} using PyPDF2.")
                return text, "pdf_text"
            else:
                self.logger.info(f"Low text extracted from {doc_path.name}. Attempting OCR fallback.")
        except Exception as e:
            self.logger.warning(f"Direct text extraction failed for {doc_path.name}: {e}. Proceeding with OCR.")

        # Fallback to OCR
        if not convert_from_path:
            self.logger.error("'pdf2image' is not installed, and OCR on PDFs will fail. Please run 'pip install pdf2image' and install Poppler.")
            return text, "ocr_failed" # Return any text we got

        try:
            ocr_text = ""
            dpi = self.config.get('ocr', {}).get('resolution_dpi', 300)
            images = convert_from_path(doc_path, dpi=dpi)
            for i, img in enumerate(images):
                self.logger.debug(f"OCR processing page {i+1} of {doc_path.name}")
                ocr_text += pytesseract.image_to_string(img) + "\n"

            # Use OCR text only if it's significantly better
            if len(ocr_text) > len(text):
                self.logger.info(f"OCR provided more text for {doc_path.name}. Using OCR result.")
                return ocr_text, "ocr"
            else:
                self.logger.info(f"OCR did not provide more text for {doc_path.name}. Using original extract.")
                return text, "pdf_text"
        except Exception as e:
            self.logger.error(f"OCR fallback failed for {doc_path.name}: {e}")
            return text, "ocr_failed" # Return original text on failure
