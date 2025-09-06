import logging
import json
import re
from typing import Dict, Any, List, Tuple, Optional

# It's better to have a dedicated module for LLM interactions
# but for now, we can import them here.
try:
    import ollama
except ImportError:
    ollama = None

try:
    import requests
except ImportError:
    requests = None

from .ingestion import Document


class ClassifiedDocument(Document):
    """Extends Document to include classification results."""
    def __init__(self, document: Document, doc_type: str, confidence: float, method: str):
        super().__init__(document.path, document.text, document.method)
        self.doc_type = doc_type
        self.confidence = confidence
        self.classification_method = method

    def __repr__(self):
        return f"ClassifiedDocument(path={self.path.name}, type='{self.doc_type}', confidence={self.confidence:.2f})"


class DocumentClassifier:
    """
    Classifies documents using a tiered approach: filename, content rules, and LLM fallback.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DocumentClassifier with the pipeline's configuration.

        Args:
            config: The configuration dictionary from config.yaml.
        """
        self.config = config.get('classification', {})
        self.llm_config = config.get('llm', {})
        self.logger = logging.getLogger(__name__)

    def classify_documents(self, documents: List[Document]) -> List[ClassifiedDocument]:
        """
        Classifies a list of documents.

        Args:
            documents: A list of Document objects from the ingestion stage.

        Returns:
            A list of ClassifiedDocument objects with classification info.
        """
        self.logger.info(f"Starting classification for {len(documents)} documents.")
        classified_docs = []
        for doc in documents:
            classified_doc = self._classify_single_document(doc)
            classified_docs.append(classified_doc)

        self.logger.info("Classification process completed.")
        return classified_docs

    def _classify_single_document(self, doc: Document) -> ClassifiedDocument:
        """Applies the tiered classification logic to a single document."""
        # Tier 1: Filename Rules
        result = self._classify_by_filename(doc)
        if result:
            doc_type, confidence, method = result
            return ClassifiedDocument(doc, doc_type, confidence, method)

        # Tier 2: Content Rules
        result = self._classify_by_content(doc)
        if result:
            doc_type, confidence, method = result
            return ClassifiedDocument(doc, doc_type, confidence, method)

        # Tier 3: LLM Fallback
        if self.config.get('use_llm_fallback', False):
            self.logger.debug(f"Rules failed for {doc.path.name}, using LLM fallback.")
            result = self._classify_with_llm(doc)
            if result:
                doc_type, confidence, method = result
                return ClassifiedDocument(doc, doc_type, confidence, method)

        # Default to 'other' if all else fails
        self.logger.warning(f"Could not classify {doc.path.name}. Defaulting to 'other'.")
        return ClassifiedDocument(doc, "other", 0.0, "failed")

    def _classify_by_filename(self, doc: Document) -> Optional[Tuple[str, float, str]]:
        """Classifies based on keywords in the filename."""
        filename_rules = self.config.get('filename_rules', {})
        fn_lower = doc.path.name.lower()
        for doc_type, keywords in filename_rules.items():
            if any(kw in fn_lower for kw in keywords):
                self.logger.debug(f"Classified {doc.path.name} as '{doc_type}' by filename.")
                return doc_type, 0.98, "rules_filename"
        return None

    def _classify_by_content(self, doc: Document) -> Optional[Tuple[str, float, str]]:
        """Classifies based on keywords in the document text."""
        content_rules = self.config.get('content_rules', {})
        text_lower = doc.text.lower()
        scores = {doc_type: sum(1 for kw in kws if kw in text_lower) for doc_type, kws in content_rules.items()}

        if not scores:
            return None

        best_match, max_score = max(scores.items(), key=lambda item: item[1])

        # A simple confidence score based on number of keyword hits
        if max_score > 0:
            confidence = min(0.5 + (max_score * 0.1), 0.95)
            if confidence >= self.config.get('confidence_threshold', 0.7):
                self.logger.debug(f"Classified {doc.path.name} as '{best_match}' by content.")
                return best_match, confidence, "rules_content"
        return None

    def _classify_with_llm(self, doc: Document) -> Optional[Tuple[str, float, str]]:
        """Classifies using a configured LLM provider."""
        provider = self.llm_config.get('active_provider')
        self.logger.info(f"Using LLM provider: {provider} for classification.")

        prompt = f"""Analyze the following document text and identify its type (e.g., invoice, receipt, contract). Respond with ONLY a valid JSON object in this format: {{"predicted_type": "The document type", "confidence": 0.85}}. TEXT: "{doc.text[:4000]}" """

        try:
            if provider == 'ollama':
                return self._call_ollama(prompt)
            elif provider == 'openai':
                return self._call_openai(prompt)
            else:
                self.logger.error(f"Unsupported LLM provider: {provider}")
                return None
        except Exception as e:
            self.logger.error(f"LLM classification failed for {doc.path.name}: {e}")
            return None

    def _call_ollama(self, prompt: str) -> Optional[Tuple[str, float, str]]:
        """Helper to call the Ollama API."""
        if not ollama:
            self.logger.error("Ollama library not installed.")
            return None

        cfg = self.llm_config.get('ollama', {})
        response = ollama.chat(
            model=cfg.get('model'),
            messages=[{'role': 'user', 'content': prompt}],
            format='json'
        )
        raw_json = response['message']['content']
        data = json.loads(self._clean_json_response(raw_json))
        return data.get("predicted_type", "other").lower(), float(data.get("confidence", 0.0)), "llm_ollama"

    def _call_openai(self, prompt: str) -> Optional[Tuple[str, float, str]]:
        """Helper to call the OpenAI API."""
        if not requests:
            self.logger.error("Requests library not installed.")
            return None

        cfg = self.llm_config.get('openai', {})
        import os # For getting API key from environment

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
        }
        payload = {
            "model": cfg.get('model'),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }

        response = requests.post(cfg.get('base_url'), headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        raw_json = response.json()['choices'][0]['message']['content']
        data = json.loads(self._clean_json_response(raw_json))
        return data.get("predicted_type", "other").lower(), float(data.get("confidence", 0.0)), "llm_openai"

    def _clean_json_response(self, response: str) -> str:
        """Extracts a JSON object from a string, even if it's embedded in other text."""
        match = re.search(r'\{.*\}', response, re.DOTALL)
        return match.group(0) if match else "{}"
