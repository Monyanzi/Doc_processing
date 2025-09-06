import logging
import json
import re
import os
from typing import Dict, Any, List, Optional
import concurrent.futures

# Re-using the same LLM interaction pattern
try:
    import ollama
except ImportError:
    ollama = None

try:
    import requests
except ImportError:
    requests = None

from .classification import ClassifiedDocument


class ExtractedDocument(ClassifiedDocument):
    """Extends ClassifiedDocument to include the extracted data."""
    def __init__(self, classified_doc: ClassifiedDocument, extracted_data: Dict[str, Any]):
        super().__init__(classified_doc, classified_doc.doc_type, classified_doc.confidence, classified_doc.classification_method)
        self.extracted_data = extracted_data

    def __repr__(self):
        status = "success" if self.extracted_data and not self.extracted_data.get('error') else "failed"
        return f"ExtractedDocument(path={self.path.name}, type='{self.doc_type}', status='{status}')"


class DocumentExtractor:
    """
    Extracts structured data from documents using a hybrid regex and LLM approach.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DocumentExtractor with the pipeline's configuration.

        Args:
            config: The configuration dictionary from config.yaml.
        """
        self.config = config.get('extraction', {})
        self.llm_config = config.get('llm', {})
        self.logger = logging.getLogger(__name__)

    def extract_data(self, documents: List[ClassifiedDocument]) -> List[ExtractedDocument]:
        """
        Extracts structured data from a list of classified documents.

        Args:
            documents: A list of ClassifiedDocument objects.

        Returns:
            A list of ExtractedDocument objects with the extracted data.
        """
        self.logger.info(f"Starting data extraction for {len(documents)} documents.")

        docs_for_llm = []
        extracted_docs = []

        # Pass 1: Regex Extraction
        for doc in documents:
            target_schema = self.config.get('schemas', {}).get(doc.doc_type)
            if not target_schema:
                self.logger.warning(f"No extraction schema found for document type '{doc.doc_type}'. Skipping {doc.path.name}.")
                extracted_docs.append(ExtractedDocument(doc, {"status": "skipped", "reason": f"No schema for {doc.doc_type}"}))
                continue

            partial_data = self._extract_with_rules(doc.text, doc.doc_type)

            # If regex found all fields, we are done with this doc.
            if len(partial_data) == len(target_schema):
                 partial_data['extraction_method'] = 'regex_only'
                 extracted_docs.append(ExtractedDocument(doc, partial_data))
                 self.logger.debug(f"Fully extracted {doc.path.name} using regex.")
            else:
                docs_for_llm.append((doc, partial_data))

        # Pass 2: LLM Gap-filling (in parallel)
        if docs_for_llm:
            self.logger.info(f"{len(docs_for_llm)} documents require LLM gap-filling.")
            max_workers = self.config.get('max_workers', 4)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_doc = {executor.submit(self._extract_with_llm, doc, partial_data): doc for doc, partial_data in docs_for_llm}
                for future in concurrent.futures.as_completed(future_to_doc):
                    doc = future_to_doc[future]
                    try:
                        llm_result_data = future.result()
                        extracted_docs.append(ExtractedDocument(doc, llm_result_data))
                    except Exception as exc:
                        self.logger.error(f'{doc.path.name} generated an exception during LLM extraction: {exc}')
                        extracted_docs.append(ExtractedDocument(doc, {"status": "failed", "error": str(exc)}))

        self.logger.info("Extraction process completed.")
        return extracted_docs

    def _extract_with_rules(self, text: str, doc_type: str) -> Dict[str, Any]:
        """Extracts data using regex rules from the config."""
        rules = self.config.get('regex_rules', {}).get(doc_type, {})
        data = {}
        for field, pattern in rules.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                data[field] = match.group(1).strip()
        return data

    def _extract_with_llm(self, doc: ClassifiedDocument, partial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fills in missing data using the configured LLM."""
        target_schema = self.config.get('schemas', {}).get(doc.doc_type, {})
        missing_fields = {k: v for k, v in target_schema.items() if k not in partial_data}

        if not missing_fields:
            return partial_data

        prompt = f"""You are a data extraction expert. From the text below, extract the values for the following JSON schema. Only extract the information for the fields provided in the schema. If a value is not found, use null. Respond with ONLY a valid JSON object. SCHEMA: {json.dumps(missing_fields)} TEXT: "{doc.text[:8000]}" """

        provider = self.llm_config.get('active_provider')
        self.logger.debug(f"Using LLM provider: {provider} for extraction on {doc.path.name}.")

        try:
            if provider == 'ollama':
                llm_data = self._call_ollama(prompt)
            elif provider == 'openai':
                llm_data = self._call_openai(prompt)
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")

            # Combine regex and LLM results
            final_data = partial_data.copy()
            final_data.update(llm_data)
            final_data['extraction_method'] = 'hybrid_regex_llm'
            return final_data

        except Exception as e:
            self.logger.error(f"LLM extraction failed for {doc.path.name}: {e}")
            return {"status": "failed_llm", "error": str(e), **partial_data}

    def _call_ollama(self, prompt: str) -> Dict[str, Any]:
        """Helper to call the Ollama API."""
        if not ollama: raise ImportError("Ollama library not installed.")
        cfg = self.llm_config.get('ollama', {})
        response = ollama.chat(model=cfg.get('model'), messages=[{'role': 'user', 'content': prompt}], format='json')
        raw_json = response['message']['content']
        return json.loads(self._clean_json_response(raw_json))

    def _call_openai(self, prompt: str) -> Dict[str, Any]:
        """Helper to call the OpenAI API."""
        if not requests: raise ImportError("Requests library not installed.")
        cfg = self.llm_config.get('openai', {})
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
        payload = {"model": cfg.get('model'), "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "response_format": {"type": "json_object"}}
        response = requests.post(cfg.get('base_url'), headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        raw_json = response.json()['choices'][0]['message']['content']
        return json.loads(self._clean_json_response(raw_json))

    def _clean_json_response(self, response: str) -> str:
        """Extracts a JSON object from a string."""
        match = re.search(r'\{.*\}', response, re.DOTALL)
        return match.group(0) if match else "{}"
