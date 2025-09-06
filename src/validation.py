import logging
from typing import Dict, Any, List, Tuple

from .extraction import ExtractedDocument


class ValidatedDocument(ExtractedDocument):
    """Extends ExtractedDocument to include validation results."""
    def __init__(self, extracted_doc: ExtractedDocument, validation_summary: Dict[str, Any]):
        super().__init__(extracted_doc, extracted_doc.extracted_data)
        self.validation_summary = validation_summary

    def __repr__(self):
        status = "valid" if self.validation_summary.get('is_valid') else "invalid"
        return f"ValidatedDocument(path={self.path.name}, type='{self.doc_type}', validation_status='{status}')"


class DataValidator:
    """
    Validates the structured data extracted from documents.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DataValidator.

        Args:
            config: The configuration dictionary from config.yaml.
        """
        self.config = config.get('validation', {}) # For future validation rules
        self.logger = logging.getLogger(__name__)

    def validate_documents(self, documents: List[ExtractedDocument]) -> List[ValidatedDocument]:
        """
        Validates a list of extracted documents.

        Args:
            documents: A list of ExtractedDocument objects.

        Returns:
            A list of ValidatedDocument objects with validation info.
        """
        self.logger.info(f"Starting validation for {len(documents)} documents.")
        validated_docs = []
        for doc in documents:
            if doc.extracted_data.get('status', '').startswith('fail'):
                summary = {'is_valid': False, 'errors': [f"Extraction failed: {doc.extracted_data.get('error')}"], 'warnings': []}
            elif doc.extracted_data.get('status') == 'skipped':
                 summary = {'is_valid': False, 'errors': [f"Extraction skipped: {doc.extracted_data.get('reason')}"], 'warnings': []}
            else:
                summary = self._validate_single_document(doc)

            validated_docs.append(ValidatedDocument(doc, summary))

        self.logger.info("Validation process completed.")
        return validated_docs

    def _validate_single_document(self, doc: ExtractedDocument) -> Dict[str, Any]:
        """
        Performs validation checks on a single document's extracted data.

        This is a placeholder for more sophisticated validation logic.
        For now, it checks for the presence of key fields based on the schema.
        """
        errors = []
        warnings = []

        doc_type = doc.doc_type
        data = doc.extracted_data

        # Example validation: Check for required fields based on a hypothetical schema extension
        # In a real system, this would be more configurable.
        required_fields = {
            "invoice": ["invoice_number", "total_amount"],
            "receipt": ["transaction_date", "total_amount"],
            "contract": ["contract_title", "effective_date"]
        }

        fields_to_check = required_fields.get(doc_type, [])
        for field in fields_to_check:
            if not data.get(field):
                errors.append(f"Missing required field: '{field}'")

        # Example: Check if total_amount is a number
        if "total_amount" in data and data.get("total_amount"):
            try:
                float(data["total_amount"])
            except (ValueError, TypeError):
                errors.append(f"Field 'total_amount' is not a valid number: {data['total_amount']}")

        return {
            'is_valid': not errors,
            'errors': errors,
            'warnings': warnings
        }
