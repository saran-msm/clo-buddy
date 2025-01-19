from typing import List, Dict
import re
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

class GuardRails:
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        
        self.sensitive_patterns = {
            'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        }
        
        self.forbidden_terms = [
            'top secret',
            'classified document',
            'strictly confidential',
            'do not distribute'
        ]

    def validate_content(self, content: str) -> str:
        """
        Validates and sanitizes document content
        """
        if not content or len(content.strip()) == 0:
            raise ValueError("Empty document content")
            
        # Instead of blocking, just anonymize sensitive content
        safe_content = self._anonymize_pii(content)
        safe_content = self._remove_sensitive_patterns(safe_content)
        
        return safe_content

    def validate_response(self, response: str) -> str:
        """
        Validates and sanitizes model responses
        """
        if not response or len(response.strip()) == 0:
            raise ValueError("Empty response not allowed")
        
        # Anonymize any PII that might have slipped through
        safe_response = self._anonymize_pii(response)
        
        # Ensure response length is reasonable
        if len(safe_response) > 10000:  # Arbitrary limit
            safe_response = safe_response[:10000] + "..."
        
        return safe_response

    def _contains_forbidden_terms(self, text: str) -> bool:
        """
        Checks if text contains any forbidden terms
        """
        text_lower = text.lower()
        # Count how many forbidden terms are found
        forbidden_count = sum(1 for term in self.forbidden_terms if term in text_lower)
        # Only return True if multiple forbidden terms are found
        return forbidden_count >= 2

    def _anonymize_pii(self, text: str) -> str:
        """
        Anonymizes personally identifiable information
        """
        try:
            # Analyze text for PII
            analyzer_results = self.analyzer.analyze(
                text=text,
                language='en'
            )
            
            # Anonymize detected PII
            if analyzer_results:
                anonymized_text = self.anonymizer.anonymize(
                    text=text,
                    analyzer_results=analyzer_results
                ).text
                return anonymized_text
            return text
        except Exception as e:
            print(f"Warning: PII anonymization failed: {str(e)}")
            return text

    def _remove_sensitive_patterns(self, text: str) -> str:
        """
        Removes sensitive patterns like credit card numbers
        """
        for pattern in self.sensitive_patterns.values():
            text = re.sub(pattern, '[REDACTED]', text)
        
        return text

    def validate_file_size(self, file_size: int, max_size: int = 16 * 1024 * 1024) -> bool:
        """
        Validates file size
        """
        return file_size <= max_size

    def validate_file_type(self, file_type: str, allowed_types: List[str]) -> bool:
        """
        Validates file type
        """
        return file_type.lower() in allowed_types 