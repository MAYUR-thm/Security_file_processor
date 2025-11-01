"""
PII (Personally Identifiable Information) Cleaner Module

This module detects and removes/masks PII from text content including:
- Names (using spaCy NER)
- Email addresses
- Phone numbers
- Addresses
- Government IDs (SSN, etc.)
- Credit card numbers
- Client names (configurable list)
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
import json

# NLP libraries
import spacy
from spacy import displacy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PIICleaner:
    """Main class for detecting and cleaning PII from text."""
    
    def __init__(self, spacy_model: str = "en_core_web_sm", client_names_file: Optional[str] = None):
        """
        Initialize the PII Cleaner.
        
        Args:
            spacy_model: spaCy model to use for NER
            client_names_file: Path to JSON file containing client names to mask
        """
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logger.error(f"spaCy model '{spacy_model}' not found. Please install it with: python -m spacy download {spacy_model}")
            raise
        
        # Load client names if provided
        self.client_names = set()
        if client_names_file and Path(client_names_file).exists():
            self.load_client_names(client_names_file)
        
        # PII patterns (regex)
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            'address': r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl)\b',
            'employee_id': r'\b(?:EMP|ID|E)\d{5,7}\b',
            'token': r'\b[A-Z]{2}-\d{5,6}-[A-Z]{2,3}\b',
            'organization': r'\b(?:Department|Dept|Organization|Org|Company|Corp|Corporation|Inc|Ltd)\b',
            'system_path': r'(?i)\b(?:C:\\|/usr/|/etc/|/var/|\\\\)[a-zA-Z0-9\\_\\-\\/\\s]+\b',
            'sensitive_terms': r'\b(?:confidential|secret|internal use only|privileged access|restricted)\b'
        }
        
        # Column name mappings for structured data
        self.column_pii_types = {
            'employee full name': 'PERSON',
            'employee id': 'employee_id',
            'verified by (ra name)': 'PERSON',
            'ra signature / initials': 'PERSON',
            'authorized by (supervisor)': 'PERSON',
            'supervisor email': 'email',
            'token serial number': 'token',
            'date of authorization': 'date',
            'date of in-person verification': 'date'
        }
        
        # Masking options
        self.mask_char = '*'
        self.replacement_text = {
            'PERSON': '[PERSON]',
            'ORG': '[ORGANIZATION]',
            'GPE': '[LOCATION]',
            'email': '[EMAIL]',
            'phone': '[PHONE]',
            'ssn': '[SSN]',
            'credit_card': '[CREDIT_CARD]',
            'ip_address': '[IP_ADDRESS]',
            'url': '[URL]',
            'date': '[DATE]',
            'address': '[ADDRESS]',
            'employee_id': '[EMPLOYEE_ID]',
            'token': '[TOKEN]',
            'client_name': '[CLIENT_NAME]'
        }
    
    def load_client_names(self, file_path: str) -> None:
        """Load client names from a JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.client_names = set(name.lower() for name in data)
                elif isinstance(data, dict) and 'client_names' in data:
                    self.client_names = set(name.lower() for name in data['client_names'])
                else:
                    logger.warning(f"Invalid format in {file_path}. Expected list or dict with 'client_names' key.")
        except Exception as e:
            logger.error(f"Error loading client names from {file_path}: {str(e)}")
    
    def detect_pii(self, text: str) -> Dict[str, List[Dict[str, any]]]:
        """
        Detect PII in text and return detailed information about findings.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing detected PII by category
        """
        pii_findings = {
            'named_entities': [],
            'regex_patterns': [],
            'client_names': [],
            'statistics': {}
        }
        
        # Named Entity Recognition using spaCy
        doc = self.nlp(text)
        
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE']:
                pii_findings['named_entities'].append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': getattr(ent, 'confidence', 1.0)
                })
        
        # Regex pattern matching
        for pattern_name, pattern in self.pii_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                pii_findings['regex_patterns'].append({
                    'text': match.group(),
                    'pattern': pattern_name,
                    'start': match.start(),
                    'end': match.end()
                })
        
        # Client name detection
        if self.client_names:
            text_lower = text.lower()
            for client_name in self.client_names:
                if client_name in text_lower:
                    # Find all occurrences
                    start = 0
                    while True:
                        pos = text_lower.find(client_name, start)
                        if pos == -1:
                            break
                        
                        pii_findings['client_names'].append({
                            'text': text[pos:pos+len(client_name)],
                            'client_name': client_name,
                            'start': pos,
                            'end': pos + len(client_name)
                        })
                        start = pos + 1
        
        # Calculate statistics
        pii_findings['statistics'] = {
            'total_entities': len(pii_findings['named_entities']),
            'total_patterns': len(pii_findings['regex_patterns']),
            'total_client_names': len(pii_findings['client_names']),
            'entity_types': {},
            'pattern_types': {}
        }
        
        # Count by type
        for entity in pii_findings['named_entities']:
            label = entity['label']
            pii_findings['statistics']['entity_types'][label] = pii_findings['statistics']['entity_types'].get(label, 0) + 1
        
        for pattern in pii_findings['regex_patterns']:
            pattern_type = pattern['pattern']
            pii_findings['statistics']['pattern_types'][pattern_type] = pii_findings['statistics']['pattern_types'].get(pattern_type, 0) + 1
        
        return pii_findings
    
    def remove_pii(self, text: str, mask_mode: str = "replace") -> Dict[str, any]:
        """
        Remove or mask PII from text.
        
        Args:
            text: Input text to clean
            mask_mode: "replace" (with placeholders) or "mask" (with asterisks) or "remove" (delete entirely)
            
        Returns:
            Dictionary containing cleaned text and removal statistics
        """
        if mask_mode not in ["replace", "mask", "remove"]:
            raise ValueError("mask_mode must be 'replace', 'mask', or 'remove'")
        
        # Detect PII first
        pii_findings = self.detect_pii(text)
        
        # Collect all PII locations for replacement
        replacements = []
        
        # Add named entities
        for entity in pii_findings['named_entities']:
            replacements.append({
                'start': entity['start'],
                'end': entity['end'],
                'original': entity['text'],
                'replacement': self._get_replacement(entity['text'], entity['label'], mask_mode)
            })
        
        # Add regex patterns
        for pattern in pii_findings['regex_patterns']:
            replacements.append({
                'start': pattern['start'],
                'end': pattern['end'],
                'original': pattern['text'],
                'replacement': self._get_replacement(pattern['text'], pattern['pattern'], mask_mode)
            })
        
        # Add client names
        for client in pii_findings['client_names']:
            replacements.append({
                'start': client['start'],
                'end': client['end'],
                'original': client['text'],
                'replacement': self._get_replacement(client['text'], 'client_name', mask_mode)
            })
        
        # Sort replacements by start position (descending) to avoid index shifting
        replacements.sort(key=lambda x: x['start'], reverse=True)
        
        # Apply replacements while preserving formatting
        cleaned_text = text
        for replacement in replacements:
            start, end = replacement['start'], replacement['end']
            # Add spaces around replacement if it's in the middle of text
            prefix = " " if start > 0 and cleaned_text[start-1].isalnum() else ""
            suffix = " " if end < len(cleaned_text) and cleaned_text[end:end+1].isalnum() else ""
            cleaned_text = cleaned_text[:start] + prefix + replacement['replacement'] + suffix + cleaned_text[end:]
        
        # Clean up any double spaces that might have been created
        cleaned_text = " ".join(cleaned_text.split())
        
        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'pii_findings': pii_findings,
            'replacements_made': len(replacements),
            'mask_mode': mask_mode
        }
    
    def _get_replacement(self, original_text: str, pii_type: str, mask_mode: str) -> str:
        """Get the replacement text based on mask mode and PII type."""
        if mask_mode == "remove":
            return ""
        elif mask_mode == "mask":
            return self.mask_char * len(original_text)
        else:  # replace
            return self.replacement_text.get(pii_type, '[PII]')
    
    def analyze_pii_risk(self, text: str) -> Dict[str, any]:
        """
        Analyze the PII risk level of the text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing risk assessment
        """
        pii_findings = self.detect_pii(text)
        
        # Calculate risk score
        risk_weights = {
            'PERSON': 3,
            'ORG': 2,
            'GPE': 1,
            'email': 4,
            'phone': 4,
            'ssn': 5,
            'credit_card': 5,
            'ip_address': 2,
            'url': 1,
            'date': 1,
            'address': 3,
            'employee_id': 3,
            'token': 3,
            'client_name': 4
        }
        
        total_score = 0
        
        # Score named entities
        for entity in pii_findings['named_entities']:
            total_score += risk_weights.get(entity['label'], 1)
        
        # Score regex patterns
        for pattern in pii_findings['regex_patterns']:
            total_score += risk_weights.get(pattern['pattern'], 1)
        
        # Score client names
        total_score += len(pii_findings['client_names']) * risk_weights.get('client_name', 4)
        
        # Determine risk level
        if total_score == 0:
            risk_level = "LOW"
        elif total_score <= 5:
            risk_level = "MEDIUM"
        elif total_score <= 15:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        return {
            'risk_level': risk_level,
            'risk_score': total_score,
            'pii_count': (len(pii_findings['named_entities']) + 
                         len(pii_findings['regex_patterns']) + 
                         len(pii_findings['client_names'])),
            'recommendations': self._get_risk_recommendations(risk_level, pii_findings)
        }
    
    def clean_structured_data(self, data: Dict[str, Any], mask_mode: str = "replace") -> Dict[str, Any]:
        """
        Clean PII from structured data (e.g., Excel columns).
        
        Args:
            data: Dictionary containing column names and values
            mask_mode: How to handle PII
            
        Returns:
            Dictionary with cleaned data and statistics
        """
        cleaned_data = {}
        stats = {
            'total_fields': len(data),
            'fields_with_pii': 0,
            'pii_instances': 0,
            'pii_types': {}
        }
        
        for column, value in data.items():
            column_lower = str(column).lower()
            
            if column_lower in self.column_pii_types:
                pii_type = self.column_pii_types[column_lower]
                replacement = self._get_replacement(str(value), pii_type, mask_mode)
                cleaned_data[column] = replacement
                
                if str(value) != replacement:
                    stats['fields_with_pii'] += 1
                    stats['pii_instances'] += 1
                    stats['pii_types'][pii_type] = stats['pii_types'].get(pii_type, 0) + 1
            else:
                # Check for PII in unstructured text fields
                if isinstance(value, str) and len(value) > 2:
                    result = self.remove_pii(value, mask_mode)
                    cleaned_data[column] = result['cleaned_text']
                    
                    if result['replacements_made'] > 0:
                        stats['fields_with_pii'] += 1
                        stats['pii_instances'] += result['replacements_made']
                        # Update PII type counts
                        for ent in result['pii_findings']['named_entities']:
                            stats['pii_types'][ent['label']] = stats['pii_types'].get(ent['label'], 0) + 1
                else:
                    cleaned_data[column] = value
        
        return {
            'cleaned_data': cleaned_data,
            'statistics': stats
        }

    def _get_risk_recommendations(self, risk_level: str, pii_findings: Dict) -> List[str]:
        """Get recommendations based on risk level."""
        recommendations = []
        
        if risk_level == "LOW":
            recommendations.append("Text appears to have minimal PII risk.")
        
        elif risk_level == "MEDIUM":
            recommendations.append("Consider reviewing and masking identified PII before sharing.")
            
        elif risk_level == "HIGH":
            recommendations.append("High PII risk detected. Mask or remove PII before processing.")
            recommendations.append("Review all named entities and contact information.")
            
        else:  # CRITICAL
            recommendations.append("CRITICAL: Do not share this content without thorough PII removal.")
            recommendations.append("Multiple high-risk PII types detected (SSN, credit cards, etc.).")
            recommendations.append("Consider legal compliance requirements (GDPR, CCPA, etc.).")
        
        # Specific recommendations based on findings
        if pii_findings['statistics']['pattern_types'].get('ssn', 0) > 0:
            recommendations.append("SSN detected - ensure compliance with privacy regulations.")
        
        if pii_findings['statistics']['pattern_types'].get('credit_card', 0) > 0:
            recommendations.append("Credit card numbers detected - PCI DSS compliance required.")
        
        if pii_findings['statistics']['pattern_types'].get('email', 0) > 0:
            recommendations.append("Email addresses detected - consider data protection implications.")
        
        return recommendations
    
    def create_pii_report(self, text: str, output_file: Optional[str] = None) -> Dict[str, any]:
        """
        Create a comprehensive PII analysis report.
        
        Args:
            text: Input text to analyze
            output_file: Optional file path to save the report
            
        Returns:
            Complete PII analysis report
        """
        pii_findings = self.detect_pii(text)
        risk_analysis = self.analyze_pii_risk(text)
        cleaned_result = self.remove_pii(text, mask_mode="replace")
        
        from datetime import datetime
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'text_length': len(text),
            'pii_findings': pii_findings,
            'risk_analysis': risk_analysis,
            'cleaned_preview': cleaned_result['cleaned_text'][:500] + "..." if len(cleaned_result['cleaned_text']) > 500 else cleaned_result['cleaned_text'],
            'summary': {
                'total_pii_items': (len(pii_findings['named_entities']) + 
                                  len(pii_findings['regex_patterns']) + 
                                  len(pii_findings['client_names'])),
                'risk_level': risk_analysis['risk_level'],
                'requires_cleaning': risk_analysis['risk_level'] in ['HIGH', 'CRITICAL']
            }
        }
        
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                logger.info(f"PII report saved to {output_file}")
            except Exception as e:
                logger.error(f"Error saving report to {output_file}: {str(e)}")
        
        return report


def remove_pii(text: str, mask_mode: str = "replace", client_names_file: Optional[str] = None) -> str:
    """
    Convenience function to remove PII from text.
    
    Args:
        text: Input text to clean
        mask_mode: "replace", "mask", or "remove"
        client_names_file: Path to client names JSON file
        
    Returns:
        Cleaned text
    """
    cleaner = PIICleaner(client_names_file=client_names_file)
    result = cleaner.remove_pii(text, mask_mode=mask_mode)
    return result['cleaned_text']


def detect_pii(text: str, client_names_file: Optional[str] = None) -> Dict[str, List[Dict[str, any]]]:
    """
    Convenience function to detect PII in text.
    
    Args:
        text: Input text to analyze
        client_names_file: Path to client names JSON file
        
    Returns:
        PII findings dictionary
    """
    cleaner = PIICleaner(client_names_file=client_names_file)
    return cleaner.detect_pii(text)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        test_text = " ".join(sys.argv[1:])
    else:
        test_text = """
        John Doe's email is john.doe@example.com and his phone number is (555) 123-4567.
        His SSN is 123-45-6789 and he lives at 123 Main Street, Anytown, NY.
        The meeting with Acme Corp is scheduled for tomorrow.
        """
    
    print("Original text:")
    print(test_text)
    print("\n" + "="*50 + "\n")
    
    # Create cleaner
    cleaner = PIICleaner()
    
    # Detect PII
    findings = cleaner.detect_pii(test_text)
    print("PII Findings:")
    print(json.dumps(findings['statistics'], indent=2))
    
    # Risk analysis
    risk = cleaner.analyze_pii_risk(test_text)
    print(f"\nRisk Level: {risk['risk_level']}")
    print(f"Risk Score: {risk['risk_score']}")
    
    # Clean text
    cleaned = cleaner.remove_pii(test_text)
    print(f"\nCleaned text:")
    print(cleaned['cleaned_text'])
