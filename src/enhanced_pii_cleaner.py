"""
Enhanced PII Cleaner with specialized patterns for security documents and infrastructure configurations.
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
import json
import spacy
from spacy.matcher import Matcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPIICleaner:
    """Enhanced PII Cleaner with specialized patterns for security documents."""
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Initialize the enhanced PII cleaner."""
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logger.error(f"spaCy model '{spacy_model}' not found.")
            raise

        # Define specialized patterns for security documents
        self.specialized_patterns = {
            # Network and Infrastructure
            'ip_cidr': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}/[0-9]{1,2}\b',
            'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'port_range': r'\b\d{1,5}(?:-\d{1,5})?\b(?=\s*(?:\/(?:tcp|udp|icmp)|$))',
            
            # Cloud Resources
            'aws_account_id': r'\b\d{12}\b',
            'aws_arn': r'arn:aws:[a-zA-Z0-9\-]*:[a-zA-Z0-9\-]*:\d{12}:.*',
            'aws_role_id': r'\b[A-Z]{2}[A-Z0-9]{16,32}\b',
            'aws_resource': r'\b(?:role|user|group|policy|bucket|function)/[a-zA-Z0-9_\-\.]+\b',
            
            # Security Document Elements
            'certificate_titles': r'\b(?:CERTIFICATE\s+OF|CERTIFICATION\s+OF)\s+[A-Z\s]+\b',
            'authorization_codes': r'\b[A-Z]{2,}-\d{3,}-[A-Z0-9]{2,}\b',
            'security_levels': r'\b(?:CONFIDENTIAL|SECRET|TOP SECRET|RESTRICTED|INTERNAL USE ONLY)\b',
            
            # Professional Titles and Departments
            'job_titles': r'\b(?:Security Officer|Administrator|Director|Manager|Supervisor|Engineer|Analyst)\b',
            'departments': r'\b(?:Security|IT|Information Technology|Operations|Infrastructure)\s+(?:Department|Dept|Team|Group)\b',
            
            # Group and Policy Naming
            'policy_names': r'\b[A-Za-z0-9\-]+(?:Policy|Role|Group|Permission)s?\b',
            'resource_names': r'\b(?:Test|Prod|Dev|Stage)-[A-Za-z0-9\-]+\b'
        }
        
        # Replacement patterns
        self.replacements = {
            'ip_cidr': '[NETWORK_RANGE]',
            'ip_address': '[IP_ADDRESS]',
            'port_range': '[PORT_RANGE]',
            'aws_account_id': '[AWS_ACCOUNT]',
            'aws_arn': '[AWS_ARN]',
            'aws_role_id': '[AWS_ROLE_ID]',
            'aws_resource': '[AWS_RESOURCE]',
            'certificate_titles': '[CERTIFICATE_TYPE]',
            'authorization_codes': '[AUTH_CODE]',
            'security_levels': '[SECURITY_LEVEL]',
            'job_titles': '[JOB_TITLE]',
            'departments': '[DEPARTMENT]',
            'policy_names': '[POLICY_NAME]',
            'resource_names': '[RESOURCE_NAME]',
            'PERSON': '[PERSON]',
            'ORG': '[ORGANIZATION]',
            'DATE': '[DATE]'
        }

        # Initialize spaCy matcher for custom patterns
        self.matcher = Matcher(self.nlp.vocab)
        self._add_custom_patterns()

    def _add_custom_patterns(self):
        """Add custom patterns to the spaCy matcher."""
        # Add patterns for detecting job titles with names
        self.matcher.add("JOB_TITLE_WITH_NAME", [
            [{"TEXT": {"REGEX": r"[A-Z][a-z]+"}}, 
             {"TEXT": {"REGEX": r"[A-Z][a-z]+"}}, 
             {"LOWER": {"IN": ["officer", "administrator", "director", "manager", "engineer"]}}]
        ])

    def clean_text(self, text: str, mask_mode: str = "replace") -> Dict[str, Any]:
        """
        Clean PII and sensitive information from text.
        
        Args:
            text: Text to clean
            mask_mode: 'replace', 'mask', or 'remove'
            
        Returns:
            Dict containing cleaned text and detection statistics
        """
        if not text:
            return {"cleaned_text": text, "detections": {}}

        # Initialize detection tracking
        detections = {pattern_name: [] for pattern_name in self.specialized_patterns.keys()}
        detections.update({'named_entities': [], 'custom_patterns': []})

        # Process with spaCy
        doc = self.nlp(text)
        
        # Get all matches that need to be replaced
        replacements = []

        # 1. Check for named entities
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'DATE']:
                replacements.append({
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'text': ent.text,
                    'type': ent.label_
                })
                detections['named_entities'].append({
                    'text': ent.text,
                    'type': ent.label_
                })

        # 2. Check for custom patterns
        matcher_matches = self.matcher(doc)
        for match_id, start, end in matcher_matches:
            pattern_name = self.nlp.vocab.strings[match_id]
            replacements.append({
                'start': doc[start].idx,
                'end': doc[end-1].idx + len(doc[end-1].text),
                'text': doc[start:end].text,
                'type': pattern_name
            })
            detections['custom_patterns'].append({
                'text': doc[start:end].text,
                'type': pattern_name
            })

        # 3. Check for specialized patterns
        for pattern_name, pattern in self.specialized_patterns.items():
            for match in re.finditer(pattern, text):
                replacements.append({
                    'start': match.start(),
                    'end': match.end(),
                    'text': match.group(),
                    'type': pattern_name
                })
                detections[pattern_name].append({
                    'text': match.group(),
                    'type': pattern_name
                })

        # Sort replacements by start position (reverse order to maintain string indices)
        replacements.sort(key=lambda x: x['start'], reverse=True)

        # Apply replacements
        cleaned_text = text
        for repl in replacements:
            replacement_text = self.replacements.get(repl['type'], '[REDACTED]')
            if mask_mode == "mask":
                replacement_text = '*' * len(repl['text'])
            elif mask_mode == "remove":
                replacement_text = ''
            
            cleaned_text = (
                cleaned_text[:repl['start']] +
                replacement_text +
                cleaned_text[repl['end']:]
            )

        return {
            "cleaned_text": cleaned_text,
            "detections": detections
        }

    def clean_document(self, doc_dict: Dict[str, Any], mask_mode: str = "replace") -> Dict[str, Any]:
        """
        Clean PII from a document dictionary (for structured documents).
        
        Args:
            doc_dict: Dictionary containing document fields
            mask_mode: 'replace', 'mask', or 'remove'
            
        Returns:
            Cleaned document dictionary
        """
        cleaned_doc = {}
        
        for key, value in doc_dict.items():
            if isinstance(value, str):
                cleaned_doc[key] = self.clean_text(value, mask_mode)["cleaned_text"]
            elif isinstance(value, dict):
                cleaned_doc[key] = self.clean_document(value, mask_mode)
            elif isinstance(value, list):
                cleaned_doc[key] = [
                    self.clean_document(item, mask_mode) if isinstance(item, dict)
                    else self.clean_text(item, mask_mode)["cleaned_text"] if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                cleaned_doc[key] = value
                
        return cleaned_doc

def create_pii_cleaner():
    """Create and return an instance of the EnhancedPIICleaner."""
    return EnhancedPIICleaner()