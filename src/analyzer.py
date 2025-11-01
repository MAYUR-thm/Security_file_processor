"""
Security Content Analyzer Module

This module analyzes cleansed text to extract security-related information including:
- IAM policy statements
- Firewall rule entries
- IDS/IPS log snippets
- Network security configurations
- Vulnerability information
- Security compliance data
"""

import re
import json
import logging
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from datetime import datetime

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecurityAnalyzer:
    """Main class for analyzing security-related content in text."""
    
    def __init__(self):
        """Initialize the SecurityAnalyzer."""
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        
        # Security-related patterns and keywords
        self.security_patterns = {
            'iam_policies': {
                'patterns': [
                    r'(?i)\b(?:allow|deny)\s+(?:user|group|role)\s+\w+\s+(?:to|from)\s+\w+',
                    r'(?i)\b(?:grant|revoke)\s+(?:permission|access|privilege)\s+\w+',
                    r'(?i)\b(?:policy|permission|role|user|group)\s*:\s*["\']?[\w\-\.]+["\']?',
                    r'(?i)\b(?:arn|resource)\s*:\s*["\']?arn:[\w\-:\/\*]+["\']?',
                    r'(?i)\b(?:effect|action|resource|condition)\s*:\s*["\']?[\w\-\.\*\/]+["\']?',
                    r'(?i)\b(?:assume|trust)\s+(?:role|policy)',
                ],
                'keywords': [
                    'iam', 'policy', 'permission', 'role', 'user', 'group', 'access',
                    'allow', 'deny', 'grant', 'revoke', 'assume', 'trust', 'arn',
                    'resource', 'action', 'effect', 'condition', 'principal'
                ]
            },
            'firewall_rules': {
                'patterns': [
                    r'(?i)\b(?:allow|deny|block|drop)\s+(?:tcp|udp|icmp|ip)\s+(?:port\s+)?\d+',
                    r'(?i)\b(?:source|destination|src|dst)\s+(?:ip|address)?\s*:\s*\d+\.\d+\.\d+\.\d+',
                    r'(?i)\b(?:port|protocol)\s*:\s*\d+',
                    r'(?i)\b(?:inbound|outbound|ingress|egress)\s+(?:rule|traffic)',
                    r'(?i)\b(?:security\s+group|network\s+acl|firewall\s+rule)',
                    r'(?i)\b(?:tcp|udp|icmp|http|https|ssh|ftp|smtp|dns)\s+(?:port\s+)?\d+',
                ],
                'keywords': [
                    'firewall', 'rule', 'allow', 'deny', 'block', 'drop', 'port',
                    'protocol', 'tcp', 'udp', 'icmp', 'ip', 'address', 'source',
                    'destination', 'inbound', 'outbound', 'ingress', 'egress',
                    'security group', 'network acl', 'iptables', 'netfilter'
                ]
            },
            'ids_ips_logs': {
                'patterns': [
                    r'(?i)\b(?:alert|warning|critical|high|medium|low)\s+(?:severity|priority)',
                    r'(?i)\b(?:attack|intrusion|malware|virus|trojan|exploit)\s+(?:detected|found|blocked)',
                    r'(?i)\b(?:signature|rule)\s+id\s*:\s*\d+',
                    r'(?i)\b(?:src|source)\s+ip\s*:\s*\d+\.\d+\.\d+\.\d+',
                    r'(?i)\b(?:dst|destination)\s+ip\s*:\s*\d+\.\d+\.\d+\.\d+',
                    r'(?i)\b(?:snort|suricata|bro|zeek)\s+(?:alert|log|event)',
                ],
                'keywords': [
                    'ids', 'ips', 'intrusion', 'detection', 'prevention', 'alert',
                    'signature', 'rule', 'attack', 'malware', 'virus', 'exploit',
                    'snort', 'suricata', 'bro', 'zeek', 'severity', 'priority'
                ]
            },
            'network_security': {
                'patterns': [
                    r'(?i)\b(?:vpn|ssl|tls|ipsec)\s+(?:connection|tunnel|encryption)',
                    r'(?i)\b(?:certificate|cert)\s+(?:expired|invalid|revoked)',
                    r'(?i)\b(?:encryption|cipher|algorithm)\s*:\s*[\w\-]+',
                    r'(?i)\b(?:vlan|subnet|network)\s+(?:id|address)\s*:\s*[\d\.\/]+',
                    r'(?i)\b(?:nat|proxy|load\s+balancer)\s+(?:rule|configuration)',
                ],
                'keywords': [
                    'network', 'security', 'vpn', 'ssl', 'tls', 'ipsec', 'encryption',
                    'certificate', 'cipher', 'algorithm', 'vlan', 'subnet', 'nat',
                    'proxy', 'load balancer', 'dmz', 'perimeter'
                ]
            },
            'vulnerability_info': {
                'patterns': [
                    r'(?i)\bcve-\d{4}-\d{4,}',
                    r'(?i)\b(?:vulnerability|exploit|cve|cwe)\s+(?:id|number)\s*:\s*[\w\-]+',
                    r'(?i)\b(?:cvss|score)\s*:\s*\d+\.?\d*',
                    r'(?i)\b(?:critical|high|medium|low)\s+(?:vulnerability|risk|severity)',
                    r'(?i)\b(?:patch|update|fix)\s+(?:available|required|installed)',
                ],
                'keywords': [
                    'vulnerability', 'exploit', 'cve', 'cwe', 'cvss', 'score',
                    'patch', 'update', 'fix', 'critical', 'high', 'medium', 'low',
                    'severity', 'risk', 'exposure'
                ]
            },
            'compliance': {
                'patterns': [
                    r'(?i)\b(?:pci|dss|hipaa|gdpr|sox|iso\s*27001|nist)\s+(?:compliance|standard|requirement)',
                    r'(?i)\b(?:audit|assessment|review)\s+(?:finding|result|recommendation)',
                    r'(?i)\b(?:control|requirement)\s+(?:id|number)\s*:\s*[\w\-\.]+',
                    r'(?i)\b(?:compliant|non-compliant|violation|exception)',
                ],
                'keywords': [
                    'compliance', 'audit', 'assessment', 'control', 'requirement',
                    'pci', 'dss', 'hipaa', 'gdpr', 'sox', 'iso', '27001', 'nist',
                    'standard', 'framework', 'regulation', 'policy'
                ]
            }
        }
        
        # Initialize TF-IDF vectorizer for similarity analysis
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def analyze_security_content(self, text: str) -> Dict[str, any]:
        """
        Analyze text for security-related content.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing security analysis results
        """
        logger.info("Starting security content analysis")
        
        # Initialize results structure
        results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'text_length': len(text),
            'categories': {},
            'summary': {
                'total_findings': 0,
                'categories_found': [],
                'confidence_score': 0.0
            },
            'key_findings': [],
            'recommendations': []
        }
        
        # Analyze each security category
        for category, config in self.security_patterns.items():
            category_results = self._analyze_category(text, category, config)
            results['categories'][category] = category_results
            
            if category_results['findings']:
                results['summary']['categories_found'].append(category)
                results['summary']['total_findings'] += len(category_results['findings'])
        
        # Extract key findings
        results['key_findings'] = self._extract_key_findings(results['categories'])
        
        # Calculate overall confidence score
        results['summary']['confidence_score'] = self._calculate_confidence_score(results['categories'])
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results['categories'])
        
        logger.info(f"Analysis complete: {results['summary']['total_findings']} findings across {len(results['summary']['categories_found'])} categories")
        
        return results
    
    def _analyze_category(self, text: str, category: str, config: Dict) -> Dict[str, any]:
        """Analyze text for a specific security category."""
        findings = []
        
        # Pattern-based detection
        for pattern in config['patterns']:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                findings.append({
                    'type': 'pattern_match',
                    'pattern': pattern,
                    'text': match.group().strip(),
                    'start': match.start(),
                    'end': match.end(),
                    'context': self._get_context(text, match.start(), match.end())
                })
        
        # Keyword-based detection with context
        keyword_findings = self._find_keyword_contexts(text, config['keywords'])
        findings.extend(keyword_findings)
        
        # Calculate category confidence
        confidence = self._calculate_category_confidence(findings, text, config)
        
        return {
            'category': category,
            'findings': findings,
            'finding_count': len(findings),
            'confidence': confidence,
            'keywords_found': list(set([f['keyword'] for f in keyword_findings if 'keyword' in f]))
        }
    
    def _find_keyword_contexts(self, text: str, keywords: List[str]) -> List[Dict[str, any]]:
        """Find keyword occurrences with their contexts."""
        findings = []
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            for keyword in keywords:
                if keyword.lower() in sentence_lower:
                    # Find the exact position in the original text
                    start_pos = text.lower().find(sentence_lower)
                    keyword_pos = sentence_lower.find(keyword.lower())
                    
                    if start_pos != -1 and keyword_pos != -1:
                        findings.append({
                            'type': 'keyword_context',
                            'keyword': keyword,
                            'sentence': sentence.strip(),
                            'context': sentence.strip(),
                            'start': start_pos + keyword_pos,
                            'end': start_pos + keyword_pos + len(keyword)
                        })
        
        return findings
    
    def _get_context(self, text: str, start: int, end: int, context_size: int = 100) -> str:
        """Get context around a match."""
        context_start = max(0, start - context_size)
        context_end = min(len(text), end + context_size)
        
        context = text[context_start:context_end].strip()
        
        # Add ellipsis if truncated
        if context_start > 0:
            context = "..." + context
        if context_end < len(text):
            context = context + "..."
        
        return context
    
    def _calculate_category_confidence(self, findings: List[Dict], text: str, config: Dict) -> float:
        """Calculate confidence score for a category."""
        if not findings:
            return 0.0
        
        # Base score from number of findings
        finding_score = min(len(findings) / 10.0, 1.0)  # Normalize to 0-1
        
        # Keyword density score
        keyword_count = sum(1 for f in findings if f['type'] == 'keyword_context')
        total_words = len(word_tokenize(text))
        keyword_density = keyword_count / total_words if total_words > 0 else 0
        density_score = min(keyword_density * 100, 1.0)  # Scale and cap at 1.0
        
        # Pattern match score (higher weight for regex matches)
        pattern_count = sum(1 for f in findings if f['type'] == 'pattern_match')
        pattern_score = min(pattern_count / 5.0, 1.0)
        
        # Weighted average
        confidence = (finding_score * 0.3 + density_score * 0.3 + pattern_score * 0.4)
        
        return round(confidence, 3)
    
    def _extract_key_findings(self, categories: Dict[str, Dict]) -> List[str]:
        """Extract the most significant findings across all categories."""
        key_findings = []
        
        for category, data in categories.items():
            if data['confidence'] > 0.3:  # Only include high-confidence categories
                # Get top findings from this category
                findings = data['findings']
                
                # Prioritize pattern matches over keyword matches
                pattern_matches = [f for f in findings if f['type'] == 'pattern_match']
                keyword_matches = [f for f in findings if f['type'] == 'keyword_context']
                
                # Add top pattern matches
                for finding in pattern_matches[:3]:  # Top 3 pattern matches
                    description = f"{category.replace('_', ' ').title()}: {finding['text']}"
                    if description not in key_findings:
                        key_findings.append(description)
                
                # Add representative keyword findings if no pattern matches
                if not pattern_matches and keyword_matches:
                    unique_sentences = list(set([f['sentence'] for f in keyword_matches]))
                    for sentence in unique_sentences[:2]:  # Top 2 unique sentences
                        description = f"{category.replace('_', ' ').title()}: {sentence[:100]}..."
                        if description not in key_findings:
                            key_findings.append(description)
        
        return key_findings[:10]  # Return top 10 key findings
    
    def _calculate_confidence_score(self, categories: Dict[str, Dict]) -> float:
        """Calculate overall confidence score for the analysis."""
        if not categories:
            return 0.0
        
        # Weight categories by their importance
        category_weights = {
            'iam_policies': 0.25,
            'firewall_rules': 0.25,
            'ids_ips_logs': 0.20,
            'network_security': 0.15,
            'vulnerability_info': 0.10,
            'compliance': 0.05
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for category, data in categories.items():
            weight = category_weights.get(category, 0.1)
            weighted_score += data['confidence'] * weight
            total_weight += weight
        
        return round(weighted_score / total_weight if total_weight > 0 else 0.0, 3)
    
    def _generate_recommendations(self, categories: Dict[str, Dict]) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        for category, data in categories.items():
            if data['confidence'] > 0.5:  # High confidence findings
                if category == 'iam_policies':
                    recommendations.append("Review IAM policies for least privilege compliance")
                    recommendations.append("Audit user permissions and role assignments")
                
                elif category == 'firewall_rules':
                    recommendations.append("Validate firewall rules for security gaps")
                    recommendations.append("Ensure proper network segmentation")
                
                elif category == 'ids_ips_logs':
                    recommendations.append("Investigate security alerts and incidents")
                    recommendations.append("Update IDS/IPS signatures and rules")
                
                elif category == 'network_security':
                    recommendations.append("Review network security configurations")
                    recommendations.append("Validate encryption and certificate settings")
                
                elif category == 'vulnerability_info':
                    recommendations.append("Prioritize vulnerability remediation")
                    recommendations.append("Implement security patching procedures")
                
                elif category == 'compliance':
                    recommendations.append("Address compliance gaps and violations")
                    recommendations.append("Schedule regular compliance assessments")
        
        # Add general recommendations
        if len([c for c in categories.values() if c['confidence'] > 0.3]) > 2:
            recommendations.append("Conduct comprehensive security review")
            recommendations.append("Implement security monitoring and alerting")
        
        return list(set(recommendations))  # Remove duplicates
    
    def extract_structured_data(self, text: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Extract structured security data from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing structured security data
        """
        structured_data = {
            'ip_addresses': [],
            'ports': [],
            'protocols': [],
            'domains': [],
            'file_paths': [],
            'usernames': [],
            'timestamps': []
        }
        
        # IP addresses
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        for match in re.finditer(ip_pattern, text):
            structured_data['ip_addresses'].append({
                'value': match.group(),
                'context': self._get_context(text, match.start(), match.end(), 50)
            })
        
        # Ports
        port_pattern = r'\b(?:port\s+)?(\d{1,5})\b'
        for match in re.finditer(port_pattern, text, re.IGNORECASE):
            port_num = int(match.group(1))
            if 1 <= port_num <= 65535:
                structured_data['ports'].append({
                    'value': str(port_num),
                    'context': self._get_context(text, match.start(), match.end(), 50)
                })
        
        # Protocols
        protocol_pattern = r'\b(tcp|udp|icmp|http|https|ssh|ftp|smtp|dns|ssl|tls)\b'
        for match in re.finditer(protocol_pattern, text, re.IGNORECASE):
            structured_data['protocols'].append({
                'value': match.group().upper(),
                'context': self._get_context(text, match.start(), match.end(), 50)
            })
        
        # Domains
        domain_pattern = r'\b[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)*[a-zA-Z]{2,}\b'
        for match in re.finditer(domain_pattern, text):
            structured_data['domains'].append({
                'value': match.group(),
                'context': self._get_context(text, match.start(), match.end(), 50)
            })
        
        # File paths
        path_pattern = r'(?:[a-zA-Z]:\\|/)[^\s<>"\'|?*\n\r]+'
        for match in re.finditer(path_pattern, text):
            structured_data['file_paths'].append({
                'value': match.group(),
                'context': self._get_context(text, match.start(), match.end(), 50)
            })
        
        # Usernames (simple pattern)
        username_pattern = r'\b(?:user|username|account)\s*[:=]\s*([a-zA-Z0-9_\-\.]+)'
        for match in re.finditer(username_pattern, text, re.IGNORECASE):
            structured_data['usernames'].append({
                'value': match.group(1),
                'context': self._get_context(text, match.start(), match.end(), 50)
            })
        
        # Timestamps
        timestamp_pattern = r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}[\sT]\d{1,2}:\d{1,2}:\d{1,2}'
        for match in re.finditer(timestamp_pattern, text):
            structured_data['timestamps'].append({
                'value': match.group(),
                'context': self._get_context(text, match.start(), match.end(), 50)
            })
        
        return structured_data
    
    def generate_file_description(self, analysis_results: Dict[str, any]) -> str:
        """Generate a concise file description based on analysis results."""
        categories_found = analysis_results['summary']['categories_found']
        confidence = analysis_results['summary']['confidence_score']
        
        if not categories_found:
            return "No significant security content detected"
        
        # Create description based on dominant categories
        descriptions = []
        
        for category in categories_found:
            category_data = analysis_results['categories'][category]
            
            if category_data['confidence'] > 0.5:
                if category == 'iam_policies':
                    descriptions.append("IAM policy configurations")
                elif category == 'firewall_rules':
                    descriptions.append("firewall rule definitions")
                elif category == 'ids_ips_logs':
                    descriptions.append("intrusion detection logs")
                elif category == 'network_security':
                    descriptions.append("network security settings")
                elif category == 'vulnerability_info':
                    descriptions.append("vulnerability information")
                elif category == 'compliance':
                    descriptions.append("compliance documentation")
        
        if descriptions:
            if len(descriptions) == 1:
                return f"Contains {descriptions[0]}"
            elif len(descriptions) == 2:
                return f"Contains {descriptions[0]} and {descriptions[1]}"
            else:
                return f"Contains {', '.join(descriptions[:-1])}, and {descriptions[-1]}"
        else:
            return f"Contains security-related content ({', '.join(categories_found)})"


def analyze_security_content(text: str) -> Dict[str, any]:
    """
    Convenience function to analyze security content in text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Security analysis results
    """
    analyzer = SecurityAnalyzer()
    return analyzer.analyze_security_content(text)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        # Read from file
        file_path = sys.argv[1]
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                test_text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
    else:
        # Use sample text
        test_text = """
        IAM Policy: Allow user john.doe to access S3 bucket company-data
        Firewall Rule: Deny TCP port 22 from external networks
        Alert: High severity intrusion detected from IP 192.168.1.100
        CVE-2023-1234: Critical vulnerability in Apache HTTP Server
        Network: SSL/TLS encryption enabled on all connections
        Compliance: PCI DSS requirement 3.4 - Encrypt cardholder data
        """
    
    print("Analyzing security content...")
    print("=" * 50)
    
    analyzer = SecurityAnalyzer()
    results = analyzer.analyze_security_content(test_text)
    
    print(f"Analysis Results:")
    print(f"Total findings: {results['summary']['total_findings']}")
    print(f"Categories found: {', '.join(results['summary']['categories_found'])}")
    print(f"Confidence score: {results['summary']['confidence_score']}")
    
    print(f"\nKey Findings:")
    for finding in results['key_findings']:
        print(f"- {finding}")
    
    print(f"\nRecommendations:")
    for rec in results['recommendations']:
        print(f"- {rec}")
    
    # Generate file description
    description = analyzer.generate_file_description(results)
    print(f"\nFile Description: {description}")
