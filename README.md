# Security File Processor

An automated solution that helps security consultants by cleansing files and analyzing security-related content. The tool processes multiple file types, removes sensitive data (PII, client names, logos), and extracts useful security information.

## 🚀 Features

### Workflow A - File Cleansing
- **PII Detection & Removal**: Automatically detects and masks personally identifiable information including:
  - Names, emails, phone numbers, addresses
  - Government IDs (SSN, etc.)
  - Credit card numbers
  - Custom client names
- **Logo Detection & Masking**: Identifies and masks logos in images using computer vision
- **Multi-format Support**: Processes PDFs, images, Excel, and PowerPoint files

### Workflow B - Security Analysis
- **Security Content Extraction**: Identifies and analyzes:
  - IAM policy statements
  - Firewall rule entries
  - IDS/IPS log snippets
  - Network security configurations
  - Vulnerability information
  - Compliance data
- **Structured Data Extraction**: Extracts IP addresses, ports, protocols, domains, etc.
- **Risk Assessment**: Provides confidence scores and recommendations

### Comprehensive Reporting
- **JSON Reports**: Structured data for programmatic access
- **CSV Reports**: Spreadsheet-compatible analysis results
- **PowerPoint Reports**: Executive summaries with tables and charts
- **Workflow Diagrams**: Visual representation of the processing pipeline

## 📁 Project Structure

```
security_file_processor/
├── data/                    # Input files directory
├── output/                  # Processed files and reports
│   ├── cleaned_files/       # PII-cleansed and logo-masked files
│   ├── reports/            # Generated reports (JSON, CSV, PPTX)
│   └── logs/               # Processing logs and statistics
├── src/                    # Source code modules
│   ├── ocr.py              # Text extraction and OCR
│   ├── pii_cleaner.py      # PII detection and removal
│   ├── logo_remover.py     # Logo detection and masking
│   ├── analyzer.py         # Security content analysis
│   └── report_generator.py # Report generation
├── main.py                 # Main orchestrator
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.10 - 3.13 (✨ **Optimized for Python 3.13.5**)
- Tesseract OCR (for image text extraction)

### Install Tesseract OCR

**Windows:**
```bash
# Download and install from: https://github.com/UB-Mannheim/tesseract/wiki
# Or using chocolatey:
choco install tesseract
```

**macOS:**
```bash
brew install tesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr
```

### Install Python Dependencies

1. Clone or download this repository
2. Navigate to the project directory
3. Install dependencies:

```bash
pip install -r requirements.txt
```

> **🎯 Python 3.13 Users**: See [PYTHON_313_SETUP.md](PYTHON_313_SETUP.md) for optimized installation and performance tips!

### Install spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

## 🚀 Quick Start

### Basic Usage

Process a single file:
```bash
python main.py --input document.pdf
```

Process a directory of files:
```bash
python main.py --input data/files --output results
```

Process with custom settings:
```bash
python main.py --input data/ --client-names client_names.json --logo-templates logos/ --workflow both
```

### Command Line Options

```bash
python main.py [OPTIONS]

Required:
  --input, -i PATH          Input file or directory path

Optional:
  --output, -o PATH         Output directory (default: output)
  --workflow, -w CHOICE     Processing workflow: cleansing, analysis, both (default: both)
  --client-names, -c PATH   JSON file containing client names to mask
  --logo-templates, -l PATH Directory containing logo template images
  --ocr-engine CHOICE       OCR engine: pytesseract, easyocr (default: pytesseract)
  --verbose, -v             Enable verbose logging
  --help, -h                Show help message
  --version                 Show version information
```

## 📋 Supported File Types

| File Type | Extensions | Processing Method |
|-----------|------------|-------------------|
| PDF | `.pdf` | Direct text extraction + OCR for scanned pages |
| Images | `.jpg`, `.jpeg`, `.png` | OCR text extraction + logo detection |
| Excel | `.xlsx` | Cell content extraction |
| PowerPoint | `.pptx` | Slide text extraction |
| Archives | `.zip` | Extract and process individual files |

## 🔧 Configuration

### Client Names Configuration

Create a JSON file with client names to be masked:

```json
{
  "client_names": [
    "Acme Corporation",
    "TechCorp Inc",
    "Global Solutions Ltd"
  ]
}
```

Or as a simple array:
```json
[
  "Acme Corporation",
  "TechCorp Inc",
  "Global Solutions Ltd"
]
```

### Logo Templates

Place logo template images in a directory structure like:
```
logos/
├── company_logo.png
├── client_logo.jpg
└── brand_mark.png
```

The system will use these templates for logo detection and masking.

## 📊 Output Structure

### Cleaned Files
- **Text files**: `cleaned_<filename>.txt` - PII-cleansed text content
- **Images**: `cleaned_<filename>.<ext>` - Logo-masked images

### Reports
- **JSON Report**: Complete structured analysis results
- **CSV Report**: Tabular data for spreadsheet analysis
- **PowerPoint Report**: Executive presentation with charts and tables
- **Statistics**: Processing metrics and performance data
- **Workflow Diagram**: Visual pipeline representation

### Example JSON Output Structure
```json
{
  "metadata": {
    "generated_at": "2024-01-15T10:30:00",
    "total_files_processed": 5
  },
  "summary": {
    "total_files": 5,
    "successful_files": 4,
    "failed_files": 1,
    "success_rate": 80.0
  },
  "files": [
    {
      "file_name": "security_policy.pdf",
      "file_type": "pdf",
      "file_description": "Contains IAM policy configurations",
      "pii_analysis": {
        "risk_level": "MEDIUM",
        "pii_count": 3,
        "recommendations": ["Review and mask identified PII"]
      },
      "security_analysis": {
        "summary": {
          "total_findings": 15,
          "confidence_score": 0.85,
          "categories_found": ["iam_policies", "firewall_rules"]
        },
        "key_findings": [
          "IAM Policy: Allow user access to S3 bucket",
          "Firewall Rule: Deny TCP port 22"
        ]
      }
    }
  ]
}
```

## 🔍 Security Analysis Categories

The system analyzes content for these security categories:

1. **IAM Policies**: User permissions, role assignments, access controls
2. **Firewall Rules**: Network access controls, port configurations
3. **IDS/IPS Logs**: Intrusion detection alerts, security events
4. **Network Security**: VPN, SSL/TLS, encryption configurations
5. **Vulnerability Information**: CVE references, security patches
6. **Compliance**: Regulatory requirements, audit findings

## 🎯 Use Cases

### Security Consultants
- Cleanse client documents before sharing
- Extract security insights from multiple document types
- Generate executive reports for stakeholders
- Maintain compliance with data protection regulations

### Security Teams
- Analyze security configurations across different formats
- Standardize security documentation
- Create comprehensive security assessments
- Track security findings and recommendations

### Compliance Officers
- Remove PII from documents for safe processing
- Generate audit-ready reports
- Ensure data protection compliance
- Document security control implementations

## 🔧 Advanced Usage

### Programmatic Usage

You can also use the components programmatically:

```python
from src.ocr import extract_text_from_file
from src.pii_cleaner import remove_pii
from src.analyzer import analyze_security_content
from src.report_generator import generate_report

# Extract text
text_result = extract_text_from_file("document.pdf")

# Clean PII
cleaned_text = remove_pii(text_result['text'])

# Analyze security content
security_analysis = analyze_security_content(cleaned_text)

# Generate reports
results = [{"file_name": "document.pdf", "security_analysis": security_analysis}]
report_files = generate_report(results)
```

### Batch Processing

Process multiple files efficiently:

```python
from main import SecurityFileProcessor

processor = SecurityFileProcessor(
    output_dir="results",
    client_names_file="clients.json"
)

results = processor.process_files("data/documents", workflow="both")
```

## 🐛 Troubleshooting

### Common Issues

**OCR Not Working:**
- Ensure Tesseract is installed and in PATH
- Try switching OCR engines: `--ocr-engine easyocr`

**Memory Issues with Large Files:**
- Process files individually instead of batch processing
- Increase system memory or use cloud processing

**Missing Dependencies:**
- Run `pip install -r requirements.txt` again
- Check Python version compatibility (3.10+)

**spaCy Model Missing:**
- Install language model: `python -m spacy download en_core_web_sm`

### Performance Optimization

- Use `easyocr` for better accuracy on complex images
- Process images at lower resolution for faster processing
- Use SSD storage for better I/O performance
- Consider GPU acceleration for large-scale processing

## 📝 Logging

The system generates comprehensive logs:

- **Console Output**: Real-time processing status
- **Log Files**: Detailed processing logs in `output/logs/`
- **Statistics**: Performance metrics and processing times

Enable verbose logging:
```bash
python main.py --input data/ --verbose
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the logs in `output/logs/`
3. Create an issue with detailed error information
4. Include sample files (with PII removed) if possible

## 🔮 Future Enhancements

- [ ] Web interface for easier file uploads
- [ ] API endpoints for integration with other tools
- [ ] Additional OCR engines and language support
- [ ] Machine learning models for better security content detection
- [ ] Cloud storage integration (AWS S3, Azure Blob, etc.)
- [ ] Real-time processing dashboard
- [ ] Advanced logo detection using deep learning
- [ ] Custom security pattern definitions
- [ ] Integration with security tools (SIEM, vulnerability scanners)

---

**Security File Processor v1.0.0** - Automated security document processing and analysis solution.
