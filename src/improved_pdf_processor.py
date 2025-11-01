"""
Improved PDF processor with enhanced text detection and PII masking.
"""

import fitz
import pdfplumber
from PIL import Image
import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import tempfile
import os
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import io

from .enhanced_text_processor import EnhancedTextProcessor, OCRConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedPDFProcessor:
    """PDF processor with enhanced text detection and PII masking."""
    
    def __init__(self, ocr_config: Optional[OCRConfig] = None):
        """Initialize processor with enhanced text processing."""
        self.text_processor = EnhancedTextProcessor(ocr_config)
        
    def process_file(self, 
                    input_file: Union[str, Path], 
                    output_file: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Process PDF file with enhanced PII detection and masking.
        
        Args:
            input_file: Input PDF path
            output_file: Output PDF path
            
        Returns:
            Processing results
        """
        input_file = Path(input_file)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        if not output_file:
            output_file = input_file.parent / f"cleaned_{input_file.name}"
        output_file = Path(output_file)

        stats = {
            'total_pages': 0,
            'pages_processed': 0,
            'pii_instances': 0,
            'processing_time': 0
        }

        start_time = datetime.now()
        text_content = []

        try:
            # Open PDF
            pdf_doc = fitz.open(input_file)
            stats['total_pages'] = len(pdf_doc)

            # Create output PDF
            out_doc = fitz.open()

            # Process each page
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                new_page = out_doc.new_page(width=page.rect.width,
                                          height=page.rect.height)

                # Process page
                result = self._process_page(page, new_page)
                stats['pages_processed'] += 1
                stats['pii_instances'] += result['pii_instances']
                text_content.append(result['text_content'])

            # Save processed PDF and text summary
            output_file.parent.mkdir(parents=True, exist_ok=True)
            out_doc.save(output_file)

            # Create text summary
            summary_file = output_file.with_suffix('.txt')
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(text_content))

            stats['processing_time'] = (datetime.now() - start_time).total_seconds()
            stats['success'] = True
            stats['output_file'] = str(output_file)
            stats['text_summary'] = str(summary_file)

            return stats

        except Exception as e:
            logger.error(f"Error processing PDF file {input_file}: {str(e)}")
            raise RuntimeError(f"PDF processing failed: {str(e)}")

    def _process_page(self, page: fitz.Page, new_page: fitz.Page) -> Dict[str, Any]:
        """Process a single PDF page."""
        # Copy original content
        new_page.show_pdf_page(new_page.rect, page.parent, page.number)

        # Get page image
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Convert to numpy array for processing
        np_image = np.array(img)

        # Process with enhanced text processor
        result = self.text_processor.process_image(np_image)

        if result['pii_detected']:
            # Convert masked image back to PDF
            masked_img = Image.fromarray(result['masked_image'])
            
            # Create temporary image stream
            img_stream = io.BytesIO()
            masked_img.save(img_stream, format='PNG')
            img_stream.seek(0)

            # Replace page content with masked image
            new_page.insert_image(new_page.rect, stream=img_stream)
            
        # Extract text for summary
        text = page.get_text("text")
        if text.strip():
            text_content = f"\n=== Page {page.number + 1} ===\n{text}"
        else:
            # Use OCR text from enhanced text processor
            text_content = f"\n=== Page {page.number + 1} (OCR) ===\n{result['text']}"
            
        return {
            'text_content': text_content,
            'pii_instances': len(result['pii_detected'])
        }

    def generate_report(self, 
                   files: List[Union[str, Path]], 
                   output_file: Optional[Union[str, Path]] = None) -> str:
        """Generate analysis report for processed files."""
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not output_file:
            report_dir = Path("data/reports")
            report_dir.mkdir(parents=True, exist_ok=True)
            output_file = report_dir / f"security_analysis_report_{timestamp}.md"
        output_file = Path(output_file)

        # Process all files
        results = []
        for file_path in files:
            file_path = Path(file_path)
            
            try:
                if file_path.suffix.lower() == '.pdf':
                    # Process PDF file
                    result = self.process_file(file_path)
                    results.append({
                        'file_name': file_path.name,
                        'file_type': 'PDF',
                        'pages': result['total_pages'],
                        'pii_detected': result['pii_instances'],
                        'status': 'Processed' if result['success'] else 'Error',
                        'findings': [
                            f"{result['total_pages']} pages processed",
                            f"{result['pii_instances']} PII instances found",
                            f"Processing time: {result['processing_time']:.2f}s"
                        ]
                    })
                else:
                    # Process image file
                    img = cv2.imread(str(file_path))
                    if img is None:
                        raise ValueError(f"Could not load image: {file_path}")
                    
                    result = self.text_processor.process_image(img)
                    
                    results.append({
                        'file_name': file_path.name,
                        'file_type': file_path.suffix[1:].upper(),
                        'pii_detected': len(result['pii_detected']),
                        'status': 'Processed',
                        'findings': [
                            f"{len(result['pii_detected'])} PII instances found",
                            f"Text confidence: {result['confidence']:.1f}%"
                        ]
                    })

            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                results.append({
                    'file_name': file_path.name,
                    'file_type': file_path.suffix[1:].upper(),
                    'status': 'Error',
                    'findings': [f"Error: {str(e)}"]
                })

        # Generate report in markdown format
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Security Analysis Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## File Analysis Results\n\n")
            f.write("| File Name | Type | Status | Findings |\n")
            f.write("|-----------|------|--------|----------|\n")
            
            for result in results:
                findings = '<br>'.join(f'• {finding}' for finding in result['findings'])
                status_icon = "✓" if result['status'] == 'Processed' else "⚠"
                f.write(f"| {result['file_name']} | {result['file_type']} | "
                       f"{status_icon} {result['status']} | {findings} |\n")

        return str(output_file)