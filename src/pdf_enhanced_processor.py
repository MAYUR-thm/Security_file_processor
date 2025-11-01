"""
Enhanced PDF Processor Module

This module combines multiple PDF processing libraries to achieve better results:
- PyMuPDF (fitz) for fast text extraction and PDF manipulation
- pdf2image for converting PDF pages to images
- Tesseract for OCR with specialized configurations
- pdfplumber for accurate table detection and extraction
- PyPDF2 for metadata and structure analysis
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import io
import fitz  # PyMuPDF
import pdfplumber
import PyPDF2
from PIL import Image, ImageEnhance
import pytesseract
from .enhanced_pii_cleaner import create_pii_cleaner
from .enhanced_ocr import EnhancedOCR
from pdf2image import convert_from_path
import numpy as np
import cv2
import tempfile
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFEnhancedProcessor:
    """Enhanced PDF processor that combines multiple libraries for better results."""

    def __init__(self, pii_cleaner: Optional[Any] = None, tesseract_path: Optional[str] = None):
        """
        Initialize the enhanced PDF processor.
        
        Args:
            pii_cleaner: Optional PIICleaner instance
            tesseract_path: Optional path to tesseract executable
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        # Initialize enhanced PII cleaner and OCR
        self.pii_cleaner = pii_cleaner if pii_cleaner else create_pii_cleaner()
        self.ocr_processor = EnhancedOCR(tesseract_path)
        
        # Configure OCR settings
        self.ocr_configs = {
            'standard': r'--oem 3 --psm 6',
            'handwriting': r'--oem 3 --psm 6 -c tessedit_char_blacklist=|',
            'form': r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.-_@/',
            'table': r'--oem 3 --psm 6 -c preserve_interword_spaces=1',
            'code': r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        }
        self.ocr_config = self.ocr_configs['standard']

    def process_file(self, 
                    input_file: Union[str, Path], 
                    output_file: Optional[Union[str, Path]] = None,
                    mask_mode: str = "replace",
                    create_text_summary: bool = True) -> Dict[str, Any]:
        """
        Process a PDF file using multiple libraries for optimal results.
        
        Args:
            input_file: Path to input PDF file
            output_file: Path for output PDF file
            mask_mode: How to handle PII ('replace', 'mask', or 'remove')
            create_text_summary: Whether to create text summary
            
        Returns:
            Dictionary containing processing results
        """
        input_file = Path(input_file)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        if not output_file:
            output_file = input_file.parent / f"cleaned_{input_file.name}"
        else:
            output_file = Path(output_file)

        try:
            stats = {
                'total_pages': 0,
                'pages_with_text': 0,
                'pages_requiring_ocr': 0,
                'tables_extracted': 0,
                'pii_instances': 0,
                'success': False
            }

            # Step 1: Initial analysis using PyPDF2
            with open(input_file, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                stats['total_pages'] = len(pdf_reader.pages)
                metadata = pdf_reader.metadata

            # Step 2: Process with combined approach
            out_doc = fitz.open()
            text_content = []
            
            # Create temporary directory for page images
            with tempfile.TemporaryDirectory() as temp_dir:
                # Convert PDF to images for consistent processing
                pages = convert_from_path(input_file)
                
                for page_num, page_image in enumerate(pages):
                    # Save page image temporarily
                    image_path = os.path.join(temp_dir, f'page_{page_num}.png')
                    page_image.save(image_path, 'PNG')
                    
                    # Process page with multiple methods
                    page_result = self._process_page_multi_method(
                        page_num,
                        image_path,
                        input_file,
                        mask_mode
                    )
                    
                    # Update statistics
                    if page_result['has_text']:
                        stats['pages_with_text'] += 1
                    if page_result['required_ocr']:
                        stats['pages_requiring_ocr'] += 1
                    stats['tables_extracted'] += len(page_result['tables'])
                    
                    # Add processed page to output PDF
                    if page_result['processed_page']:
                        out_doc.insert_pdf(page_result['processed_page'])
                    
                    # Add to text content
                    text_content.extend(page_result['text_content'])

            # Save processed PDF
            output_file.parent.mkdir(parents=True, exist_ok=True)
            out_doc.save(str(output_file))
            
            # Create text summary if requested
            if create_text_summary:
                txt_output = output_file.with_suffix('.txt')
                with open(txt_output, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(text_content))
                stats['text_summary'] = str(txt_output)
            
            stats['success'] = True
            stats['output_file'] = str(output_file)
            
            return stats

        except Exception as e:
            error_msg = f"Error processing PDF file {input_file}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(f"PDF processing failed: {error_msg}")

    def _process_page_multi_method(self, 
                                 page_num: int,
                                 image_path: str,
                                 pdf_path: str,
                                 mask_mode: str) -> Dict[str, Any]:
        """
        Process a single page using multiple methods for optimal results.
        
        Args:
            page_num: Page number
            image_path: Path to page image
            pdf_path: Path to original PDF
            mask_mode: How to handle PII
            
        Returns:
            Dictionary containing page processing results
        """
        result = {
            'has_text': False,
            'required_ocr': False,
            'tables': [],
            'text_content': [],
            'processed_page': None
        }

        try:
            # Method 1: Extract text with PyMuPDF
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            pymupdf_text = page.get_text()
            
            # Method 2: Use pdfplumber for table detection
            with pdfplumber.open(pdf_path) as pdf:
                plumber_page = pdf.pages[page_num]
                tables = plumber_page.extract_tables()
                plumber_text = plumber_page.extract_text()
                result['tables'] = tables

            # Method 3: OCR on the page image
            image = Image.open(image_path)
            # Detect content type
            content_type = self._detect_content_type(image)
            # Preprocess image
            processed_image = self._preprocess_image_for_ocr(image, content_type)
            # Perform OCR
            ocr_text = pytesseract.image_to_string(
                processed_image, 
                config=self.ocr_configs[content_type]
            )

            # Combine and clean text
            combined_text = self._combine_text_results(pymupdf_text, plumber_text, ocr_text)
            if combined_text.strip():
                result['has_text'] = True
                cleaned_text = self._clean_text(combined_text, mask_mode)
                result['text_content'].append(f"=== Page {page_num + 1} ===\n{cleaned_text}")

            # Create processed page
            result['processed_page'] = self._create_processed_page(
                page,
                tables,
                cleaned_text if combined_text.strip() else "",
                mask_mode
            )

            return result

        except Exception as e:
            logger.error(f"Error processing page {page_num}: {str(e)}")
            return result

    def _preprocess_image_for_ocr(self, image: Image.Image, content_type: str) -> Image.Image:
        """Enhanced image preprocessing for better OCR results."""
        # Convert to CV2 format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Apply different preprocessing based on content type
        if content_type == 'handwriting':
            # Enhance contrast
            gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
            # Denoise
            gray = cv2.fastNlMeansDenoising(gray)
            # Adaptive thresholding
            gray = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
        elif content_type in ['form', 'table']:
            # Enhance contrast
            gray = cv2.convertScaleAbs(gray, alpha=1.3, beta=0)
            # Remove noise
            gray = cv2.GaussianBlur(gray, (3,3), 0)
            # Adaptive thresholding
            gray = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            # Deskew
            angle = self._get_skew_angle(gray)
            if abs(angle) > 0.5:
                gray = self._rotate_image(gray, angle)
                
        else:  # standard or code
            # Basic preprocessing
            gray = cv2.GaussianBlur(gray, (3,3), 0)
            gray = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]

        # Convert back to PIL
        return Image.fromarray(gray)

    def _detect_content_type(self, image: Image.Image) -> str:
        """Enhanced content type detection."""
        # Convert to CV2
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Detect lines for table/form detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is not None and len(lines) > 10:
            horizontal_lines = 0
            vertical_lines = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) > abs(y2 - y1):
                    horizontal_lines += 1
                else:
                    vertical_lines += 1
            
            if horizontal_lines > 5 and vertical_lines > 5:
                return 'table'
            elif horizontal_lines > 3 or vertical_lines > 3:
                return 'form'

        # Check for handwriting
        pixel_std = np.std(gray)
        gradient_magnitude = np.gradient(gray)
        gradient_std = np.std(gradient_magnitude)
        
        if gradient_std > 20 and pixel_std > 40:  # Thresholds may need tuning
            return 'handwriting'

        return 'standard'

    def _combine_text_results(self, pymupdf_text: str, plumber_text: str, ocr_text: str) -> str:
        """Intelligently combine text from different extraction methods."""
        texts = [pymupdf_text, plumber_text, ocr_text]
        texts = [t.strip() for t in texts if t and t.strip()]
        
        if not texts:
            return ""
            
        # If only one result, use it
        if len(texts) == 1:
            return texts[0]
            
        # Score each text
        scores = []
        for text in texts:
            score = 0
            # Length score
            score += len(text.split())
            # Readable characters ratio
            readable = sum(c.isalnum() or c.isspace() for c in text)
            score += readable / len(text) if text else 0
            # Word recognition score
            words = text.split()
            score += sum(len(w) > 1 for w in words)
            scores.append(score)
            
        # Use the text with highest score
        return texts[scores.index(max(scores))]

    def _create_processed_page(self, 
                             original_page: fitz.Page,
                             tables: List[List[List[str]]],
                             cleaned_text: str,
                             mask_mode: str) -> fitz.Document:
        """Create a processed page with cleaned content."""
        # Create new document for the page
        doc = fitz.open()
        page = doc.new_page(width=original_page.rect.width,
                           height=original_page.rect.height)
        
        # Copy original content
        page.show_pdf_page(page.rect, original_page.parent, original_page.number)
        
        if cleaned_text:
            # Add cleaned text overlay
            text_point = fitz.Point(50, 50)
            page.insert_text(text_point, cleaned_text,
                           fontname="helv",
                           fontsize=10,
                           color=(1, 0, 0))  # Red text
                           
        if tables:
            # Add processed tables
            y_position = 50
            for table in tables:
                x_position = 50
                for row in table:
                    for cell in row:
                        if cell:
                            cleaned_cell = self._clean_text(str(cell), mask_mode)
                            page.insert_text(
                                (x_position, y_position),
                                cleaned_cell,
                                fontname="helv",
                                fontsize=10,
                                color=(1, 0, 0) if cleaned_cell != cell else (0, 0, 0)
                            )
                        x_position += 100
                    y_position += 20
                y_position += 30
        
        return doc

    def _get_skew_angle(self, image: np.ndarray) -> float:
        """Get the skew angle of the image."""
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        return -angle

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate the image by given angle."""
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def _clean_text(self, text: str, mask_mode: str) -> str:
        """Clean PII from text."""
        if not text or not text.strip():
            return text
            
        try:
            if self.pii_cleaner is None:
                from .pii_cleaner import PIICleaner
                self.pii_cleaner = PIICleaner()
            
            # Normalize text
            text = self._normalize_text(text)
            
            # Clean PII
            result = self.pii_cleaner.remove_pii(text, mask_mode=mask_mode)
            cleaned_text = result['cleaned_text']
            
            # Post-process
            cleaned_text = self._post_process_text(cleaned_text)
            
            return cleaned_text
            
        except Exception as e:
            logger.warning(f"Error cleaning text: {str(e)}")
            return text

    def _normalize_text(self, text: str) -> str:
        """Normalize text for cleaning."""
        # Remove irregular whitespace
        text = re.sub(r'[\r\n\t\f\v]+', ' ', text)
        
        # Fix common OCR errors
        text = re.sub(r'(?<=\w)~(?=\w)', '', text)
        text = re.sub(r'(?<=\w)-(?=\w)', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Fix punctuation spacing
        text = re.sub(r'\s*([.,!?])\s*', r'\1 ', text)
        
        return text.strip()

    def _post_process_text(self, text: str) -> str:
        """Post-process cleaned text."""
        # Fix spacing around redacted content
        text = re.sub(r'\s*\[([^\]]+)\]\s*', r' [\1] ', text)
        
        # Fix multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix sentence spacing
        text = re.sub(r'\.(?=\w)', '. ', text)
        
        return text.strip()

    def generate_analysis_report(self, files: List[Union[str, Path]], output_file: Optional[Union[str, Path]] = None) -> str:
        """
        Generate a detailed analysis report for multiple files in a structured format.
        
        Args:
            files: List of file paths to analyze
            output_file: Optional path for output report file (supports .csv, .md, or .html)
            
        Returns:
            Path to the generated report file
        """
        # Initialize report data
        report_data = []
        
        for file_path in files:
            file_path = Path(file_path)
            
            try:
                # Basic file info
                file_info = {
                    'File Name': file_path.name,
                    'File Type': file_path.suffix.lower()[1:],  # Remove the dot
                    'File Description': '',
                    'Key Findings': []
                }
                
                # Process file based on type
                if file_info['File Type'] == 'pdf':
                    # Analyze PDF structure
                    analysis = self.analyze_structure(file_path)
                    
                    # Generate description
                    desc_parts = []
                    if analysis['pages_with_text'] > 0:
                        desc_parts.append(f"Text-based document with {analysis['total_pages']} pages")
                    if analysis['total_tables'] > 0:
                        desc_parts.append(f"Contains {analysis['total_tables']} tables")
                    if analysis['pages_with_images'] > 0:
                        desc_parts.append(f"Includes {analysis['total_images']} images")
                        
                    file_info['File Description'] = ". ".join(desc_parts)
                    
                    # Add key findings
                    findings = []
                    if analysis['pages_requiring_ocr'] > 0:
                        findings.append(f"Contains {analysis['pages_requiring_ocr']} scanned pages requiring OCR")
                    if analysis.get('metadata', {}).get('Title'):
                        findings.append(f"Document title: {analysis['metadata']['Title']}")
                    if analysis['total_tables'] > 0:
                        findings.append("Contains structured data in table format")
                    
                    file_info['Key Findings'] = findings
                
                elif file_info['File Type'] in ['png', 'jpg', 'jpeg']:
                    # Load image for analysis
                    img = Image.open(file_path)
                    
                    # Detect content type
                    content_type = self._detect_content_type(img)
                    
                    # Generate description based on image analysis
                    desc_parts = [f"{img.width}x{img.height} {content_type} image"]
                    if content_type == 'handwriting':
                        desc_parts.append("Contains handwritten content")
                    elif content_type == 'form':
                        desc_parts.append("Form or structured document")
                    elif content_type == 'table':
                        desc_parts.append("Contains tabular data")
                    
                    file_info['File Description'] = ". ".join(desc_parts)
                    
                    # Add key findings
                    findings = []
                    if content_type == 'handwriting':
                        findings.append("Manual entry system, dependent on handwriting")
                        findings.append("May require enhanced OCR processing")
                    elif content_type == 'form':
                        findings.append("Contains form fields and structured data")
                        findings.append("Suitable for automated data extraction")
                    elif content_type == 'table':
                        findings.append("Contains structured tabular data")
                        findings.append("Can be processed for data extraction")
                    
                    file_info['Key Findings'] = findings
                
                report_data.append(file_info)
                
            except Exception as e:
                logger.error(f"Error analyzing file {file_path}: {str(e)}")
                # Add error entry to report
                report_data.append({
                    'File Name': file_path.name,
                    'File Type': file_path.suffix.lower()[1:],
                    'File Description': 'Error during analysis',
                    'Key Findings': [f"Error: {str(e)}"]
                })
        
        # Generate report file
        if not output_file:
            # Get current timestamp for unique filename
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Create reports directory in data folder if it doesn't exist
            report_dir = Path("data/reports")
            report_dir.mkdir(parents=True, exist_ok=True)
            output_file = report_dir / f"security_analysis_report_{timestamp}.md"
        else:
            output_file = Path(output_file)
            # If output_file doesn't include directory, put it in data/reports
            if '/' not in str(output_file) and '\\' not in str(output_file):
                report_dir = Path("data/reports")
                report_dir.mkdir(parents=True, exist_ok=True)
                output_file = report_dir / output_file
            
        with open(output_file, 'w', encoding='utf-8') as f:
            if output_file.suffix == '.csv':
                # CSV format
                import csv
                writer = csv.DictWriter(f, fieldnames=['File Name', 'File Type', 'File Description', 'Key Findings'])
                writer.writeheader()
                for entry in report_data:
                    entry['Key Findings'] = '; '.join(entry['Key Findings'])
                    writer.writerow(entry)
                    
            elif output_file.suffix == '.html':
                # HTML format
                f.write('<!DOCTYPE html><html><head><style>')
                f.write('table {border-collapse: collapse; width: 100%;} ')
                f.write('th, td {border: 1px solid #ddd; padding: 8px; text-align: left;} ')
                f.write('th {background-color: #4CAF50; color: white;} ')
                f.write('tr:nth-child(even) {background-color: #f2f2f2;} ')
                f.write('</style></head><body>')
                f.write('<table>')
                f.write('<tr><th>File Name</th><th>File Type</th><th>File Description</th><th>Key Findings</th></tr>')
                for entry in report_data:
                    f.write('<tr>')
                    f.write(f'<td>{entry["File Name"]}</td>')
                    f.write(f'<td>{entry["File Type"]}</td>')
                    f.write(f'<td>{entry["File Description"]}</td>')
                    f.write(f'<td><ul>')
                    for finding in entry['Key Findings']:
                        f.write(f'<li>{finding}</li>')
                    f.write('</ul></td>')
                    f.write('</tr>')
                f.write('</table></body></html>')
                
            else:
                # Markdown format
                f.write('| File Name | File Type | File Description | Key Findings |\n')
                f.write('|-----------|-----------|------------------|---------------|\n')
                for entry in report_data:
                    findings = '<br>'.join(f'• {finding}' for finding in entry['Key Findings'])
                    f.write(f'| {entry["File Name"]} | {entry["File Type"]} | {entry["File Description"]} | {findings} |\n')
        
        return str(output_file)