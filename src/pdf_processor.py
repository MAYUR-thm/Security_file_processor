"""
PDF Processor Module

This module handles processing of PDF files including:
- Text extraction from text-based PDFs
- OCR for scanned PDFs
- Table extraction and processing
- Structure preservation
- PII detection and cleaning
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import io
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import numpy as np
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Main class for processing PDF files."""

    def __init__(self, pii_cleaner: Optional[Any] = None, tesseract_path: Optional[str] = None):
        """
        Initialize the PDF processor.
        
        Args:
            pii_cleaner: Optional PIICleaner instance to use for PII detection/cleaning
            tesseract_path: Optional path to tesseract executable
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        # Initialize or use provided PII cleaner
        self.pii_cleaner = pii_cleaner
        
        # Configure OCR settings for different content types
        self.ocr_configs = {
            'standard': r'--oem 3 --psm 6',
            'handwriting': r'--oem 3 --psm 6 -c tessedit_char_blacklist=|',
            'form': r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.-_@/',
            'table': r'--oem 3 --psm 6 -c preserve_interword_spaces=1',
            'code': r'--oem 3 --psm 6 -c preserve_interword_spaces=1 tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        }
        
        # Set default OCR config
        self.ocr_config = self.ocr_configs['standard']

    def process_file(self, 
                    input_file: Union[str, Path], 
                    output_file: Optional[Union[str, Path]] = None,
                    mask_mode: str = "replace",
                    create_text_summary: bool = True) -> Dict[str, Any]:
        """
        Process a PDF file, detecting and cleaning PII while preserving structure.
        Creates both a cleaned PDF file and optionally a text summary.
        
        Args:
            input_file: Path to input PDF file
            output_file: Path for output PDF file
            mask_mode: How to handle PII ('replace', 'mask', or 'remove')
            create_text_summary: Whether to create additional text summary file
            
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
            # Process with PyMuPDF first
            doc = fitz.open(input_file)
            
            stats = {
                'total_pages': len(doc),
                'pages_with_text': 0,
                'pages_requiring_ocr': 0,
                'tables_extracted': 0,
                'pii_instances': 0,
                'success': False
            }

            # Create output PDF
            out_doc = fitz.open()
            text_content = []
            
            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                # Create a new page in output document with same dimensions
                new_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
                
                if text.strip():
                    stats['pages_with_text'] += 1
                    # Process text-based content
                    text_content.append(f"\n=== Page {page_num + 1} ===\n{text}")
                    
                    # Copy original page content first
                    new_page.show_pdf_page(new_page.rect, page.parent, page.number)
                    
                    # Get text blocks with all layout information
                    blocks = page.get_text("dict", flags=11)["blocks"]  # Include more layout info
                    
                    for block in blocks:
                        if "lines" in block:
                            for line in block["lines"]:
                                # Collect all spans in the line first
                                line_spans = []
                                line_text = ""
                                for span in line["spans"]:
                                    line_spans.append(span)
                                    line_text += span["text"] + " "
                                
                                # Clean the entire line as one unit
                                line_text = line_text.strip()
                                if line_text:
                                    cleaned_text = self._clean_text(line_text, mask_mode)
                                    if cleaned_text != line_text:
                                        # Calculate line rectangle covering all spans
                                        first_span = line_spans[0]
                                        last_span = line_spans[-1]
                                        rect = fitz.Rect(
                                            first_span["bbox"][0],  # x0
                                            min(s["bbox"][1] for s in line_spans),  # y0
                                            last_span["bbox"][2],   # x1
                                            max(s["bbox"][3] for s in line_spans)   # y1
                                        )
                                        
                                        # Clear original line
                                        new_page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))
                                        
                                        # Add cleaned text in red, preserving original positioning
                                        # Create text insertion point
                                        point = fitz.Point(first_span["origin"][0], first_span["origin"][1])
                                        
                                        # Insert text with default font (we'll use fontname instead of font)
                                        new_page.insert_text(
                                            point,
                                            cleaned_text,
                                            fontsize=first_span["size"],
                                            fontname="helv",  # Use fontname instead of font
                                            color=(1, 0, 0)  # Red
                                        )
                else:
                    stats['pages_requiring_ocr'] += 1
                    # Process scanned page with OCR
                    # Copy original page content first
                    new_page.show_pdf_page(new_page.rect, page.parent, page.number)
                    
                    # Get OCR text
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Detect content type and set appropriate OCR config
                    content_type = self._detect_content_type(img)
                    self.ocr_config = self.ocr_configs[content_type]
                    
                    # Preprocess image and perform OCR
                    processed_img = self._preprocess_image_for_ocr(img, content_type)
                    ocr_text = pytesseract.image_to_string(processed_img, config=self.ocr_config)
                    cleaned_text = self._clean_text(ocr_text, mask_mode)
                    
                    text_content.append(f"\n=== Page {page_num + 1} (OCR) ===\n{ocr_text}")
                    
                    if cleaned_text != ocr_text:
                        # Add cleaned text as overlay
                        # Create text insertion points
                        header_point = fitz.Point(50, 50)
                        text_point = fitz.Point(50, 70)
                        
                        # Insert header text
                        new_page.insert_text(
                            header_point,
                            "REDACTED CONTENT BELOW",
                            fontname="helv",
                            fontsize=12,
                            color=(1, 0, 0)  # Red
                        )
                        
                        # Insert cleaned text
                        new_page.insert_text(
                            text_point,
                            cleaned_text,
                            fontname="helv",
                            fontsize=10,
                            color=(1, 0, 0)  # Red
                        )

            # Optional text summary output
            if create_text_summary:
                txt_output = Path(str(output_file)).with_suffix('.txt')
                with open(txt_output, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(text_content))

            # Process tables with pdfplumber
            with pdfplumber.open(input_file) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    if tables:
                        stats['tables_extracted'] += len(tables)
                        # Get the page from our output document
                        out_page = out_doc[page_num]  # Page already exists from previous processing
                        
                        # Process and add tables to the page
                        y_position = 50
                        for table in tables:
                            for row in table:
                                x_position = 50
                                row_text = []
                                for cell in row:
                                    if cell:
                                        cleaned_cell = self._clean_text(str(cell), mask_mode)
                                        if cleaned_cell != cell:
                                            color = (1, 0, 0)  # Red for redacted
                                        else:
                                            color = (0, 0, 0)  # Black
                                        
                                        out_page.insert_text(
                                            (x_position, y_position),
                                            cleaned_cell,
                                            color=color,
                                            fontsize=10
                                        )
                                        row_text.append(cleaned_cell)
                                    x_position += 100
                                y_position += 20
                            y_position += 30  # Space between tables

            # Ensure output directory exists
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save processed PDF file
            logger.info(f"Saving cleaned PDF to: {output_file}")
            out_doc.save(str(output_file))  # Ensure path is string
            
            # Clean up
            out_doc.close()
            doc.close()
            
            # Update statistics
            stats['success'] = True
            stats['output_file'] = str(output_file)
            if create_text_summary:
                stats['text_summary'] = str(txt_output)
            
            logger.info(f"Successfully processed PDF file: {input_file}")
            return stats

        except Exception as e:
            error_msg = f"Error processing PDF file {input_file}: {str(e)}"
            logger.error(error_msg)
            
            # Clean up on error
            try:
                out_doc.close()
                doc.close()
            except:
                pass
                
            # Reraise with more context
            raise RuntimeError(f"PDF processing failed: {error_msg}")

    def _process_text_page(self, page: fitz.Page, mask_mode: str) -> fitz.Document:
        """Process a text-based PDF page."""
        # Create temporary document for the page
        temp_doc = fitz.open()
        new_page = temp_doc.new_page(width=page.rect.width, height=page.rect.height)
        
        try:
            # Copy original page content first to preserve layout, images, etc.
            new_page.show_pdf_page(new_page.rect, page.parent, page.number)
            
            # Get text blocks with all layout information
            blocks = page.get_text("dict", flags=11)["blocks"]  # Include more layout info
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        # Collect all spans in the line first
                        line_spans = []
                        line_text = ""
                        for span in line["spans"]:
                            line_spans.append(span)
                            line_text += span["text"] + " "
                        
                        # Clean the entire line as one unit
                        line_text = line_text.strip()
                        if line_text:
                            cleaned_text = self._clean_text(line_text, mask_mode)
                            if cleaned_text != line_text:
                                # Calculate line rectangle covering all spans
                                first_span = line_spans[0]
                                last_span = line_spans[-1]
                                rect = fitz.Rect(
                                    first_span["bbox"][0],  # x0
                                    min(s["bbox"][1] for s in line_spans),  # y0
                                    last_span["bbox"][2],   # x1
                                    max(s["bbox"][3] for s in line_spans)   # y1
                                )
                                
                                # Clear original line
                                new_page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))
                                
                                # Add cleaned text in red, preserving original positioning
                                new_page.insert_text(
                                    (first_span["origin"][0], first_span["origin"][1]),
                                    cleaned_text,
                                    fontsize=first_span["size"],
                                    font=first_span.get("font", "helv"),
                                    color=(1, 0, 0)  # Red
                                )
        except Exception as e:
            logger.warning(f"Error processing text page: {str(e)}")
            # On error, copy original page content
            new_page.show_pdf_page(new_page.rect, page.parent, page.number)
        
        return temp_doc

    def _process_scanned_page(self, page: fitz.Page, mask_mode: str) -> fitz.Document:
        """Process a scanned PDF page using OCR."""
        # Convert page to image
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Perform OCR
        text = pytesseract.image_to_string(img, config=self.ocr_config)
        
        # Clean detected text
        cleaned_text = self._clean_text(text, mask_mode)
        
        # Create new page with cleaned text
        temp_doc = fitz.open()
        new_page = temp_doc.new_page(width=page.rect.width, height=page.rect.height)
        
        # Add background image
        new_page.insert_image(new_page.rect, pixmap=pix)
        
        # Add cleaned text as overlay if different from original
        if cleaned_text != text:
            new_page.insert_text((50, 50), "REDACTED CONTENT BELOW", 
                               color=(1, 0, 0),
                               fontsize=12)
            new_page.insert_text((50, 70), cleaned_text,
                               color=(1, 0, 0))
        
        return temp_doc

    def _process_tables(self, tables: List[List[List[str]]], mask_mode: str) -> Optional[fitz.Document]:
        """Process tables extracted from PDF."""
        if not tables:
            return None

        # Create document for tables
        temp_doc = fitz.open()
        new_page = temp_doc.new_page()
        
        y_position = 50
        for table in tables:
            for row in table:
                x_position = 50
                for cell in row:
                    if cell:
                        cleaned_cell = self._clean_text(str(cell), mask_mode)
                        if cleaned_cell != cell:
                            color = (1, 0, 0)  # Red for redacted
                        else:
                            color = (0, 0, 0)  # Black
                        
                        new_page.insert_text((x_position, y_position),
                                          cleaned_cell,
                                          color=color)
                    x_position += 100
                y_position += 20
            y_position += 30  # Space between tables
        
        return temp_doc

    def _preprocess_image_for_ocr(self, image: Image.Image, content_type: str = 'standard') -> Image.Image:
        """
        Preprocess image for better OCR results based on content type.
        
        Args:
            image: PIL Image to process
            content_type: Type of content ('standard', 'handwriting', 'form', etc.)
            
        Returns:
            Processed PIL Image
        """
        # Convert PIL to CV2
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        if content_type == 'handwriting':
            # Enhanced processing for handwritten text
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        elif content_type in ['form', 'table']:
            # Enhanced processing for structured content
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            # Deskew if needed
            angle = self._get_skew_angle(gray)
            if abs(angle) > 0.5:
                gray = self._rotate_image(gray, angle)
        
        elif content_type == 'code':
            # Enhanced processing for code snippets
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        else:  # standard
            # Standard image preprocessing
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Convert back to PIL
        return Image.fromarray(gray)

    def _get_skew_angle(self, image: np.ndarray) -> float:
        """
        Detect the skew angle of the image.
        
        Args:
            image: CV2 image in grayscale
            
        Returns:
            Skew angle in degrees
        """
        # Detect edges
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta) - 90
                if abs(angle) < 45:  # Filter out vertical lines
                    angles.append(angle)
            
            if angles:
                # Return the most common angle
                return np.median(angles)
        
        return 0.0

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by given angle.
        
        Args:
            image: CV2 image
            angle: Angle in degrees
            
        Returns:
            Rotated image
        """
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Perform rotation
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated

    def _detect_content_type(self, image: Image.Image, text: str = '') -> str:
        """
        Detect the type of content in the image/text.
        
        Args:
            image: PIL Image to analyze
            text: Optional text content
            
        Returns:
            Content type string
        """
        # Convert PIL to CV2
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Check for table-like structures
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        if lines is not None and len(lines) > 10:
            horizontal_lines = sum(1 for rho, theta in lines[:, 0] if abs(np.degrees(theta) - 0) < 10)
            vertical_lines = sum(1 for rho, theta in lines[:, 0] if abs(np.degrees(theta) - 90) < 10)
            if horizontal_lines > 3 and vertical_lines > 3:
                return 'table'
        
        # Check for form-like content
        if text:
            form_indicators = ['name:', 'date:', 'signature:', 'address:', 'phone:', 'email:']
            if any(indicator in text.lower() for indicator in form_indicators):
                return 'form'
        
        # Check for code snippets
        code_indicators = ['{', '}', 'function', 'class', 'def', 'return', 'import', 'from']
        if text and any(indicator in text for indicator in code_indicators):
            return 'code'
        
        # Check for handwriting
        # Handwriting typically has more variance in pixel intensity
        if cv2.Laplacian(gray, cv2.CV_64F).var() > 500:
            return 'handwriting'
        
        return 'standard'

    def _clean_text(self, text: str, mask_mode: str) -> str:
        """Clean PII from text using the PII cleaner."""
        if not text or not text.strip():
            return text
            
        try:
            # Use instance PII cleaner if available, otherwise create new one
            if self.pii_cleaner is None:
                from .pii_cleaner import PIICleaner
                self.pii_cleaner = PIICleaner()
            
            # Pre-process text to fix common OCR/formatting issues
            text = self._normalize_text(text)
            
            # Clean PII
            result = self.pii_cleaner.remove_pii(text, mask_mode=mask_mode)
            cleaned_text = result['cleaned_text']
            
            # Post-process to ensure proper formatting
            cleaned_text = self._post_process_text(cleaned_text)
            
            return cleaned_text
        except Exception as e:
            logger.warning(f"Error cleaning text with PIICleaner: {str(e)}")
            return text  # Return original text if cleaning fails
            
    def _normalize_text(self, text: str) -> str:
        """Pre-process text to fix common OCR and formatting issues."""
        # Remove any non-standard whitespace characters
        text = re.sub(r'[\r\n\t\f\v]+', ' ', text)
        
        # Fix common OCR errors
        text = re.sub(r'(?<=\w)~(?=\w)', '', text)  # Remove ~ between words
        text = re.sub(r'(?<=\w)-(?=\w)', ' ', text)  # Convert incorrect hyphens to spaces
        text = re.sub(r'\s+', ' ', text)  # Normalize multiple spaces
        
        # Fix spacing around punctuation
        text = re.sub(r'\s*([.,!?])\s*', r'\1 ', text)
        
        return text.strip()
        
    def _post_process_text(self, text: str) -> str:
        """Post-process cleaned text to ensure proper formatting."""
        # Fix spacing around redacted content
        text = re.sub(r'\s*\[([^\]]+)\]\s*', r' [\1] ', text)
        
        # Fix multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix spacing after periods
        text = re.sub(r'\.(?=\w)', '. ', text)
        
        # Clean up any remaining formatting issues
        text = text.strip()
        
        return text

    def extract_text(self, file_path: Union[str, Path], use_ocr: bool = True) -> str:
        """
        Extract text content from PDF file.
        
        Args:
            file_path: Path to PDF file
            use_ocr: Whether to use OCR for scanned pages
            
        Returns:
            Extracted text content
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            text_content = []
            
            # Try text extraction first
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                if text.strip():
                    text_content.append(text)
                elif use_ocr:
                    # Page appears to be scanned, use OCR
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = pytesseract.image_to_string(img, config=self.ocr_config)
                    text_content.append(ocr_text)
            
            doc.close()
            
            # Extract tables
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables()
                    if tables:
                        text_content.append("\n=== Tables ===\n")
                        for table in tables:
                            for row in table:
                                text_content.append(" | ".join(str(cell) for cell in row if cell))
            
            return "\n".join(text_content)

        except Exception as e:
            logger.error(f"Error extracting text from PDF file {file_path}: {str(e)}")
            raise

    def analyze_structure(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Analyze the structure of a PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary containing structural analysis
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            doc = fitz.open(file_path)
            
            analysis = {
                'total_pages': len(doc),
                'pages_with_text': 0,
                'pages_with_images': 0,
                'total_images': 0,
                'total_tables': 0,
                'metadata': doc.metadata,
                'page_sizes': []
            }

            # Analyze each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Check for text
                if page.get_text().strip():
                    analysis['pages_with_text'] += 1
                
                # Check for images
                image_list = page.get_images()
                if image_list:
                    analysis['pages_with_images'] += 1
                    analysis['total_images'] += len(image_list)
                
                # Record page size
                analysis['page_sizes'].append({
                    'width': page.rect.width,
                    'height': page.rect.height
                })

            doc.close()
            
            # Count tables using pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables()
                    analysis['total_tables'] += len(tables)

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing PDF file {file_path}: {str(e)}")
            raise


def process_pdf_file(input_file: Union[str, Path], 
                    output_file: Optional[Union[str, Path]] = None,
                    mask_mode: str = "replace") -> Dict[str, Any]:
    """
    Convenience function to process a single PDF file.
    
    Args:
        input_file: Path to input PDF file
        output_file: Path for output PDF file
        mask_mode: How to handle PII ('replace', 'mask', or 'remove')
        
    Returns:
        Processing results dictionary
    """
    processor = PDFProcessor()
    return processor.process_file(input_file, output_file, mask_mode)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        try:
            processor = PDFProcessor()
            
            # First analyze the structure
            print("\nAnalyzing PDF structure...")
            analysis = processor.analyze_structure(input_file)
            print("\nFile Structure:")
            print(f"Total pages: {analysis['total_pages']}")
            print(f"Pages with text: {analysis['pages_with_text']}")
            print(f"Pages with images: {analysis['pages_with_images']}")
            print(f"Total images: {analysis['total_images']}")
            print(f"Total tables: {analysis['total_tables']}")
            
            # Process the file
            print("\nProcessing file...")
            results = processor.process_file(input_file, output_file)
            
            print("\nProcessing Results:")
            print(f"Total pages: {results['total_pages']}")
            print(f"Pages with text: {results['pages_with_text']}")
            print(f"Pages requiring OCR: {results['pages_requiring_ocr']}")
            print(f"Tables extracted: {results['tables_extracted']}")
            print(f"\nOutput file: {results['output_file']}")
            
        except Exception as e:
            print(f"Error: {str(e)}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Usage: python pdf_processor.py input.pdf [output.pdf]")
        sys.exit(1)