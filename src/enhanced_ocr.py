"""
Enhanced OCR Processor with improved text recognition and PII masking
"""

import pytesseract
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class EnhancedOCR:
    """Enhanced OCR processor with specialized preprocessing for different document types."""
    
    def __init__(self, tesseract_path: Optional[str] = None):
        """Initialize the OCR processor."""
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
        # Configuration for different content types
        self.ocr_configs = {
            'default': r'--oem 3 --psm 6',
            'single_column': r'--oem 3 --psm 6',
            'multi_column': r'--oem 3 --psm 1',
            'table': r'--oem 3 --psm 6 -c preserve_interword_spaces=1',
            'certificate': r'--oem 3 --psm 1 -c preserve_interword_spaces=1',
            'code': r'--oem 3 --psm 6 -c preserve_interword_spaces=1',
            'handwriting': r'--oem 3 --psm 13'
        }

    def preprocess_image(self, image: Image.Image, doc_type: str = 'default') -> Image.Image:
        """
        Apply specialized preprocessing based on document type.
        
        Args:
            image: PIL Image to process
            doc_type: Type of document ('default', 'certificate', 'table', etc.)
            
        Returns:
            Preprocessed PIL Image
        """
        # Convert PIL to CV2
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        if doc_type == 'certificate':
            # Enhanced processing for certificates
            # Increase contrast and sharpness
            gray = cv2.convertScaleAbs(gray, alpha=1.3, beta=0)
            gray = cv2.GaussianBlur(gray, (3,3), 0)
            gray = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )
            
        elif doc_type == 'table':
            # Enhanced processing for tables
            # Use Otsu's thresholding after Gaussian filtering
            blur = cv2.GaussianBlur(gray, (3,3), 0)
            gray = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Detect and straighten lines
            kernel = np.ones((1,5), np.uint8)
            gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
        elif doc_type == 'code':
            # Enhanced processing for code snippets
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
        else:  # default
            # Apply adaptive thresholding
            gray = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )
        
        # Denoise
        gray = cv2.fastNlMeansDenoising(gray)
        
        # Check and correct skew
        angle = self._get_skew_angle(gray)
        if abs(angle) > 0.5:
            gray = self._rotate_image(gray, angle)
        
        # Convert back to PIL
        return Image.fromarray(gray)

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
        return cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

    def detect_document_type(self, image: Image.Image) -> str:
        """
        Detect the type of document from the image.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Document type string
        """
        # Convert to CV2
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Detect lines for table detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is not None and len(lines) > 10:
            # Count horizontal and vertical lines
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
        
        # Check for certificate characteristics
        text = pytesseract.image_to_string(image)
        if any(word in text.upper() for word in ['CERTIFICATE', 'CERTIFICATION', 'DIPLOMA']):
            return 'certificate'
        
        # Check for code characteristics
        if any(word in text for word in ['{', '}', 'function', 'class', 'def']):
            return 'code'
        
        return 'default'

    def process_image(self, image: Image.Image, enhance_text: bool = True) -> Dict[str, Any]:
        """
        Process image and extract text with optimal settings.
        
        Args:
            image: PIL Image to process
            enhance_text: Whether to enhance text recognition
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        # Detect document type
        doc_type = self.detect_document_type(image)
        
        # Preprocess image
        processed_image = self.preprocess_image(image, doc_type)
        
        # Get OCR config
        config = self.ocr_configs.get(doc_type, self.ocr_configs['default'])
        
        # Extract text
        text = pytesseract.image_to_string(processed_image, config=config)
        
        # Get detailed OCR data
        ocr_data = pytesseract.image_to_data(
            processed_image,
            config=config,
            output_type=pytesseract.Output.DICT
        )
        
        # Extract high-confidence text blocks
        text_blocks = []
        for i in range(len(ocr_data['text'])):
            if int(ocr_data['conf'][i]) > 60:  # Filter by confidence
                text_blocks.append({
                    'text': ocr_data['text'][i],
                    'conf': ocr_data['conf'][i],
                    'bbox': (
                        ocr_data['left'][i],
                        ocr_data['top'][i],
                        ocr_data['width'][i],
                        ocr_data['height'][i]
                    )
                })
        
        return {
            'text': text,
            'doc_type': doc_type,
            'text_blocks': text_blocks,
            'confidence': np.mean([block['conf'] for block in text_blocks]) if text_blocks else 0
        }

    def extract_structured_text(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extract text with structure preservation.
        
        Args:
            image: PIL Image to process
            
        Returns:
            Dictionary containing structured text data
        """
        # Process image
        result = self.process_image(image)
        
        # Get detailed OCR data
        ocr_data = pytesseract.image_to_data(
            image,
            config=self.ocr_configs[result['doc_type']],
            output_type=pytesseract.Output.DICT
        )
        
        # Organize text by layout
        structured_data = {
            'title': [],
            'headers': [],
            'body': [],
            'tables': []
        }
        
        current_block = []
        last_top = None
        
        for i in range(len(ocr_data['text'])):
            if not ocr_data['text'][i].strip():
                continue
                
            # Check if new line
            if last_top is not None and abs(ocr_data['top'][i] - last_top) > ocr_data['height'][i]:
                if current_block:
                    text = ' '.join(current_block)
                    
                    # Classify text block
                    if len(current_block) == 1 and ocr_data['height'][i] > 30:
                        structured_data['title'].append(text)
                    elif ocr_data['height'][i] > 20:
                        structured_data['headers'].append(text)
                    else:
                        structured_data['body'].append(text)
                        
                    current_block = []
            
            current_block.append(ocr_data['text'][i])
            last_top = ocr_data['top'][i]
        
        # Add last block
        if current_block:
            structured_data['body'].append(' '.join(current_block))
        
        return structured_data