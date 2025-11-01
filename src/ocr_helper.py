"""
OCR Helper Module

This module provides specialized OCR configurations and processing
for different types of content including:
- Handwritten text
- Forms and structured documents
- Code snippets and configurations
- Diagrams and technical drawings
- Surveillance footage
"""

import logging
from typing import Dict, List, Optional, Union, Any
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCRHelper:
    """Helper class for OCR operations with content-specific processing."""

    def __init__(self, tesseract_path: Optional[str] = None):
        """Initialize OCR helper with optional Tesseract path."""
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        # Configure OCR settings for different content types
        self.ocr_configs = {
            'standard': {
                'config': r'--oem 3 --psm 6',
                'preprocess': ['denoise', 'contrast', 'sharpen']
            },
            'handwriting': {
                'config': r'--oem 3 --psm 6 -c tessedit_char_blacklist=|',
                'preprocess': ['denoise', 'adaptive_threshold', 'deskew']
            },
            'form': {
                'config': r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.-_@/',
                'preprocess': ['denoise', 'threshold', 'deskew']
            },
            'code': {
                'config': r'--oem 3 --psm 6 -c preserve_interword_spaces=1 tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~',
                'preprocess': ['threshold']
            },
            'diagram': {
                'config': r'--oem 3 --psm 6 -c preserve_interword_spaces=1',
                'preprocess': ['denoise', 'threshold']
            },
            'surveillance': {
                'config': r'--oem 3 --psm 6 -c tessedit_char_blacklist=|~',
                'preprocess': ['denoise', 'enhance', 'stabilize']
            }
        }

    def process_image(self, image: Union[np.ndarray, Image.Image], content_type: str = 'standard') -> Dict[str, Any]:
        """
        Process image with appropriate OCR configuration.
        
        Args:
            image: Input image (CV2 or PIL)
            content_type: Type of content to process
            
        Returns:
            Dictionary containing OCR results and metadata
        """
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Get configuration for content type
        config = self.ocr_configs.get(content_type, self.ocr_configs['standard'])
        
        # Preprocess image
        processed = self._preprocess_image(pil_image, config['preprocess'])
        
        try:
            # Extract text with position and confidence
            data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT,
                                           config=config['config'])
            
            # Analyze results
            results = {
                'content_type': content_type,
                'text_blocks': [],
                'confidence': [],
                'word_count': 0,
                'line_count': 0,
                'average_confidence': 0
            }
            
            # Process text blocks
            current_block = ''
            current_conf = []
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:  # Valid recognition
                    if data['text'][i].strip():
                        results['word_count'] += 1
                        current_block += data['text'][i] + ' '
                        current_conf.append(float(data['conf'][i]))
                        
                if data['text'][i].strip() and data['block_num'][i] != data['block_num'][i-1]:
                    if current_block:
                        results['text_blocks'].append({
                            'text': current_block.strip(),
                            'confidence': sum(current_conf) / len(current_conf) if current_conf else 0,
                            'bbox': (data['left'][i], data['top'][i],
                                   data['left'][i] + data['width'][i],
                                   data['top'][i] + data['height'][i])
                        })
                        current_block = ''
                        current_conf = []
                        results['line_count'] += 1
            
            # Add final block if any
            if current_block:
                results['text_blocks'].append({
                    'text': current_block.strip(),
                    'confidence': sum(current_conf) / len(current_conf) if current_conf else 0
                })
                results['line_count'] += 1
            
            # Calculate average confidence
            if results['text_blocks']:
                results['average_confidence'] = sum(block['confidence'] 
                                                  for block in results['text_blocks']) / len(results['text_blocks'])
            
            return results
            
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            return {
                'content_type': content_type,
                'error': str(e),
                'text_blocks': [],
                'word_count': 0,
                'line_count': 0,
                'average_confidence': 0
            }

    def _preprocess_image(self, image: Image.Image, steps: List[str]) -> Image.Image:
        """
        Preprocess image for better OCR results.
        
        Args:
            image: PIL Image to process
            steps: List of preprocessing steps to apply
            
        Returns:
            Processed PIL Image
        """
        # Convert to CV2 format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        processed = cv_image.copy()
        
        # Convert to grayscale
        if len(processed.shape) == 3:
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed

        for step in steps:
            if step == 'denoise':
                gray = cv2.fastNlMeansDenoising(gray)
            elif step == 'contrast':
                gray = cv2.equalizeHist(gray)
            elif step == 'sharpen':
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                gray = cv2.filter2D(gray, -1, kernel)
            elif step == 'adaptive_threshold':
                gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
            elif step == 'threshold':
                gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            elif step == 'deskew':
                angle = self._get_skew_angle(gray)
                if abs(angle) > 0.5:
                    gray = self._rotate_image(gray, angle)
            elif step == 'enhance':
                gray = cv2.detailEnhance(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            elif step == 'stabilize':
                gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Convert back to PIL
        return Image.fromarray(gray)

    def _get_skew_angle(self, image: np.ndarray) -> float:
        """Calculate skew angle of text in image."""
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta) - 90
                if abs(angle) < 45:  # Filter out vertical lines
                    angles.append(angle)
            
            if angles:
                return np.median(angles)
        
        return 0.0

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle."""
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, 
                            borderMode=cv2.BORDER_REPLICATE)

    def detect_content_type(self, image: Union[np.ndarray, Image.Image]) -> str:
        """
        Automatically detect the type of content in the image.
        
        Args:
            image: Input image (CV2 or PIL)
            
        Returns:
            Detected content type
        """
        # Convert to CV2 if needed
        if isinstance(image, Image.Image):
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            cv_image = image

        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY) if len(cv_image.shape) == 3 else cv_image
        
        # Check for surveillance footage characteristics
        text = pytesseract.image_to_string(gray)
        if any(re.search(pattern, text) for pattern in [r'\d{2}:\d{2}:\d{2}', r'\d{2}/\d{2}/\d{4}']):
            return 'surveillance'
        
        # Check for code snippets
        if any(keyword in text for keyword in ['function', 'class', 'def', 'return', 'import']):
            return 'code'
        
        # Check for forms
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        if lines is not None:
            h_lines = sum(1 for rho, theta in lines[:, 0] if abs(np.degrees(theta) - 0) < 10)
            v_lines = sum(1 for rho, theta in lines[:, 0] if abs(np.degrees(theta) - 90) < 10)
            if h_lines > 3 and v_lines > 3:
                return 'form'
        
        # Check for diagrams
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        geometric_shapes = sum(1 for c in contours 
                             if len(cv2.approxPolyDP(c, 0.01*cv2.arcLength(c, True), True)) in [3,4,5,6,8])
        if geometric_shapes > 3:
            return 'diagram'
        
        # Check for handwriting
        if cv2.Laplacian(gray, cv2.CV_64F).var() > 500:
            return 'handwriting'
        
        return 'standard'