"""
Image Processor Module

This module handles processing of image files (PNG, JPEG) including:
- OCR text extraction
- Logo detection and masking
- Image preprocessing for better OCR
- PII detection in extracted text
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import cv2
import numpy as np
from PIL import Image
import pytesseract
import io
from .ocr_helper import OCRHelper
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    """Main class for processing image files."""

    def __init__(self, tesseract_path: Optional[str] = None):
        """
        Initialize the Image processor.
        
        Args:
            tesseract_path: Optional path to tesseract executable
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        # Configure OCR settings for different content types
        self.ocr_configs = {
            'standard': r'--oem 3 --psm 6',
            'handwriting': r'--oem 3 --psm 6 -c tessedit_char_blacklist=|',
            'form': r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.-_@/',
            'diagram': r'--oem 3 --psm 6 -c preserve_interword_spaces=1',
            'surveillance': r'--oem 3 --psm 6 -c tessedit_char_blacklist=|~'
        }
        
        # Image preprocessing parameters
        self.target_dpi = 300
        self.preprocessing_steps = {
            'standard': ['denoise', 'contrast', 'sharpen'],
            'handwriting': ['denoise', 'adaptive_threshold', 'deskew'],
            'form': ['denoise', 'threshold', 'deskew'],
            'diagram': ['denoise', 'threshold'],
            'surveillance': ['denoise', 'enhance', 'stabilize']
        }

    def process_file(self, 
                    input_file: Union[str, Path], 
                    output_file: Optional[Union[str, Path]] = None,
                    mask_mode: str = "replace") -> Dict[str, Any]:
        """
        Process an image file, detecting and cleaning PII while preserving quality.
        
        Args:
            input_file: Path to input image file
            output_file: Path for output image file
            mask_mode: How to handle PII ('replace', 'mask', or 'remove')
            
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
            # Read image
            image = cv2.imread(str(input_file))
            if image is None:
                raise ValueError(f"Could not read image: {input_file}")
            
            stats = {
                'original_size': input_file.stat().st_size,
                'dimensions': image.shape,
                'text_regions': 0,
                'logo_regions': 0,
                'pii_instances': 0,
                'success': False
            }

            # Detect content type
            content_type = self.ocr_helper.detect_content_type(image)
            stats['content_type'] = content_type
            
            # Process with OCR helper
            ocr_results = self.ocr_helper.process_image(image, content_type)
            
            # Update statistics
            stats['text_regions'] = len(ocr_results['text_blocks'])
            stats['average_confidence'] = ocr_results['average_confidence']
            stats['word_count'] = ocr_results['word_count']
            stats['line_count'] = ocr_results['line_count']
            
            # Detect and mask logos
            logo_regions = self._detect_logos(image)
            stats['logo_regions'] = len(logo_regions)
            
            # Create mask for sensitive regions
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cleaned_image = image.copy()

            # Process text blocks
            for block in ocr_results['text_blocks']:
                x, y, w, h = block['bbox']
                mask[y:y+h, x:x+w] = 255
                
                # Process text for PII
                if block['text'].strip():
                    if mask_mode == "mask":
                        cleaned_image[y:y+h, x:x+w] = 255  # White mask
                    elif mask_mode == "remove":
                        # Try to use inpainting to remove the text naturally
                        cleaned_image[y:y+h, x:x+w] = cv2.inpaint(
                            cleaned_image[y:y+h, x:x+w],
                            mask[y:y+h, x:x+w],
                            3,
                            cv2.INPAINT_TELEA
                        )

            # Mask logo regions
            for region in logo_regions:
                x, y, w, h = region
                mask[y:y+h, x:x+w] = 255
                if mask_mode in ["mask", "remove"]:
                    # Use inpainting to remove logos naturally
                    cleaned_image[y:y+h, x:x+w] = cv2.inpaint(
                        cleaned_image[y:y+h, x:x+w],
                        mask[y:y+h, x:x+w],
                        3,
                        cv2.INPAINT_TELEA
                    )

            # Save processed image
            cv2.imwrite(str(output_file), cleaned_image)
            
            stats['success'] = True
            stats['output_file'] = str(output_file)
            stats['output_size'] = output_file.stat().st_size
            
            logger.info(f"Successfully processed image file: {input_file}")
            
            return stats

        except Exception as e:
            logger.error(f"Error processing image file {input_file}: {str(e)}")
            raise

    def _detect_content_type(self, image: np.ndarray) -> str:
        """Detect the type of content in the image."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Check for surveillance footage characteristics
        if self._is_surveillance_footage(gray):
            return 'surveillance'
        
        # Check for diagrams
        if self._is_diagram(gray):
            return 'diagram'
        
        # Check for forms
        if self._is_form(gray):
            return 'form'
        
        # Check for handwritten content
        if self._has_handwriting(gray):
            return 'handwriting'
        
        return 'standard'

    def _is_surveillance_footage(self, gray: np.ndarray) -> bool:
        """Detect if image is from surveillance footage."""
        # Check for timestamp overlay
        has_timestamp = self._detect_timestamp_overlay(gray)
        
        # Check for typical surveillance camera distortion
        distortion = self._measure_lens_distortion(gray)
        
        # Check for low light characteristics
        is_low_light = np.mean(gray) < 100 and np.std(gray) < 40
        
        return has_timestamp or distortion > 0.2 or is_low_light

    def _is_diagram(self, gray: np.ndarray) -> bool:
        """Detect if image is a diagram or technical drawing."""
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Line detection
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        # Check for geometric shapes
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        geometric_shapes = 0
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
            if len(approx) in [3, 4, 5, 6, 8]:  # Common geometric shapes
                geometric_shapes += 1
        
        # Criteria for diagram
        has_many_lines = lines is not None and len(lines) > 10
        has_geometric_shapes = geometric_shapes > 3
        
        return has_many_lines and has_geometric_shapes

    def _is_form(self, gray: np.ndarray) -> bool:
        """Detect if image is a form."""
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Line detection focused on form-like structures
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        if lines is not None:
            horizontal_lines = sum(1 for rho, theta in lines[:, 0] if abs(np.degrees(theta) - 0) < 10)
            vertical_lines = sum(1 for rho, theta in lines[:, 0] if abs(np.degrees(theta) - 90) < 10)
            
            # Forms typically have aligned horizontal and vertical lines
            return horizontal_lines > 3 and vertical_lines > 3
        
        return False

    def _has_handwriting(self, gray: np.ndarray) -> bool:
        """Detect if image contains handwritten content."""
        # Calculate image statistics
        blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Handwriting typically has high variance and irregular patterns
        return blur_var > 500

    def _detect_timestamp_overlay(self, image: np.ndarray) -> bool:
        """Detect timestamp overlay in surveillance footage."""
        # Use OCR to detect timestamp patterns
        text = pytesseract.image_to_string(image, config='--psm 6')
        timestamp_patterns = [
            r'\d{2}:\d{2}:\d{2}',
            r'\d{2}/\d{2}/\d{4}',
            r'\d{4}-\d{2}-\d{2}'
        ]
        return any(re.search(pattern, text) for pattern in timestamp_patterns)

    def _measure_lens_distortion(self, image: np.ndarray) -> float:
        """Measure lens distortion typical in surveillance cameras."""
        # Simplified method - measure deviation from straight lines
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        if lines is not None:
            deviations = []
            for rho, theta in lines[:, 0]:
                # Measure deviation from perfect horizontal/vertical
                angle = np.degrees(theta) % 90
                deviation = min(angle, 90 - angle) / 90
                deviations.append(deviation)
            return np.mean(deviations) if deviations else 0.0
        return 0.0

    def _preprocess_image(self, image: np.ndarray, content_type: str = 'standard') -> np.ndarray:
        """Preprocess image based on content type."""
        processed = image.copy()
        
        # Convert to grayscale if needed
        if len(processed.shape) == 3:
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed

        if 'denoise' in self.preprocessing_steps:
            # Apply denoising
            gray = cv2.fastNlMeansDenoising(gray)

        if 'contrast' in self.preprocessing_steps:
            # Enhance contrast
            gray = cv2.equalizeHist(gray)

        if 'sharpen' in self.preprocessing_steps:
            # Sharpen image
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            gray = cv2.filter2D(gray, -1, kernel)

        return gray

    def _detect_text_regions(self, image: np.ndarray) -> List[tuple]:
        """Detect regions containing text in the image."""
        # Use pytesseract to get bounding boxes for text
        boxes = pytesseract.image_to_boxes(image)
        regions = []
        
        height = image.shape[0]
        
        for box in boxes.splitlines():
            b = box.split()
            if len(b) == 6:
                x, y, w, h = int(b[1]), height - int(b[2]), int(b[3]), int(b[4])
                regions.append((x, y, w-x, h-y))

        # Merge overlapping regions
        return self._merge_regions(regions)

    def _detect_logos(self, image: np.ndarray) -> List[tuple]:
        """Detect regions containing logos in the image."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and shape
        logo_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum size threshold
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w)/h
                if 0.5 <= aspect_ratio <= 2.0:  # Typical logo aspect ratios
                    logo_regions.append((x, y, w, h))
        
        return logo_regions

    def _merge_regions(self, regions: List[tuple], threshold: int = 10) -> List[tuple]:
        """Merge overlapping or nearby regions."""
        if not regions:
            return []
            
        # Sort regions by x coordinate
        sorted_regions = sorted(regions, key=lambda r: r[0])
        merged = []
        current = list(sorted_regions[0])
        
        for region in sorted_regions[1:]:
            if (region[0] <= current[0] + current[2] + threshold and
                region[1] <= current[1] + current[3] + threshold):
                # Merge regions
                current[2] = max(current[0] + current[2], region[0] + region[2]) - current[0]
                current[3] = max(current[1] + current[3], region[1] + region[3]) - current[1]
            else:
                merged.append(tuple(current))
                current = list(region)
                
        merged.append(tuple(current))
        return merged

    def extract_text(self, file_path: Union[str, Path]) -> str:
        """
        Extract text content from image file.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Extracted text content
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # Read and preprocess image
            image = cv2.imread(str(file_path))
            processed_image = self._preprocess_image(image)
            
            # Extract text
            text = pytesseract.image_to_string(processed_image, config=self.ocr_config)
            
            # Add position information
            text_with_positions = []
            boxes = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT)
            
            for i in range(len(boxes['text'])):
                if boxes['text'][i].strip():
                    text_with_positions.append(
                        f"[{boxes['left'][i]}, {boxes['top'][i]}] {boxes['text'][i]}"
                    )
            
            return "\n".join(text_with_positions)

        except Exception as e:
            logger.error(f"Error extracting text from image file {file_path}: {str(e)}")
            raise

    def analyze_structure(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Analyze the structure of an image file.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Dictionary containing structural analysis
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # Read image with both OpenCV and PIL for different analyses
            cv_image = cv2.imread(str(file_path))
            pil_image = Image.open(file_path)
            
            analysis = {
                'format': pil_image.format,
                'mode': pil_image.mode,
                'dimensions': pil_image.size,
                'dpi': pil_image.info.get('dpi', (72, 72)),
                'file_size': file_path.stat().st_size,
                'channels': cv_image.shape[2] if len(cv_image.shape) == 3 else 1,
                'bit_depth': pil_image.bits,
                'has_alpha': 'A' in pil_image.getbands()
            }
            
            # Detect text regions
            processed = self._preprocess_image(cv_image)
            text_regions = self._detect_text_regions(processed)
            analysis['text_regions'] = len(text_regions)
            
            # Detect potential logos
            logo_regions = self._detect_logos(cv_image)
            analysis['logo_regions'] = len(logo_regions)
            
            # Basic image quality metrics
            if len(cv_image.shape) == 3:
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = cv_image
                
            analysis['quality_metrics'] = {
                'blur_score': cv2.Laplacian(gray, cv2.CV_64F).var(),
                'contrast': gray.std(),
                'brightness': gray.mean()
            }

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing image file {file_path}: {str(e)}")
            raise


def process_image_file(input_file: Union[str, Path], 
                      output_file: Optional[Union[str, Path]] = None,
                      mask_mode: str = "replace") -> Dict[str, Any]:
    """
    Convenience function to process a single image file.
    
    Args:
        input_file: Path to input image file
        output_file: Path for output image file
        mask_mode: How to handle PII ('replace', 'mask', or 'remove')
        
    Returns:
        Processing results dictionary
    """
    processor = ImageProcessor()
    return processor.process_file(input_file, output_file, mask_mode)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        try:
            processor = ImageProcessor()
            
            # First analyze the structure
            print("\nAnalyzing image structure...")
            analysis = processor.analyze_structure(input_file)
            print("\nImage Structure:")
            print(f"Format: {analysis['format']}")
            print(f"Dimensions: {analysis['dimensions']}")
            print(f"DPI: {analysis['dpi']}")
            print(f"Text regions: {analysis['text_regions']}")
            print(f"Logo regions: {analysis['logo_regions']}")
            
            # Process the file
            print("\nProcessing file...")
            results = processor.process_file(input_file, output_file)
            
            print("\nProcessing Results:")
            print(f"Text regions processed: {results['text_regions']}")
            print(f"Logo regions masked: {results['logo_regions']}")
            print(f"Original size: {results['original_size']} bytes")
            print(f"Output size: {results['output_size']} bytes")
            print(f"\nOutput file: {results['output_file']}")
            
        except Exception as e:
            print(f"Error: {str(e)}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Usage: python image_processor.py input.png [output.png]")
        sys.exit(1)