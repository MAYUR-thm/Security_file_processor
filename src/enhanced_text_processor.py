"""
Enhanced OCR module with improved text detection and PII masking.
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import re
from rapidfuzz import fuzz
import easyocr
from pathlib import Path
import concurrent.futures

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OCRConfig:
    """Configuration for OCR processing."""
    UPSCALE_MIN_HEIGHT: int = 20
    USE_SUPERRES: bool = True
    USE_EASYOCR: bool = True
    CONFIDENCE_THRESHOLD: float = 60.0
    PADDING_PCT: float = 0.15
    FUZZY_THRESHOLD: float = 80.0
    MASK_STRICTNESS: str = "medium"
    MAX_WORKERS: int = 4
    
    # CLI-specific settings
    CLI_PSM_MODE: int = 6  # Uniform text block
    CLI_OEM_MODE: int = 1  # LSTM OCR Engine
    CLI_CHARS: str = r"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.:/_[]{}()=\\\"'"
    
    @property
    def cli_config(self) -> str:
        """Get Tesseract config string for CLI content."""
        return f"--oem {self.CLI_OEM_MODE} --psm {self.CLI_PSM_MODE} -c tessedit_char_whitelist={self.CLI_CHARS}"
    
    # CLI-specific settings
    CLI_PSM_MODE: int = 6  # Uniform text block
    CLI_OEM_MODE: int = 1  # LSTM OCR Engine
    CLI_CONFIG: str = "--oem 1 --psm 6 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.:/_[]{}()\"'=\\"
    
    # CLI-specific settings
    CLI_PSM_MODE: int = 6  # Uniform text block
    CLI_OEM_MODE: int = 1  # LSTM OCR Engine
    CLI_WHITELIST: str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.:/_[]{}\"'=\\"
    
    # CLI-specific settings
    CLI_PSM_MODE: int = 6  # Uniform text block
    CLI_OEM_MODE: int = 1  # LSTM OCR Engine
    CLI_WHITELIST: str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.:/_[]{}\"'="

class EnhancedTextProcessor:
    """Advanced text processing with improved detection and masking."""
    
    def __init__(self, config: Optional[OCRConfig] = None):
        """Initialize the text processor."""
        self.config = config or OCRConfig()
        
        # Initialize EasyOCR if enabled
        if self.config.USE_EASYOCR:
            self.reader = easyocr.Reader(['en'])
        
        # Initialize super-resolution if enabled
        if self.config.USE_SUPERRES:
            self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
            model_path = Path(__file__).parent / "models" / "ESPCN_x4.pb"
            if model_path.exists():
                self.sr.readModel(str(model_path))
                self.sr.setModel("espcn", 4)
            else:
                logger.warning("Super-resolution model not found. Falling back to cubic interpolation.")
                self.config.USE_SUPERRES = False

        # Compile regex patterns for CLI and structured content
        self.patterns = {
            'arn': re.compile(r'arn:aws:iam::\d{12}:(?:role|user|group|policy)[/:][\w+=,.@\-_/]+', re.I),
            's3_action': re.compile(r'\bs3[:/]\s*(?:get|list|put|delete)\w*\b', re.I),
            'ip_address': re.compile(r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'),
            'sensitive_words': re.compile(r'\b(?:role|user|group|policy|secret|key|password|token|credential)\b', re.I),
            'phone': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            'date': re.compile(r'\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}[-+]\d{2}:\d{2})?\b'),
            'account_id': re.compile(r'\b\d{12}\b'),
            'role_id': re.compile(r'\b[A-Z0-9]{21}EXAMPLE\b', re.I),
            'json_key': re.compile(r'"(?:Role(?:Name|Id|Arn)|Principal|Effect|Action|Version)":', re.I),
            'cli_param': re.compile(r'--[a-z-]+\s+["\']?[\w\-./]+["\']?', re.I)
        }

        # Keywords for fuzzy matching with CLI context
        self.keywords = [
            'role', 'arn', 'aws', 'iam', 'policy', 'user', 'group',
            'secret', 'key', 'password', 'token', 'credential',
            'create-role', 'assume-role', 'trust-policy', 'statement',
            'principal', 'effect', 'action'
        ]
        
        # Special handling for CLI output
        self.cli_markers = [
            'aws', 'iam', 'create-role', 'assume-role',
            'Output:', '{', '}', '[', ']'
        ]

    def preprocess_image(self, image: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Preprocess image or ROI for better text detection.
        
        Args:
            image: Input image
            roi: Optional region of interest (x, y, w, h)
            
        Returns:
            Preprocessed image
        """
        # Extract ROI if provided
        if roi:
            x, y, w, h = roi
            image = image[y:y+h, x:x+w]
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Check if upscaling needed
        if h < self.config.UPSCALE_MIN_HEIGHT:
            scale = self.config.UPSCALE_MIN_HEIGHT / h
            if self.config.USE_SUPERRES:
                gray = self.sr.upsample(gray)
            else:
                gray = cv2.resize(gray, None, fx=scale, fy=scale, 
                                interpolation=cv2.INTER_CUBIC)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Denoise
        gray = cv2.fastNlMeansDenoising(gray)
        
        # Optional sharpening
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        gray = cv2.filter2D(gray, -1, kernel)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(gray, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY,
                                     11, 2)
        
        return binary

    def detect_text_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect text regions in image using multiple methods.
        
        Args:
            image: Input image
            
        Returns:
            List of detected regions with coordinates and confidence
        """
        regions = []
        
        # Get Tesseract word boxes
        word_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        for i in range(len(word_data['text'])):
            if int(word_data['conf'][i]) > 0:  # Filter empty results
                x, y, w, h = (word_data['left'][i], word_data['top'][i],
                             word_data['width'][i], word_data['height'][i])
                
                # Add padding
                pad_x = int(w * self.config.PADDING_PCT)
                pad_y = int(h * self.config.PADDING_PCT)
                
                regions.append({
                    'bbox': (max(0, x - pad_x),
                            max(0, y - pad_y),
                            w + 2 * pad_x,
                            h + 2 * pad_y),
                    'text': word_data['text'][i],
                    'conf': float(word_data['conf'][i]),
                    'method': 'tesseract'
                })
        
        # Add EasyOCR results if enabled
        if self.config.USE_EASYOCR:
            try:
                easy_results = self.reader.readtext(image)
                for bbox, text, conf in easy_results:
                    x, y = map(int, bbox[0])  # Top-left corner
                    w = int(bbox[2][0] - bbox[0][0])  # Width
                    h = int(bbox[2][1] - bbox[0][1])  # Height
                    
                    # Add padding
                    pad_x = int(w * self.config.PADDING_PCT)
                    pad_y = int(h * self.config.PADDING_PCT)
                    
                    regions.append({
                        'bbox': (max(0, x - pad_x),
                                max(0, y - pad_y),
                                w + 2 * pad_x,
                                h + 2 * pad_y),
                        'text': text,
                        'conf': conf * 100,  # Convert to percentage
                        'method': 'easyocr'
                    })
            except Exception as e:
                logger.warning(f"EasyOCR detection failed: {str(e)}")
        
        return regions

    def process_region(self, image: np.ndarray, region: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single text region with multiple OCR configs.
        
        Args:
            image: Full image
            region: Region information
            
        Returns:
            Updated region information
        """
        x, y, w, h = region['bbox']
        roi = image[y:y+h, x:x+w]
        
        # Preprocess ROI
        processed_roi = self.preprocess_image(roi)
        
        results = []
        
        # Check for CLI content
        is_cli = False
        try:
            full_text = pytesseract.image_to_string(processed_roi)
            is_cli = any(marker in full_text.lower() for marker in self.cli_markers)
        except Exception:
            pass
        
        # Use appropriate OCR configs
        if is_cli:
            # Use CLI-specific config
            try:
                text = pytesseract.image_to_string(
                    processed_roi, 
                    config=self.config.cli_config
                ).strip()
                
                if text:
                    data = pytesseract.image_to_data(
                        processed_roi,
                        config=self.config.cli_config,
                        output_type=pytesseract.Output.DICT
                    )
                    conf = float(data['conf'][0]) if len(data['conf']) > 0 else 0.0
                    results.append({
                        'text': text,
                        'conf': conf,
                        'method': 'tesseract_cli',
                        'is_cli': True
                    })
            except Exception as e:
                logger.warning(f"CLI-specific OCR failed: {str(e)}")
        
        # Try standard configs if no CLI content or CLI processing failed
        if not results:
            configs = [
                '--oem 1 --psm 6',  # Single uniform block
                '--oem 1 --psm 11',  # Sparse text
                '--oem 1 --psm 3'   # Full page
            ]
            
            for config in configs:
                try:
                    text = pytesseract.image_to_string(processed_roi, config=config).strip()
                    if text:
                        data = pytesseract.image_to_data(
                            processed_roi,
                            config=config,
                            output_type=pytesseract.Output.DICT
                        )
                        conf = float(data['conf'][0]) if len(data['conf']) > 0 else 0.0
                        results.append({
                            'text': text,
                            'conf': conf,
                            'method': f'tesseract_{config}',
                            'is_cli': False
                        })
                except Exception as e:
                    logger.warning(f"Tesseract processing failed with config {config}: {str(e)}")
        
        # Try EasyOCR if enabled and Tesseract confidence is low
        if (self.config.USE_EASYOCR and 
            (not results or max(r['conf'] for r in results) < self.config.CONFIDENCE_THRESHOLD)):
            try:
                easy_results = self.reader.readtext(processed_roi)
                for _, text, conf in easy_results:
                    results.append({
                        'text': text,
                        'conf': conf * 100,
                        'method': 'easyocr',
                        'is_cli': False
                    })
            except Exception as e:
                logger.warning(f"EasyOCR processing failed: {str(e)}")
        
        # Choose best result
        if results:
            # Prefer CLI results for CLI content
            if is_cli:
                cli_results = [r for r in results if r.get('is_cli', False)]
                if cli_results:
                    best_result = max(cli_results, key=lambda x: x['conf'])
                else:
                    best_result = max(results, key=lambda x: x['conf'])
            else:
                best_result = max(results, key=lambda x: x['conf'])
            
            region.update(best_result)
        
        return region

    def detect_pii(self, text: str, neighbors: List[str] = None) -> Dict[str, Any]:
        """
        Detect PII in text using regex and fuzzy matching.
        
        Args:
            text: Text to check
            neighbors: Optional neighboring text for context
            
        Returns:
            Dictionary with detection results
        """
        result = {
            'is_pii': False,
            'type': None,
            'confidence': 0.0,
            'context': {}
        }
        
        # Normalize text for checking
        normalized = text.lower().strip()
        
        # Check for CLI context
        is_cli = any(marker in normalized for marker in self.cli_markers)
        
        # Check if this is part of a JSON structure
        is_json = bool(self.patterns['json_key'].search(text))
        
        # Track context for smarter detection
        context = {
            'is_cli': is_cli,
            'is_json': is_json,
            'has_neighbors': bool(neighbors)
        }
        
        # Check for direct PII matches
        for name, pattern in self.patterns.items():
            if match := pattern.search(text):
                result.update({
                    'is_pii': True,
                    'type': name,
                    'confidence': 95.0,
                    'match': match.group(0),
                    'context': context
                })
                return result
        
        # Special handling for CLI parameters
        if is_cli:
            if param_match := self.patterns['cli_param'].search(text):
                param_name = param_match.group(0).split()[0]
                # Check if this is a sensitive parameter
                if any(keyword in param_name for keyword in ['role', 'policy', 'arn', 'key']):
                    result.update({
                        'is_pii': True,
                        'type': 'cli_parameter',
                        'confidence': 90.0,
                        'match': param_match.group(0),
                        'context': context
                    })
                    return result
        
        # Check combined text with neighbors for context
        if neighbors:
            combined_text = ' '.join([text] + neighbors)
            for name, pattern in self.patterns.items():
                if match := pattern.search(combined_text):
                    result.update({
                        'is_pii': True,
                        'type': f'contextual_{name}',
                        'confidence': 85.0,
                        'match': match.group(0),
                        'context': context
                    })
                    return result
        
        # Enhanced fuzzy matching for structured content
        for keyword in self.keywords:
            # Direct match
            ratio = fuzz.ratio(normalized, keyword)
            if ratio >= self.config.FUZZY_THRESHOLD:
                result.update({
                    'is_pii': True,
                    'type': 'fuzzy_keyword',
                    'confidence': ratio,
                    'match': text,
                    'context': context
                })
                return result
            
            # Partial match for longer text
            if len(normalized) > len(keyword):
                for i in range(len(normalized) - len(keyword) + 1):
                    ratio = fuzz.ratio(normalized[i:i+len(keyword)], keyword)
                    if ratio >= self.config.FUZZY_THRESHOLD:
                        result.update({
                            'is_pii': True,
                            'type': 'partial_keyword',
                            'confidence': ratio,
                            'match': text,
                            'context': context
                        })
                        return result
        
        # Special handling for JSON structure values
        if is_json and any(normalized.strip('" ') in self.keywords for k in self.keywords):
            result.update({
                'is_pii': True,
                'type': 'json_value',
                'confidence': 80.0,
                'match': text,
                'context': context
            })
            return result
        
        result['context'] = context
        return result

    def process_image(self, image: Union[np.ndarray, str, Path]) -> Dict[str, Any]:
        """
        Process image and detect/mask PII.
        
        Args:
            image: Input image or path
            
        Returns:
            Dictionary with results and masked image
        """
        # Load image if needed
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"Could not load image: {image}")
        
        # Create copy for masking
        masked_image = image.copy()
        
        # Check for CLI-style content
        is_cli_content = False
        try:
            # Convert image to text to check for CLI markers
            full_text = pytesseract.image_to_string(image)
            is_cli_content = any(marker in full_text.lower() for marker in self.cli_markers)
        except Exception:
            pass
        
        # Detect text regions
        regions = self.detect_text_regions(image)
        
        # Process regions in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            futures = [executor.submit(self.process_region, image, region)
                      for region in regions]
            regions = [f.result() for f in futures]
        
        # Group nearby regions
        grouped_regions = self._group_nearby_regions(regions)
        
        # Process each group for PII
        pii_detected = []
        masked_regions = []
        
        for group in grouped_regions:
            group_has_pii = False
            group_detections = []
            
            # Get text and neighbors
            for i, region in enumerate(group):
                neighbors = [r['text'] for r in group[:i] + group[i+1:]]
                
                # Enhanced PII detection
                detection = self.detect_pii(region['text'], neighbors)
                
                if detection['is_pii']:
                    group_has_pii = True
                    region.update({
                        'pii_type': detection['type'],
                        'confidence': detection['confidence'],
                        'context': detection['context']
                    })
                    group_detections.append(region)
                    pii_detected.append(region)
            
            if group_has_pii:
                # Special handling for CLI and JSON content
                is_structured = any(r.get('context', {}).get('is_cli') or 
                                  r.get('context', {}).get('is_json') 
                                  for r in group_detections)
                
                if is_structured or self.config.MASK_STRICTNESS == "high":
                    # Mask entire group for structured content
                    bbox = self._merge_bboxes([r['bbox'] for r in group])
                    masked_regions.append(bbox)
                else:
                    # Mask individual PII regions
                    for region in group_detections:
                        masked_regions.append(region['bbox'])
        
        # Apply masking with consideration for content type
        for bbox in masked_regions:
            if is_cli_content:
                # Use solid redaction for CLI content
                self._apply_cli_mask(masked_image, bbox)
            else:
                # Use blur for regular content
                self._apply_mask(masked_image, bbox)
        
        return {
            'masked_image': masked_image,
            'pii_detected': pii_detected,
            'total_regions': len(regions),
            'pii_regions': len(pii_detected),
            'is_cli_content': is_cli_content
        }

    def _group_nearby_regions(self, regions: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group regions that are close to each other."""
        if not regions:
            return []
            
        groups = [[regions[0]]]
        
        for region in regions[1:]:
            x1, y1, w1, h1 = region['bbox']
            added = False
            
            for group in groups:
                # Check distance to any region in group
                for r in group:
                    x2, y2, w2, h2 = r['bbox']
                    
                    # Calculate distance between regions
                    dist_x = abs((x1 + w1/2) - (x2 + w2/2))
                    dist_y = abs((y1 + h1/2) - (y2 + h2/2))
                    
                    # Group if close enough
                    if dist_x < max(w1, w2) and dist_y < max(h1, h2):
                        group.append(region)
                        added = True
                        break
                
                if added:
                    break
            
            if not added:
                groups.append([region])
        
        return groups

    def _merge_bboxes(self, bboxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        """Merge multiple bounding boxes into one."""
        x1 = min(bbox[0] for bbox in bboxes)
        y1 = min(bbox[1] for bbox in bboxes)
        x2 = max(bbox[0] + bbox[2] for bbox in bboxes)
        y2 = max(bbox[1] + bbox[3] for bbox in bboxes)
        return (x1, y1, x2 - x1, y2 - y1)

    def _apply_mask(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
        """Apply blur masking to a region."""
        x, y, w, h = bbox
        # Add padding
        pad_x = int(w * self.config.PADDING_PCT)
        pad_y = int(h * self.config.PADDING_PCT)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(image.shape[1], x + w + pad_x)
        y2 = min(image.shape[0], y + h + pad_y)
        
        # Apply gaussian blur
        roi = image[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(roi, (99, 99), 30)
        image[y1:y2, x1:x2] = blurred
        
    def _apply_cli_mask(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
        """Apply solid redaction masking for CLI content."""
        x, y, w, h = bbox
        # Add padding
        pad_x = int(w * self.config.PADDING_PCT)
        pad_y = int(h * self.config.PADDING_PCT)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(image.shape[1], x + w + pad_x)
        y2 = min(image.shape[0], y + h + pad_y)
        
        # Create solid color redaction
        roi = image[y1:y2, x1:x2]
        mask_color = (0, 0, 0) if len(image.shape) == 3 else 0
        
        # Create redaction overlay
        overlay = np.full(roi.shape, mask_color, dtype=np.uint8)
        
        # Add red highlight for CLI content
        if len(image.shape) == 3:
            overlay[:, :, 2] = 255  # Red channel
        
        # Apply redaction with alpha blending
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)
        image[y1:y2, x1:x2] = roi