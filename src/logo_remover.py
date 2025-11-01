"""
Logo Detection and Removal Module

This module detects and masks logos from images using computer vision techniques.
It supports various logo detection methods including:
- Template matching
- Feature-based detection (SIFT, ORB)
- Contour-based detection
- Color-based detection
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json

from PIL import Image, ImageDraw

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogoRemover:
    """Main class for detecting and removing logos from images."""
    
    def __init__(self, logo_templates_dir: Optional[str] = None):
        """
        Initialize the LogoRemover.
        
        Args:
            logo_templates_dir: Directory containing logo template images
        """
        self.logo_templates_dir = Path(logo_templates_dir) if logo_templates_dir else None
        self.logo_templates = []
        
        # Detection parameters
        self.template_threshold = 0.8
        self.feature_threshold = 0.75
        self.contour_area_threshold = 1000
        
        # Load logo templates if directory provided
        if self.logo_templates_dir and self.logo_templates_dir.exists():
            self.load_logo_templates()
        
        # Initialize feature detectors
        try:
            self.sift = cv2.SIFT_create()
        except AttributeError:
            # Fallback for older OpenCV versions
            self.sift = cv2.xfeatures2d.SIFT_create()
        
        try:
            self.orb = cv2.ORB_create()
        except:
            logger.warning("ORB detector not available")
            self.orb = None
        
        # FLANN matcher for feature matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    def load_logo_templates(self) -> None:
        """Load logo templates from the specified directory."""
        if not self.logo_templates_dir:
            return
        
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for template_file in self.logo_templates_dir.rglob('*'):
            if template_file.suffix.lower() in supported_formats:
                try:
                    template_img = cv2.imread(str(template_file))
                    if template_img is not None:
                        # Convert to grayscale for template matching
                        template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
                        
                        # Extract features
                        kp, des = self.sift.detectAndCompute(template_gray, None)
                        
                        self.logo_templates.append({
                            'name': template_file.stem,
                            'path': str(template_file),
                            'image': template_img,
                            'gray': template_gray,
                            'keypoints': kp,
                            'descriptors': des,
                            'shape': template_gray.shape
                        })
                        
                        logger.info(f"Loaded logo template: {template_file.name}")
                
                except Exception as e:
                    logger.warning(f"Failed to load template {template_file}: {str(e)}")
        
        logger.info(f"Loaded {len(self.logo_templates)} logo templates")
    
    def detect_and_mask_logo(self, image_path: Union[str, Path], 
                           output_path: Optional[Union[str, Path]] = None,
                           mask_method: str = "blur") -> Dict[str, any]:
        """
        Detect and mask logos in an image.
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the processed image (optional)
            mask_method: Method to mask logos ("blur", "fill", "pixelate", "remove")
            
        Returns:
            Dictionary containing detection results and processed image info
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect logos using multiple methods
        detections = []
        
        # Method 1: Template matching
        template_detections = self._detect_by_template_matching(gray)
        detections.extend(template_detections)
        
        # Method 2: Feature-based detection
        feature_detections = self._detect_by_features(gray)
        detections.extend(feature_detections)
        
        # Method 3: Contour-based detection (for simple logos)
        contour_detections = self._detect_by_contours(gray)
        detections.extend(contour_detections)
        
        # Method 4: Color-based detection (for colorful logos)
        color_detections = self._detect_by_color(image)
        detections.extend(color_detections)
        
        # Remove duplicate detections
        filtered_detections = self._filter_overlapping_detections(detections)
        
        # Apply masking
        processed_image = image.copy()
        for detection in filtered_detections:
            processed_image = self._apply_mask(processed_image, detection, mask_method)
        
        # Save processed image if output path provided
        if output_path:
            cv2.imwrite(str(output_path), processed_image)
            logger.info(f"Processed image saved to: {output_path}")
        
        return {
            'original_image_path': str(image_path),
            'processed_image_path': str(output_path) if output_path else None,
            'detections': filtered_detections,
            'detection_count': len(filtered_detections),
            'mask_method': mask_method,
            'image_shape': image.shape,
            'success': True
        }
    
    def _detect_by_template_matching(self, gray_image: np.ndarray) -> List[Dict[str, any]]:
        """Detect logos using template matching."""
        detections = []
        
        for template in self.logo_templates:
            template_gray = template['gray']
            
            # Multi-scale template matching
            scales = [0.5, 0.75, 1.0, 1.25, 1.5]
            
            for scale in scales:
                # Resize template
                width = int(template_gray.shape[1] * scale)
                height = int(template_gray.shape[0] * scale)
                
                if width > gray_image.shape[1] or height > gray_image.shape[0]:
                    continue
                
                resized_template = cv2.resize(template_gray, (width, height))
                
                # Template matching
                result = cv2.matchTemplate(gray_image, resized_template, cv2.TM_CCOEFF_NORMED)
                
                # Find matches above threshold
                locations = np.where(result >= self.template_threshold)
                
                for pt in zip(*locations[::-1]):
                    detections.append({
                        'method': 'template_matching',
                        'template_name': template['name'],
                        'bbox': (pt[0], pt[1], width, height),
                        'confidence': float(result[pt[1], pt[0]]),
                        'scale': scale
                    })
        
        return detections
    
    def _detect_by_features(self, gray_image: np.ndarray) -> List[Dict[str, any]]:
        """Detect logos using feature matching (SIFT)."""
        detections = []
        
        # Extract features from input image
        kp_img, des_img = self.sift.detectAndCompute(gray_image, None)
        
        if des_img is None:
            return detections
        
        for template in self.logo_templates:
            if template['descriptors'] is None:
                continue
            
            try:
                # Match features
                matches = self.flann.knnMatch(template['descriptors'], des_img, k=2)
                
                # Filter good matches using Lowe's ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < self.feature_threshold * n.distance:
                            good_matches.append(m)
                
                # Need at least 4 matches for homography
                if len(good_matches) >= 4:
                    # Extract matched keypoints
                    src_pts = np.float32([template['keypoints'][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_img[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    
                    # Find homography
                    homography, mask = cv2.findHomography(src_pts, dst_pts, 
                                                        cv2.RANSAC, 5.0)
                    
                    if homography is not None:
                        # Transform template corners to image space
                        h, w = template['shape']
                        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                        transformed_corners = cv2.perspectiveTransform(corners, homography)
                        
                        # Calculate bounding box
                        x_coords = transformed_corners[:, 0, 0]
                        y_coords = transformed_corners[:, 0, 1]
                        
                        x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
                        y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
                        
                        # Validate bounding box
                        if (x_min >= 0 and y_min >= 0 and 
                            x_max < gray_image.shape[1] and y_max < gray_image.shape[0] and
                            x_max > x_min and y_max > y_min):
                            
                            detections.append({
                                'method': 'feature_matching',
                                'template_name': template['name'],
                                'bbox': (x_min, y_min, x_max - x_min, y_max - y_min),
                                'confidence': len(good_matches) / len(template['keypoints']),
                                'matches': len(good_matches),
                                'corners': transformed_corners.tolist()
                            })
            
            except Exception as e:
                logger.debug(f"Feature matching failed for {template['name']}: {str(e)}")
        
        return detections
    
    def _detect_by_contours(self, gray_image: np.ndarray) -> List[Dict[str, any]]:
        """Detect logos using contour analysis."""
        detections = []
        
        # Apply edge detection
        edges = cv2.Canny(gray_image, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > self.contour_area_threshold:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate shape features
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                aspect_ratio = w / h if h > 0 else 0
                
                # Filter based on shape characteristics (typical for logos)
                if (0.2 < aspect_ratio < 5.0 and  # Reasonable aspect ratio
                    0.1 < circularity < 1.2 and   # Not too irregular
                    w > 20 and h > 20):           # Minimum size
                    
                    detections.append({
                        'method': 'contour_detection',
                        'bbox': (x, y, w, h),
                        'confidence': min(circularity, 1.0),
                        'area': area,
                        'circularity': circularity,
                        'aspect_ratio': aspect_ratio
                    })
        
        return detections
    
    def _detect_by_color(self, image: np.ndarray) -> List[Dict[str, any]]:
        """Detect logos using color-based analysis."""
        detections = []
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for common logo colors
        color_ranges = {
            'red': [(0, 50, 50), (10, 255, 255)],
            'blue': [(100, 50, 50), (130, 255, 255)],
            'green': [(40, 50, 50), (80, 255, 255)],
            'yellow': [(20, 50, 50), (40, 255, 255)],
        }
        
        for color_name, (lower, upper) in color_ranges.items():
            # Create mask for color range
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Apply morphological operations to clean up mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours in mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area > 500:  # Minimum area for color regions
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate color density in bounding box
                    roi_mask = mask[y:y+h, x:x+w]
                    color_density = np.sum(roi_mask > 0) / (w * h) if w * h > 0 else 0
                    
                    if color_density > 0.3:  # At least 30% of the region should be the target color
                        detections.append({
                            'method': 'color_detection',
                            'color': color_name,
                            'bbox': (x, y, w, h),
                            'confidence': color_density,
                            'area': area
                        })
        
        return detections
    
    def _filter_overlapping_detections(self, detections: List[Dict[str, any]], 
                                     overlap_threshold: float = 0.5) -> List[Dict[str, any]]:
        """Filter out overlapping detections, keeping the ones with higher confidence."""
        if not detections:
            return []
        
        # Sort by confidence (descending)
        sorted_detections = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)
        
        filtered = []
        
        for detection in sorted_detections:
            bbox1 = detection['bbox']
            
            # Check overlap with already selected detections
            is_overlapping = False
            
            for selected in filtered:
                bbox2 = selected['bbox']
                
                if self._calculate_overlap(bbox1, bbox2) > overlap_threshold:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                filtered.append(detection)
        
        return filtered
    
    def _calculate_overlap(self, bbox1: Tuple[int, int, int, int], 
                          bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate the overlap ratio between two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def _apply_mask(self, image: np.ndarray, detection: Dict[str, any], 
                   mask_method: str) -> np.ndarray:
        """Apply masking to detected logo region."""
        x, y, w, h = detection['bbox']
        
        # Ensure coordinates are within image bounds
        x = max(0, min(x, image.shape[1] - 1))
        y = max(0, min(y, image.shape[0] - 1))
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return image
        
        roi = image[y:y+h, x:x+w]
        
        if mask_method == "blur":
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(roi, (51, 51), 0)
            image[y:y+h, x:x+w] = blurred
        
        elif mask_method == "fill":
            # Fill with average color of surrounding area
            # Get surrounding region
            pad = 10
            y_start = max(0, y - pad)
            y_end = min(image.shape[0], y + h + pad)
            x_start = max(0, x - pad)
            x_end = min(image.shape[1], x + w + pad)
            
            surrounding = image[y_start:y_end, x_start:x_end]
            avg_color = np.mean(surrounding, axis=(0, 1))
            
            image[y:y+h, x:x+w] = avg_color
        
        elif mask_method == "pixelate":
            # Apply pixelation effect
            small = cv2.resize(roi, (w//10, h//10), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            image[y:y+h, x:x+w] = pixelated
        
        elif mask_method == "remove":
            # Fill with white
            image[y:y+h, x:x+w] = [255, 255, 255]
        
        return image
    
    def batch_process_images(self, input_dir: Union[str, Path], 
                           output_dir: Union[str, Path],
                           mask_method: str = "blur") -> Dict[str, any]:
        """Process multiple images in a directory."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        results = []
        
        for image_file in input_dir.rglob('*'):
            if image_file.suffix.lower() in supported_formats:
                try:
                    output_file = output_dir / f"cleaned_{image_file.name}"
                    result = self.detect_and_mask_logo(image_file, output_file, mask_method)
                    results.append(result)
                    
                    logger.info(f"Processed: {image_file.name} -> {result['detection_count']} logos detected")
                
                except Exception as e:
                    logger.error(f"Failed to process {image_file}: {str(e)}")
                    results.append({
                        'original_image_path': str(image_file),
                        'error': str(e),
                        'success': False
                    })
        
        return {
            'processed_count': len(results),
            'successful_count': sum(1 for r in results if r.get('success', False)),
            'total_detections': sum(r.get('detection_count', 0) for r in results),
            'results': results
        }


def detect_and_mask_logo(image_path: Union[str, Path], 
                        output_path: Optional[Union[str, Path]] = None,
                        mask_method: str = "blur",
                        logo_templates_dir: Optional[str] = None) -> Dict[str, any]:
    """
    Convenience function to detect and mask logos in an image.
    
    Args:
        image_path: Path to input image
        output_path: Path to save processed image
        mask_method: Masking method ("blur", "fill", "pixelate", "remove")
        logo_templates_dir: Directory containing logo templates
        
    Returns:
        Detection and processing results
    """
    remover = LogoRemover(logo_templates_dir=logo_templates_dir)
    return remover.detect_and_mask_logo(image_path, output_path, mask_method)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python logo_remover.py <image_path> [output_path] [mask_method]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    mask_method = sys.argv[3] if len(sys.argv) > 3 else "blur"
    
    try:
        result = detect_and_mask_logo(image_path, output_path, mask_method)
        
        if result['success']:
            print(f"Successfully processed image: {image_path}")
            print(f"Detected {result['detection_count']} logos")
            if output_path:
                print(f"Processed image saved to: {output_path}")
        else:
            print(f"Failed to process image: {image_path}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
