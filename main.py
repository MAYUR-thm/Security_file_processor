"""
Security File Processor - Main Orchestrator

This is the main entry point for the automated security file processing solution.
It orchestrates the entire pipeline including:
- File upload and preprocessing
- Text extraction (OCR)
- PII detection and cleansing
- Logo detection and masking
- Security content analysis
- Report generation

Usage:
    python main.py --input <input_path> [options]
    python main.py --input data/files --output results --client-names data/client_names.json
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import zipfile
import shutil

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

# Import our modules
from ocr import TextExtractor, extract_text_from_file
from pii_cleaner import PIICleaner, remove_pii
from logo_remover import LogoRemover, detect_and_mask_logo
from analyzer import SecurityAnalyzer, analyze_security_content
from report_generator import ReportGenerator, generate_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('security_processor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SecurityFileProcessor:
    """Main class that orchestrates the entire security file processing pipeline."""
    
    def __init__(self, 
                 output_dir: Union[str, Path] = "output",
                 client_names_file: Optional[str] = None,
                 logo_templates_dir: Optional[str] = None,
                 ocr_engine: str = "pytesseract"):
        """
        Initialize the Security File Processor.
        
        Args:
            output_dir: Directory to save processed files and reports
            client_names_file: Path to JSON file containing client names to mask
            logo_templates_dir: Directory containing logo template images
            ocr_engine: OCR engine to use ("pytesseract" or "easyocr")
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.cleaned_files_dir = self.output_dir / "cleaned_files"
        self.reports_dir = self.output_dir / "reports"
        self.logs_dir = self.output_dir / "logs"
        
        for directory in [self.cleaned_files_dir, self.reports_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.text_extractor = TextExtractor(ocr_engine=ocr_engine)
        self.pii_cleaner = PIICleaner(client_names_file=client_names_file)
        self.logo_remover = LogoRemover(logo_templates_dir=logo_templates_dir)
        self.security_analyzer = SecurityAnalyzer()
        self.report_generator = ReportGenerator(output_dir=self.reports_dir)
        
        # Processing statistics
        self.processing_stats = {
            'start_time': None,
            'end_time': None,
            'total_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'files_processed': []
        }
        
        # Supported file extensions
        self.supported_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.xlsx', '.pptx', '.zip', '.txt'}
    
    def process_files(self, input_path: Union[str, Path], 
                     workflow: str = "both") -> Dict[str, any]:
        """
        Process files through the complete pipeline.
        
        Args:
            input_path: Path to input file or directory
            workflow: Processing workflow ("cleansing", "analysis", or "both")
            
        Returns:
            Dictionary containing processing results and statistics
        """
        logger.info(f"Starting security file processing pipeline")
        logger.info(f"Input path: {input_path}")
        logger.info(f"Workflow: {workflow}")
        logger.info(f"Output directory: {self.output_dir}")
        
        self.processing_stats['start_time'] = datetime.now()
        
        try:
            # Get list of files to process
            files_to_process = self._get_files_to_process(input_path)
            self.processing_stats['total_files'] = len(files_to_process)
            
            logger.info(f"Found {len(files_to_process)} files to process")
            
            if not files_to_process:
                logger.warning("No supported files found to process")
                return self._generate_final_results([])
            
            # Process each file
            results = []
            
            for file_path in files_to_process:
                try:
                    logger.info(f"Processing file: {file_path}")
                    result = self._process_single_file(file_path, workflow)
                    results.append(result)
                    
                    if result.get('success', False):
                        self.processing_stats['successful_files'] += 1
                    else:
                        self.processing_stats['failed_files'] += 1
                    
                    self.processing_stats['files_processed'].append({
                        'file_path': str(file_path),
                        'success': result.get('success', False),
                        'processing_time': result.get('processing_time', 0)
                    })
                
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
                    self.processing_stats['failed_files'] += 1
                    
                    results.append({
                        'file_name': file_path.name,
                        'file_path': str(file_path),
                        'success': False,
                        'error': str(e),
                        'processing_timestamp': datetime.now().isoformat()
                    })
            
            # Generate final results and reports
            final_results = self._generate_final_results(results)
            
            self.processing_stats['end_time'] = datetime.now()
            
            logger.info(f"Processing completed successfully")
            logger.info(f"Total files: {self.processing_stats['total_files']}")
            logger.info(f"Successful: {self.processing_stats['successful_files']}")
            logger.info(f"Failed: {self.processing_stats['failed_files']}")
            
            return final_results
        
        except Exception as e:
            logger.error(f"Critical error in processing pipeline: {str(e)}")
            self.processing_stats['end_time'] = datetime.now()
            raise
    
    def _get_files_to_process(self, input_path: Union[str, Path]) -> List[Path]:
        """Get list of files to process from input path."""
        input_path = Path(input_path)
        files_to_process = []
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")
        
        if input_path.is_file():
            # Single file
            if input_path.suffix.lower() in self.supported_extensions:
                files_to_process.append(input_path)
            else:
                logger.warning(f"Unsupported file type: {input_path}")
        
        elif input_path.is_dir():
            # Directory - find all supported files
            for file_path in input_path.rglob('*'):
                if (file_path.is_file() and 
                    file_path.suffix.lower() in self.supported_extensions):
                    files_to_process.append(file_path)
        
        return sorted(files_to_process)
    
    def _process_single_file(self, file_path: Path, workflow: str) -> Dict[str, any]:
        """Process a single file through the pipeline."""
        start_time = datetime.now()
        
        result = {
            'file_name': file_path.name,
            'file_path': str(file_path),
            'file_type': file_path.suffix.lower().lstrip('.'),
            'processing_timestamp': start_time.isoformat(),
            'workflow': workflow,
            'success': False
        }
        
        try:
            # Step 1: Text Extraction
            logger.info(f"  Step 1: Extracting text from {file_path.name}")
            extraction_result = self.text_extractor.extract_text_from_file(file_path)
            
            if not extraction_result.get('success', False):
                result['error'] = f"Text extraction failed: {extraction_result.get('error', 'Unknown error')}"
                return result
            
            extracted_text = extraction_result.get('text', '')
            result['text_length'] = len(extracted_text)
            result['extraction_metadata'] = {
                'method': 'ocr' if extraction_result.get('is_scanned', False) else 'direct',
                'file_type': extraction_result.get('file_type', 'unknown')
            }
            
            # Step 2: Workflow A - File Cleansing (if requested)
            if workflow in ['cleansing', 'both']:
                logger.info(f"  Step 2: Performing PII cleansing")
                
                # PII Detection and Cleansing
                pii_result = self.pii_cleaner.remove_pii(extracted_text, mask_mode="replace")
                cleaned_text = pii_result['cleaned_text']
                
                # Risk analysis
                risk_analysis = self.pii_cleaner.analyze_pii_risk(extracted_text)
                
                result['pii_analysis'] = {
                    'risk_level': risk_analysis['risk_level'],
                    'risk_score': risk_analysis['risk_score'],
                    'pii_count': risk_analysis['pii_count'],
                    'pii_findings': pii_result['pii_findings'],
                    'replacements_made': pii_result['replacements_made'],
                    'recommendations': risk_analysis['recommendations']
                }
                
                # Logo Detection and Masking (for image files)
                if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    logger.info(f"  Step 2b: Detecting and masking logos")
                    
                    cleaned_image_path = self.cleaned_files_dir / f"cleaned_{file_path.name}"
                    logo_result = self.logo_remover.detect_and_mask_logo(
                        file_path, cleaned_image_path, mask_method="blur"
                    )
                    
                    result['logo_analysis'] = {
                        'detections': logo_result.get('detections', []),
                        'detection_count': logo_result.get('detection_count', 0),
                        'cleaned_image_path': str(cleaned_image_path) if logo_result.get('success') else None
                    }
                
                # Save cleaned content in original format
                if file_path.suffix.lower() == '.pdf':
                    from src.pdf_processor import PDFProcessor
                    pdf_processor = PDFProcessor(pii_cleaner=self.pii_cleaner)
                    cleaned_file_path = self.cleaned_files_dir / f"cleaned_{file_path.name}"
                    pdf_result = pdf_processor.process_file(file_path, cleaned_file_path)
                    result['cleaned_file_path'] = str(cleaned_file_path)
                
                elif file_path.suffix.lower() == '.pptx':
                    from src.powerpoint_processor import PowerPointProcessor
                    ppt_processor = PowerPointProcessor(pii_cleaner=self.pii_cleaner)
                    cleaned_file_path = self.cleaned_files_dir / f"cleaned_{file_path.name}"
                    ppt_result = ppt_processor.process_file(file_path, cleaned_file_path)
                    result['cleaned_file_path'] = str(cleaned_file_path)
                
                elif file_path.suffix.lower() == '.xlsx':
                    from src.excel_processor import ExcelProcessor
                    excel_processor = ExcelProcessor()
                    cleaned_file_path = self.cleaned_files_dir / f"cleaned_{file_path.name}"
                    excel_result = excel_processor.process_file(file_path, cleaned_file_path)
                    result['cleaned_file_path'] = str(cleaned_file_path)
                
                else:
                    # For other files, save as text
                    cleaned_text_path = self.cleaned_files_dir / f"cleaned_{file_path.stem}.txt"
                    with open(cleaned_text_path, 'w', encoding='utf-8') as f:
                        f.write(cleaned_text)
                    result['cleaned_file_path'] = str(cleaned_text_path)
                
                # Also save text version for analysis
                text_path = self.cleaned_files_dir / f"cleaned_{file_path.stem}.txt"
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)
                result['cleaned_text_path'] = str(text_path)
                
                # Use cleaned text for further analysis
                analysis_text = cleaned_text
            else:
                # Use original text for analysis
                analysis_text = extracted_text
            
            # Step 3: Workflow B - Security Analysis (if requested)
            if workflow in ['analysis', 'both']:
                logger.info(f"  Step 3: Performing security analysis")
                
                security_result = self.security_analyzer.analyze_security_content(analysis_text)
                
                # Generate file description
                file_description = self.security_analyzer.generate_file_description(security_result)
                
                result['security_analysis'] = security_result
                result['file_description'] = file_description
                
                # Extract structured data
                structured_data = self.security_analyzer.extract_structured_data(analysis_text)
                result['structured_data'] = structured_data
            
            # Calculate processing time
            end_time = datetime.now()
            result['processing_time'] = (end_time - start_time).total_seconds()
            result['success'] = True
            
            logger.info(f"  Successfully processed {file_path.name} in {result['processing_time']:.2f} seconds")
            
        except Exception as e:
            logger.error(f"  Error processing {file_path.name}: {str(e)}")
            result['error'] = str(e)
            result['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def _generate_final_results(self, processing_results: List[Dict[str, any]]) -> Dict[str, any]:
        """Generate final results and reports."""
        logger.info("Generating comprehensive reports")
        
        try:
            # Generate reports
            report_files = self.report_generator.generate_comprehensive_report(
                processing_results, 
                "security_analysis_report"
            )
            
            # Save processing statistics
            stats_file = self.logs_dir / "processing_statistics.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.processing_stats, f, indent=2, default=str)
            
            # Create summary
            summary = {
                'processing_summary': {
                    'total_files': self.processing_stats['total_files'],
                    'successful_files': self.processing_stats['successful_files'],
                    'failed_files': self.processing_stats['failed_files'],
                    'success_rate': (self.processing_stats['successful_files'] / 
                                   max(self.processing_stats['total_files'], 1)) * 100,
                    'total_processing_time': str(self.processing_stats['end_time'] - 
                                               self.processing_stats['start_time']) if self.processing_stats['end_time'] else None
                },
                'output_files': {
                    'cleaned_files_directory': str(self.cleaned_files_dir),
                    'reports_directory': str(self.reports_dir),
                    'logs_directory': str(self.logs_dir),
                    'generated_reports': report_files,
                    'processing_statistics': str(stats_file)
                },
                'processing_results': processing_results
            }
            
            return summary
        
        except Exception as e:
            logger.error(f"Error generating final results: {str(e)}")
            raise
    
    def process_zip_file(self, zip_path: Union[str, Path]) -> Dict[str, any]:
        """
        Process a ZIP file by extracting and processing individual files.
        
        Args:
            zip_path: Path to ZIP file
            
        Returns:
            Processing results
        """
        zip_path = Path(zip_path)
        
        if not zip_path.exists() or zip_path.suffix.lower() != '.zip':
            raise ValueError(f"Invalid ZIP file: {zip_path}")
        
        # Create temporary extraction directory
        temp_dir = self.output_dir / f"temp_extract_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Extract ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            logger.info(f"Extracted ZIP file to: {temp_dir}")
            
            # Process extracted files
            results = self.process_files(temp_dir, workflow="both")
            
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
            
            return results
        
        except Exception as e:
            # Clean up on error
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise


def main():
    """Main entry point for the security file processor."""
    parser = argparse.ArgumentParser(
        description="Automated Security File Processing Solution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input data/files --output results
  python main.py --input document.pdf --workflow cleansing
  python main.py --input files.zip --client-names clients.json --logo-templates logos/
  python main.py --input data/ --workflow analysis --ocr-engine easyocr
        """
    )
    
    # Required arguments
    parser.add_argument('--input', '-i', required=True,
                       help='Input file or directory path')
    
    # Optional arguments
    parser.add_argument('--output', '-o', default='output',
                       help='Output directory (default: output)')
    
    parser.add_argument('--workflow', '-w', 
                       choices=['cleansing', 'analysis', 'both'],
                       default='both',
                       help='Processing workflow (default: both)')
    
    parser.add_argument('--client-names', '-c',
                       help='Path to JSON file containing client names to mask')
    
    parser.add_argument('--logo-templates', '-l',
                       help='Directory containing logo template images')
    
    parser.add_argument('--ocr-engine', 
                       choices=['pytesseract', 'easyocr'],
                       default='pytesseract',
                       help='OCR engine to use (default: pytesseract)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    parser.add_argument('--version', action='version', version='Security File Processor 1.0.0')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize processor
        processor = SecurityFileProcessor(
            output_dir=args.output,
            client_names_file=args.client_names,
            logo_templates_dir=args.logo_templates,
            ocr_engine=args.ocr_engine
        )
        
        # Process files
        input_path = Path(args.input)
        
        if input_path.suffix.lower() == '.zip':
            results = processor.process_zip_file(input_path)
        else:
            results = processor.process_files(input_path, workflow=args.workflow)
        
        # Print summary
        print("\n" + "="*60)
        print("PROCESSING COMPLETED SUCCESSFULLY")
        print("="*60)
        
        summary = results['processing_summary']
        print(f"Total files processed: {summary['total_files']}")
        print(f"Successful: {summary['successful_files']}")
        print(f"Failed: {summary['failed_files']}")
        print(f"Success rate: {summary['success_rate']:.1f}%")
        
        if summary['total_processing_time']:
            print(f"Total processing time: {summary['total_processing_time']}")
        
        print(f"\nOutput directory: {args.output}")
        
        output_files = results['output_files']
        print(f"Cleaned files: {output_files['cleaned_files_directory']}")
        print(f"Reports: {output_files['reports_directory']}")
        print(f"Logs: {output_files['logs_directory']}")
        
        print(f"\nGenerated reports:")
        for report_type, file_path in output_files['generated_reports'].items():
            print(f"  - {report_type}: {file_path}")
        
        print("\n" + "="*60)
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
