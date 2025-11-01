#!/usr/bin/env python3
"""
Test script to verify Security File Processor installation and dependencies.

Run this script to check if all required components are properly installed.
"""

import sys
import importlib
from pathlib import Path

def test_python_version():
    """Test Python version compatibility."""
    print("Testing Python version...")
    version = sys.version_info
    
    if version.major == 3 and 10 <= version.minor <= 13:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        if version.minor == 13:
            print("   Note: Python 3.13 detected - using latest compatible packages")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.10-3.13")
        return False

def test_dependencies():
    """Test required Python packages."""
    print("\nTesting Python dependencies...")
    
    required_packages = [
        'pytesseract',
        'easyocr', 
        'PIL',
        'pdfplumber',
        'PyPDF2',
        'openpyxl',
        'pptx',
        'spacy',
        'cv2',
        'numpy',
        'pandas',
        'sklearn',
        'matplotlib',
        'nltk'
    ]
    
    optional_packages = [
        'fitz',  # PyMuPDF
        'graphviz'
    ]
    
    success_count = 0
    
    for package in required_packages:
        try:
            if package == 'PIL':
                importlib.import_module('PIL')
            elif package == 'cv2':
                importlib.import_module('cv2')
            elif package == 'pptx':
                importlib.import_module('pptx')
            elif package == 'sklearn':
                importlib.import_module('sklearn')
            else:
                importlib.import_module(package)
            
            print(f"✅ {package} - Available")
            success_count += 1
        except ImportError:
            print(f"❌ {package} - Missing (required)")
    
    print(f"\nOptional packages:")
    for package in optional_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} - Available")
        except ImportError:
            print(f"⚠️  {package} - Missing (optional)")
    
    return success_count == len(required_packages)

def test_spacy_model():
    """Test spaCy language model."""
    print("\nTesting spaCy language model...")
    
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy English model - Available")
        return True
    except OSError:
        print("❌ spaCy English model - Missing")
        print("   Install with: python -m spacy download en_core_web_sm")
        return False
    except ImportError:
        print("❌ spaCy - Not installed")
        return False

def test_tesseract():
    """Test Tesseract OCR availability."""
    print("\nTesting Tesseract OCR...")
    
    try:
        import pytesseract
        from PIL import Image
        import numpy as np
        
        # Create a simple test image
        test_image = Image.fromarray(np.ones((50, 200, 3), dtype=np.uint8) * 255)
        
        # Try to run OCR
        pytesseract.image_to_string(test_image)
        print("✅ Tesseract OCR - Available")
        return True
    except Exception as e:
        print(f"❌ Tesseract OCR - Error: {str(e)}")
        print("   Make sure Tesseract is installed and in PATH")
        return False

def test_project_structure():
    """Test project directory structure."""
    print("\nTesting project structure...")
    
    required_dirs = ['src', 'data', 'output']
    required_files = [
        'main.py',
        'requirements.txt',
        'src/ocr.py',
        'src/pii_cleaner.py',
        'src/logo_remover.py',
        'src/analyzer.py',
        'src/report_generator.py'
    ]
    
    all_good = True
    
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"✅ Directory: {directory}")
        else:
            print(f"❌ Directory: {directory} - Missing")
            all_good = False
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ File: {file_path}")
        else:
            print(f"❌ File: {file_path} - Missing")
            all_good = False
    
    return all_good

def test_basic_functionality():
    """Test basic functionality of core modules."""
    print("\nTesting basic functionality...")
    
    try:
        # Add src to path
        sys.path.append(str(Path(__file__).parent / 'src'))
        
        # Test PII cleaner
        from pii_cleaner import PIICleaner
        cleaner = PIICleaner()
        test_text = "John Doe's email is john.doe@example.com"
        result = cleaner.remove_pii(test_text)
        print("✅ PII Cleaner - Working")
        
        # Test security analyzer
        from analyzer import SecurityAnalyzer
        analyzer = SecurityAnalyzer()
        analysis = analyzer.analyze_security_content("Allow user access to S3 bucket")
        print("✅ Security Analyzer - Working")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("Security File Processor - Installation Test")
    print("=" * 50)
    
    tests = [
        ("Python Version", test_python_version),
        ("Dependencies", test_dependencies),
        ("spaCy Model", test_spacy_model),
        ("Tesseract OCR", test_tesseract),
        ("Project Structure", test_project_structure),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} - Error: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The Security File Processor is ready to use.")
        print("\nTry running:")
        print("  python main.py --input data/sample_security_document.txt")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please fix the issues above.")
        print("\nInstallation steps:")
        print("1. pip install -r requirements.txt")
        print("2. python -m spacy download en_core_web_sm")
        print("3. Install Tesseract OCR for your operating system")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

