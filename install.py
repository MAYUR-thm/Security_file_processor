#!/usr/bin/env python3
"""
Installation script for Security File Processor

This script automates the installation of all required dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"[RUNNING] {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"[OK] {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} - Failed")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and 10 <= version.minor <= 13:
        print(f"[OK] Python {version.major}.{version.minor}.{version.micro} - Compatible")
        if version.minor == 13:
            print("   [INFO] Python 3.13 detected - optimized for latest features")
        return True
    else:
        print(f"[ERROR] Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.10-3.13")
        return False

def install_pip_packages():
    """Install Python packages from requirements.txt."""
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install -r requirements.txt", "Installing Python packages"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    return True

def install_spacy_model():
    """Install spaCy English language model."""
    return run_command("python -m spacy download en_core_web_sm", 
                      "Installing spaCy English model")

def check_tesseract():
    """Check if Tesseract is available."""
    try:
        result = subprocess.run("tesseract --version", shell=True, 
                              capture_output=True, text=True)
        print("[OK] Tesseract OCR - Available")
        return True
    except:
        print("[WARNING] Tesseract OCR - Not found")
        print("   Please install Tesseract manually:")
        print("   Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   macOS: brew install tesseract")
        print("   Linux: sudo apt-get install tesseract-ocr")
        return False

def install_optional_packages():
    """Install optional packages."""
    optional_commands = [
        ("pip install graphviz", "Installing Graphviz Python wrapper (optional)"),
    ]
    
    for command, description in optional_commands:
        run_command(command, description)  # Don't fail if optional packages fail

def main():
    """Main installation process."""
    print("Security File Processor - Installation Script")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        print("\n[ERROR] Installation failed: Incompatible Python version")
        sys.exit(1)
    
    # Install pip packages
    print(f"\n[STEP] Installing Python packages...")
    if not install_pip_packages():
        print("\n[ERROR] Installation failed: Could not install Python packages")
        sys.exit(1)
    
    # Install spaCy model
    print(f"\n[STEP] Installing spaCy language model...")
    if not install_spacy_model():
        print("[WARNING] spaCy model installation failed")
        print("   You can install it manually later with:")
        print("   python -m spacy download en_core_web_sm")
    
    # Check Tesseract
    print(f"\n[STEP] Checking Tesseract OCR...")
    check_tesseract()
    
    # Install optional packages
    print(f"\n[STEP] Installing optional packages...")
    install_optional_packages()
    
    # Run verification
    print(f"\n[STEP] Running installation verification...")
    if Path("test_installation.py").exists():
        run_command("python test_installation.py", "Verifying installation")
    
    print(f"\n" + "=" * 60)
    print("[SUCCESS] Installation completed!")
    print("=" * 60)
    
    print(f"\nNext steps:")
    print(f"1. If Tesseract failed, install it manually for your OS")
    print(f"2. Test the installation: python test_installation.py")
    print(f"3. Run examples: python example_usage.py")
    print(f"4. Process your first file: python main.py --input data/sample_security_document.txt")
    
    print(f"\nDocumentation:")
    print(f"- README.md - Complete usage guide")
    print(f"- PROJECT_SUMMARY.md - Technical overview")
    print(f"- example_usage.py - Usage examples")

if __name__ == "__main__":
    main()

