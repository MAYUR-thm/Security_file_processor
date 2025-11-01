#!/usr/bin/env python3
"""
Example Usage Script for Security File Processor

This script demonstrates various ways to use the Security File Processor
for different scenarios and use cases.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

def example_1_single_file_processing():
    """Example 1: Process a single file with both workflows."""
    print("Example 1: Single File Processing")
    print("-" * 40)
    
    from main import SecurityFileProcessor
    
    # Initialize processor
    processor = SecurityFileProcessor(
        output_dir="example_output",
        client_names_file="data/client_names.json"
    )
    
    # Process the sample document
    results = processor.process_files(
        input_path="data/sample_security_document.txt",
        workflow="both"
    )
    
    print(f"✅ Processed {results['processing_summary']['total_files']} files")
    print(f"📊 Success rate: {results['processing_summary']['success_rate']:.1f}%")
    print(f"📁 Output directory: {results['output_files']['reports_directory']}")
    
    return results

def example_2_pii_cleaning_only():
    """Example 2: PII cleaning workflow only."""
    print("\nExample 2: PII Cleaning Only")
    print("-" * 40)
    
    from pii_cleaner import PIICleaner
    
    # Sample text with PII
    sample_text = """
    Contact Information:
    Name: John Doe
    Email: john.doe@acme.com
    Phone: (555) 123-4567
    SSN: 123-45-6789
    Address: 123 Main Street, Anytown, NY 12345
    
    Meeting with Acme Corporation scheduled for tomorrow.
    """
    
    # Initialize PII cleaner with client names
    cleaner = PIICleaner(client_names_file="data/client_names.json")
    
    # Analyze PII risk
    risk_analysis = cleaner.analyze_pii_risk(sample_text)
    print(f"🔍 PII Risk Level: {risk_analysis['risk_level']}")
    print(f"📊 Risk Score: {risk_analysis['risk_score']}")
    print(f"🔢 PII Items Found: {risk_analysis['pii_count']}")
    
    # Clean the text
    cleaned_result = cleaner.remove_pii(sample_text, mask_mode="replace")
    print(f"🧹 Cleaned Text Preview:")
    print(cleaned_result['cleaned_text'][:200] + "...")
    
    return cleaned_result

def example_3_security_analysis_only():
    """Example 3: Security analysis workflow only."""
    print("\nExample 3: Security Analysis Only")
    print("-" * 40)
    
    from analyzer import SecurityAnalyzer
    
    # Sample security content
    security_text = """
    IAM Policy Configuration:
    - Allow user developers to access S3 bucket company-data
    - Deny role junior-developers from deleting RDS production database
    
    Firewall Rules:
    - Allow TCP port 443 from 0.0.0.0/0 to web servers
    - Deny TCP port 22 from external networks
    - Block ICMP from source IP 10.0.0.0/8
    
    Security Alert:
    High severity intrusion detected from IP 192.168.1.100
    CVE-2023-1234: Critical vulnerability in Apache HTTP Server
    """
    
    # Initialize analyzer
    analyzer = SecurityAnalyzer()
    
    # Perform analysis
    analysis = analyzer.analyze_security_content(security_text)
    
    print(f"🔍 Security Categories Found: {', '.join(analysis['summary']['categories_found'])}")
    print(f"📊 Confidence Score: {analysis['summary']['confidence_score']}")
    print(f"🔢 Total Findings: {analysis['summary']['total_findings']}")
    
    print(f"\n🎯 Key Findings:")
    for finding in analysis['key_findings'][:3]:
        print(f"  • {finding}")
    
    print(f"\n💡 Recommendations:")
    for rec in analysis['recommendations'][:3]:
        print(f"  • {rec}")
    
    return analysis

def example_4_batch_processing():
    """Example 4: Batch processing multiple files."""
    print("\nExample 4: Batch Processing")
    print("-" * 40)
    
    from main import SecurityFileProcessor
    
    # Create some sample files for demonstration
    sample_files = [
        ("data/policy1.txt", "IAM Policy: Allow user access to S3\nFirewall: Deny port 22"),
        ("data/policy2.txt", "CVE-2023-1234: Critical vulnerability\nAlert: Intrusion detected"),
        ("data/policy3.txt", "PCI DSS Compliance required\nEncryption: AES-256 enabled")
    ]
    
    # Create sample files
    for file_path, content in sample_files:
        Path(file_path).parent.mkdir(exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(content)
    
    # Initialize processor
    processor = SecurityFileProcessor(output_dir="batch_output")
    
    # Process all files in data directory
    results = processor.process_files(
        input_path="data",
        workflow="analysis"  # Only security analysis
    )
    
    print(f"✅ Batch processed {results['processing_summary']['total_files']} files")
    print(f"📊 Success rate: {results['processing_summary']['success_rate']:.1f}%")
    
    # Clean up sample files
    for file_path, _ in sample_files:
        Path(file_path).unlink(missing_ok=True)
    
    return results

def example_5_programmatic_usage():
    """Example 5: Using individual components programmatically."""
    print("\nExample 5: Programmatic Component Usage")
    print("-" * 40)
    
    # Import individual components
    from ocr import extract_text_from_file
    from pii_cleaner import remove_pii
    from analyzer import analyze_security_content
    from report_generator import generate_report
    
    # Sample workflow
    sample_text = """
    Security Policy Document
    Contact: admin@company.com
    
    IAM Policy: Allow users to access S3 bucket
    Firewall Rule: Deny TCP port 22 from external
    Alert: CVE-2023-1234 detected
    """
    
    print("🔄 Step 1: Text extraction (simulated)")
    extracted_text = sample_text
    
    print("🔄 Step 2: PII cleaning")
    cleaned_text = remove_pii(extracted_text, mask_mode="replace")
    
    print("🔄 Step 3: Security analysis")
    security_analysis = analyze_security_content(cleaned_text)
    
    print("🔄 Step 4: Report generation")
    results = [{
        "file_name": "sample_document.txt",
        "file_type": "txt",
        "file_description": "Security policy document",
        "security_analysis": security_analysis,
        "pii_analysis": {"risk_level": "LOW", "pii_count": 1}
    }]
    
    report_files = generate_report(results, output_dir="programmatic_output")
    
    print(f"✅ Generated reports:")
    for report_type, file_path in report_files.items():
        print(f"  📄 {report_type}: {file_path}")
    
    return report_files

def example_6_logo_detection():
    """Example 6: Logo detection and masking (requires image file)."""
    print("\nExample 6: Logo Detection (Image Processing)")
    print("-" * 40)
    
    from logo_remover import LogoRemover
    import numpy as np
    from PIL import Image
    
    # Create a simple test image (since we don't have real images)
    print("📷 Creating test image...")
    test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    pil_image = Image.fromarray(test_image)
    test_image_path = "data/test_image.png"
    pil_image.save(test_image_path)
    
    # Initialize logo remover
    logo_remover = LogoRemover()
    
    # Detect and mask logos
    result = logo_remover.detect_and_mask_logo(
        image_path=test_image_path,
        output_path="output/masked_image.png",
        mask_method="blur"
    )
    
    print(f"🔍 Logo detections: {result['detection_count']}")
    print(f"✅ Processed image saved: {result.get('processed_image_path', 'N/A')}")
    
    # Clean up test image
    Path(test_image_path).unlink(missing_ok=True)
    
    return result

def main():
    """Run all examples."""
    print("🚀 Security File Processor - Usage Examples")
    print("=" * 60)
    
    examples = [
        example_1_single_file_processing,
        example_2_pii_cleaning_only,
        example_3_security_analysis_only,
        example_4_batch_processing,
        example_5_programmatic_usage,
        example_6_logo_detection
    ]
    
    results = {}
    
    for i, example_func in enumerate(examples, 1):
        try:
            print(f"\n{'='*60}")
            result = example_func()
            results[f"example_{i}"] = {"success": True, "result": result}
        except Exception as e:
            print(f"❌ Example {i} failed: {str(e)}")
            results[f"example_{i}"] = {"success": False, "error": str(e)}
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 EXAMPLES SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for r in results.values() if r["success"])
    total = len(results)
    
    for example_name, result in results.items():
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        print(f"{status} - {example_name}")
        if not result["success"]:
            print(f"    Error: {result['error']}")
    
    print(f"\n🎯 Overall: {successful}/{total} examples completed successfully")
    
    if successful == total:
        print("\n🎉 All examples completed! The Security File Processor is working correctly.")
    else:
        print(f"\n⚠️  Some examples failed. Check the error messages above.")
    
    print(f"\n💡 Next steps:")
    print(f"  • Run: python main.py --input data/sample_security_document.txt")
    print(f"  • Check output files in the generated directories")
    print(f"  • Customize client_names.json for your specific needs")
    print(f"  • Add logo templates to improve logo detection")

if __name__ == "__main__":
    main()
