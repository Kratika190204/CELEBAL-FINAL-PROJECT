#!/usr/bin/env python3
"""
Test script for documentation handler functionality
"""

import os
import sys
from documentation_handler import DocumentationHandler

def test_documentation_handler():
    """Test the documentation handler functionality"""
    print("🧪 Testing Documentation Handler...")
    
    # Initialize handler
    handler = DocumentationHandler()
    
    # Test 1: Check if file exists
    print(f"📁 Checking if {handler.doc_file} exists...")
    if os.path.exists(handler.doc_file):
        file_size = os.path.getsize(handler.doc_file)
        print(f"   File exists with size: {file_size} bytes")
    else:
        print("   File does not exist")
    
    # Test 2: Create sample documentation
    print("📝 Creating sample documentation...")
    try:
        handler.create_sample_documentation()
        print("   ✅ Sample documentation created successfully")
    except Exception as e:
        print(f"   ❌ Error creating documentation: {e}")
        return False
    
    # Test 3: Verify file was created
    print("🔍 Verifying created file...")
    if os.path.exists(handler.doc_file):
        file_size = os.path.getsize(handler.doc_file)
        print(f"   File size after creation: {file_size} bytes")
        if file_size > 0:
            print("   ✅ File created successfully with content")
        else:
            print("   ❌ File is still empty")
            return False
    else:
        print("   ❌ File was not created")
        return False
    
    # Test 4: Try to read the document
    print("📖 Testing document reading...")
    try:
        from docx import Document
        doc = Document(handler.doc_file)
        paragraph_count = len([p for p in doc.paragraphs if p.text.strip()])
        print(f"   ✅ Document read successfully with {paragraph_count} non-empty paragraphs")
    except Exception as e:
        print(f"   ❌ Error reading document: {e}")
        return False
    
    print("🎉 All documentation tests passed!")
    return True

if __name__ == "__main__":
    success = test_documentation_handler()
    if success:
        print("\n✅ Documentation handler is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ Documentation handler has issues!")
        sys.exit(1) 