import pytest
from io import BytesIO
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.document_processor import DocumentProcessor

def test_extract_text():
    processor = DocumentProcessor()
    
    # Test PDF extraction
    pdf_content = b"%PDF-1.4 test content"
    pdf_file = BytesIO(pdf_content)
    pdf_file.name = "test.pdf"
    
    # This should raise an error for invalid PDF
    with pytest.raises(Exception):
        processor.extract_text(pdf_file, "test.pdf")
    
    # Test text file extraction
    text_content = b"Test content"
    text_file = BytesIO(text_content)
    text_file.name = "test.txt"
    
    result = processor.extract_text(text_file, "test.txt")
    assert result == "Test content"

def test_chunk_text():
    processor = DocumentProcessor()
    
    # Test text chunking
    long_text = "This is a test sentence. " * 100
    chunks = processor.chunk_text(long_text)
    
    assert len(chunks) > 0
    assert all(len(chunk) > 0 for chunk in chunks) 