"""Test that input files with extra columns work correctly."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import load_input_data

def test_load_with_extra_columns():
    """Test loading a file with extra columns beyond the required ones."""
    # Test with passepartout.tsv which has extra columns
    input_file = os.path.join(os.path.dirname(__file__), '..', 'in_private', 'passepartout.tsv')
    
    if not os.path.exists(input_file):
        print(f"Skipping test: {input_file} not found")
        return
    
    print(f"Testing with: {input_file}")
    
    # Load in sentences mode
    data = load_input_data(input_file, format_type='sentences')
    
    # Verify we got data
    assert len(data) > 0, "Should have loaded some data"
    
    # Check structure of first few items
    print(f"\nLoaded {len(data)} sentences")
    print("\nFirst 3 entries:")
    for i, (doc_id, sent_id, sentence) in enumerate(data[:3]):
        print(f"  {i+1}. doc_id='{doc_id}', sent_id='{sent_id}', sentence='{sentence[:50]}...'")
    
    # Verify structure
    for doc_id, sent_id, sentence in data[:10]:
        assert isinstance(doc_id, str) and doc_id, "doc_id should be non-empty string"
        assert isinstance(sent_id, str) and sent_id, "sent_id should be non-empty string"
        assert isinstance(sentence, str) and sentence, "sentence should be non-empty string"
    
    print("\n✅ Test passed: Extra columns are properly ignored")

if __name__ == '__main__':
    test_load_with_extra_columns()
