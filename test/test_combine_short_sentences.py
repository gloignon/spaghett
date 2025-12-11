"""
Tests for combining short sentences functionality.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from utils import combine_short_sentences


def test_combine_short_sentences_basic():
    """Test basic sentence combining."""
    rows = [
        ('doc1', '1', 'Short.'),
        ('doc1', '2', 'This is a longer sentence with more words.'),
        ('doc1', '3', 'Tiny'),
        ('doc1', '4', 'Also quite short here'),
    ]
    
    result, stats = combine_short_sentences(rows, min_sentence_words=3)
    
    # First sentence (1 word) should combine with second
    # Third sentence (1 word) should combine with fourth
    assert len(result) == 2
    assert result[0] == ('doc1', '1+2', 'Short. This is a longer sentence with more words.')
    assert result[1] == ('doc1', '3+4', 'Tiny Also quite short here')
    assert stats['combined_sentences'] == 2
    assert stats['total_units'] == 2


def test_combine_respects_doc_boundaries():
    """Test that sentences are not combined across different documents."""
    rows = [
        ('doc1', '1', 'Short.'),
        ('doc2', '1', 'Different document.'),
    ]
    
    result, stats = combine_short_sentences(rows, min_sentence_words=3)
    
    # Should not combine across doc boundaries
    assert len(result) == 2
    assert result[0] == ('doc1', '1', 'Short.')
    assert result[1] == ('doc2', '1', 'Different document.')
    assert stats['combined_sentences'] == 0


def test_combine_last_sentence_not_combined():
    """Test that the last sentence in a document is not combined if short."""
    rows = [
        ('doc1', '1', 'This is a long enough sentence.'),
        ('doc1', '2', 'Short.'),
    ]
    
    result, stats = combine_short_sentences(rows, min_sentence_words=3)
    
    # Last sentence should remain as is even if short
    assert len(result) == 2
    assert result[1] == ('doc1', '2', 'Short.')
    assert stats['combined_sentences'] == 0


def test_combine_disabled_when_zero():
    """Test that combining is disabled when min_sentence_words is 0."""
    rows = [
        ('doc1', '1', 'A'),
        ('doc1', '2', 'B'),
    ]
    
    result, stats = combine_short_sentences(rows, min_sentence_words=0)
    
    assert len(result) == 2
    assert result == rows
    assert stats['combined_sentences'] == 0


def test_combine_multiple_consecutive():
    """Test combining multiple consecutive short sentences."""
    rows = [
        ('doc1', '1', 'A'),
        ('doc1', '2', 'B'),
        ('doc1', '3', 'C'),
        ('doc1', '4', 'This is a longer sentence.'),
    ]
    
    result, stats = combine_short_sentences(rows, min_sentence_words=3)
    
    # A+B (combined), then C+D (combined)
    assert len(result) == 2
    assert result[0] == ('doc1', '1+2', 'A B')
    assert result[1] == ('doc1', '3+4', 'C This is a longer sentence.')
    assert stats['combined_sentences'] == 2


def test_combine_preserves_long_sentences():
    """Test that sentences meeting the minimum are not combined."""
    rows = [
        ('doc1', '1', 'This sentence has enough words already.'),
        ('doc1', '2', 'This one also has enough.'),
    ]
    
    result, stats = combine_short_sentences(rows, min_sentence_words=3)
    
    assert len(result) == 2
    assert result == rows
    assert stats['combined_sentences'] == 0


if __name__ == '__main__':
    test_combine_short_sentences_basic()
    test_combine_respects_doc_boundaries()
    test_combine_last_sentence_not_combined()
    test_combine_disabled_when_zero()
    test_combine_multiple_consecutive()
    test_combine_preserves_long_sentences()
    print("✓ All tests passed!")
