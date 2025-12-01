import sys
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import split_sentence_by_words, split_long_inputs


def test_split_sentence_by_words_respects_limit():
    sentence = "one two three four five"
    chunks = split_sentence_by_words(sentence, max_words=2)
    assert chunks == ["one two", "three four", "five"]


def test_split_long_inputs_adds_suffix_and_stats():
    rows = [
        ("doc1", "1", "one two three four"),
        ("doc1", "2", "short")
    ]
    expanded, stats = split_long_inputs(rows, max_sentence_words=2, logger=None)
    assert expanded == [
        ("doc1", "1.1", "one two"),
        ("doc1", "1.2", "three four"),
        ("doc1", "2", "short"),
    ]
    assert stats["split_sentences"] == 1
    assert stats["total_chunks"] == 3


def test_split_long_inputs_with_word_limit():
    rows = [("docX", "1", "a b c d e f g")]
    expanded, stats = split_long_inputs(rows, max_sentence_words=3, logger=None)
    assert expanded == [
        ("docX", "1.1", "a b c"),
        ("docX", "1.2", "d e f"),
        ("docX", "1.3", "g"),
    ]
    assert stats["split_sentences"] == 1
    assert stats["total_chunks"] == 3

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
