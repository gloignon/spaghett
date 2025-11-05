import sys
from pathlib import Path
import pytest


# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from scorer import score_autoregressive, score_masked_lm, score_masked_lm_l2r
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM

import torch

@pytest.fixture
def ar_model():
    """Load a small autoregressive model for testing."""
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_attentions=True
    )
    return tokenizer, model


@pytest.fixture
def mlm_model():
    """Load a small masked language model for testing."""
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(
        model_name,
        output_attentions=True
    )
    return tokenizer, model


def test_score_autoregressive_basic(ar_model):
    """Test basic autoregressive scoring."""
    tokenizer, model = ar_model
    sentence = "The cat sat on the mat."
    
    result = score_autoregressive(
        sentence=sentence,
        left_context="",
        tokenizer=tokenizer,
        model=model,
        top_k=3,
        lookahead_n=0,
        output_attentions=False
    )
    
    assert len(result.scored_tokens) > 0
    assert len(result.surprisals) == len(result.scored_tokens)
    assert len(result.entropies) == len(result.scored_tokens)
    assert all(s >= 0 for s in result.surprisals if not math.isnan(s))


def test_score_autoregressive_with_lookahead(ar_model):
    """Test autoregressive scoring with greedy lookahead."""
    tokenizer, model = ar_model
    sentence = "Hello world"
    
    result = score_autoregressive(
        sentence=sentence,
        left_context="",
        tokenizer=tokenizer,
        model=model,
        top_k=3,
        lookahead_n=2,
        lookahead_strategy='greedy',
        output_attentions=False
    )
    
    assert len(result.scored_tokens) > 0
    assert len(result.pred_columns) == len(result.scored_tokens)
    # Each pred column should have top_k + lookahead tokens
    for pred_col in result.pred_columns:
        assert len(pred_col) >= 3  # At least top_k tokens


def test_score_masked_lm(mlm_model):
    """Test masked language model scoring."""
    tokenizer, model = mlm_model
    sentence = "The cat sat on the mat."
    
    result = score_masked_lm(
        sentence=sentence,
        tokenizer=tokenizer,
        model=model,
        top_k=3,
        output_attentions=False
    )
    
    assert len(result.scored_tokens) > 0
    assert len(result.surprisals) == len(result.scored_tokens)
    assert len(result.entropies) == len(result.scored_tokens)
    assert all(s >= 0 for s in result.surprisals if not math.isnan(s))


def test_score_masked_lm_l2r(mlm_model):
    """Test masked LM with left-to-right within-word scoring."""
    tokenizer, model = mlm_model
    sentence = "The quick brown fox"
    
    result = score_masked_lm_l2r(
        sentence=sentence,
        tokenizer=tokenizer,
        model=model,
        top_k=3,
        output_attentions=False
    )
    
    assert len(result.scored_tokens) > 0
    assert len(result.surprisals) == len(result.scored_tokens)
    assert len(result.pred_columns) == len(result.scored_tokens)


def test_empty_sentence_ar(ar_model):
    """Test that empty sentences are handled gracefully."""
    tokenizer, model = ar_model
    
    result = score_autoregressive(
        sentence="",
        left_context="",
        tokenizer=tokenizer,
        model=model,
        output_attentions=False
    )
    
    assert len(result.scored_tokens) == 0
    assert len(result.surprisals) == 0


def test_empty_sentence_mlm(mlm_model):
    """Test that empty sentences are handled gracefully in MLM."""
    tokenizer, model = mlm_model
    
    result = score_masked_lm(
        sentence="",
        tokenizer=tokenizer,
        model=model,
        output_attentions=False
    )
    
    assert len(result.scored_tokens) == 0
    assert len(result.surprisals) == 0


def test_attention_extraction_ar(ar_model):
    """Test that attention extraction works."""
    tokenizer, model = ar_model
    sentence = "Hello world"
    
    result = score_autoregressive(
        sentence=sentence,
        left_context="",
        tokenizer=tokenizer,
        model=model,
        output_attentions=True
    )
    
    assert result.attention_tuples is not None
    # Should have some attention tuples
    assert len(result.attention_tuples) > 0


import math  # Add this import at the top


if __name__ == "__main__":
    pytest.main([__file__, "-v"])