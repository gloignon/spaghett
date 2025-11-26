def test_write_output_tsv_cf_surprisal(ar_model):
    """Test writing AR results to TSV with counterfactual surprisal for top_k tokens."""
    tokenizer, model = ar_model
    sentence = "The cat sat on the mat."
    result = score_autoregressive(
        sentence=sentence,
        left_context="",
        tokenizer=tokenizer,
        model=model,
        top_k=3,
        lookahead_n=0
    )
    results = []
    for idx, (raw_token, token, is_special, surp, ent, preds) in enumerate(
        zip(result.raw_tokens, result.scored_tokens, result.is_special_flags, result.surprisals, result.entropies, result.pred_columns), 1):
        row = {
            'doc_id': 'doc1',
            'sentence_id': '1',
            'token_index': idx,
            'token': raw_token,
            'token_decoded': token,
            'is_special': is_special,
            'surprisal_bits': '' if math.isnan(surp) else f'{surp:.4f}',
            'entropy_bits': '' if math.isnan(ent) else f'{ent:.4f}'
        }
        for i in range(1, 4):
            row[f'pred_alt_{i}'] = preds[i - 1] if i - 1 < len(preds) else ''
        results.append(row)
    import tempfile
    import csv
    with tempfile.TemporaryDirectory() as tmpdir:
        tsv_path = os.path.join(tmpdir, "test_output.tsv")
        from utils import write_output
        write_output(tsv_path, results, top_k=3, lookahead_n=0, mode='ar', top_k_cf_surprisal=True, output_format='tsv')
        # Read back and check
        with open(tsv_path, encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            rows = list(reader)
            assert len(rows) == len(results)
            # Check pred_alt_1 format
            val = rows[0]['pred_alt_1']
            assert isinstance(val, str) and '|' in val
            # Check that the first row matches what we wrote (except pred_alt columns, which are formatted)
            orig_row = results[0]
            for col in ['doc_id', 'sentence_id', 'token_index', 'token', 'token_decoded', 'is_special', 'surprisal_bits', 'entropy_bits']:
                assert str(rows[0][col]) == str(orig_row[col])
import tempfile
import os
import pandas as pd
import sys
from pathlib import Path
import pytest
import math


# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from scorer import score_autoregressive, score_masked_lm, score_masked_lm_l2r
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM


@pytest.fixture
def ar_model():
    """Load a small autoregressive model for testing."""
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name
    )
    return tokenizer, model


@pytest.fixture
def mlm_model():
    """Load a small masked language model for testing."""
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(
        model_name
    )
    return tokenizer, model


def test_write_output_parquet(ar_model):
    """Test writing AR results to parquet file and reading it back."""
    tokenizer, model = ar_model
    sentence = "The cat sat on the mat."
    result = score_autoregressive(
        sentence=sentence,
        left_context="",
        tokenizer=tokenizer,
        model=model,
        top_k=3,
        lookahead_n=0
    )
    # Build a minimal results list as expected by write_output
    results = []
    for idx, (raw_token, token, is_special, surp, ent, preds) in enumerate(
        zip(result.raw_tokens, result.scored_tokens, result.is_special_flags, result.surprisals, result.entropies, result.pred_columns), 1):
        row = {
            'doc_id': 'doc1',
            'sentence_id': '1',
            'token_index': idx,
            'token': raw_token,
            'token_decoded': token,
            'is_special': is_special,
            'surprisal_bits': '' if math.isnan(surp) else f'{surp:.4f}',
            'entropy_bits': '' if math.isnan(ent) else f'{ent:.4f}'
        }
        for i in range(1, 4):
            row[f'pred_alt_{i}'] = preds[i - 1] if i - 1 < len(preds) else ''
        results.append(row)
    # Write to a temporary parquet file
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = os.path.join(tmpdir, "test_output.parquet")
        from utils import write_output
        write_output(parquet_path, results, top_k=3, lookahead_n=0, mode='ar', top_k_cf_surprisal=True, output_format='parquet')
        # Read back and check
        df = pd.read_parquet(parquet_path)
        assert not df.empty
        assert 'pred_alt_1' in df.columns
        # Check that pred_alt_1 is a string with token|surprisal
        val = df.loc[0, 'pred_alt_1']
        assert isinstance(val, str) and '|' in val
        # Check that the first row matches what we wrote
        orig_row = results[0]
        for col in ['doc_id', 'sentence_id', 'token_index', 'token', 'token_decoded', 'is_special', 'surprisal_bits', 'entropy_bits']:
            assert str(df.loc[0, col]) == str(orig_row[col])
    """Test basic autoregressive scoring."""
    tokenizer, model = ar_model
    sentence = "The cat sat on the mat."
    
    result = score_autoregressive(
        sentence=sentence,
        left_context="",
        tokenizer=tokenizer,
        model=model,
        top_k=3,
        lookahead_n=0
    )
    
    assert len(result.scored_tokens) > 0
    assert len(result.surprisals) == len(result.scored_tokens)
    assert len(result.entropies) == len(result.scored_tokens)
    assert all(s >= 0 for s in result.surprisals if not math.isnan(s))
    # Check top_k predictions include token and counterfactual surprisal
    for pred_col in result.pred_columns:
        for i in range(3):
            val = pred_col[i]
            assert isinstance(val, tuple)
            token, cf_surprisal = val
            assert isinstance(token, str)
            assert isinstance(cf_surprisal, float)


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
        lookahead_strategy='greedy'
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
        top_k=3
    )
    
    assert len(result.scored_tokens) > 0
    assert len(result.surprisals) == len(result.scored_tokens)
    assert len(result.entropies) == len(result.scored_tokens)
    assert all(s >= 0 for s in result.surprisals if not math.isnan(s))
    # Check top_k predictions include token and counterfactual surprisal
    for pred_col in result.pred_columns:
        for i in range(3):
            val = pred_col[i]
            assert isinstance(val, tuple)
            token, cf_surprisal = val
            assert isinstance(token, str)
            assert isinstance(cf_surprisal, float)


def test_score_masked_lm_l2r(mlm_model):
    """Test masked LM with left-to-right within-word scoring."""
    tokenizer, model = mlm_model
    sentence = "The quick brown fox"
    
    result = score_masked_lm_l2r(
        sentence=sentence,
        tokenizer=tokenizer,
        model=model,
        top_k=3
    )
    
    assert len(result.scored_tokens) > 0
    assert len(result.surprisals) == len(result.scored_tokens)
    assert len(result.pred_columns) == len(result.scored_tokens)
    # Check top_k predictions include token and counterfactual surprisal
    for pred_col in result.pred_columns:
        for i in range(3):
            val = pred_col[i]
            assert isinstance(val, tuple)
            token, cf_surprisal = val
            assert isinstance(token, str)
            assert isinstance(cf_surprisal, float)


def test_empty_sentence_ar(ar_model):
    """Test that empty sentences are handled gracefully."""
    tokenizer, model = ar_model
    
    result = score_autoregressive(
        sentence="",
        left_context="",
        tokenizer=tokenizer,
        model=model
    )
    
    assert len(result.scored_tokens) == 0
    assert len(result.surprisals) == 0


def test_empty_sentence_mlm(mlm_model):
    """Test that empty sentences are handled gracefully in MLM."""
    tokenizer, model = mlm_model
    
    result = score_masked_lm(
        sentence="",
        tokenizer=tokenizer,
        model=model
    )
    
    assert len(result.scored_tokens) == 0
    assert len(result.surprisals) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])