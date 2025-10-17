from __future__ import annotations
import unittest
import sys
import os
from typing import List, Tuple
import numpy as np

# Add src directory to path before importing main
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

#!/usr/bin/env python3
"""
test_minicons_compare.py

Test that surprisal values from main.py match those from the minicons library.

Usage:
    python -m pytest test_minicons_compare.py -v
    python -m unittest test_minicons_compare.py -v
    python test_minicons_compare.py  # runs as unittest
"""


# Test configuration
TEST_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "This is a short test sentence.",
    "Minicons surprisal should be comparable to main.py output.",
]

THRESHOLD = 0.1  # Maximum allowed difference in bits
AR_MODEL = "gpt2"
MLM_MODEL = "bert-base-uncased"
PLL_METRIC = "original"


def compute_with_main(texts: List[str], model_name: str, mode: str) -> List[Tuple[List[str], List[float]]]:
    """
    Compute surprisal using main.py's process_sentences function.
    
    Returns:
        List of (tokens, surprisals) tuples, one per text
    """
    import importlib
    
    try:
        main = importlib.import_module("main")
    except Exception as e:
        raise RuntimeError(f"Failed to import main.py: {e}")
    
    if not hasattr(main, "process_sentences"):
        raise RuntimeError(
            "main.py does not export process_sentences(). "
            "Please ensure your main.py has this function."
        )
    
    # Call process_sentences
    results = main.process_sentences(
        sentences=texts,
        mode=mode,
        model_name=model_name,
        left_context='',
        top_k=1,
        lookahead_n=0,
        doc_ids=[f"doc_{i}" for i in range(len(texts))],
        sentence_ids=[f"sent_{i}" for i in range(len(texts))],
        progress=False
    )
    
    # Group results by sentence
    output = []
    current_sent_id = None
    current_tokens = []
    current_surprisals = []
    
    for row in results:
        if row['sentence_id'] != current_sent_id:
            if current_tokens:
                output.append((current_tokens, current_surprisals))
            current_sent_id = row['sentence_id']
            current_tokens = []
            current_surprisals = []
        
        current_tokens.append(row['token_decoded'])
        # Handle empty surprisal values
        if row['surprisal_bits'] != '':
            current_surprisals.append(float(row['surprisal_bits']))
        else:
            current_surprisals.append(np.nan)
    
    if current_tokens:
        output.append((current_tokens, current_surprisals))
    
    return output


def compute_with_minicons(texts: List[str], model_name: str, mode: str, pll_metric: str = 'original') -> List[Tuple[List[str], List[float]]]:
    """
    Compute surprisal using minicons library.
    
    Args:
        texts: List of sentences to score
        model_name: HuggingFace model name
        mode: 'ar' for autoregressive or 'mlm' for masked language model
        pll_metric: For MLM only - 'original' or 'within_word_l2r'
    
    Returns:
        List of (tokens, surprisals) tuples, one per text
    """
    try:
        from minicons import scorer
    except ImportError as e:
        raise RuntimeError(
            "Failed to import minicons. Install it with: pip install minicons\n"
            f"Error: {e}"
        )
    
    # Initialize scorer based on mode
    try:
        if mode == 'ar':
            model = scorer.IncrementalLMScorer(model_name, 'cpu')
        elif mode == 'mlm':
            model = scorer.MaskedLMScorer(model_name, 'cpu')
        else:
            raise ValueError(f"Invalid mode: {mode}")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize minicons scorer: {e}")
    
    output = []
    
    if mode == 'ar':
        # Use prepare_text and compute_stats for autoregressive models
        for text in texts:
            # Get tokenized text (including special tokens)
            tokens_batch = model.prepare_text([text])
            
            # Get token strings (including special tokens)
            token_ids = model.tokenizer([text], add_special_tokens=True)['input_ids'][0]
            token_strings = model.tokenizer.convert_ids_to_tokens(token_ids)
            
            # Compute log probabilities in base 2, then negate to get surprisal
            # compute_stats returns log probabilities, not surprisal
            logprobs = model.compute_stats(tokens_batch, base_two=True)
            
            # Convert log probabilities to surprisal (negate them)
            surprisal_values = [-lp for lp in logprobs[0]]
            
            # In AR models, minicons returns surprisal for tokens 1..N
            # (the first token has no context, so no surprisal is computed)
            # We need to prepend NaN for the first token to align with main.py
            surprisal_values = [np.nan] + surprisal_values
            
            output.append((token_strings, surprisal_values))
    
    elif mode == 'mlm':
        # Use sequence_score with identity reduction for masked models
        for text in texts:
            # Get ALL tokens including special tokens
            toks = model.tokenizer([text], add_special_tokens=True)
            token_ids = toks['input_ids'][0]
            token_strings = model.tokenizer.convert_ids_to_tokens(token_ids)
            
            # Get surprisal scores using sequence_score with identity reduction
            # sequence_score returns surprisal in NATS by default, need to convert to BITS
            raw_scores = model.sequence_score(
                [text],
                reduction=lambda x: x,
                PLL_metric=pll_metric
            )
            
            # Convert to list
            surprisal_values = raw_scores[0].tolist()
            
            # Convert from nats to bits by multiplying with log2(e) ≈ 1.4427
            # OR check if values are negative and negate first
            if surprisal_values and any(v < 0 for v in surprisal_values if not np.isnan(v)):
                # Values are negative log probs, negate them first
                surprisal_values = [-v for v in surprisal_values]
            
            # Convert from nats to bits: bits = nats / ln(2) = nats * log2(e)
            surprisal_values = [v / np.log(2) if not np.isnan(v) else v for v in surprisal_values]
            
            # Check if we need to pad for special tokens
            # minicons may not return surprisals for special tokens in some cases
            if len(surprisal_values) < len(token_strings):
                # Pad with NaN for missing special token surprisals
                cls_tok = model.tokenizer.cls_token
                sep_tok = model.tokenizer.sep_token
                
                padded_surprisals = []
                surprisal_idx = 0
                for tok in token_strings:
                    if tok in [cls_tok, sep_tok] and surprisal_idx >= len(surprisal_values):
                        padded_surprisals.append(np.nan)
                    elif tok == cls_tok and surprisal_idx == 0:
                        # CLS token at start
                        padded_surprisals.append(np.nan)
                    else:
                        if surprisal_idx < len(surprisal_values):
                            padded_surprisals.append(surprisal_values[surprisal_idx])
                            surprisal_idx += 1
                        else:
                            padded_surprisals.append(np.nan)
                
                surprisal_values = padded_surprisals
            
            output.append((token_strings, surprisal_values))
    
    return output


class TestMiniconsSurprisal(unittest.TestCase):
    """Test that main.py surprisal values match minicons."""
    
    @classmethod
    def setUpClass(cls):
        """Load models once for all tests."""
        print(f"\nLoading models for testing...")
        print(f"AR model: {AR_MODEL}")
        print(f"MLM model: {MLM_MODEL}")
    
    def _compare_surprisals(self, main_results, minicons_results, model_name, mode):
        """Helper to compare surprisal values."""
        for i, (main_data, minicons_data) in enumerate(zip(main_results, minicons_results)):
            main_tokens, main_surprisals = main_data
            minicons_tokens, minicons_surprisals = minicons_data
            
            # Check token count matches
            self.assertEqual(
                len(main_tokens), 
                len(minicons_tokens),
                f"Text {i}: Token count mismatch in {mode.upper()} mode ({model_name})"
            )
            
            # Compare token by token
            for j, (mt, ms, mct, mcs) in enumerate(zip(
                main_tokens, main_surprisals, minicons_tokens, minicons_surprisals
            )):
                # Skip NaN values
                if np.isnan(ms) or np.isnan(mcs):
                    continue
                
                diff = abs(ms - mcs)
                
                self.assertLessEqual(
                    diff,
                    THRESHOLD,
                    f"Text {i}, token {j} '{mt}': "
                    f"main.py={ms:.4f}, minicons={mcs:.4f}, diff={diff:.4f} "
                    f"exceeds threshold {THRESHOLD} in {mode.upper()} mode ({model_name})"
                )
    
    def test_ar_model_surprisal(self):
        """Test that AR model surprisal matches minicons."""
        print(f"\nTesting AR model: {AR_MODEL}")
        
        main_results = compute_with_main(TEST_TEXTS, AR_MODEL, 'ar')
        minicons_results = compute_with_minicons(TEST_TEXTS, AR_MODEL, 'ar')
        
        self._compare_surprisals(main_results, minicons_results, AR_MODEL, 'ar')
        print(f"✓ AR model passed")
    
    def test_mlm_model_surprisal(self):
        """Test that MLM model surprisal matches minicons."""
        print(f"\nTesting MLM model: {MLM_MODEL}")
        
        main_results = compute_with_main(TEST_TEXTS, MLM_MODEL, 'mlm')
        minicons_results = compute_with_minicons(TEST_TEXTS, MLM_MODEL, 'mlm', PLL_METRIC)
        
        self._compare_surprisals(main_results, minicons_results, MLM_MODEL, 'mlm')
        print(f"✓ MLM model passed")


if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)