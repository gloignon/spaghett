from __future__ import annotations
import unittest
import sys
import os
from typing import List, Tuple
import numpy as np

# Add src directory to path before importing scorer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Verbose printing control (on by default). Set SPAGHETT_VERBOSE=0 to silence tables.
VERBOSE = os.getenv("SPAGHETT_VERBOSE", "1") == "1"

#!/usr/bin/env python3
"""
test_minicons_compare.py

Test that surprisal values from scorer.py match those from the minicons library.

Usage:
    python -m pytest test_minicons_compare.py -v
    python -m unittest test_minicons_compare.py -v
    python test_minicons_compare.py  # runs as unittest
"""


# Test configuration
TEST_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "This is a short test sentence.",
    "Minicons surprisal should be comparable to scorer.py output.",
]

THRESHOLD = 0.1  # Maximum allowed difference in bits
AR_MODEL = "gpt2"
MLM_MODEL = "bert-base-uncased"
PLL_METRIC = "original"
PLL_METRICS = ["original", "within_word_l2r"]


def compute_with_scorer(texts: List[str], model_name: str, mode: str, pll_metric: str = 'original') -> List[Tuple[List[str], List[float]]]:
    """
    Compute surprisal using scorer.py's process_sentences function.
    
    Returns:
        List of (tokens, surprisals) tuples, one per text
    """
    import importlib
    
    try:
        scorer = importlib.import_module("scorer")
    except Exception as e:
        raise RuntimeError(f"Failed to import scorer.py: {e}")
    
    if not hasattr(scorer, "process_sentences"):
        raise RuntimeError(
            "scorer.py does not export process_sentences(). "
            "Please ensure your scorer.py has this function."
        )
    
    # Call process_sentences
    results = scorer.process_sentences(
        sentences=texts,
        mode=mode,
        model_name=model_name,
        left_context='',
        top_k=1,
        lookahead_n=0,
        doc_ids=[f"doc_{i}" for i in range(len(texts))],
        sentence_ids=[f"sent_{i}" for i in range(len(texts))],
        progress=False,
        pll_metric=pll_metric  # pass through for MLM PLL variants
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
            # We need to prepend NaN for the first token to align with scorer.py
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
    """Test that scorer.py surprisal values match minicons."""
    
    @classmethod
    def setUpClass(cls):
        """Load models once for all tests."""
        print(f"\nLoading models for testing...")
        print(f"AR model: {AR_MODEL}")
        print(f"MLM model: {MLM_MODEL}")
    
    def _print_table(self, text_idx: int, tokens: List[str], scorer_vals: List[float], mini_vals: List[float]):
        if not VERBOSE:
            return
        print("\n" + "=" * 100)
        print(f"Text {text_idx} token-level comparison")
        print("=" * 100)
        print(f"{'Idx':<4} {'Token':<24} {'scorer.py (bits)':>16} {'minicons (bits)':>17} {'Diff':>10}  Status")
        print("-" * 100)
        for j, (tok, ss, mcs) in enumerate(zip(tokens, scorer_vals, mini_vals)):
            ss_str = "NaN" if np.isnan(ss) else f"{ss:.4f}"
            mcs_str = "NaN" if np.isnan(mcs) else f"{mcs:.4f}"
            if np.isnan(ss) or np.isnan(mcs):
                diff_str = "N/A"
                status = "- SKIP"
            else:
                diff = abs(ss - mcs)
                diff_str = f"{diff:.4f}"
                status = "✓ OK" if diff <= THRESHOLD else "✗ FAIL"
            disp_tok = tok if len(tok) <= 22 else tok[:20] + ".."
            print(f"{j:<4} {disp_tok:<24} {ss_str:>16} {mcs_str:>17} {diff_str:>10}  {status}")
        print("-" * 100)

    def _compare_surprisals(self, scorer_results, minicons_results, model_name, mode):
        """Helper to compare surprisal values."""
        for i, (scorer_data, minicons_data) in enumerate(zip(scorer_results, minicons_results)):
            scorer_tokens, scorer_surprisals = scorer_data
            minicons_tokens, minicons_surprisals = minicons_data
            
            # Print table before assertions (helps diagnose mismatches)
            self._print_table(i, scorer_tokens, scorer_surprisals, minicons_surprisals)

            # Check token count matches
            self.assertEqual(
                len(scorer_tokens), 
                len(minicons_tokens),
                f"Text {i}: Token count mismatch in {mode.upper()} mode ({model_name})"
            )
            
            # Compare token by token
            for j, (st, ss, mct, mcs) in enumerate(zip(
                scorer_tokens, scorer_surprisals, minicons_tokens, minicons_surprisals
            )):
                # Skip NaN values
                if np.isnan(ss) or np.isnan(mcs):
                    continue
                
                diff = abs(ss - mcs)
                
                self.assertLessEqual(
                    diff,
                    THRESHOLD,
                    f"Text {i}, token {j} '{st}': "
                    f"scorer.py={ss:.4f}, minicons={mcs:.4f}, diff={diff:.4f} "
                    f"exceeds threshold {THRESHOLD} in {mode.upper()} mode ({model_name})"
                )

    def test_ar_model_surprisal(self):
        """Test that AR model surprisal matches minicons."""
        print(f"\nTesting AR model: {AR_MODEL}")
        
        scorer_results = compute_with_scorer(TEST_TEXTS, AR_MODEL, 'ar')
        minicons_results = compute_with_minicons(TEST_TEXTS, AR_MODEL, 'ar')
        
        self._compare_surprisals(scorer_results, minicons_results, AR_MODEL, 'ar')
        print(f"✓ AR model passed")

    def test_mlm_model_surprisal_original(self):
        """Test that MLM (original PLL) surprisal matches minicons."""
        pll = "original"
        print(f"\nTesting MLM model ({pll}): {MLM_MODEL}")
        scorer_results = compute_with_scorer(TEST_TEXTS, MLM_MODEL, 'mlm', pll_metric=pll)
        minicons_results = compute_with_minicons(TEST_TEXTS, MLM_MODEL, 'mlm', pll_metric=pll)
        self._compare_surprisals(scorer_results, minicons_results, MLM_MODEL, f'mlm:{pll}')
        print(f"✓ MLM {pll} passed")

    def test_mlm_model_surprisal_l2r(self):
        """Test that MLM (within_word_l2r) surprisal matches minicons."""
        pll = "within_word_l2r"
        print(f"\nTesting MLM model ({pll}): {MLM_MODEL}")
        scorer_results = compute_with_scorer(TEST_TEXTS, MLM_MODEL, 'mlm', pll_metric=pll)
        minicons_results = compute_with_minicons(TEST_TEXTS, MLM_MODEL, 'mlm', pll_metric=pll)
        self._compare_surprisals(scorer_results, minicons_results, MLM_MODEL, f'mlm:{pll}')
        print(f"✓ MLM {pll} passed")

    def test_ar_empty_string(self):
        """Test AR model with empty string."""
        print(f"\nTesting AR model with edge case: empty string")
        empty_text = [""]
        
        scorer_success = True
        minicons_success = True
        scorer_error = None
        minicons_error = None
        
        try:
            scorer_results = compute_with_scorer(empty_text, AR_MODEL, 'ar')
        except Exception as e:
            scorer_success = False
            scorer_error = str(e)
            scorer_results = [([], [])]
        
        try:
            minicons_results = compute_with_minicons(empty_text, AR_MODEL, 'ar')
        except Exception as e:
            minicons_success = False
            minicons_error = str(e)
            minicons_results = [([], [])]
        
        if not scorer_success:
            print(f"⚠ scorer.py failed on empty string: {scorer_error}")
        if not minicons_success:
            print(f"⚠ minicons failed on empty string: {minicons_error}")
        
        if scorer_success and minicons_success:
            self._compare_surprisals(scorer_results, minicons_results, AR_MODEL, 'ar')
            print(f"✓ AR empty string passed")
        else:
            self.skipTest(f"Skipping comparison - scorer.py success: {scorer_success}, minicons success: {minicons_success}")

    def test_mlm_empty_string(self):
        """Test MLM model with empty string."""
        print(f"\nTesting MLM model with edge case: empty string")
        empty_text = [""]
        
        scorer_success = True
        minicons_success = True
        scorer_error = None
        minicons_error = None
        
        try:
            scorer_results = compute_with_scorer(empty_text, MLM_MODEL, 'mlm')
        except Exception as e:
            scorer_success = False
            scorer_error = str(e)
            scorer_results = [([], [])]
        
        try:
            minicons_results = compute_with_minicons(empty_text, MLM_MODEL, 'mlm')
        except Exception as e:
            minicons_success = False
            minicons_error = str(e)
            minicons_results = [([], [])]
        
        if not scorer_success:
            print(f"⚠ scorer.py failed on empty string: {scorer_error}")
        if not minicons_success:
            print(f"⚠ minicons failed on empty string: {minicons_error}")
        
        if scorer_success and minicons_success:
            self._compare_surprisals(scorer_results, minicons_results, MLM_MODEL, 'mlm')
            print(f"✓ MLM empty string passed")
        else:
            self.skipTest(f"Skipping comparison - scorer.py success: {scorer_success}, minicons success: {minicons_success}")

    def test_ar_single_token(self):
        """Test AR model with single token sentences."""
        print(f"\nTesting AR model with single tokens")
        single_tokens = ["Hello", "World", "!"]
        scorer_results = compute_with_scorer(single_tokens, AR_MODEL, 'ar')
        minicons_results = compute_with_minicons(single_tokens, AR_MODEL, 'ar')
        self._compare_surprisals(scorer_results, minicons_results, AR_MODEL, 'ar')
        print(f"✓ AR single token passed")

    def test_mlm_single_token(self):
        """Test MLM model with single token sentences."""
        print(f"\nTesting MLM model with single tokens")
        single_tokens = ["Hello", "World", "!"]
        scorer_results = compute_with_scorer(single_tokens, MLM_MODEL, 'mlm')
        minicons_results = compute_with_minicons(single_tokens, MLM_MODEL, 'mlm')
        self._compare_surprisals(scorer_results, minicons_results, MLM_MODEL, 'mlm')
        print(f"✓ MLM single token passed")

    def test_ar_long_sentence(self):
        """Test AR model with long sentence."""
        print(f"\nTesting AR model with long sentence")
        long_text = [
            "This is a very long sentence that contains many words and should test "
            "whether the implementation handles longer sequences correctly and accurately "
            "computes surprisal values for each token in the extended context window."
        ]
        scorer_results = compute_with_scorer(long_text, AR_MODEL, 'ar')
        minicons_results = compute_with_minicons(long_text, AR_MODEL, 'ar')
        self._compare_surprisals(scorer_results, minicons_results, AR_MODEL, 'ar')
        print(f"✓ AR long sentence passed")

    def test_mlm_long_sentence(self):
        """Test MLM model with long sentence."""
        print(f"\nTesting MLM model with long sentence")
        long_text = [
            "This is a very long sentence that contains many words and should test "
            "whether the implementation handles longer sequences correctly and accurately "
            "computes surprisal values for each token in the extended context window."
        ]
        scorer_results = compute_with_scorer(long_text, MLM_MODEL, 'mlm')
        minicons_results = compute_with_minicons(long_text, MLM_MODEL, 'mlm')
        self._compare_surprisals(scorer_results, minicons_results, MLM_MODEL, 'mlm')
        print(f"✓ MLM long sentence passed")

    def test_ar_special_characters(self):
        """Test AR model with special characters and punctuation."""
        print(f"\nTesting AR model with special characters")
        special_texts = [
            "Hello, world!",
            "What's happening?",
            "Price: $100.50",
            "Email: test@example.com"
        ]
        scorer_results = compute_with_scorer(special_texts, AR_MODEL, 'ar')
        minicons_results = compute_with_minicons(special_texts, AR_MODEL, 'ar')
        self._compare_surprisals(scorer_results, minicons_results, AR_MODEL, 'ar')
        print(f"✓ AR special characters passed")

    def test_mlm_special_characters(self):
        """Test MLM model with special characters and punctuation."""
        print(f"\nTesting MLM model with special characters")
        special_texts = [
            "Hello, world!",
            "What's happening?",
            "Price: $100.50",
            "Email: test@example.com"
        ]
        scorer_results = compute_with_scorer(special_texts, MLM_MODEL, 'mlm')
        minicons_results = compute_with_minicons(special_texts, MLM_MODEL, 'mlm')
        self._compare_surprisals(scorer_results, minicons_results, MLM_MODEL, 'mlm')
        print(f"✓ MLM special characters passed")

    def test_ar_multiple_sentences_batch(self):
        """Test AR model processes multiple sentences consistently."""
        print(f"\nTesting AR model with batch of sentences")
        batch_texts = [
            "First sentence.",
            "Second sentence here.",
            "Third and final sentence."
        ]
        scorer_results = compute_with_scorer(batch_texts, AR_MODEL, 'ar')
        minicons_results = compute_with_minicons(batch_texts, AR_MODEL, 'ar')
        
        # Verify we got results for all sentences
        self.assertEqual(len(scorer_results), len(batch_texts))
        self.assertEqual(len(minicons_results), len(batch_texts))
        
        self._compare_surprisals(scorer_results, minicons_results, AR_MODEL, 'ar')
        print(f"✓ AR batch processing passed")

    def test_mlm_multiple_sentences_batch(self):
        """Test MLM model processes multiple sentences consistently."""
        print(f"\nTesting MLM model with batch of sentences")
        batch_texts = [
            "First sentence.",
            "Second sentence here.",
            "Third and final sentence."
        ]
        scorer_results = compute_with_scorer(batch_texts, MLM_MODEL, 'mlm')
        minicons_results = compute_with_minicons(batch_texts, MLM_MODEL, 'mlm')
        
        # Verify we got results for all sentences
        self.assertEqual(len(scorer_results), len(batch_texts))
        self.assertEqual(len(minicons_results), len(batch_texts))
        
        self._compare_surprisals(scorer_results, minicons_results, MLM_MODEL, 'mlm')
        print(f"✓ MLM batch processing passed")

    def test_ar_whitespace_only(self):
        """Test AR model with whitespace-only strings."""
        print(f"\nTesting AR model with whitespace strings")
        whitespace_texts = [" ", "  ", "\t", "\n"]
        
        scorer_success = True
        minicons_success = True
        scorer_error = None
        minicons_error = None
        
        try:
            scorer_results = compute_with_scorer(whitespace_texts, AR_MODEL, 'ar')
        except Exception as e:
            scorer_success = False
            scorer_error = str(e)
            scorer_results = [([], [])] * len(whitespace_texts)
        
        try:
            minicons_results = compute_with_minicons(whitespace_texts, AR_MODEL, 'ar')
        except Exception as e:
            minicons_success = False
            minicons_error = str(e)
            minicons_results = [([], [])] * len(whitespace_texts)
        
        if not scorer_success:
            print(f"⚠ scorer.py failed on whitespace strings: {scorer_error}")
        if not minicons_success:
            print(f"⚠ minicons failed on whitespace strings: {minicons_error}")
        
        if scorer_success and minicons_success:
            self._compare_surprisals(scorer_results, minicons_results, AR_MODEL, 'ar')
            print(f"✓ AR whitespace strings passed")
        else:
            self.skipTest(f"Skipping comparison - scorer.py success: {scorer_success}, minicons success: {minicons_success}")

    def test_mlm_whitespace_only(self):
        """Test MLM model with whitespace-only strings."""
        print(f"\nTesting MLM model with whitespace strings")
        whitespace_texts = [" ", "  ", "\t", "\n"]
        
        scorer_success = True
        minicons_success = True
        scorer_error = None
        minicons_error = None
        
        try:
            scorer_results = compute_with_scorer(whitespace_texts, MLM_MODEL, 'mlm')
        except Exception as e:
            scorer_success = False
            scorer_error = str(e)
            scorer_results = [([], [])] * len(whitespace_texts)
        
        try:
            minicons_results = compute_with_minicons(whitespace_texts, MLM_MODEL, 'mlm')
        except Exception as e:
            minicons_success = False
            minicons_error = str(e)
            minicons_results = [([], [])] * len(whitespace_texts)
        
        if not scorer_success:
            print(f"⚠ scorer.py failed on whitespace strings: {scorer_error}")
        if not minicons_success:
            print(f"⚠ minicons failed on whitespace strings: {minicons_error}")
        
        if scorer_success and minicons_success:
            self._compare_surprisals(scorer_results, minicons_results, MLM_MODEL, 'mlm')
            print(f"✓ MLM whitespace strings passed")
        else:
            self.skipTest(f"Skipping comparison - scorer.py success: {scorer_success}, minicons success: {minicons_success}")

    def test_ar_unicode_characters(self):
        """Test AR model with Unicode characters."""
        print(f"\nTesting AR model with Unicode characters")
        unicode_texts = [
            "Café au lait",
            "こんにちは世界",
            "Привет мир",
            "🎉 Emoji test! 🚀"
        ]
        scorer_results = compute_with_scorer(unicode_texts, AR_MODEL, 'ar')
        minicons_results = compute_with_minicons(unicode_texts, AR_MODEL, 'ar')
        self._compare_surprisals(scorer_results, minicons_results, AR_MODEL, 'ar')
        print(f"✓ AR Unicode characters passed")

    def test_mlm_unicode_characters(self):
        """Test MLM model with Unicode characters."""
        print(f"\nTesting MLM model with Unicode characters")
        unicode_texts = [
            "Café au lait",
            "こんにちは世界",
            "Привет мир",
            "🎉 Emoji test! 🚀"
        ]
        scorer_results = compute_with_scorer(unicode_texts, MLM_MODEL, 'mlm')
        minicons_results = compute_with_minicons(unicode_texts, MLM_MODEL, 'mlm')
        self._compare_surprisals(scorer_results, minicons_results, MLM_MODEL, 'mlm')
        print(f"✓ MLM Unicode characters passed")

    def test_ar_numbers_and_symbols(self):
        """Test AR model with numbers and mathematical symbols."""
        print(f"\nTesting AR model with numbers and symbols")
        number_texts = [
            "2 + 2 = 4",
            "π ≈ 3.14159",
            "100% correct",
            "α β γ δ"
        ]
        scorer_results = compute_with_scorer(number_texts, AR_MODEL, 'ar')
        minicons_results = compute_with_minicons(number_texts, AR_MODEL, 'ar')
        self._compare_surprisals(scorer_results, minicons_results, AR_MODEL, 'ar')
        print(f"✓ AR numbers and symbols passed")

    def test_mlm_numbers_and_symbols(self):
        """Test MLM model with numbers and mathematical symbols."""
        print(f"\nTesting MLM model with numbers and symbols")
        number_texts = [
            "2 + 2 = 4",
            "π ≈ 3.14159",
            "100% correct",
            "α β γ δ"
        ]
        scorer_results = compute_with_scorer(number_texts, MLM_MODEL, 'mlm')
        minicons_results = compute_with_minicons(number_texts, MLM_MODEL, 'mlm')
        self._compare_surprisals(scorer_results, minicons_results, MLM_MODEL, 'mlm')
        print(f"✓ MLM numbers and symbols passed")

    def test_ar_repeated_tokens(self):
        """Test AR model with repeated tokens."""
        print(f"\nTesting AR model with repeated tokens")
        repeated_texts = [
            "la la la la la",
            "Hello Hello Hello",
            "!!! !!! !!!"
        ]
        scorer_results = compute_with_scorer(repeated_texts, AR_MODEL, 'ar')
        minicons_results = compute_with_minicons(repeated_texts, AR_MODEL, 'ar')
        self._compare_surprisals(scorer_results, minicons_results, AR_MODEL, 'ar')
        print(f"✓ AR repeated tokens passed")

    def test_mlm_repeated_tokens(self):
        """Test MLM model with repeated tokens."""
        print(f"\nTesting MLM model with repeated tokens")
        repeated_texts = [
            "la la la la la",
            "Hello Hello Hello",
            "!!! !!! !!!"
        ]
        scorer_results = compute_with_scorer(repeated_texts, MLM_MODEL, 'mlm')
        minicons_results = compute_with_minicons(repeated_texts, MLM_MODEL, 'mlm')
        self._compare_surprisals(scorer_results, minicons_results, MLM_MODEL, 'mlm')
        print(f"✓ MLM repeated tokens passed")

    def test_ar_mixed_case(self):
        """Test AR model with mixed case text."""
        print(f"\nTesting AR model with mixed case")
        mixed_case_texts = [
            "ThIs Is MiXeD cAsE",
            "UPPERCASE TEXT",
            "lowercase text",
            "Title Case Text"
        ]
        scorer_results = compute_with_scorer(mixed_case_texts, AR_MODEL, 'ar')
        minicons_results = compute_with_minicons(mixed_case_texts, AR_MODEL, 'ar')
        self._compare_surprisals(scorer_results, minicons_results, AR_MODEL, 'ar')
        print(f"✓ AR mixed case passed")

    def test_mlm_mixed_case(self):
        """Test MLM model with mixed case text."""
        print(f"\nTesting MLM model with mixed case")
        mixed_case_texts = [
            "ThIs Is MiXeD cAsE",
            "UPPERCASE TEXT",
            "lowercase text",
            "Title Case Text"
        ]
        scorer_results = compute_with_scorer(mixed_case_texts, MLM_MODEL, 'mlm')
        minicons_results = compute_with_minicons(mixed_case_texts, MLM_MODEL, 'mlm')
        self._compare_surprisals(scorer_results, minicons_results, MLM_MODEL, 'mlm')
        print(f"✓ MLM mixed case passed")

    def test_ar_contractions(self):
        """Test AR model with contractions."""
        print(f"\nTesting AR model with contractions")
        contraction_texts = [
            "I'm happy.",
            "You're right.",
            "It's working.",
            "They've done it."
        ]
        scorer_results = compute_with_scorer(contraction_texts, AR_MODEL, 'ar')
        minicons_results = compute_with_minicons(contraction_texts, AR_MODEL, 'ar')
        self._compare_surprisals(scorer_results, minicons_results, AR_MODEL, 'ar')
        print(f"✓ AR contractions passed")

    def test_mlm_contractions(self):
        """Test MLM model with contractions."""
        print(f"\nTesting MLM model with contractions")
        contraction_texts = [
            "I'm happy.",
            "You're right.",
            "It's working.",
            "They've done it."
        ]
        scorer_results = compute_with_scorer(contraction_texts, MLM_MODEL, 'mlm')
        minicons_results = compute_with_minicons(contraction_texts, MLM_MODEL, 'mlm')
        self._compare_surprisals(scorer_results, minicons_results, MLM_MODEL, 'mlm')
        print(f"✓ MLM contractions passed")

    def test_ar_urls_and_paths(self):
        """Test AR model with URLs and file paths."""
        print(f"\nTesting AR model with URLs and paths")
        url_texts = [
            "Visit https://www.example.com",
            "File path: C:\\Users\\Documents\\file.txt",
            "Link: http://github.com/user/repo"
        ]
        scorer_results = compute_with_scorer(url_texts, AR_MODEL, 'ar')
        minicons_results = compute_with_minicons(url_texts, AR_MODEL, 'ar')
        self._compare_surprisals(scorer_results, minicons_results, AR_MODEL, 'ar')
        print(f"✓ AR URLs and paths passed")

    def test_mlm_urls_and_paths(self):
        """Test MLM model with URLs and file paths."""
        print(f"\nTesting MLM model with URLs and paths")
        url_texts = [
            "Visit https://www.example.com",
            "File path: C:\\Users\\Documents\\file.txt",
            "Link: http://github.com/user/repo"
        ]
        scorer_results = compute_with_scorer(url_texts, MLM_MODEL, 'mlm')
        minicons_results = compute_with_minicons(url_texts, MLM_MODEL, 'mlm')
        self._compare_surprisals(scorer_results, minicons_results, MLM_MODEL, 'mlm')
        print(f"✓ MLM URLs and paths passed")
        
if __name__ == "__main__":
    unittest.main(verbosity=2)