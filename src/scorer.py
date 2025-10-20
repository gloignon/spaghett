# pyright: reportGeneralTypeIssues=false
"""
Simple script for computing per-token surprisal and entropy, by sentence, with extra left context,
and exporting the top-k most probable next tokens for each scored token.

CLI parameters:
    --input_file: Path to the input TSV file with documents or sentences.
    --output_file: Path to the output TSV file (default: will create one based on input file and timestamp). You can also provide a folder path for the output file, and it will create a timestamped file inside that folder. 
    --mode: 'ar' for autoregressive (GPT-style) or 'mlm' for masked language model (BERT-style).
    --model: Name of the pre-trained model to use (e.g., 'gpt2', 'bert-base-uncased').
    --format: 'documents' or 'sentences' to specify input format.
    --left_context_file: Path to a .txt file whose contents are prepended to every sentence.
    --top_k: Number of top probable tokens to output (default: 5).
    --lookahead_n: (AR only) Number of follow tokens to generate (default: 3).
    --lookahead_strategy: (AR only) Strategy for generating follow tokens: 'greedy' or 'beam' (default: greedy).
    --beam_width: (AR only) Beam width for beam search (default: 3, only used when --lookahead_strategy=beam).
    --pll_metric: (MLM only) PLL variant to use: 'original' for original PLL, 'within_word_l2r' for within-word left-to-right scoring (default: original).

Input TSV formats:
    - documents: doc_id<TAB>text (with header)
    - sentences: doc_id<TAB>sentence_id<TAB>sentence (with header)

Output TSV columns:
    doc_id (str): 
        Document identifier from input file.
    
    sentence_id (str): 
        Sentence identifier within the document. Auto-generated for 'documents' format.
    
    token_index (int): 
        Position of the token within the sentence (1-indexed).
    
    token (str): 
        The actual token text

    token_decoded (str):
        Human-readable decoded token text (e.g., "Il" instead of "▁Il" or "<s>").

    is_special (int): 
        Flag indicating if token is a special token (1) or regular content token (0).
        Special tokens include BOS, EOS, CLS, SEP, PAD, MASK, etc.
    
    surprisal_bits (float): 
        Surprisal value in bits: -log2(p(token|context)).
        Measures how unexpected the token is given prior context.
        Higher values = more surprising/unexpected.
        Empty for first token in AR mode (no prior context to condition on).
        AR mode: conditioned on all previous tokens.
        MLM mode: conditioned on bidirectional context (all other tokens).
    
    entropy_bits (float): 
        Shannon entropy of the probability distribution over all possible next tokens, in bits.
        Measures uncertainty/predictability at this position.
        Higher values = more uncertain/less predictable.
        Empty for first token in AR mode.
        Range: 0 (completely certain) to log2(vocab_size) (uniform distribution).
    
    pred_alt_1 through pred_alt_N (str): 
        The N most probable tokens the model predicted would appear (instead of the current token), 
        ranked by probability (1=highest).
        Number of columns determined by --top_k parameter.
        Useful for analyzing alternative predictions and model confidence.
    
    pred_next_1 through pred_next_M (str): 
        (AR mode only) Continuation tokens generated from current position.
        Number of columns determined by --lookahead_n parameter.
        Generation strategy determined by --lookahead_strategy:
            - 'greedy': selects highest probability token at each step
            - 'beam': uses beam search to find most probable sequence
        Useful for understanding what word/phrase the model expects to follow.
Notes:
    - The general philosophy is "fix it in post-processing" rather than complicating the scoring code.
    - All tokens are scored, including special tokens (BOS/EOS/CLS/SEP/etc) when the model produces them.
    - In AR mode we also compute surprisal and entropy for the fist token if no prior context is provided. Filter out if undesired.
    - Don't want special tokens? You can also filter them out easily using the is_special column in post-processing.
    - We avoid making assumptions about special characters in tokenization (e.g., "Ġ" vs "▁") and output both raw and decoded forms.
    - Designed for simplicity and robustness over performance (limited batching, no GPU acceleration).
    - Surprisal and entropy are in bits (log base 2).
    - For MLM, we use parallel masking (one forward pass with all tokens masked) for efficiency. AR has no batching for now.

"""

import argparse
import csv
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
from tqdm import tqdm

@dataclass
class ScoringConfig:
    mode: str
    model_name: str
    pll_metric: str = 'original'
    lookahead_strategy: str = 'greedy'
    top_k: int = 3
    lookahead_n: int = 3  # Default for AR, will be adjusted in __post_init__
    beam_width: int = 3

    _VALID_MODES = frozenset({'ar', 'mlm'})
    _VALID_STRATEGIES = frozenset({'greedy', 'beam'})
    _VALID_PLL = frozenset({'original', 'within_word_l2r'})

    def __post_init__(self):
        """Adjust defaults based on mode before validation."""
        # Auto-adjust lookahead_n for MLM mode if user didn't explicitly set it
        # We can't detect "user didn't set it" in __init__, but we can make it smart:
        # If mode is MLM and lookahead_n > 0, set it to 0
        if self.mode == 'mlm' and self.lookahead_n > 0:
            object.__setattr__(self, 'lookahead_n', 0)
        
        # Similarly for lookahead_strategy
        if self.mode == 'mlm' and self.lookahead_strategy != 'greedy':
            object.__setattr__(self, 'lookahead_strategy', 'greedy')

    def validate(self) -> None:
        errors = []
        
        # Basic type/range validation
        if self.mode not in self._VALID_MODES:
            errors.append(f"mode must be one of {sorted(self._VALID_MODES)}, got '{self.mode}'")
        if self.top_k < 0:  # Changed from < 1 to < 0
            errors.append(f"top_k must be >= 0, got {self.top_k}")
        if self.lookahead_n < 0:
            errors.append(f"lookahead_n must be >= 0, got {self.lookahead_n}")
        if self.lookahead_strategy not in self._VALID_STRATEGIES:
            errors.append(f"lookahead_strategy must be one of {sorted(self._VALID_STRATEGIES)}, got '{self.lookahead_strategy}'")
        if self.beam_width < 1:
            errors.append(f"beam_width must be >= 1, got {self.beam_width}")
        if self.pll_metric not in self._VALID_PLL:
            errors.append(f"pll_metric must be one of {sorted(self._VALID_PLL)}, got '{self.pll_metric}'")
        
        # Mode-specific validation
        if self.mode == 'ar':
            # AR mode: pll_metric not applicable
            if self.pll_metric != 'original':
                errors.append(f"pll_metric is only applicable in MLM mode, got '{self.pll_metric}'")
            # Beam search requires beam_width
            if self.lookahead_strategy == 'beam' and self.beam_width < 1:
                errors.append(f"beam_width must be >= 1 when using beam search, got {self.beam_width}")
        else:  # MLM mode
            # MLM mode: top_k must be >= 1
            if self.top_k < 1:
                errors.append(f"top_k must be >= 1 in MLM mode, got {self.top_k}")
        
        # Warnings (don't block execution, just inform)
        if self.mode == 'ar' and self.lookahead_strategy == 'beam' and self.beam_width > self.top_k and self.top_k > 0:
            print(f"Warning: beam_width ({self.beam_width}) > top_k ({self.top_k}) may generate many sequences", file=sys.stderr)
        
        if self.top_k > 50:
            print(f"Warning: top_k ({self.top_k}) is quite large, this may slow down processing", file=sys.stderr)
        
        if self.lookahead_n > 20:
            print(f"Warning: lookahead_n ({self.lookahead_n}) is quite large, this may slow down processing", file=sys.stderr)
        
        if errors:
            bullet = "\n  • ".join(errors)
            raise ValueError(f"Invalid scoring configuration:\n  • {bullet}")

LN2 = math.log(2.0)


# ============================================================================
# Helper functions
# ============================================================================

def simple_sentence_split(text: str) -> List[str]:
    """Split text into sentences using basic punctuation."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]


def read_left_context(path: str) -> str:
    """Read left context from file, strip whitespace."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Left context file not found: '{path}'\n"
            f"Please check that the file path is correct and the file exists."
        )
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # Check if file is empty or contains only whitespace
        if not content:
            raise ValueError(
                f"Left context file is empty: '{path}'\n"
                f"The context file must contain text to use as left context.\n"
            )
            
        return content
        
    except PermissionError:
        raise PermissionError(
            f"Permission denied when trying to read: '{path}'\n"
            f"Please check file permissions."
        )
    except UnicodeDecodeError as e:
        raise ValueError(
            f"Failed to decode file '{path}' as UTF-8.\n"
            f"Please ensure the file is UTF-8 encoded.\n"
            f"Error: {str(e)}"
        )


def combine_context_and_sentence(context: str, sentence: str) -> str:
    """Combine left context with sentence, adding space if both non-empty."""
    if context and sentence:
        return f"{context} {sentence}"
    return context or sentence


def decode_token(tokenizer, token_id: int, is_special: bool) -> Tuple[str, str]:
    """
    Decode token, returning both raw and human-readable forms.
    
    Returns:
        (raw_token, decoded_token) where:
        - raw_token: token as model outputs it (e.g., "▁Il", "ĠIl", "<s>")
        - decoded_token: human-readable form (e.g., "Il", "Il", "<s>")
    """
    raw_token = tokenizer.convert_ids_to_tokens([token_id])[0]
    if is_special:
        decoded_token = raw_token  # Special tokens are already readable
    else:
        decoded_token = tokenizer.decode([token_id])
    return raw_token, decoded_token


def prepare_input_with_context(sentence_ids: torch.Tensor, context_ids: List[int] = None) -> Tuple[torch.Tensor, int]:
    """
    Combine context and sentence IDs.
    
    Returns:
        (combined_ids, sentence_start_position)
    """
    if context_ids is not None and len(context_ids) > 0:
        input_ids = torch.cat([torch.tensor(context_ids), sentence_ids])
        sentence_start = len(context_ids)
    else:
        input_ids = sentence_ids
        sentence_start = 0
    return input_ids, sentence_start


def compute_surprisal_entropy(logits: torch.Tensor, target_id: int) -> Tuple[float, float]:
    """Compute surprisal and entropy from logits."""
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    
    surprisal = -log_probs[target_id].item() / LN2
    entropy = -(probs * log_probs).sum().item() / LN2
    
    return surprisal, entropy


def get_top_k_predictions(logits: torch.Tensor, tokenizer, k: int) -> List[str]:
    """
    Get the top-k most probable tokens.
    Returns raw tokens with special characters preserved.
    """
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k)
    
    # Use convert_ids_to_tokens to preserve special characters
    top_tokens = tokenizer.convert_ids_to_tokens(top_ids.tolist())
    
    return top_tokens


# ============================================================================
# Lookahead generation functions
# ============================================================================

def greedy_lookahead(model, tokenizer, prefix_ids: torch.Tensor, n: int) -> List[str]:
    """
    Generate n tokens greedily (argmax at each step).
    Returns raw tokens with special characters preserved.
    """
    current = prefix_ids.clone()
    tokens = []
    
    for _ in range(n):
        with torch.no_grad():
            outputs = model(current.unsqueeze(0))
            logits = outputs.logits[0, -1]
        
        next_id = logits.argmax().item()
        # Use convert_ids_to_tokens to preserve special characters
        token_str = tokenizer.convert_ids_to_tokens([next_id])[0]
        tokens.append(token_str)
        
        current = torch.cat([current, torch.tensor([next_id])])
    
    return tokens


def beam_search_lookahead(model, tokenizer, prefix_ids: torch.Tensor, n: int, beam_width: int = 3) -> List[str]:
    """
    Generate n tokens using beam search.
    Returns raw tokens with special characters preserved.
    """
    beams = [(prefix_ids.clone(), 0.0, [])]
    
    for step in range(n):
        candidates = []
        
        for current_ids, current_score, current_tokens in beams:
            with torch.no_grad():
                outputs = model(current_ids.unsqueeze(0))
                logits = outputs.logits[0, -1]
            
            log_probs = torch.log_softmax(logits, dim=-1)
            top_log_probs, top_ids = torch.topk(log_probs, beam_width)
            
            for log_prob, token_id in zip(top_log_probs, top_ids):
                new_score = current_score + log_prob.item()
                new_ids = torch.cat([current_ids, token_id.unsqueeze(0)])
                # Use convert_ids_to_tokens to preserve special characters
                token_str = tokenizer.convert_ids_to_tokens([token_id.item()])[0]
                new_tokens = current_tokens + [token_str]
                candidates.append((new_ids, new_score, new_tokens))
        
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    
    best_beam = beams[0]
    return best_beam[2]


# ============================================================================
# Scoring functions
# ============================================================================

def score_autoregressive(
    sentence: str,
    left_context: str,
    tokenizer,
    model,
    top_k: int = 5,
    lookahead_n: int = 3,
    lookahead_strategy: str = 'greedy',
    beam_width: int = 3,
    context_ids: List[int] = None
) -> Tuple[List[str], List[str], List[int], List[float], List[float], List[List[str]]]:
    """
    Score sentence with AR model.

    Returns:
        Tuple containing:
            - List[str]: Raw tokens as model outputs them.
            - List[str]: Decoded tokens (human-readable).
            - List[int]: Flags indicating if each token is a special token (1) or not (0).
            - List[float]: Surprisal values (in bits) for each token.
            - List[float]: Entropy values (in bits) for each token.
            - List[List[str]]: For each token, a list containing the most probable token, top-k predictions, and lookahead tokens.
    """
    # Handle empty or whitespace-only sentences
    if not sentence or not sentence.strip():
        return [], [], [], [], [], []
    
    # Tokenize sentence
    encoding = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
    sentence_ids = encoding["input_ids"][0]
    
    if not sentence_ids.numel():
        return [], [], [], [], [], []
    
    # Prepare input with context
    input_ids, sentence_start = prepare_input_with_context(sentence_ids, context_ids)
    special_mask = tokenizer.get_special_tokens_mask(sentence_ids.tolist(), already_has_special_tokens=True)
    
    # Get logits for the full sequence
    with torch.no_grad():
        outputs = model(input_ids.unsqueeze(0))
        logits = outputs.logits[0]  # shape: (seq_len, vocab_size)
    
    # Score each token
    raw_tokens = []
    scored_tokens = []
    is_special_flags = []
    surprisals = []
    entropies = []
    pred_columns = []
    
    for sent_pos in range(len(sentence_ids)):
        combined_pos = sentence_start + sent_pos
        target_id = sentence_ids[sent_pos].item()
        is_special = 1 if special_mask[sent_pos] else 0
        
        # Decode token (both raw and readable)
        raw_token, token_str = decode_token(tokenizer, target_id, is_special)
        
        # Score based on position
        if combined_pos == 0:
            # First token with no context: use unconditional distribution
            with torch.no_grad():
                if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
                    minimal_input = torch.tensor([[tokenizer.bos_token_id]])
                    minimal_logits = model(minimal_input).logits[0, -1]
                else:
                    minimal_logits = logits[0] if len(logits) > 0 else None
            
            if minimal_logits is not None:
                surprisal, entropy = compute_surprisal_entropy(minimal_logits, target_id)
                top_k_tokens = get_top_k_predictions(minimal_logits, tokenizer, top_k)
                
                if lookahead_n > 0:
                    if lookahead_strategy == 'beam':
                        lookahead = beam_search_lookahead(model, tokenizer, input_ids[:1], lookahead_n, beam_width)
                    else:
                        lookahead = greedy_lookahead(model, tokenizer, input_ids[:1], lookahead_n)
                else:
                    lookahead = []
                
                pred_col = top_k_tokens + lookahead
            else:
                surprisal = entropy = float('nan')
                pred_col = [''] * (1 + top_k + lookahead_n)
        
        else:
            # Regular token: use previous position's logits
            surprisal, entropy = compute_surprisal_entropy(logits[combined_pos - 1], target_id)
            top_k_tokens = get_top_k_predictions(logits[combined_pos - 1], tokenizer, top_k)
            
            if lookahead_n > 0:
                if lookahead_strategy == 'beam':
                    lookahead = beam_search_lookahead(model, tokenizer, input_ids[:combined_pos + 1], lookahead_n, beam_width)
                else:
                    lookahead = greedy_lookahead(model, tokenizer, input_ids[:combined_pos + 1], lookahead_n)
            else:
                lookahead = []
            
            pred_col = top_k_tokens + lookahead
        
        raw_tokens.append(raw_token)
        scored_tokens.append(token_str)
        is_special_flags.append(is_special)
        surprisals.append(surprisal)
        entropies.append(entropy)
        pred_columns.append(pred_col)
    
    return raw_tokens, scored_tokens, is_special_flags, surprisals, entropies, pred_columns


def score_masked_lm(
    sentence: str,
    tokenizer,
    model,
    top_k: int = 5,
    context_ids: List[int] = None
) -> Tuple[List[str], List[str], List[int], List[float], List[float], List[List[str]]]:
    """
    Score sentence with MLM using parallel masking (original PLL).
    """
    if tokenizer.mask_token_id is None:
        raise ValueError("Model doesn't have a [MASK] token. Use --mode ar instead.")
    
    # Handle empty or whitespace-only sentences
    if not sentence or not sentence.strip():
        return [], [], [], [], [], []
    
    # Tokenize sentence
    encoding = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
    sentence_ids = encoding["input_ids"][0]
    
    if not sentence_ids.numel():
        return [], [], [], [], [], []
    
    # Prepare input with context
    base_ids, sentence_start = prepare_input_with_context(sentence_ids, context_ids)
    special_mask = tokenizer.get_special_tokens_mask(sentence_ids.tolist(), already_has_special_tokens=True)
    
    num_tokens = len(sentence_ids)
    
    # Create ALL masked versions at once (mask current token only)
    masked_batch = []
    for sent_pos in range(num_tokens):
        combined_pos = sentence_start + sent_pos
        masked_ids = base_ids.clone()
        masked_ids[combined_pos] = tokenizer.mask_token_id
        masked_batch.append(masked_ids)
    
    masked_batch = torch.stack(masked_batch)
    
    with torch.no_grad():
        outputs = model(masked_batch)
        all_logits = outputs.logits  # (num_tokens, seq_len, vocab_size)
    
    raw_tokens, scored_tokens, is_special_flags = [], [], []
    surprisals, entropies, pred_columns = [], [], []
    
    for sent_pos in range(num_tokens):
        combined_pos = sentence_start + sent_pos
        target_id = sentence_ids[sent_pos].item()
        is_special = 1 if special_mask[sent_pos] else 0
        
        logits = all_logits[sent_pos, combined_pos]
        
        surprisal, entropy = compute_surprisal_entropy(logits, target_id)
        top_k_tokens = get_top_k_predictions(logits, tokenizer, top_k)
        
        raw_token, token_str = decode_token(tokenizer, target_id, is_special)
        
        raw_tokens.append(raw_token)
        scored_tokens.append(token_str)
        is_special_flags.append(is_special)
        surprisals.append(surprisal)
        entropies.append(entropy)
        pred_columns.append(top_k_tokens)
    
    return raw_tokens, scored_tokens, is_special_flags, surprisals, entropies, pred_columns


def score_masked_lm_l2r(
    sentence: str,
    tokenizer,
    model,
    top_k: int = 5,
    context_ids: List[int] = None
) -> Tuple[List[str], List[str], List[int], List[float], List[float], List[List[str]]]:
    """
    Score sentence with MLM using within_word_l2r (minicons-compatible):
    - For each word (group of subtokens), predict subtokens left-to-right.
    - For subtoken k in a word, mask the current subtoken and all future subtokens
      in that word; keep previous subtokens and all other tokens visible.
    - Special tokens are scored by masking only themselves.
    """
    if tokenizer.mask_token_id is None:
        raise ValueError("Model doesn't have a [MASK] token. Use --mode ar instead.")
    
    # Handle empty or whitespace-only sentences
    if not sentence or not sentence.strip():
        return [], [], [], [], [], []
    
    # Tokenize sentence
    encoding = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
    sentence_ids = encoding["input_ids"][0]  # (seq_len,)
    if not sentence_ids.numel():
        return [], [], [], [], [], []
    
    # Base + left context
    base_ids, sentence_start = prepare_input_with_context(sentence_ids, context_ids)
    seq_len = len(sentence_ids)
    special_mask = tokenizer.get_special_tokens_mask(sentence_ids.tolist(), already_has_special_tokens=True)
    
    # Word alignment (requires fast tokenizer)
    if getattr(tokenizer, "is_fast", False) and hasattr(encoding, "word_ids"):
        word_ids = encoding.word_ids(0)  # list length == seq_len, None for specials
    else:
        # Fallback: treat each token as its own word
        word_ids = list(range(seq_len))
    
    # Build mapping word_id -> list of sentence positions (subtoken indices)
    word_to_positions = {}
    for pos, wid in enumerate(word_ids):
        if wid is None:
            continue
        word_to_positions.setdefault(wid, []).append(pos)
    
    # Prepare one masked variant per sentence position (same batch size as seq_len)
    masked_variants = []
    variant_targets = []  # (sent_pos, combined_pos)
    
    mask_id = tokenizer.mask_token_id
    
    # For normal words: L2R within-word masking
    for _, positions in word_to_positions.items():
        for k, sent_pos in enumerate(positions):
            combined_pos = sentence_start + sent_pos
            ids = base_ids.clone()
            # Mask current subtoken
            ids[combined_pos] = mask_id
            # Mask future subtokens within the same word
            for future_pos in positions[k+1:]:
                ids[sentence_start + future_pos] = mask_id
            masked_variants.append(ids)
            variant_targets.append((sent_pos, combined_pos))
    
    # For special tokens (word_id is None): mask only themselves
    for sent_pos, wid in enumerate(word_ids):
        if wid is None:
            combined_pos = sentence_start + sent_pos
            ids = base_ids.clone()
            ids[combined_pos] = mask_id
            masked_variants.append(ids)
            variant_targets.append((sent_pos, combined_pos))
    
    if not masked_variants:
        return [], [], [], [], [], []
    
    batch = torch.stack(masked_variants)  # (num_variants == seq_len, seq_total_len)
    with torch.no_grad():
        outputs = model(batch)
        logits_batch = outputs.logits  # (num_variants, seq_total_len, vocab)
    
    # Prepare outputs aligned to sentence tokens (include specials)
    raw_tokens, scored_tokens, is_special_flags = [], [], []
    for sent_pos in range(seq_len):
        tid = int(sentence_ids[sent_pos].item())
        is_special = 1 if special_mask[sent_pos] else 0
        raw_token, token_str = decode_token(tokenizer, tid, is_special)
        raw_tokens.append(raw_token)
        scored_tokens.append(token_str)
        is_special_flags.append(is_special)
    
    surprisals = [float("nan")] * seq_len
    entropies = [float("nan")] * seq_len
    pred_columns: List[List[str]] = [[] for _ in range(seq_len)]
    
    # Fill values from each variant at its target position
    for variant_idx, (sent_pos, combined_pos) in enumerate(variant_targets):
        target_id = int(sentence_ids[sent_pos].item())
        logits = logits_batch[variant_idx, combined_pos]
        surp, ent = compute_surprisal_entropy(logits, target_id)  # returns bits
        top_k_tokens = get_top_k_predictions(logits, tokenizer, top_k)
        
        surprisals[sent_pos] = surp
        entropies[sent_pos] = ent
        pred_columns[sent_pos] = top_k_tokens
    
    return raw_tokens, scored_tokens, is_special_flags, surprisals, entropies, pred_columns


# ============================================================================
# I/O functions
# ============================================================================

def load_input_data(input_file: str, format_type: str) -> List[Tuple[str, str, str]]:
    """Load input TSV and return list of (doc_id, sentence_id, sentence)."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Input file not found: '{input_file}'\n"
            f"Please check that the file path is correct and the file exists."
        )
    
    # Define expected format once
    required_cols = ['doc_id', 'text'] if format_type == 'documents' else ['doc_id', 'sentence_id', 'sentence']
    min_cols = len(required_cols)
    
    data = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            
            # Try to read header
            try:
                header = next(reader)
            except StopIteration:
                raise ValueError(
                    f"Input file is empty: '{input_file}'\n"
                    f"Expected format: {chr(9).join(required_cols)}"
                )
            
            # Validate header
            if len(header) < min_cols:
                raise ValueError(
                    f"Invalid header in '{input_file}'\n"
                    f"Expected {min_cols} columns: {', '.join(required_cols)}\n"
                    f"Got {len(header)}: {', '.join(header) if header else '(empty)'}"
                )
            
            # Warn if names don't match
            header_lower = [col.lower().strip() for col in header[:min_cols]]
            if header_lower != required_cols:
                print(f"Warning: Expected columns {required_cols}, got {header[:min_cols]}")
            
            # Read data rows
            row_count = 0
            for row_num, row in enumerate(reader, start=2):
                # Skip completely empty rows
                if not row or all(cell.strip() == '' for cell in row):
                    continue
                
                row_count += 1
                  
                if len(row) < min_cols:
                    print(f"Warning: Skipping row {row_num}: expected {min_cols} columns, got {len(row)}")
                    continue
                
                # Extract and validate
                if format_type == 'documents':
                    doc_id, text = row[0].strip(), row[1].strip()
                    if not doc_id or not text:
                        print(f"Warning: Skipping row {row_num}: empty field(s)")
                        continue
                    sentences = simple_sentence_split(text)
                    data.extend((doc_id, str(i), s) for i, s in enumerate(sentences, 1))
                else:
                    doc_id, sent_id, sentence = row[0].strip(), row[1].strip(), row[2].strip()
                    if not (doc_id and sent_id and sentence):
                        print(f"Warning: Skipping row {row_num}: empty field(s)")
                        continue
                    data.append((doc_id, sent_id, sentence))
            
            # Check results
            if row_count == 0:
                raise ValueError(f"No data rows in '{input_file}'")
            if not data:
                raise ValueError(
                    f"No valid data in '{input_file}' ({row_count} rows were malformed)\n"
                    f"Expected format: {chr(9).join(required_cols)}"
                )
                
    except PermissionError:
        raise PermissionError(f"Permission denied: '{input_file}'")
    except UnicodeDecodeError as e:
        raise ValueError(f"File '{input_file}' is not UTF-8 encoded: {e}")
    
    return data


def write_output(output_file: str, results: List[dict], top_k: int, lookahead_n: int, mode: str):
    """Write results to TSV with dynamic column headers."""
    if not results:
        return
    
    # Build column names
    columns = ['doc_id', 'sentence_id', 'token_index', 'token', 'token_decoded', 'is_special', 
               'surprisal_bits', 'entropy_bits']
    
    columns += [f'pred_alt_{i}' for i in range(1, top_k + 1)]
    
    if mode == 'ar':
        columns += [f'pred_next_{i}' for i in range(1, lookahead_n + 1)]
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter='\t')
        writer.writeheader()
        writer.writerows(results)


# ============================================================================
# Core processing function (I/O independent)
# ============================================================================

def process_sentences(
    sentences: List[str],
    mode: str,
    model_name: str,
    left_context: str = '',
    top_k: int = 5,
    lookahead_n: int = 3,
    lookahead_strategy: str = 'greedy',
    beam_width: int = 3,
    doc_ids: List[str] = None,
    sentence_ids: List[str] = None,
    progress: bool = True,
    pll_metric: str = 'original'
) -> List[dict]:
    config = ScoringConfig(
        mode=mode,
        model_name=model_name,
        top_k=top_k,
        lookahead_n=lookahead_n,
        lookahead_strategy=lookahead_strategy,
        beam_width=beam_width,
        pll_metric=pll_metric
    )
    config.validate()
    # Auto-generate IDs if not provided
    if doc_ids is None:
        doc_ids = ['doc1'] * len(sentences)
    if sentence_ids is None:
        sentence_ids = [str(i) for i in range(1, len(sentences) + 1)]
    
    if len(doc_ids) != len(sentences) or len(sentence_ids) != len(sentences):
        raise ValueError("doc_ids and sentence_ids must match length of sentences")
    
    # Load model with error handling
    print(f"Loading {mode.upper()} model: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        raise ValueError(
            f"Failed to load tokenizer for model '{model_name}'. "
            f"Please check that the model exists on HuggingFace Hub or provide a valid local path. "
            f"Error: {type(e).__name__}: {str(e)}"
        )
    
    # Pre-tokenize context once
    context_ids = tokenizer(left_context, add_special_tokens=False)["input_ids"] if left_context else []
    
    try:
        if mode == 'ar':
            model = AutoModelForCausalLM.from_pretrained(model_name)
            score_fn = lambda s: score_autoregressive(
                s, left_context, tokenizer, model, top_k, lookahead_n,
                lookahead_strategy, beam_width, context_ids
            )
        elif mode == 'mlm':
            model = AutoModelForMaskedLM.from_pretrained(model_name)
            if pll_metric == 'within_word_l2r':
                score_fn = lambda s: score_masked_lm_l2r(s, tokenizer, model, top_k, context_ids)
            else:
                score_fn = lambda s: score_masked_lm(s, tokenizer, model, top_k, context_ids)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'ar' or 'mlm'")
    except Exception as e:
        if "Invalid mode" in str(e):
            raise  # Re-raise our own validation error
        raise ValueError(
            f"Failed to load model '{model_name}' for mode '{mode}'. "
            f"Please verify: (1) model exists, (2) model is compatible with {mode.upper()} mode "
            f"(AR models like GPT for 'ar', MLM models like BERT for 'mlm'). "
            f"Error: {type(e).__name__}: {str(e)}"
        )
    
    model.eval()
    
    # Process sentences
    results = []
    iterator = zip(doc_ids, sentence_ids, sentences)
    if progress:
        iterator = tqdm(iterator, total=len(sentences), desc="Processing")
    
    for doc_id, sent_id, sentence in iterator:
        raw_tokens, tokens, is_special_flags, surprisals, entropies, pred_cols = score_fn(sentence)
        
        for idx, (raw_token, token, is_special, surp, ent, preds) in enumerate(
            zip(raw_tokens, tokens, is_special_flags, surprisals, entropies, pred_cols), 1
        ):
            row = {
                'doc_id': doc_id,
                'sentence_id': sent_id,
                'token_index': idx,
                'token': raw_token,
                'token_decoded': token,
                'is_special': is_special,
                'surprisal_bits': '' if math.isnan(surp) else f'{surp:.4f}',
                'entropy_bits': '' if math.isnan(ent) else f'{ent:.4f}'
            }
            
            # Add top-k columns
            for i in range(1, top_k + 1):
                row[f'pred_alt_{i}'] = preds[i - 1] if i - 1 < len(preds) else ''
            
            # Add lookahead columns (AR only)
            if mode == 'ar':
                offset = top_k
                for i in range(1, lookahead_n + 1):
                    row[f'pred_next_{i}'] = preds[offset + i - 1] if offset + i - 1 < len(preds) else ''
            
            results.append(row)
    
    return results


# ============================================================================
# CLI-specific functions
# ============================================================================

def process_from_file(
    input_file: str,
    output_file: str,
    mode: str,
    model_name: str,
    format_type: str,
    left_context_file: str = '',
    top_k: int = 5,
    lookahead_n: int = 3,
    lookahead_strategy: str = 'greedy',
    beam_width: int = 3,
    pll_metric: str = 'original'
):
    """
    Process input TSV file and write output TSV file.
    
    This function handles I/O for CLI usage.
    """
    try:
        # Load context
        left_context = ''
        if left_context_file:
            print(f"Loading left context from: {left_context_file}")
            left_context = read_left_context(left_context_file)
        
        # Load data
        print(f"Loading input from: {input_file}")
        data = load_input_data(input_file, format_type)
        
        if not data:
            print(f"Warning: No valid data found in {input_file}")
            print("Please check that:")
            print("  1. File has correct format (TSV with header)")
            print("  2. File contains data rows (not just header)")
            print("  3. Rows have correct number of columns")
            return
        
        # Extract components
        doc_ids = [item[0] for item in data]
        sentence_ids = [item[1] for item in data]
        sentences = [item[2] for item in data]
        
        # Process using core function
        results = process_sentences(
            sentences=sentences,
            mode=mode,
            model_name=model_name,
            left_context=left_context,
            top_k=top_k,
            lookahead_n=lookahead_n,
            lookahead_strategy=lookahead_strategy,
            beam_width=beam_width,
            doc_ids=doc_ids,
            sentence_ids=sentence_ids,
            progress=True,
            pll_metric=pll_metric
        )
        
        # Write output
        write_output(output_file, results, top_k, lookahead_n, mode)
        print(f"Results written to: {output_file}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except PermissionError as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compute per-token surprisal and entropy.",
        epilog="""
Examples:
  # Score sentences with GPT-2 (autoregressive)
  python scorer.py --input_file data.tsv --mode ar --model gpt2
  
  # Score documents with BERT (masked LM)
  python scorer.py --input_file docs.tsv --mode mlm --model bert-base-uncased --format documents
  
  # Output to folder (auto-generates filename)
  python scorer.py --input_file data.tsv --mode ar --model gpt2 --output_file ./results/
  
  # With custom output and context
  python scorer.py --input_file data.tsv --output_file results.tsv --mode ar --model gpt2 \\
                   --left_context_file context.txt --top_k 10
  
  # With within-word L2R scoring (MLM only)
  python scorer.py --input_file data.tsv --mode mlm --model bert-base-uncased \\
                   --pll_metric within_word_l2r

For more information, see the documentation at the top of this file.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input_file', required=True, help='Input TSV file')
    parser.add_argument('--output_file', default='simple_output.tsv', help='Output TSV file or folder (default: auto-generated filename in current directory)')
    parser.add_argument('--mode', choices=['ar', 'mlm'], required=True, 
                       help='Model mode: "ar" for autoregressive (GPT), "mlm" for masked LM (BERT)')
    parser.add_argument('--model', required=True, 
                       help='HuggingFace model name (e.g., "gpt2", "bert-base-uncased")')
    parser.add_argument('--format', choices=['documents', 'sentences'], default="sentences", 
                       help='Input format: "documents" (doc_id, text) or "sentences" (doc_id, sent_id, sentence)')
    parser.add_argument('--left_context_file', default='', help='File with left context (optional)')
    parser.add_argument('--top_k', type=int, default=3, help='Number of top-k predictions (default: 3)')
    parser.add_argument('--lookahead_n', type=int, default=3, help='AR: number of lookahead tokens (default: 3)')
    parser.add_argument('--lookahead_strategy', choices=['greedy', 'beam'], default='greedy', 
                       help='AR: lookahead strategy - greedy or beam search (default: greedy)')
    parser.add_argument('--beam_width', type=int, default=3, help='AR: beam width for beam search (default: 3)')
    parser.add_argument('--pll_metric', choices=['original', 'within_word_l2r'],
                        default='original', help='MLM: PLL variant - "original" or "within_word_l2r" (default: original)')
    
    # Check if no arguments provided
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        print("\n❌ Error: No arguments provided. At minimum, you need:", file=sys.stderr)
        print("  --input_file <file> --mode <ar|mlm> --model <name>\n", file=sys.stderr)
        sys.exit(1)
    
    args = parser.parse_args()

    # Validate parameters using ScoringConfig
    try:
        ScoringConfig(
            mode=args.mode,
            model_name=args.model,
            top_k=args.top_k,
            lookahead_n=args.lookahead_n,
            lookahead_strategy=args.lookahead_strategy,
            beam_width=args.beam_width,
            pll_metric=args.pll_metric
        ).validate()
    except ValueError as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Build filename parts
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_short = args.model.replace('/', '_').split('-')[0]
    parts = [Path(args.input_file).stem, args.mode, model_short, f'k{args.top_k}']
    
    if args.left_context_file:
        parts.append('extra')
    if args.mode == 'ar' and args.lookahead_n > 0:
        parts.append(f'look{args.lookahead_n}')
        if args.lookahead_strategy == 'beam':
            parts.append(f'beam{args.beam_width}')
    if args.pll_metric == 'within_word_l2r':
        parts.append('L2R')
    
    parts.append(timestamp)
    generated_filename = '_'.join(parts) + '.tsv'
    
    # Determine output path
    output_path = Path(args.output_file)
    
    if output_path.is_dir() or (not output_path.exists() and output_path.suffix == ''):
        # Directory path (existing or to-be-created)
        output_path.mkdir(parents=True, exist_ok=True)
        final_output = output_path / generated_filename
        print(f"→ Output: {final_output}")
    elif args.output_file == 'simple_output.tsv':
        # Default
        final_output = Path(generated_filename)
        print(f"→ Output: {final_output}")
    else:
        # Specific filename
        final_output = output_path
        final_output.parent.mkdir(parents=True, exist_ok=True)
        print(f"→ Output: {final_output}")
    
    process_from_file(
        input_file=args.input_file,
        output_file=str(final_output),
        mode=args.mode,
        model_name=args.model,
        format_type=args.format,
        left_context_file=args.left_context_file,
        top_k=args.top_k,
        lookahead_n=args.lookahead_n,
        lookahead_strategy=args.lookahead_strategy,
        beam_width=args.beam_width,
        pll_metric=args.pll_metric
    )

if __name__ == "__main__":
    main()

