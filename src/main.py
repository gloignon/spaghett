# pyright: reportGeneralTypeIssues=false
"""
Simple script for computing per-token surprisal and entropy, by sentence, with extra left context,
and exporting the top-k most probable next tokens for each scored token.

CLI parameters:
    --input_file: Path to the input TSV file with documents or sentences.
    --output_file: Path to the output TSV file (default: simple_output.tsv).
    --mode: 'ar' for autoregressive (GPT-style) or 'mlm' for masked language model (BERT-style).
    --model: Name of the pre-trained model to use (e.g., 'gpt2', 'bert-base-uncased').
    --format: 'documents' or 'sentences' to specify input format.
    --left_context_file: Path to a .txt file whose contents are prepended to every sentence.
    --top_k: Number of top probable tokens to output (default: 5).
    --lookahead_n: (AR only) Number of follow tokens to generate (default: 3).
    --lookahead_strategy: (AR only) Strategy for generating follow tokens: 'greedy' or 'beam' (default: greedy).
    --beam_width: (AR only) Beam width for beam search (default: 3, only used when --lookahead_strategy=beam).

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
        The actual token text. Special tokens (BOS/EOS/CLS/SEP) shown as token symbols (e.g., '<s>', '</s>').
    
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
    - All tokens are scored, including special tokens (BOS/EOS/CLS/SEP/etc) when the model produces them.
    - In AR mode we also compute surprisal and entropy for the fist token if no prior context is provided. Filter out if undesired.
    - Don't want special tokens? You can filter them out easily using the is_special column in post-processing.
    - Surprisal and entropy are in bits (log base 2) for interpretability.
    - Designed for simplicity and robustness over performance (limited batching, no GPU acceleration).
"""

import argparse
import csv
import math
import re
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
from tqdm import tqdm

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
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def combine_context_and_sentence(context: str, sentence: str) -> str:
    """Combine left context with sentence, adding space if both non-empty."""
    if context and sentence:
        return f"{context} {sentence}"
    return context or sentence


def decode_token(tokenizer, token_id: int, is_special: bool) -> str:
    """Decode token, handling special tokens appropriately."""
    if is_special:
        return tokenizer.convert_ids_to_tokens([token_id])[0]
    return tokenizer.decode([token_id])


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
    """Get top-k most probable tokens from logits."""
    probs = torch.softmax(logits, dim=-1)
    top_k_probs, top_k_ids = torch.topk(probs, k=min(k, len(probs)))
    return [tokenizer.decode([tid]) for tid in top_k_ids.tolist()]


# ============================================================================
# Lookahead generation functions
# ============================================================================

def greedy_lookahead(model, tokenizer, prefix_ids: torch.Tensor, n: int) -> List[str]:
    """Generate n greedy next tokens from prefix."""
    if n <= 0:
        return []
    
    current = prefix_ids.clone()
    lookahead = []
    
    for _ in range(n):
        with torch.no_grad():
            logits = model(current.unsqueeze(0)).logits[0, -1]
        
        next_id = logits.argmax().item()
        lookahead.append(tokenizer.decode([next_id]))
        current = torch.cat([current, torch.tensor([next_id])])
    
    return lookahead

# This will be find the most probable sequence of n tokens (forming a word) using beam search
def beam_search_lookahead(model, tokenizer, prefix_ids: torch.Tensor, n: int, beam_width: int = 3) -> List[str]:
    """Generate n tokens using beam search and return the best sequence."""
    if n <= 0:
        return []
    
    # Initialize beams: [(sequence, cumulative_log_prob)]
    beams = [(prefix_ids.clone(), 0.0)]
    
    for step in range(n):
        candidates = []
        
        for seq, score in beams:
            with torch.no_grad():
                logits = model(seq.unsqueeze(0)).logits[0, -1]
            
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Get top beam_width candidates
            top_log_probs, top_ids = torch.topk(log_probs, k=min(beam_width, len(log_probs)))
            
            for log_prob, token_id in zip(top_log_probs.tolist(), top_ids.tolist()):
                new_seq = torch.cat([seq, torch.tensor([token_id])])
                new_score = score + log_prob
                candidates.append((new_seq, new_score))
        
        # Keep top beam_width sequences
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    
    # Return best sequence (excluding prefix)
    best_seq = beams[0][0]
    lookahead_ids = best_seq[len(prefix_ids):].tolist()
    return [tokenizer.decode([tid]) for tid in lookahead_ids]


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
) -> Tuple[List[str], List[int], List[float], List[float], List[List[str]]]:
    """
    Score sentence with AR model.

    Returns:
        Tuple containing:
            - List[str]: Decoded tokens for the sentence.
            - List[int]: Flags indicating if each token is a special token (1) or not (0).
            - List[float]: Surprisal values (in bits) for each token.
            - List[float]: Entropy values (in bits) for each token.
            - List[List[str]]: For each token, a list containing the most probable token, top-k predictions, and lookahead tokens.
    """
    # Tokenize sentence
    encoding = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
    sentence_ids = encoding["input_ids"][0]
    
    if not sentence_ids.numel():
        return [], [], [], [], []
    
    # Prepare input with context
    input_ids, sentence_start = prepare_input_with_context(sentence_ids, context_ids)
    special_mask = tokenizer.get_special_tokens_mask(sentence_ids.tolist(), already_has_special_tokens=True)
    
    # Get logits for the full sequence
    with torch.no_grad():
        outputs = model(input_ids.unsqueeze(0))
        logits = outputs.logits[0]  # shape: (seq_len, vocab_size)
    
    # Score each token
    scored_tokens = []
    is_special_flags = []
    surprisals = []
    entropies = []
    pred_columns = []
    
    for sent_pos in range(len(sentence_ids)):
        combined_pos = sentence_start + sent_pos
        target_id = sentence_ids[sent_pos].item()
        is_special = 1 if special_mask[sent_pos] else 0
        
        # Decode token
        token_str = decode_token(tokenizer, target_id, is_special)
        
        # Score based on position
        if combined_pos == 0:
            # First token with no context: use unconditional distribution
            # This is p(token | start_of_sequence)
            # For most AR models, this is the distribution after BOS/empty prompt
            
            # We can't condition on anything, so we use the model's implicit
            # prior distribution. We'll use a minimal input to get this.
            with torch.no_grad():
                # Get logits for empty/minimal context
                # Some models need at least BOS token, others work with empty
                if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
                    # Use BOS token as minimal context
                    minimal_input = torch.tensor([[tokenizer.bos_token_id]])
                    minimal_logits = model(minimal_input).logits[0, -1]
                else:
                    # No BOS - use the first position logits from current forward pass
                    # This represents p(first_token | empty_context)
                    minimal_logits = logits[0] if len(logits) > 0 else None
            
            if minimal_logits is not None:
                # Calculate surprisal and entropy
                surprisal, entropy = compute_surprisal_entropy(minimal_logits, target_id)
                
                # Get predictions
                top_k_tokens = get_top_k_predictions(minimal_logits, tokenizer, top_k)
                
                # Get lookahead (from current first token)
                if lookahead_n > 0:
                    if lookahead_strategy == 'beam':
                        lookahead = beam_search_lookahead(model, tokenizer, input_ids[:1], lookahead_n, beam_width)
                    else:
                        lookahead = greedy_lookahead(model, tokenizer, input_ids[:1], lookahead_n)
                else:
                    lookahead = []
                
                pred_col = top_k_tokens + lookahead
            else:
                # Fallback if we can't get minimal logits
                surprisal = entropy = float('nan')
                pred_col = [''] * (1 + top_k + lookahead_n)
        
        else:
            # Regular token: use previous position's logits
            surprisal, entropy = compute_surprisal_entropy(logits[combined_pos - 1], target_id)
            
            # Get predictions
            top_k_tokens = get_top_k_predictions(logits[combined_pos - 1], tokenizer, top_k)
            
            # Get lookahead
            if lookahead_n > 0:
                if lookahead_strategy == 'beam':
                    lookahead = beam_search_lookahead(model, tokenizer, input_ids[:combined_pos + 1], lookahead_n, beam_width)
                else:
                    lookahead = greedy_lookahead(model, tokenizer, input_ids[:combined_pos + 1], lookahead_n)
            else:
                lookahead = []
            
            pred_col = top_k_tokens + lookahead
        
        scored_tokens.append(token_str)
        is_special_flags.append(is_special)
        surprisals.append(surprisal)
        entropies.append(entropy)
        pred_columns.append(pred_col)
    
    return scored_tokens, is_special_flags, surprisals, entropies, pred_columns


def score_masked_lm(
    sentence: str,
    tokenizer,
    model,
    top_k: int = 5,
    context_ids: List[int] = None
) -> Tuple[List[str], List[int], List[float], List[float], List[List[str]]]:
    """
    Score sentence with MLM using parallel masking.
    
    Instead of N forward passes (one per token), we do ONE forward pass
    with all masked positions batched together. This should be faster.
    
    Left context is used for conditioning but NOT included in output.
    """
    if tokenizer.mask_token_id is None:
        raise ValueError("Model doesn't have a [MASK] token. Use --mode ar instead.")
    
    # Tokenize sentence
    encoding = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
    sentence_ids = encoding["input_ids"][0]
    
    if not sentence_ids.numel():
        return [], [], [], [], []
    
    # Prepare input with context
    base_ids, sentence_start = prepare_input_with_context(sentence_ids, context_ids)
    special_mask = tokenizer.get_special_tokens_mask(sentence_ids.tolist(), already_has_special_tokens=True)
    
    num_tokens = len(sentence_ids)
    
    # Create ALL masked versions at once
    masked_batch = []
    for sent_pos in range(num_tokens):
        combined_pos = sentence_start + sent_pos
        masked_ids = base_ids.clone()
        masked_ids[combined_pos] = tokenizer.mask_token_id
        masked_batch.append(masked_ids)
    
    # Stack into batch: (num_tokens, seq_len)
    masked_batch = torch.stack(masked_batch)
    
    # ONE forward pass for all masked positions!
    with torch.no_grad():
        outputs = model(masked_batch)
        all_logits = outputs.logits  # (num_tokens, seq_len, vocab_size)
    
    # Extract logits for each masked position and score
    scored_tokens = []
    is_special_flags = []
    surprisals = []
    entropies = []
    pred_columns = []
    
    for sent_pos in range(num_tokens):
        combined_pos = sentence_start + sent_pos
        target_id = sentence_ids[sent_pos].item()
        is_special = 1 if special_mask[sent_pos] else 0
        
        # Get logits for this token's masked position
        logits = all_logits[sent_pos, combined_pos]
        
        # Calculate surprisal and entropy
        surprisal, entropy = compute_surprisal_entropy(logits, target_id)
        
        # Get top-k predictions
        top_k_tokens = get_top_k_predictions(logits, tokenizer, top_k)
        # pred_col = [top_k_tokens[0]] + top_k_tokens
        pred_col = top_k_tokens
        
        # Decode token
        token_str = decode_token(tokenizer, target_id, is_special)
        
        scored_tokens.append(token_str)
        is_special_flags.append(is_special)
        surprisals.append(surprisal)
        entropies.append(entropy)
        pred_columns.append(pred_col)
    
    return scored_tokens, is_special_flags, surprisals, entropies, pred_columns


# ============================================================================
# I/O functions
# ============================================================================

def load_input_data(input_file: str, format_type: str) -> List[Tuple[str, str, str]]:
    """Load input TSV and return list of (doc_id, sentence_id, sentence)."""
    data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # Skip header
        
        for row_num, row in enumerate(reader, start=2):
            if not row or all(cell.strip() == '' for cell in row):
                continue
                
            if format_type == 'documents':
                if len(row) < 2:
                    print(f"Warning: Skipping malformed row {row_num}: expected 2 columns, got {len(row)}")
                    continue
                doc_id, text = row[0], row[1]
                sentences = simple_sentence_split(text)
                for sent_idx, sent in enumerate(sentences, 1):
                    data.append((doc_id, str(sent_idx), sent))
            else:  # sentences
                if len(row) < 3:
                    print(f"Warning: Skipping malformed row {row_num}: expected 3 columns, got {len(row)}")
                    continue
                doc_id, sent_id, sentence = row[0], row[1], row[2]
                data.append((doc_id, sent_id, sentence))
    
    return data


def write_output(output_file: str, results: List[dict], top_k: int, lookahead_n: int, mode: str):
    """Write results to TSV with dynamic column headers."""
    if not results:
        return
    
    # Build column names
    columns = ['doc_id', 'sentence_id', 'token_index', 'token', 'is_special', 
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
    progress: bool = True
) -> List[dict]:
    """
    Process sentences and return results as list of dicts.
    
    This is the core function - independent of input/output format.
    Can be called from CLI, R, or other Python scripts.
    
    Args:
        sentences: List of sentence strings to score
        mode: 'ar' or 'mlm'
        model_name: HuggingFace model identifier
        left_context: Context string prepended to each sentence (default: '')
        top_k: Number of top predictions to return
        lookahead_n: (AR only) Number of lookahead tokens
        lookahead_strategy: (AR only) 'greedy' or 'beam'
        beam_width: (AR only) Beam width for beam search
        doc_ids: Optional document IDs (default: auto-generated)
        sentence_ids: Optional sentence IDs (default: auto-generated)
        progress: Show progress bar (default: True)
    
    Returns:
        List of dicts with columns matching output TSV format
    """
    # Auto-generate IDs if not provided
    if doc_ids is None:
        doc_ids = ['doc1'] * len(sentences)
    if sentence_ids is None:
        sentence_ids = [str(i) for i in range(1, len(sentences) + 1)]
    
    if len(doc_ids) != len(sentences) or len(sentence_ids) != len(sentences):
        raise ValueError("doc_ids and sentence_ids must match length of sentences")
    
    # Load model
    print(f"Loading {mode.upper()} model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Pre-tokenize context once
    context_ids = tokenizer(left_context, add_special_tokens=False)["input_ids"] if left_context else []
    
    if mode == 'ar':
        model = AutoModelForCausalLM.from_pretrained(model_name)
        score_fn = lambda s: score_autoregressive(
            s, left_context, tokenizer, model, top_k, lookahead_n,
            lookahead_strategy, beam_width, context_ids
        )
    elif mode == 'mlm':
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        score_fn = lambda s: score_masked_lm(s, tokenizer, model, top_k, context_ids)
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'ar' or 'mlm'")
    
    model.eval()
    
    # Process sentences
    results = []
    iterator = zip(doc_ids, sentence_ids, sentences)
    if progress:
        iterator = tqdm(iterator, total=len(sentences), desc="Processing")
    
    for doc_id, sent_id, sentence in iterator:
        tokens, is_special_flags, surprisals, entropies, pred_cols = score_fn(sentence)
        
        for idx, (token, is_special, surp, ent, preds) in enumerate(
            zip(tokens, is_special_flags, surprisals, entropies, pred_cols), 1
        ):
            row = {
                'doc_id': doc_id,
                'sentence_id': sent_id,
                'token_index': idx,
                'token': token,
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
    beam_width: int = 3
):
    """
    Process input TSV file and write output TSV file.
    
    This function handles I/O for CLI usage.
    """
    # Load context
    left_context = read_left_context(left_context_file) if left_context_file else ''
    
    # Load data
    print(f"Loading input from: {input_file}")
    data = load_input_data(input_file, format_type)
    
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
        progress=True
    )
    
    # Write output
    write_output(output_file, results, top_k, lookahead_n, mode)
    print(f"Results written to: {output_file}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Compute per-token surprisal and entropy.")
    parser.add_argument('--input_file', required=True, help='Input TSV file')
    parser.add_argument('--output_file', default='simple_output.tsv', help='Output TSV file')
    parser.add_argument('--mode', choices=['ar', 'mlm'], required=True, help='Model mode')
    parser.add_argument('--model', required=True, help='Model name from HuggingFace')
    parser.add_argument('--format', choices=['documents', 'sentences'], required=True, help='Input format')
    parser.add_argument('--left_context_file', default='', help='File with left context')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top-k predictions')
    parser.add_argument('--lookahead_n', type=int, default=3, help='AR: number of lookahead tokens')
    parser.add_argument('--lookahead_strategy', choices=['greedy', 'beam'], default='greedy', 
                       help='AR: lookahead strategy (greedy or beam search)')
    parser.add_argument('--beam_width', type=int, default=3, help='AR: beam width for beam search')
    
    args = parser.parse_args()
    
    process_from_file(
        input_file=args.input_file,
        output_file=args.output_file,
        mode=args.mode,
        model_name=args.model,
        format_type=args.format,
        left_context_file=args.left_context_file,
        top_k=args.top_k,
        lookahead_n=args.lookahead_n,
        lookahead_strategy=args.lookahead_strategy,
        beam_width=args.beam_width
    )


if __name__ == '__main__':
    main()