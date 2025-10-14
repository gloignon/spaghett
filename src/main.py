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
    --lookahead_n: (AR only) Number of greedy follow tokens after top-1 to show (default: 3).

Input TSV formats:
    - documents: doc_id<TAB>text (with header)
    - sentences: doc_id<TAB>sentence_id<TAB>sentence (with header)

Outputs TSV with columns:
    doc_id, sentence_id, token_index, token, is_special, is_context, surprisal_bits, entropy_bits, pred_top, top_k_1..top_k_N, pred_next_1..pred_next_N

Notes:
    - All tokens are now scored, including BOS, EOS, and other special tokens
    - is_special column: 1 if token is a special token (BOS/EOS/PAD/etc), 0 otherwise
    - is_context column: 1 if token belongs to left context, 0 if part of actual sentence
    - Users can filter by these columns in post-processing if desired
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


def find_token_subsequence(haystack: List[int], needle: List[int], start: int = 0) -> int:
    """Find first occurrence of needle in haystack starting at index. Returns -1 if not found."""
    if not needle:
        return -1
    
    for i in range(start, len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return -1


def get_context_boundary(token_ids: List[int], context_ids: List[int], special_mask: List[int]) -> int:
    """Get last index of context tokens. Returns -1 if no context or not found."""
    if not context_ids:
        return -1
    
    # Find first non-special token
    first_content = next((i for i, is_special in enumerate(special_mask) if not is_special), 0)
    
    # Find context subsequence
    ctx_start = find_token_subsequence(token_ids, context_ids, start=first_content)
    if ctx_start == -1:
        return -1
    
    return ctx_start + len(context_ids) - 1


# ============================================================================
# Scoring functions
# ============================================================================

def score_autoregressive(
    sentence: str,
    left_context: str,
    tokenizer,
    model,
    top_k: int = 5,
    lookahead_n: int = 3
) -> Tuple[List[str], List[int], List[int], List[float], List[float], List[List[str]]]:
    """
    Score sentence with AR model. Returns tokens, is_special flags, is_context flags,
    surprisals, entropies, and prediction columns.
    
    Now scores ALL tokens including special tokens (BOS, EOS, etc).
    """
    combined = combine_context_and_sentence(left_context, sentence)
    encoding = tokenizer(combined, return_tensors="pt", add_special_tokens=True)
    input_ids = encoding["input_ids"][0]
    
    if len(input_ids) < 2:
        return [], [], [], [], [], []
    
    # Get context boundary and special token mask
    special_mask = tokenizer.get_special_tokens_mask(input_ids.tolist(), already_has_special_tokens=True)
    context_ids = tokenizer(left_context, add_special_tokens=False)["input_ids"] if left_context else []
    ctx_last = get_context_boundary(input_ids.tolist(), context_ids, special_mask)
    
    # Get logits
    with torch.no_grad():
        logits = model(input_ids.unsqueeze(0)).logits[0]
    
    scored_tokens = []
    is_special_flags = []
    is_context_flags = []
    surprisals = []
    entropies = []
    pred_columns = []
    
    for pos in range(len(input_ids) - 1):
        target_pos = pos + 1
        target_id = input_ids[target_pos].item()
        
        # Compute probabilities
        pos_logits = logits[pos]
        probs = torch.softmax(pos_logits, dim=-1)
        log_probs = torch.log_softmax(pos_logits, dim=-1)
        
        # Surprisal and entropy
        surprisal = -log_probs[target_id].item() / LN2
        entropy = -(probs * log_probs).sum().item() / LN2
        
        # Top-k predictions
        top_k_probs, top_k_ids = torch.topk(probs, k=min(top_k, len(probs)))
        top_k_tokens = [tokenizer.decode([tid]) for tid in top_k_ids.tolist()]
        
        # Greedy lookahead
        lookahead_tokens = greedy_lookahead(model, tokenizer, input_ids[:target_pos + 1], n=lookahead_n)
        
        # Build prediction column: [pred_top, top_k_1..top_k_N, pred_next_1..pred_next_M]
        pred_col = [top_k_tokens[0]] + top_k_tokens + lookahead_tokens
        
        # Track flags
        is_special = 1 if special_mask[target_pos] else 0
        is_context = 1 if target_pos <= ctx_last else 0
        
        scored_tokens.append(tokenizer.decode([target_id]))
        is_special_flags.append(is_special)
        is_context_flags.append(is_context)
        surprisals.append(surprisal)
        entropies.append(entropy)
        pred_columns.append(pred_col)
    
    return scored_tokens, is_special_flags, is_context_flags, surprisals, entropies, pred_columns


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


def score_masked_lm(
    sentence: str,
    left_context: str,
    tokenizer,
    model,
    top_k: int = 5
) -> Tuple[List[str], List[int], List[int], List[float], List[float], List[List[str]]]:
    """
    Score sentence with MLM. Returns tokens, is_special flags, is_context flags,
    surprisals, entropies, and prediction columns.
    
    Now scores ALL tokens including special tokens (CLS, SEP, etc).
    """
    if tokenizer.mask_token_id is None:
        raise ValueError("Model doesn't have a [MASK] token. Use --mode ar instead.")
    
    combined = combine_context_and_sentence(left_context, sentence)
    encoding = tokenizer(combined, return_tensors="pt", add_special_tokens=True)
    input_ids = encoding["input_ids"][0]
    
    if len(input_ids) < 2:
        return [], [], [], [], [], []
    
    # Get context boundary and special token mask
    special_mask = tokenizer.get_special_tokens_mask(input_ids.tolist(), already_has_special_tokens=True)
    context_ids = tokenizer(left_context, add_special_tokens=False)["input_ids"] if left_context else []
    ctx_last = get_context_boundary(input_ids.tolist(), context_ids, special_mask)
    
    scored_tokens = []
    is_special_flags = []
    is_context_flags = []
    surprisals = []
    entropies = []
    pred_columns = []
    
    for pos in range(len(input_ids)):
        # Mask current position
        masked_ids = input_ids.clone()
        masked_ids[pos] = tokenizer.mask_token_id
        
        # Get predictions
        with torch.no_grad():
            logits = model(masked_ids.unsqueeze(0)).logits[0, pos]
        
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        
        actual_id = input_ids[pos].item()
        surprisal = -log_probs[actual_id].item() / LN2
        entropy = -(probs * log_probs).sum().item() / LN2
        
        # Top-k predictions
        top_k_probs, top_k_ids = torch.topk(probs, k=min(top_k, len(probs)))
        top_k_tokens = [tokenizer.decode([tid]) for tid in top_k_ids.tolist()]
        
        pred_col = [top_k_tokens[0]] + top_k_tokens
        
        # Track flags
        is_special = 1 if special_mask[pos] else 0
        is_context = 1 if pos <= ctx_last else 0
        
        # Use special token string if it's a special token, otherwise decode normally
        if is_special:
            token_str = tokenizer.convert_ids_to_tokens([actual_id])[0]
        else:
            token_str = tokenizer.decode([actual_id])
        
        scored_tokens.append(token_str)
        is_special_flags.append(is_special)
        is_context_flags.append(is_context)
        surprisals.append(surprisal)
        entropies.append(entropy)
        pred_columns.append(pred_col)
    
    return scored_tokens, is_special_flags, is_context_flags, surprisals, entropies, pred_columns


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
    
    # Build header
    base_cols = ['doc_id', 'sentence_id', 'token_index', 'token', 'is_special', 'is_context', 
                 'surprisal_bits', 'entropy_bits', 'pred_top']
    top_k_cols = [f'top_k_{i}' for i in range(1, top_k + 1)]
    
    if mode == 'ar':
        lookahead_cols = [f'pred_next_{i}' for i in range(1, lookahead_n + 1)]
        header = base_cols + top_k_cols + lookahead_cols
    else:
        header = base_cols + top_k_cols
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter='\t')
        writer.writeheader()
        writer.writerows(results)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compute per-token surprisal and entropy.")
    parser.add_argument('--input_file', required=True, help='Input TSV file')
    parser.add_argument('--output_file', default='simple_output.tsv', help='Output TSV file')
    parser.add_argument('--mode', choices=['ar', 'mlm'], required=True, help='Model mode')
    parser.add_argument('--model', required=True, help='Model name from HuggingFace')
    parser.add_argument('--format', choices=['documents', 'sentences'], required=True, help='Input format')
    parser.add_argument('--left_context_file', default='', help='File with left context')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top-k predictions')
    parser.add_argument('--lookahead_n', type=int, default=3, help='AR: greedy lookahead tokens')
    
    args = parser.parse_args()
    
    # Load context
    left_context = read_left_context(args.left_context_file) if args.left_context_file else ''
    
    # Load model
    print(f"Loading {args.mode.upper()} model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    if args.mode == 'ar':
        model = AutoModelForCausalLM.from_pretrained(args.model)
        score_fn = lambda s: score_autoregressive(
            s, left_context, tokenizer, model, args.top_k, args.lookahead_n
        )
    else:
        model = AutoModelForMaskedLM.from_pretrained(args.model)
        score_fn = lambda s: score_masked_lm(s, left_context, tokenizer, model, args.top_k)
    
    model.eval()
    
    # Load data
    print(f"Loading input from: {args.input_file}")
    data = load_input_data(args.input_file, args.format)
    
    # Process
    results = []
    for doc_id, sent_id, sentence in tqdm(data, desc="Processing"):
        tokens, is_special_flags, is_context_flags, surprisals, entropies, pred_cols = score_fn(sentence)
        
        for idx, (token, is_special, is_context, surp, ent, preds) in enumerate(
            zip(tokens, is_special_flags, is_context_flags, surprisals, entropies, pred_cols), 1
        ):
            row = {
                'doc_id': doc_id,
                'sentence_id': sent_id,
                'token_index': idx,
                'token': token,
                'is_special': is_special,
                'is_context': is_context,
                'surprisal_bits': f'{surp:.4f}',
                'entropy_bits': f'{ent:.4f}',
                'pred_top': preds[0]
            }
            
            # Add top-k columns
            for i in range(1, args.top_k + 1):
                row[f'top_k_{i}'] = preds[i] if i < len(preds) else ''
            
            # Add lookahead columns (AR only)
            if args.mode == 'ar':
                offset = args.top_k + 1
                for i in range(1, args.lookahead_n + 1):
                    row[f'pred_next_{i}'] = preds[offset + i - 1] if offset + i - 1 < len(preds) else ''
            
            results.append(row)
    
    # Write output
    write_output(args.output_file, results, args.top_k, args.lookahead_n, args.mode)
    print(f"Results written to: {args.output_file}")


if __name__ == '__main__':
    main()