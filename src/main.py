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
    
    pred_top (str): 
        The most probable token predicted by the model at this position.
        AR mode: most probable next token given context up to current position.
        MLM mode: most probable token for masked position given bidirectional context.
    
    top_k_1 through top_k_N (str): 
        The top-k most probable tokens, ranked by probability (1=highest).
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
    - All tokens are scored, including special tokens (BOS/EOS/CLS/SEP/etc).
    - First token in AR mode has empty surprisal/entropy (no prior context).
    - Users can filter rows using is_special columns in post-processing.
    - Surprisal and entropy are in bits (log base 2) for interpretability.
    - Designed for simplicity and robustness over performance (no batching, no GPU acceleration).
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
    beam_width: int = 3
) -> Tuple[List[str], List[int], List[float], List[float], List[List[str]]]:
    """
    Score sentence with AR model. Returns tokens, is_special flags,
    surprisals, entropies, and prediction columns.
    
    Left context is used for conditioning but NOT included in output.
    """
    combined = combine_context_and_sentence(left_context, sentence)
    encoding = tokenizer(combined, return_tensors="pt", add_special_tokens=True)
    input_ids = encoding["input_ids"][0]
    
    if len(input_ids) < 1:
        return [], [], [], [], []
    
    # Get context boundary and special token mask
    special_mask = tokenizer.get_special_tokens_mask(input_ids.tolist(), already_has_special_tokens=True)
    context_ids = tokenizer(left_context, add_special_tokens=False)["input_ids"] if left_context else []
    ctx_last = get_context_boundary(input_ids.tolist(), context_ids, special_mask)
    
    # Get logits
    if len(input_ids) > 1:
        with torch.no_grad():
            logits = model(input_ids.unsqueeze(0)).logits[0]
    else:
        logits = None
    
    scored_tokens = []
    is_special_flags = []
    surprisals = []
    entropies = []
    pred_columns = []
    
    # Process all tokens starting from index 0
    for pos in range(len(input_ids)):
        target_id = input_ids[pos].item()
        
        # Track flags
        is_special = 1 if special_mask[pos] else 0
        is_context = pos <= ctx_last
        
        # SKIP context tokens - don't add them to output
        if is_context:
            continue
        
        # Use special token string if it's a special token, otherwise decode normally
        if is_special:
            token_str = tokenizer.convert_ids_to_tokens([target_id])[0]
        else:
            token_str = tokenizer.decode([target_id])
        
        # For position 0 (or first sentence token after context), check if we can score it
        if pos == 0 or (ctx_last >= 0 and pos == ctx_last + 1):
            # If it's the very first token overall, can't score
            if pos == 0:
                surprisal = float('nan')
                entropy = float('nan')
                pred_col = [''] * (1 + top_k + lookahead_n)
            else:
                # First sentence token after context - CAN be scored using context
                pos_logits = logits[pos - 1]
                probs = torch.softmax(pos_logits, dim=-1)
                log_probs = torch.log_softmax(pos_logits, dim=-1)
                
                surprisal = -log_probs[target_id].item() / LN2
                entropy = -(probs * log_probs).sum().item() / LN2
                
                top_k_probs, top_k_ids = torch.topk(probs, k=min(top_k, len(probs)))
                top_k_tokens = [tokenizer.decode([tid]) for tid in top_k_ids.tolist()]
                
                if lookahead_strategy == 'beam':
                    lookahead_tokens = beam_search_lookahead(model, tokenizer, input_ids[:pos + 1], 
                                                             n=lookahead_n, beam_width=beam_width)
                else:
                    lookahead_tokens = greedy_lookahead(model, tokenizer, input_ids[:pos + 1], n=lookahead_n)
                
                pred_col = [top_k_tokens[0]] + top_k_tokens + lookahead_tokens
        else:
            # Regular token scoring
            pos_logits = logits[pos - 1]
            probs = torch.softmax(pos_logits, dim=-1)
            log_probs = torch.log_softmax(pos_logits, dim=-1)
            
            surprisal = -log_probs[target_id].item() / LN2
            entropy = -(probs * log_probs).sum().item() / LN2
            
            top_k_probs, top_k_ids = torch.topk(probs, k=min(top_k, len(probs)))
            top_k_tokens = [tokenizer.decode([tid]) for tid in top_k_ids.tolist()]
            
            if lookahead_strategy == 'beam':
                lookahead_tokens = beam_search_lookahead(model, tokenizer, input_ids[:pos + 1], 
                                                         n=lookahead_n, beam_width=beam_width)
            else:
                lookahead_tokens = greedy_lookahead(model, tokenizer, input_ids[:pos + 1], n=lookahead_n)
            
            pred_col = [top_k_tokens[0]] + top_k_tokens + lookahead_tokens
        
        scored_tokens.append(token_str)
        is_special_flags.append(is_special)
        surprisals.append(surprisal)
        entropies.append(entropy)
        pred_columns.append(pred_col)
    
    return scored_tokens, is_special_flags, surprisals, entropies, pred_columns


def score_masked_lm(
    sentence: str,
    left_context: str,
    tokenizer,
    model,
    top_k: int = 5
) -> Tuple[List[str], List[int], List[float], List[float], List[List[str]]]:
    """
    Score sentence with MLM. Returns tokens, is_special flags,
    surprisals, entropies, and prediction columns.
    
    Left context is used for conditioning but NOT included in output.
    """
    if tokenizer.mask_token_id is None:
        raise ValueError("Model doesn't have a [MASK] token. Use --mode ar instead.")
    
    combined = combine_context_and_sentence(left_context, sentence)
    encoding = tokenizer(combined, return_tensors="pt", add_special_tokens=True)
    input_ids = encoding["input_ids"][0]
    
    if len(input_ids) < 2:
        return [], [], [], [], []
    
    # Get context boundary and special token mask
    special_mask = tokenizer.get_special_tokens_mask(input_ids.tolist(), already_has_special_tokens=True)
    context_ids = tokenizer(left_context, add_special_tokens=False)["input_ids"] if left_context else []
    ctx_last = get_context_boundary(input_ids.tolist(), context_ids, special_mask)
    
    scored_tokens = []
    is_special_flags = []
    surprisals = []
    entropies = []
    pred_columns = []
    
    for pos in range(len(input_ids)):
        # Track flags
        is_special = 1 if special_mask[pos] else 0
        is_context = pos <= ctx_last
        
        # SKIP context tokens - don't add them to output
        if is_context:
            continue
        
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
        
        # Use special token string if it's a special token, otherwise decode normally
        if is_special:
            token_str = tokenizer.convert_ids_to_tokens([actual_id])[0]
        else:
            token_str = tokenizer.decode([actual_id])
        
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
    
    # Build header (removed is_context)
    base_cols = ['doc_id', 'sentence_id', 'token_index', 'token', 'is_special', 
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
    parser.add_argument('--lookahead_n', type=int, default=3, help='AR: number of lookahead tokens')
    parser.add_argument('--lookahead_strategy', choices=['greedy', 'beam'], default='greedy', 
                       help='AR: lookahead strategy (greedy or beam search)')
    parser.add_argument('--beam_width', type=int, default=3, help='AR: beam width for beam search')
    
    args = parser.parse_args()
    
    # Load context
    left_context = read_left_context(args.left_context_file) if args.left_context_file else ''
    
    # Load model
    print(f"Loading {args.mode.upper()} model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    if args.mode == 'ar':
        model = AutoModelForCausalLM.from_pretrained(args.model)
        score_fn = lambda s: score_autoregressive(
            s, left_context, tokenizer, model, args.top_k, args.lookahead_n, 
            args.lookahead_strategy, args.beam_width
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
                'entropy_bits': '' if math.isnan(ent) else f'{ent:.4f}',
                'pred_top': preds[0] if preds else ''
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