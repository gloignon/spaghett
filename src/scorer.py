import math
from typing import List, Tuple, Optional

import torch

LN2 = math.log(2.0)

# ============================================================================
# Helper functions
# ============================================================================

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


def extract_attention_tuples(
    attention_weights: torch.Tensor,
    tokenizer,
    full_token_ids: torch.Tensor,
    sentence_start: int,
    sentence_length: int
) -> List[Tuple[int, str, int, int, int, str, float]]:
    """
    Extract attention weights as tuples including context tokens.
    
    Args:
        attention_weights: Tensor of shape (num_layers, num_heads, seq_len, seq_len)
                          or (num_heads, seq_len, seq_len) if single layer
        tokenizer: Tokenizer to identify special tokens and decode tokens
        full_token_ids: Full sequence of token IDs (context + sentence)
        sentence_start: Start position of sentence in sequence
        sentence_length: Number of tokens in sentence
    
    Returns:
        List of (token_idx, token_str, is_context, is_special, rx_token_idx, rx_token_str, weight)
        where token_idx is 1-indexed position in full sequence
    """
    # Handle both single-layer and multi-layer attention
    if attention_weights.dim() == 4:
        # Average across layers and heads: (layers, heads, seq, seq) -> (seq, seq)
        attn = attention_weights.mean(dim=(0, 1))
    elif attention_weights.dim() == 3:
        # Average across heads: (heads, seq, seq) -> (seq, seq)
        attn = attention_weights.mean(dim=0)
    else:
        raise ValueError(f"Unexpected attention shape: {attention_weights.shape}")
    
    seq_len = len(full_token_ids)
    special_mask = tokenizer.get_special_tokens_mask(
        full_token_ids.tolist(), 
        already_has_special_tokens=True
    )
    
    # Convert all token IDs to strings (raw tokens with special characters)
    token_strings = tokenizer.convert_ids_to_tokens(full_token_ids.tolist())
    
    # Convert to tuples (token_idx, token_str, is_context, is_special, rx_token_idx, rx_token_str, weight)
    tuples = []
    for src_idx in range(seq_len):
        is_context_src = 1 if src_idx < sentence_start else 0
        is_special_src = special_mask[src_idx]
        src_token = token_strings[src_idx]
        
        for tgt_idx in range(seq_len):
            tgt_token = token_strings[tgt_idx]
            weight = attn[src_idx, tgt_idx].item()
            # 1-indexed token positions
            tuples.append((
                src_idx + 1,      # token_id (1-indexed)
                src_token,        # token string
                is_context_src,   # is_context
                is_special_src,   # is_special
                tgt_idx + 1,      # rx_token_id (1-indexed)
                tgt_token,        # rx_token string
                weight            # attn_score
            ))
    
    return tuples


# ============================================================================
# Core scoring functions
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
    context_ids: List[int] = None,
    output_attentions: bool = False
) -> Tuple[List[str], List[str], List[int], List[float], List[float], List[List[str]], Optional[List[Tuple[int, str, int, int, int, str, float]]]]:
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
            - Optional[List[Tuple[...]]]: Attention tuples (token_id, token, is_context, is_special, rx_token_id, rx_token, weight)
    """
    # Handle empty or whitespace-only sentences
    if not sentence or not sentence.strip():
        return [], [], [], [], [], [], None if output_attentions else []
    
    # Tokenize sentence
    encoding = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
    sentence_ids = encoding["input_ids"][0]
    
    if not sentence_ids.numel():
        return [], [], [], [], [], [], None if output_attentions else []
    
    # Prepare input with context
    input_ids, sentence_start = prepare_input_with_context(sentence_ids, context_ids)
    special_mask = tokenizer.get_special_tokens_mask(sentence_ids.tolist(), already_has_special_tokens=True)
    
    # Get logits (and optionally attention) for the full sequence
    with torch.no_grad():
        outputs = model(input_ids.unsqueeze(0), output_attentions=output_attentions)
        logits = outputs.logits[0]  # shape: (seq_len, vocab_size)
    
    # Extract attention if requested
    attention_tuples = None
    if output_attentions and outputs.attentions is not None:
        # Stack all layers: (num_layers, batch, num_heads, seq_len, seq_len)
        all_attentions = torch.stack(outputs.attentions)[:, 0, :, :, :]  # Remove batch dim
        attention_tuples = extract_attention_tuples(
            all_attentions,
            tokenizer,
            input_ids,
            sentence_start,
            len(sentence_ids)
        )
    
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
    
    return raw_tokens, scored_tokens, is_special_flags, surprisals, entropies, pred_columns, attention_tuples


def score_masked_lm(
    sentence: str,
    tokenizer,
    model,
    top_k: int = 5,
    context_ids: List[int] = None,
    output_attentions: bool = False
) -> Tuple[List[str], List[str], List[int], List[float], List[float], List[List[str]], Optional[List[Tuple[int, str, int, int, int, str, float]]]]:
    """
    Score sentence with MLM using parallel masking (original PLL).
    """
    if tokenizer.mask_token_id is None:
        raise ValueError("Model doesn't have a [MASK] token. Use --mode ar instead.")
    
    # Handle empty or whitespace-only sentences
    if not sentence or not sentence.strip():
        return [], [], [], [], [], [], None if output_attentions else []
    
    # Tokenize sentence
    encoding = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
    sentence_ids = encoding["input_ids"][0]
    
    if not sentence_ids.numel():
        return [], [], [], [], [], [], None if output_attentions else []
    
    # Prepare input with context
    base_ids, sentence_start = prepare_input_with_context(sentence_ids, context_ids)
    special_mask = tokenizer.get_special_tokens_mask(sentence_ids.tolist(), already_has_special_tokens=True)
    
    num_tokens = len(sentence_ids)
    
    # For attention extraction, we'll use the unmasked input
    attention_tuples = None
    if output_attentions:
        with torch.no_grad():
            attn_outputs = model(base_ids.unsqueeze(0), output_attentions=True)
            if attn_outputs.attentions is not None:
                all_attentions = torch.stack(attn_outputs.attentions)[:, 0, :, :, :]
                attention_tuples = extract_attention_tuples(
                    all_attentions,
                    tokenizer,
                    base_ids,
                    sentence_start,
                    num_tokens
                )
    
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
    
    return raw_tokens, scored_tokens, is_special_flags, surprisals, entropies, pred_columns, attention_tuples


def score_masked_lm_l2r(
    sentence: str,
    tokenizer,
    model,
    top_k: int = 5,
    context_ids: List[int] = None,
    output_attentions: bool = False
) -> Tuple[List[str], List[str], List[int], List[float], List[float], List[List[str]], Optional[List[Tuple[int, str, int, int, int, str, float]]]]:
    """
    Score sentence with MLM using within_word_l2r (minicons-compatible).
    """
    if tokenizer.mask_token_id is None:
        raise ValueError("Model doesn't have a [MASK] token. Use --mode ar instead.")
    
    # Handle empty or whitespace-only sentences
    if not sentence or not sentence.strip():
        return [], [], [], [], [], [], None if output_attentions else []
    
    # Tokenize sentence
    encoding = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
    sentence_ids = encoding["input_ids"][0]
    if not sentence_ids.numel():
        return [], [], [], [], [], [], None if output_attentions else []
    
    # Base + left context
    base_ids, sentence_start = prepare_input_with_context(sentence_ids, context_ids)
    seq_len = len(sentence_ids)
    special_mask = tokenizer.get_special_tokens_mask(sentence_ids.tolist(), already_has_special_tokens=True)
    
    # For attention extraction, use unmasked input
    attention_tuples = None
    if output_attentions:
        with torch.no_grad():
            attn_outputs = model(base_ids.unsqueeze(0), output_attentions=True)
            if attn_outputs.attentions is not None:
                all_attentions = torch.stack(attn_outputs.attentions)[:, 0, :, :, :]
                attention_tuples = extract_attention_tuples(
                    all_attentions,
                    tokenizer,
                    base_ids,
                    sentence_start,
                    seq_len
                )
    
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
        return [], [], [], [], [], [], None if output_attentions else []
    
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
    
    return raw_tokens, scored_tokens, is_special_flags, surprisals, entropies, pred_columns, attention_tuples



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

