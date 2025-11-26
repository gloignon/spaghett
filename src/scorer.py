'''
scorer.py
Scoring functions for autoregressive and masked language models.
Includes surprisal, entropy, top-k predictions, lookahead generation.

CLI stuff is in cli.py
attention stuff is no longer in this project!
'''
import math
import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional

LN2 = math.log(2.0)

# ============================================================================
# Data structures
# ============================================================================

@dataclass
class ScoringResult:
    """Container for scoring results."""
    scored_tokens: List[str]
    raw_tokens: List[str]
    is_special_flags: List[int]
    surprisals: List[float]
    entropies: List[float]
    pred_columns: List[List[str]] = None

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
    log_probs = torch.log_softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k)
    top_tokens = tokenizer.convert_ids_to_tokens(top_ids.tolist())
    # For each top token, also compute counterfactual surprisal
    pred_info = []
    for token, token_id in zip(top_tokens, top_ids.tolist()):
        cf_surprisal = -log_probs[token_id].item() / LN2
        pred_info.append((token, cf_surprisal))
    return pred_info

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

def validate_sentence_inputs(
    sentence: str,
    sentence_ids: Optional[torch.Tensor] = None
):
    """
    Return the canonical empty-result when the sentence or its token IDs are empty.
    """
    if not sentence or not sentence.strip():
        return ScoringResult(
            scored_tokens=[],
            raw_tokens=[],
            is_special_flags=[],
            surprisals=[],
            entropies=[],
            pred_columns=[]
        )
    if sentence_ids is not None and not sentence_ids.numel():
        return ScoringResult(
            scored_tokens=[],
            raw_tokens=[],
            is_special_flags=[],
            surprisals=[],
            entropies=[],
            pred_columns=[]
        )
    return None

def tokenize_sentence(sentence: str, tokenizer):
    """
    Tokenize the sentence with special tokens and return both the encoding and tensor IDs.
    """
    encoding = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
    sentence_ids = encoding["input_ids"][0]
    return encoding, sentence_ids

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
    context_ids: List[int] = None
) -> ScoringResult:
    """
    Score sentence with AR model.

    Returns:
        ScoringResult dataclass containing tokens, scores, predictions, and optional attentions.
    """
    empty_result = validate_sentence_inputs(sentence)
    if empty_result is not None:
        return empty_result

    _, sentence_ids = tokenize_sentence(sentence, tokenizer)
    empty_ids_result = validate_sentence_inputs(sentence, sentence_ids)
    if empty_ids_result is not None:
        return empty_ids_result

    input_ids, sentence_start = prepare_input_with_context(sentence_ids, context_ids)
    special_mask = tokenizer.get_special_tokens_mask(sentence_ids.tolist(), already_has_special_tokens=True)

    with torch.no_grad():
        outputs = model(input_ids.unsqueeze(0))
        logits = outputs.logits[0]  # shape: (seq_len, vocab_size)

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

        raw_token, token_str = decode_token(tokenizer, target_id, is_special)

        if combined_pos == 0:
            with torch.no_grad():
                if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
                    minimal_input = torch.tensor([[tokenizer.bos_token_id]])
                    minimal_logits = model(minimal_input).logits[0, -1]
                else:
                    minimal_logits = logits[0] if len(logits) > 0 else None


            if minimal_logits is not None:
                surprisal, entropy = compute_surprisal_entropy(minimal_logits, target_id)
                top_k_preds = get_top_k_predictions(minimal_logits, tokenizer, top_k)

                if lookahead_n > 0:
                    if lookahead_strategy == 'beam':
                        lookahead = beam_search_lookahead(model, tokenizer, input_ids[:1], lookahead_n, beam_width)
                    else:
                        lookahead = greedy_lookahead(model, tokenizer, input_ids[:1], lookahead_n)
                else:
                    lookahead = []

                pred_col = top_k_preds + lookahead
            else:
                surprisal = entropy = float('nan')
                pred_col = [''] * (1 + top_k + lookahead_n)

        else:
            surprisal, entropy = compute_surprisal_entropy(logits[combined_pos - 1], target_id)
            top_k_preds = get_top_k_predictions(logits[combined_pos - 1], tokenizer, top_k)

            if lookahead_n > 0:
                if lookahead_strategy == 'beam':
                    lookahead = beam_search_lookahead(model, tokenizer, input_ids[:combined_pos + 1], lookahead_n, beam_width)
                else:
                    lookahead = greedy_lookahead(model, tokenizer, input_ids[:combined_pos + 1], lookahead_n)
            else:
                lookahead = []

            pred_col = top_k_preds + lookahead

        raw_tokens.append(raw_token)
        scored_tokens.append(token_str)
        is_special_flags.append(is_special)
        surprisals.append(surprisal)
        entropies.append(entropy)
        pred_columns.append(pred_col)

    return ScoringResult(
        raw_tokens=raw_tokens,
        scored_tokens=scored_tokens,
        is_special_flags=is_special_flags,
        surprisals=surprisals,
        entropies=entropies,
        pred_columns=pred_columns
    )


def score_masked_lm(
    sentence: str,
    tokenizer,
    model,
    top_k: int = 5,
    context_ids: List[int] = None
) -> ScoringResult:
    """
    Score sentence with MLM using parallel masking (original PLL).
    """
    if tokenizer.mask_token_id is None:
        raise ValueError("Model doesn't have a [MASK] token. Use --mode ar instead.")

    empty_result = validate_sentence_inputs(sentence)
    if empty_result is not None:
        return empty_result

    encoding, sentence_ids = tokenize_sentence(sentence, tokenizer)
    empty_ids_result = validate_sentence_inputs(sentence, sentence_ids)
    if empty_ids_result is not None:
        return empty_ids_result

    base_ids, sentence_start = prepare_input_with_context(sentence_ids, context_ids)
    special_mask = tokenizer.get_special_tokens_mask(sentence_ids.tolist(), already_has_special_tokens=True)

    num_tokens = len(sentence_ids)

    masked_batch = []
    for sent_pos in range(num_tokens):
        combined_pos = sentence_start + sent_pos
        masked_ids = base_ids.clone()
        masked_ids[combined_pos] = tokenizer.mask_token_id
        masked_batch.append(masked_ids)

    masked_batch = torch.stack(masked_batch)

    with torch.no_grad():
        outputs = model(masked_batch)
        all_logits = outputs.logits

    raw_tokens, scored_tokens, is_special_flags = [], [], []
    surprisals, entropies, pred_columns = [], [], []

    for sent_pos in range(num_tokens):
        combined_pos = sentence_start + sent_pos
        target_id = sentence_ids[sent_pos].item()
        is_special = 1 if special_mask[sent_pos] else 0

        logits = all_logits[sent_pos, combined_pos]

        surprisal, entropy = compute_surprisal_entropy(logits, target_id)
        top_k_preds = get_top_k_predictions(logits, tokenizer, top_k)

        raw_token, token_str = decode_token(tokenizer, target_id, is_special)

        raw_tokens.append(raw_token)
        scored_tokens.append(token_str)
        is_special_flags.append(is_special)
        surprisals.append(surprisal)
        entropies.append(entropy)
        pred_columns.append(top_k_preds)

    return ScoringResult(
        raw_tokens=raw_tokens,
        scored_tokens=scored_tokens,
        is_special_flags=is_special_flags,
        surprisals=surprisals,
        entropies=entropies,
        pred_columns=pred_columns
    )


def score_masked_lm_l2r(
    sentence: str,
    tokenizer,
    model,
    top_k: int = 5,
    context_ids: List[int] = None
) -> ScoringResult:
    """
    Score sentence with MLM using within_word_l2r (minicons-compatible).
    """
    if tokenizer.mask_token_id is None:
        raise ValueError("Model doesn't have a [MASK] token. Use --mode ar instead.")

    empty_result = validate_sentence_inputs(sentence)
    if empty_result is not None:
        return empty_result

    encoding, sentence_ids = tokenize_sentence(sentence, tokenizer)
    empty_ids_result = validate_sentence_inputs(sentence, sentence_ids)
    if empty_ids_result is not None:
        return empty_ids_result

    base_ids, sentence_start = prepare_input_with_context(sentence_ids, context_ids)
    seq_len = len(sentence_ids)
    special_mask = tokenizer.get_special_tokens_mask(sentence_ids.tolist(), already_has_special_tokens=True)

    if getattr(tokenizer, "is_fast", False) and hasattr(encoding, "word_ids"):
        word_ids = encoding.word_ids(0)
    else:
        word_ids = list(range(seq_len))

    word_to_positions = {}
    for pos, wid in enumerate(word_ids):
        if wid is None:
            continue
        word_to_positions.setdefault(wid, []).append(pos)

    masked_variants = []
    variant_targets = []

    mask_id = tokenizer.mask_token_id

    for _, positions in word_to_positions.items():
        for k, sent_pos in enumerate(positions):
            combined_pos = sentence_start + sent_pos
            ids = base_ids.clone()
            ids[combined_pos] = mask_id
            for future_pos in positions[k+1:]:
                ids[sentence_start + future_pos] = mask_id
            masked_variants.append(ids)
            variant_targets.append((sent_pos, combined_pos))

    for sent_pos, wid in enumerate(word_ids):
        if wid is None:
            combined_pos = sentence_start + sent_pos
            ids = base_ids.clone()
            ids[combined_pos] = mask_id
            masked_variants.append(ids)
            variant_targets.append((sent_pos, combined_pos))

    if not masked_variants:
        return ScoringResult(
            scored_tokens=[],
            raw_tokens=[],
            is_special_flags=[],
            surprisals=[],
            entropies=[],
            pred_columns=[]
        )

    batch = torch.stack(masked_variants)
    with torch.no_grad():
        outputs = model(batch)
        logits_batch = outputs.logits

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

    for variant_idx, (sent_pos, combined_pos) in enumerate(variant_targets):
        target_id = int(sentence_ids[sent_pos].item())
        logits = logits_batch[variant_idx, combined_pos]
        surp, ent = compute_surprisal_entropy(logits, target_id)
        top_k_preds = get_top_k_predictions(logits, tokenizer, top_k)

        surprisals[sent_pos] = surp
        entropies[sent_pos] = ent
        pred_columns[sent_pos] = top_k_preds

    return ScoringResult(
        raw_tokens=raw_tokens,
        scored_tokens=scored_tokens,
        is_special_flags=is_special_flags,
        surprisals=surprisals,
        entropies=entropies,
        pred_columns=pred_columns
    )


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
# Layer-batched surprisal scoring functions
# ============================================================================

def score_autoregressive_by_layers(
    sentence: str,
    left_context: str,
    tokenizer,
    model,
    layers: Optional[List[int]] = None,
    top_k: int = 5,
    lookahead_n: int = 3,
    lookahead_strategy: str = 'greedy',
    beam_width: int = 3,
    context_ids: List[int] = None
) -> dict:
    """
    Compute surprisal for specified layers in AR model.
    Returns a dict: {layer_idx: ScoringResult}
    """
    empty_result = validate_sentence_inputs(sentence)
    if empty_result is not None:
        return {}

    _, sentence_ids = tokenize_sentence(sentence, tokenizer)
    empty_ids_result = validate_sentence_inputs(sentence, sentence_ids)
    if empty_ids_result is not None:
        return {}

    input_ids, sentence_start = prepare_input_with_context(sentence_ids, context_ids)
    special_mask = tokenizer.get_special_tokens_mask(sentence_ids.tolist(), already_has_special_tokens=True)

    with torch.no_grad():
        outputs = model(input_ids.unsqueeze(0), output_hidden_states=True)
        logits = outputs.logits[0]  # shape: (seq_len, vocab_size)
        hidden_states = outputs.hidden_states  # tuple: (layer0, ..., layerN)

    # If no layers specified, use last layer only
    if layers is None:
        layers = [len(hidden_states) - 1]

    results = {}
    for layer_idx in layers:
        # For AR, we need to project hidden states to logits for each layer
        # Use model.lm_head (for GPT2, etc.)
        layer_hidden = hidden_states[layer_idx][0]  # shape: (seq_len, hidden_dim)
        layer_logits = model.lm_head(layer_hidden)  # shape: (seq_len, vocab_size)

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
            raw_token, token_str = decode_token(tokenizer, target_id, is_special)

            if combined_pos == 0:
                # Use minimal input for first token
                with torch.no_grad():
                    if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
                        minimal_input = torch.tensor([[tokenizer.bos_token_id]])
                        minimal_outputs = model(minimal_input, output_hidden_states=True)
                        minimal_hidden = minimal_outputs.hidden_states[layer_idx][0, -1]
                        minimal_logits = model.lm_head(minimal_hidden)
                    else:
                        minimal_logits = layer_logits[0] if len(layer_logits) > 0 else None

                if minimal_logits is not None:
                    surprisal, entropy = compute_surprisal_entropy(minimal_logits, target_id)
                    top_k_preds = get_top_k_predictions(minimal_logits, tokenizer, top_k)
                    lookahead = []
                    if lookahead_n > 0:
                        lookahead = []  # Not supported for non-final layers
                    pred_col = top_k_preds + lookahead
                else:
                    surprisal = entropy = float('nan')
                    pred_col = [''] * (1 + top_k + lookahead_n)
            else:
                surprisal, entropy = compute_surprisal_entropy(layer_logits[combined_pos - 1], target_id)
                top_k_preds = get_top_k_predictions(layer_logits[combined_pos - 1], tokenizer, top_k)
                lookahead = []
                if lookahead_n > 0:
                    lookahead = []  # Not supported for non-final layers
                pred_col = top_k_preds + lookahead

            raw_tokens.append(raw_token)
            scored_tokens.append(token_str)
            is_special_flags.append(is_special)
            surprisals.append(surprisal)
            entropies.append(entropy)
            pred_columns.append(pred_col)

        results[layer_idx] = ScoringResult(
            raw_tokens=raw_tokens,
            scored_tokens=scored_tokens,
            is_special_flags=is_special_flags,
            surprisals=surprisals,
            entropies=entropies,
            pred_columns=pred_columns
        )
    return results

def score_masked_lm_by_layers(
    sentence: str,
    tokenizer,
    model,
    layers: Optional[List[int]] = None,
    top_k: int = 5,
    context_ids: List[int] = None
) -> dict:
    """
    Compute surprisal for specified layers in MLM model.
    Returns a dict: {layer_idx: ScoringResult}
    """
    if tokenizer.mask_token_id is None:
        raise ValueError("Model doesn't have a [MASK] token. Use --mode ar instead.")

    empty_result = validate_sentence_inputs(sentence)
    if empty_result is not None:
        return {}

    encoding, sentence_ids = tokenize_sentence(sentence, tokenizer)
    empty_ids_result = validate_sentence_inputs(sentence, sentence_ids)
    if empty_ids_result is not None:
        return {}

    base_ids, sentence_start = prepare_input_with_context(sentence_ids, context_ids)
    special_mask = tokenizer.get_special_tokens_mask(sentence_ids.tolist(), already_has_special_tokens=True)
    num_tokens = len(sentence_ids)

    masked_batch = []
    for sent_pos in range(num_tokens):
        combined_pos = sentence_start + sent_pos
        masked_ids = base_ids.clone()
        masked_ids[combined_pos] = tokenizer.mask_token_id
        masked_batch.append(masked_ids)

    masked_batch = torch.stack(masked_batch)

    with torch.no_grad():
        outputs = model(masked_batch, output_hidden_states=True)
        all_logits = outputs.logits
        hidden_states = outputs.hidden_states  # tuple: (layer0, ..., layerN)

    # If no layers specified, use last layer only
    if layers is None:
        layers = [len(hidden_states) - 1]

    results = {}
    for layer_idx in layers:
        # For MLM, project hidden states to logits for each layer
        # Use model.cls (for BERT) or model.lm_head (for Camembert, etc.)
        layer_hidden = hidden_states[layer_idx]  # shape: (batch, seq_len, hidden_dim)
        if hasattr(model, "cls"):
            layer_logits = model.cls(layer_hidden)
        elif hasattr(model, "lm_head"):
            layer_logits = model.lm_head(layer_hidden)
        else:
            # Try model.get_output_embeddings() (common HF API)
            out_emb = None
            try:
                out_emb = model.get_output_embeddings()
            except Exception:
                out_emb = None

            if out_emb is not None:
                # If it's a Linear layer (hidden_dim -> vocab_size)
                if isinstance(out_emb, torch.nn.Linear):
                    layer_logits = out_emb(layer_hidden)
                else:
                    # If embeddings (vocab_size x emb_dim), do matmul
                    try:
                        emb_weight = out_emb.weight
                        layer_logits = torch.matmul(layer_hidden, emb_weight.t())
                    except Exception:
                        raise AttributeError("Unable to project hidden states via output embeddings")
            else:
                # Last resort: try input embeddings weight
                try:
                    in_emb = model.get_input_embeddings()
                    emb_weight = in_emb.weight
                    layer_logits = torch.matmul(layer_hidden, emb_weight.t())
                except Exception:
                    raise AttributeError("Model does not have a suitable output head (cls, lm_head) or usable embeddings")

        raw_tokens, scored_tokens, is_special_flags = [], [], []
        surprisals, entropies, pred_columns = [], [], []

        for sent_pos in range(num_tokens):
            combined_pos = sentence_start + sent_pos
            target_id = sentence_ids[sent_pos].item()
            is_special = 1 if special_mask[sent_pos] else 0
            logits = layer_logits[sent_pos, combined_pos]
            surprisal, entropy = compute_surprisal_entropy(logits, target_id)
            top_k_preds = get_top_k_predictions(logits, tokenizer, top_k)
            raw_token, token_str = decode_token(tokenizer, target_id, is_special)
            raw_tokens.append(raw_token)
            scored_tokens.append(token_str)
            is_special_flags.append(is_special)
            surprisals.append(surprisal)
            entropies.append(entropy)
            pred_columns.append(top_k_preds)

        results[layer_idx] = ScoringResult(
            raw_tokens=raw_tokens,
            scored_tokens=scored_tokens,
            is_special_flags=is_special_flags,
            surprisals=surprisals,
            entropies=entropies,
            pred_columns=pred_columns
        )
    return results