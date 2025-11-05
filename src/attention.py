import torch
from typing import List, Tuple, Optional

def extract_attention_tuples(
    attention_weights: torch.Tensor,
    tokenizer,
    full_token_ids: torch.Tensor,
    sentence_start: int,
    sentence_length: int,
    selected_layers: Optional[List[int]] = None,
    selected_heads: Optional[List[int]] = None
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
        selected_layers: List of layer indices to use (None = all layers averaged)
        selected_heads: List of head indices to use (None = all heads averaged)

    Returns:
        List of (token_idx, token_str, is_context, is_special, rx_token_idx, rx_token_str, weight)
        where token_idx is 1-indexed position in full sequence
    """
    # Handle both single-layer and multi-layer attention
    if attention_weights.dim() == 4:
        num_layers, num_heads, seq_len, _ = attention_weights.shape
        # Average over selected layers
        if selected_layers:
            layer_weights = attention_weights[selected_layers]
        else:
            layer_weights = attention_weights
        avg_over_layers = layer_weights.mean(dim=0)  # (num_heads, seq_len, seq_len)
    elif attention_weights.dim() == 3:
        num_heads, seq_len, _ = attention_weights.shape
        avg_over_layers = attention_weights
    else:
        raise ValueError(f"Expected attention_weights to have 3 or 4 dimensions, got {attention_weights.dim()}")

    # Average over selected heads
    if selected_heads:
        head_weights = avg_over_layers[selected_heads]
    else:
        head_weights = avg_over_layers
    avg_attention = head_weights.mean(dim=0)  # (seq_len, seq_len)

    seq_len = len(full_token_ids)
    special_mask = tokenizer.get_special_tokens_mask(
        full_token_ids.tolist(),
        already_has_special_tokens=True
    )

    # Convert all token IDs to strings (raw tokens with special characters)
    token_strings = tokenizer.convert_ids_to_tokens(full_token_ids.tolist())

    tuples = []
    sentence_end = sentence_start + sentence_length
    
    # Iterate through sentence tokens (not context)
    for i in range(sentence_start, sentence_end):
        token_idx = i + 1  # 1-indexed
        token_str = token_strings[i]
        is_context = 0
        is_special = special_mask[i]
        
        # Get attention weights from this token to all previous tokens
        for j in range(i + 1):  # Include current token
            rx_token_idx = j + 1  # 1-indexed
            rx_token_str = token_strings[j]
            weight = avg_attention[i, j].item()
            
            tuples.append((
                token_idx,
                token_str,
                is_context,
                is_special,
                rx_token_idx,
                rx_token_str,
                weight
            ))
    
    return tuples