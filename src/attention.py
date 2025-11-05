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
        tokenizer: Tokenizer to identify special tokens and decode tokens
        full_token_ids: Full sequence of token IDs (context + sentence)
        sentence_start: Start position of sentence in sequence
        sentence_length: Number of tokens in sentence
        selected_layers: List of layer indices to use (None = last layer only)
        selected_heads: List of head indices to use (None = all heads averaged)

    Returns:
        List of (token_idx, token_str, is_context, is_special, rx_token_idx, rx_token_str, weight)
        where token_idx is 1-indexed position in full sequence
    """
    if attention_weights.dim() != 4:
        raise ValueError(f"Expected attention_weights to have 4 dimensions (layers, heads, seq, seq), got {attention_weights.dim()}")
    
    num_layers, num_heads, seq_len, _ = attention_weights.shape
    
    # Select layers: default to last layer
    if selected_layers is None:
        selected_layers = [num_layers - 1]
    else:
        # Validate layer indices
        for layer in selected_layers:
            if layer < 0 or layer >= num_layers:
                raise ValueError(f"Layer index {layer} out of range [0, {num_layers-1}]")
    
    # Average over selected layers
    layer_weights = attention_weights[selected_layers].mean(dim=0)  # (num_heads, seq_len, seq_len)
    
    # Select heads: default to average all heads
    if selected_heads is None:
        # Average all heads
        avg_attention = layer_weights.mean(dim=0)  # (seq_len, seq_len)
    else:
        # Validate head indices
        for head in selected_heads:
            if head < 0 or head >= num_heads:
                raise ValueError(f"Head index {head} out of range [0, {num_heads-1}]")
        # Average selected heads
        avg_attention = layer_weights[selected_heads].mean(dim=0)  # (seq_len, seq_len)

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