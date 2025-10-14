"""
Simple script for computing per-token surprisal and entropy, by sentence, with extra left context,
and exporting the top-k most probable next tokens for each scored token.

Variant of simple_surprisal_entropy_extra_context.py:
- Prepends a provided string (read from a .txt file) as extra left context to every sentence.
- The left context only serves as context; NO surprisal or entropy is computed for it.
- Adds --top_k (default 5) to output top_k_1..top_k_k columns with the highest-probability tokens.

CLI parameters:
    --input_file: Path to the input TSV file with documents or sentences.
    --mode: 'ar' for autoregressive (GPT-style) or 'mlm' for masked language model (BERT-style).
    --model: Name of the pre-trained model to use (e.g., 'gpt2', 'bert-base-uncased').
    --format: 'documents' or 'sentences' to specify input format.
    --left_context_file: Path to a .txt file whose contents are prepended (with a single space) to every sentence.
    --top_k: Number of top tokens to export (default=5).

Input TSV formats:
    Format 1 (documents): doc_id<TAB>text
    Format 2 (sentences): doc_id<TAB>sentence_id<TAB>sentence

Outputs:
    simple_output.tsv (TSV):
        doc_id<TAB>sentence_id<TAB>token_index<TAB>token<TAB>surprisal_bits<TAB>entropy_bits<TAB>top_k_1...top_k_k

Notes:
    - CPU-only: no batching, no CUDA/GPU, no mixed precision; intended for clarity over speed.
    - "surprisal_bits" = -log2 p(token), entropy_bits from softmax distribution in bits.
    - AR mode: uses logits at position i to score token at i+1. Special tokens are skipped (unless --include_special_tokens).
    - MLM mode: masks one (sentence) token at a time; special tokens and left-context tokens are skipped.
    - Also outputs the most probable next token (pred_top) and up to N greedy follow tokens (pred_next_1..pred_next_N) for AR, to complete a word.
"""

import argparse
import csv
import math
import re
from typing import List, Tuple, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
)
from tqdm import tqdm

LN2 = math.log(2.0)


def simple_sentence_split(text: str) -> List[str]:
    """
    Lightweight sentence splitter: splits on [.?!…] keeping delimiters.
    """
    if not text:
        return []
    t = re.sub(r"\.{3,}", "…", text)
    parts = re.split(r"([\.?!…]+)", t)
    sents, buf = [], ""
    for chunk in parts:
        if not chunk:
            continue
        buf += chunk
        if re.fullmatch(r"[\.?!…]+", chunk):
            sents.append(buf.strip())
            buf = ""
    if buf.strip():
        sents.append(buf.strip())
    return [s for s in sents if s]


def _read_left_context(path: str) -> str:
    """
    Read left context from a text file. Collapses newlines to spaces and trims.
    """
    with open(path, "r", encoding="utf-8") as f:
        ctx = f.read()
    ctx = ctx.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ").strip()
    return ctx


def _combine_left_context(ctx: str, sentence: str) -> str:
    """
    Join left context and sentence with exactly one space (if both non-empty).
    """
    if not ctx:
        return sentence
    if not sentence:
        return ctx
    return f"{ctx} {sentence}"


def _find_subsequence(haystack: List[int], needle: List[int], start: int = 0) -> Optional[int]:
    """
    Return the start index of the first occurrence of needle in haystack at or after start, else None.
    """
    if not needle:
        return start
    n, m = len(haystack), len(needle)
    if m > n:
        return None
    for i in range(start, n - m + 1):
        if haystack[i : i + m] == needle:
            return i
    return None


def _safe_piece(s: str) -> str:
    # Keep TSV stable
    return s.replace("\t", " ").replace("\r", " ").replace("\n", "⏎")


def _ar_greedy_follow(tokenizer, model, prefix_ids: List[int], start_id: int, max_follow_n: int) -> List[str]:
    """
    Given a prefix and the top-1 next token id, greedily roll out up to max_follow_n
    more tokens, stopping early if the next piece clearly starts a new word or is special/punctuation.
    Returns decoded pieces (without the first/top piece).
    """
    follow: List[str] = []
    if max_follow_n <= 0:
        return follow

    curr_ids = prefix_ids + [start_id]
    for _ in range(max_follow_n):
        with torch.no_grad():
            out = model(torch.tensor([curr_ids], dtype=torch.long))
            next_logits = out.logits[0, -1, :]
        next_id = int(torch.argmax(next_logits).item())
        piece = tokenizer.decode([next_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        s = _safe_piece(piece)

        # If the next token begins a new word (leading space), is special, or is punctuation-only, stop.
        if s[:1].isspace() or (hasattr(tokenizer, "all_special_ids") and next_id in tokenizer.all_special_ids) or re.fullmatch(r"^[\.\,\!\?\:\;\)\]\}\»\«\"\'…]+$", s or ""):
            break

        follow.append(s)
        curr_ids.append(next_id)

    return follow


def score_autoregressive_with_context(
    sentence: str,
    left_context: str,
    tokenizer,
    model,
    lookahead_n: int = 0,
    include_special_tokens: bool = False,
) -> Tuple[List[str], List[float], List[float], List[List[str]]]:
    """
    AR scoring with constant left context prepended. Only scores tokens belonging to the sentence.
    Returns:
        scored_tokens, surprisals_bits, entropies_bits, pred_cols_per_position
        where pred_cols_per_position = [pred_top, pred_next_1, ..., pred_next_N]
    """
    combined = _combine_left_context(left_context, sentence)

    enc = tokenizer(
        combined,
        return_tensors="pt",
        add_special_tokens=True,
        padding=False,
        truncation=False
    )
    input_ids = enc["input_ids"]          # [1, L]
    if input_ids.size(1) < 1:
        return [], [], [], []

    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is None:
        bos_id = getattr(tokenizer, "eos_token_id", None)
    if bos_id is None:
        bos_id = getattr(tokenizer, "cls_token_id", None)
    if bos_id is None:
        raise ValueError("No BOS/EOS/CLS token available to anchor the first-token score.")

    first_ids = input_ids[0].tolist()
    special_mask_orig = tokenizer.get_special_tokens_mask(first_ids, already_has_special_tokens=True)
    needs_bos = not bool(special_mask_orig[0])

    if needs_bos:
        input_ids = torch.cat([torch.tensor([[bos_id]], dtype=input_ids.dtype), input_ids], dim=1)

    max_len = getattr(tokenizer, "model_max_length", None)
    if max_len is not None and input_ids.size(1) > max_len:
        raise ValueError(
            f"Sequence length {input_ids.size(1)} exceeds model_max_length={max_len}. "
            "Use shorter sentences or implement a sliding window."
        )

    ids_list = input_ids[0].tolist()
    special_tokens_mask = torch.tensor(
        tokenizer.get_special_tokens_mask(ids_list, already_has_special_tokens=True),
        dtype=torch.bool
    )

    ctx_ids = tokenizer(left_context, add_special_tokens=False)["input_ids"]
    first_content_idx = 0
    while first_content_idx < len(ids_list) and (first_content_idx < len(special_tokens_mask)) and bool(special_tokens_mask[first_content_idx].item()):
        first_content_idx += 1
    ctx_start = _find_subsequence(ids_list, ctx_ids, start=first_content_idx) if ctx_ids else None
    if ctx_ids and ctx_start is None:
        ctx_last = -1
    else:
        ctx_last = (ctx_start + len(ctx_ids) - 1) if ctx_ids else -1

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # [1, L, V]

    surprisals_bits, entropies_bits, scored_tokens = [], [], []
    pred_cols_per_pos: List[List[str]] = []

    for i in range(len(ids_list) - 1):
        j = i + 1
        if j <= ctx_last:
            continue
        if bool(special_tokens_mask[j].item()) and not include_special_tokens:
            continue

        next_token_logits = logits[0, i, :]
        log_probs = torch.log_softmax(next_token_logits, dim=-1)
        probs = torch.softmax(next_token_logits, dim=-1)

        actual_token_id = input_ids[0, j].item()
        token_log_prob = log_probs[actual_token_id].item()
        surprisal_bits = -token_log_prob / LN2
        entropy_nats = -(probs * log_probs).sum().item()
        entropy_bits = entropy_nats / LN2

        # Decode human-readable piece for the scored token (keep specials visible)
        piece_true = tokenizer.decode([actual_token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        piece_true = _safe_piece(piece_true)

        # Top-1 predicted next token for this position
        top1_id = int(torch.argmax(next_token_logits).item())
        pred_top = tokenizer.decode([top1_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        pred_top = _safe_piece(pred_top)

        # Greedy follow to complete the word (up to lookahead_n)
        prefix_ids = input_ids[0, :j].tolist()  # context up to position i (exclusive of true next token)
        pred_follow = _ar_greedy_follow(tokenizer, model, prefix_ids, top1_id, max_follow_n=int(lookahead_n))
        pred_cols = [pred_top] + pred_follow

        scored_tokens.append(piece_true)
        surprisals_bits.append(surprisal_bits)
        entropies_bits.append(entropy_bits)
        pred_cols_per_pos.append(pred_cols)

    return scored_tokens, surprisals_bits, entropies_bits, pred_cols_per_pos


def score_masked_lm_with_context(
    sentence: str,
    left_context: str,
    tokenizer,
    model,
) -> Tuple[List[str], List[float], List[float], List[List[str]]]:
    """
    MLM scoring with constant left context. Only scores tokens belonging to the sentence (non-special).
    Returns:
        scored_tokens, surprisals_bits, entropies_bits, pred_cols_per_position
        For MLM, pred_cols_per_position contains only [pred_top] at each position.
    """
    if tokenizer.mask_token_id is None:
        raise ValueError("Model doesn't have a [MASK] token. Use --mode ar instead.")

    combined = _combine_left_context(left_context, sentence)
    encoding = tokenizer(combined, return_tensors="pt", add_special_tokens=True)
    input_ids = encoding["input_ids"][0]
    if len(input_ids) < 2:
        return [], [], [], []

    special_tokens_mask = tokenizer.get_special_tokens_mask(
        input_ids.tolist(), already_has_special_tokens=True
    )

    ids_list = input_ids.tolist()
    ctx_ids = tokenizer(left_context, add_special_tokens=False)["input_ids"]
    first_content_idx = 0
    while first_content_idx < len(ids_list) and special_tokens_mask[first_content_idx]:
        first_content_idx += 1
    ctx_start = _find_subsequence(ids_list, ctx_ids, start=first_content_idx) if ctx_ids else None
    if ctx_ids and ctx_start is None:
        ctx_last = -1
    else:
        ctx_last = (ctx_start + len(ctx_ids) - 1) if ctx_ids else -1

    # tokens = tokenizer.convert_ids_to_tokens(input_ids)  # not used for output
    surprisals_bits: List[float] = []
    entropies_bits: List[float] = []
    scored_tokens: List[str] = []
    pred_cols_per_pos: List[List[str]] = []

    for pos in range(len(input_ids)):
        if special_tokens_mask[pos]:
            continue
        if pos <= ctx_last:
            continue

        masked_input_ids = input_ids.clone()
        masked_input_ids[pos] = tokenizer.mask_token_id

        with torch.no_grad():
            outputs = model(masked_input_ids.unsqueeze(0))
            logits = outputs.logits

        masked_pos_logits = logits[0, pos, :]
        probs = torch.softmax(masked_pos_logits, dim=-1)
        log_probs = torch.log_softmax(masked_pos_logits, dim=-1)

        actual_token_id = input_ids[pos].item()
        token_log_prob = log_probs[actual_token_id].item()
        surprisal_bits = -token_log_prob / LN2
        entropy_nats = -(probs * log_probs).sum().item()
        entropy_bits = entropy_nats / LN2

        # Decode human-readable piece (true token)
        piece_true = tokenizer.decode([actual_token_id], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        piece_true = _safe_piece(piece_true)

        # Top-1 predicted token for the mask
        top1_id = int(torch.argmax(masked_pos_logits).item())
        pred_top = tokenizer.decode([top1_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        pred_top = _safe_piece(pred_top)

        scored_tokens.append(piece_true)
        surprisals_bits.append(surprisal_bits)
        entropies_bits.append(entropy_bits)
        pred_cols_per_pos.append([pred_top])

    return scored_tokens, surprisals_bits, entropies_bits, pred_cols_per_pos


def main():
    parser = argparse.ArgumentParser(description="Simple surprisal and entropy computation with constant left context and greedy next-token lookahead")
    parser.add_argument("--input_file", required=True, help="TSV file with doc_id and text columns")
    parser.add_argument("--mode", required=True, choices=["ar", "mlm"], help="ar=autoregressive (GPT), mlm=masked language model (BERT)")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--output_file", default="simple_output.tsv", help="Output TSV file")
    parser.add_argument("--format", choices=["documents", "sentences"], default="documents",
                        help="Input format: 'documents' (doc_id, text) or 'sentences' (doc_id, sentence_id, sentence)")
    parser.add_argument("--left_context_file", default="", help="Optional path to .txt file with left-context text; omit for no left context")
    parser.add_argument("--lookahead_n", type=int, default=3, help="AR only: number of greedy follow tokens after top-1 to show (default=3)")
    parser.add_argument("--include_special_tokens", action="store_true", help="Include special tokens (e.g., EOS) in AR outputs if present")

    args = parser.parse_args()

    print(f"Loading model: {args.model}")

    if args.mode == "ar":
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForMaskedLM.from_pretrained(args.model)

    model.eval()

    # Load optional left context
    if args.left_context_file:
        left_context = _read_left_context(args.left_context_file)
        print(f"Loaded left context ({len(left_context)} chars) from {args.left_context_file}.")
    else:
        left_context = ""
        print("No left context (empty).")

    print(f"Processing {args.format} from: {args.input_file}")

    with open(args.input_file, "r", encoding="utf-8") as fin:
        total_rows = sum(1 for _ in csv.reader(fin, delimiter="\t")) - 1  # -1 for header

    with open(args.input_file, "r", encoding="utf-8") as fin, \
         open(args.output_file, "w", encoding="utf-8", newline="") as fout:

        reader = csv.reader(fin, delimiter="\t")
        writer = csv.writer(fout, delimiter="\t")

        # Header with greedy columns
        header = ["doc_id", "sentence_id", "token_index", "token", "surprisal_bits", "entropy_bits", "pred_top"]
        header.extend([f"pred_next_{i}" for i in range(1, args.lookahead_n + 1)])
        writer.writerow(header)

        # Skip input header row
        next(reader, None)

        progress_desc = f"Processing {args.format}"
        with tqdm(total=total_rows, desc=progress_desc, unit="item") as pbar:

            if args.format == "sentences":
                # Input format: doc_id, sentence_id, sentence
                for row in reader:
                    if len(row) < 3:
                        pbar.update(1)
                        continue

                    doc_id, sentence_id, sentence = row[0], row[1], row[2]

                    if args.mode == "ar":
                        tokens, surprisals, entropies, pred_cols = score_autoregressive_with_context(
                            sentence, left_context, tokenizer, model, lookahead_n=args.lookahead_n, include_special_tokens=args.include_special_tokens
                        )
                    else:
                        tokens, surprisals, entropies, pred_cols = score_masked_lm_with_context(
                            sentence, left_context, tokenizer, model
                        )

                    for i, (token, surprisal, entropy, preds) in enumerate(zip(tokens, surprisals, entropies, pred_cols)):
                        row_out = [doc_id, sentence_id, i, token, f"{surprisal:.6f}", f"{entropy:.6f}"]
                        # Pad or trim to exactly 1 + lookahead_n columns (pred_top + pred_next_*):
                        padded = (preds + [""] * (1 + args.lookahead_n))[: 1 + args.lookahead_n]
                        row_out.extend(padded)
                        writer.writerow(row_out)

                    pbar.update(1)

            else:
                # documents format: doc_id, text
                for row in reader:
                    if len(row) < 2:
                        pbar.update(1)
                        continue

                    doc_id, text = row[0], row[1]
                    sentences = simple_sentence_split(text)

                    for sent_idx, sentence in enumerate(sentences):
                        if not sentence.strip():
                            continue

                        if args.mode == "ar":
                            tokens, surprisals, entropies, pred_cols = score_autoregressive_with_context(
                                sentence, left_context, tokenizer, model, lookahead_n=args.lookahead_n, include_special_tokens=args.include_special_tokens
                            )
                        else:
                            tokens, surprisals, entropies, pred_cols = score_masked_lm_with_context(
                                sentence, left_context, tokenizer, model
                            )

                        for i, (token, surprisal, entropy, preds) in enumerate(zip(tokens, surprisals, entropies, pred_cols)):
                            row_out = [doc_id, sent_idx, i, token, f"{surprisal:.6f}", f"{entropy:.6f}"]
                            padded = (preds + [""] * (1 + args.lookahead_n))[: 1 + args.lookahead_n]
                            row_out.extend(padded)
                            writer.writerow(row_out)

                    pbar.update(1)

    print(f"Results written to: {args.output_file}")


if __name__ == "__main__":
    main()