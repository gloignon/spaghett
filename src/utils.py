# pyright: reportGeneralTypeIssues=false
"""
Simple script for computing per-token surprisal and entropy, by sentence, with extra left context,
and exporting the top-k most probable next tokens for each scored token.
"""

import argparse
import csv
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
from tqdm import tqdm

from scorer import (
    score_autoregressive,
    score_masked_lm,
    score_masked_lm_l2r
)


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
        if self.mode == 'mlm' and self.lookahead_n > 0:
            object.__setattr__(self, 'lookahead_n', 0)
        if self.mode == 'mlm' and self.lookahead_strategy != 'greedy':
            object.__setattr__(self, 'lookahead_strategy', 'greedy')

    def validate(self) -> None:
        errors = []
        if self.mode not in self._VALID_MODES:
            errors.append(f"mode must be one of {sorted(self._VALID_MODES)}, got '{self.mode}'")
        if self.top_k < 0:
            errors.append(f"top_k must be >= 0, got {self.top_k}")
        if self.lookahead_n < 0:
            errors.append(f"lookahead_n must be >= 0, got {self.lookahead_n}")
        if self.lookahead_strategy not in self._VALID_STRATEGIES:
            errors.append(f"lookahead_strategy must be one of {sorted(self._VALID_STRATEGIES)}, got '{self.lookahead_strategy}'")
        if self.beam_width < 1:
            errors.append(f"beam_width must be >= 1, got {self.beam_width}")
        if self.pll_metric not in self._VALID_PLL:
            errors.append(f"pll_metric must be one of {sorted(self._VALID_PLL)}, got '{self.pll_metric}'")
        if self.mode == 'ar':
            if self.pll_metric != 'original':
                errors.append(f"pll_metric is only applicable in MLM mode, got '{self.pll_metric}'")
            if self.lookahead_strategy == 'beam' and self.beam_width < 1:
                errors.append(f"beam_width must be >= 1 when using beam search, got {self.beam_width}")
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

def load_input_data(input_file: str, format_type: str) -> List[Tuple[str, str, str]]:
    """Load input TSV and return list of (doc_id, sentence_id, sentence)."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Input file not found: '{input_file}'\n"
            f"Please check that the file path is correct and the file exists."
        )
    required_cols = ['doc_id', 'text'] if format_type == 'documents' else ['doc_id', 'sentence_id', 'sentence']
    min_cols = len(required_cols)
    data = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            try:
                header = next(reader)
            except StopIteration:
                raise ValueError(
                    f"Input file is empty: '{input_file}'\n"
                    f"Expected format: {chr(9).join(required_cols)}"
                )
            if len(header) < min_cols:
                raise ValueError(
                    f"Invalid header in '{input_file}'\n"
                    f"Expected {min_cols} columns: {', '.join(required_cols)}\n"
                    f"Got {len(header)}: {', '.join(header) if header else '(empty)'}"
                )
            header_lower = [col.lower().strip() for col in header[:min_cols]]
            if header_lower != required_cols:
                print(f"Warning: Expected columns {required_cols}, got {header[:min_cols]}")
            row_count = 0
            for row_num, row in enumerate(reader, start=2):
                if not row or all(cell.strip() == '' for cell in row):
                    continue
                row_count += 1
                if len(row) < min_cols:
                    print(f"Warning: Skipping row {row_num}: expected {min_cols} columns, got {len(row)}")
                    continue
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
    if doc_ids is None:
        doc_ids = ['doc1'] * len(sentences)
    if sentence_ids is None:
        sentence_ids = [str(i) for i in range(1, len(sentences) + 1)]
    if len(doc_ids) != len(sentences) or len(sentence_ids) != len(sentences):
        raise ValueError("doc_ids and sentence_ids must match length of sentences")
    print(f"Loading {mode.upper()} model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        raise ValueError(
            f"Failed to load tokenizer for model '{model_name}'. "
            f"Please check that the model exists on HuggingFace Hub or provide a valid local path. "
            f"Error: {type(e).__name__}: {str(e)}"
        )
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
                score_fn = lambda s: score_masked_lm_l2r(
                    s, tokenizer, model, top_k, context_ids
                )
            else:
                score_fn = lambda s: score_masked_lm(
                    s, tokenizer, model, top_k, context_ids
                )
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'ar' or 'mlm'")
    except Exception as e:
        if "Invalid mode" in str(e):
            raise
        raise ValueError(
            f"Failed to load model '{model_name}' for mode '{mode}'. "
            f"Please verify: (1) model exists, (2) model is compatible with {mode.upper()} mode "
            f"(AR models like GPT for 'ar', MLM models like BERT for 'mlm'). "
            f"Error: {type(e).__name__}: {str(e)}"
        )
    model.eval()
    results = []
    iterator = zip(doc_ids, sentence_ids, sentences)
    if progress:
        desc = "Processing"
        iterator = tqdm(iterator, total=len(sentences), desc=desc)
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
            for i in range(1, top_k + 1):
                row[f'pred_alt_{i}'] = preds[i - 1] if i - 1 < len(preds) else ''
            if mode == 'ar':
                offset = top_k
                for i in range(1, lookahead_n + 1):
                    row[f'pred_next_{i}'] = preds[offset + i - 1] if offset + i - 1 < len(preds) else ''
            results.append(row)
    return results

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
        left_context = ''
        if left_context_file:
            print(f"Loading left context from: {left_context_file}")
            left_context = read_left_context(left_context_file)
        print(f"Loading input from: {input_file}")
        data = load_input_data(input_file, format_type)
        if not data:
            print(f"Warning: No valid data found in {input_file}")
            print("Please check that:")
            print("  1. File has correct format (TSV with header)")
            print("  2. File contains data rows (not just header)")
            print("  3. Rows have correct number of columns")
            return
        doc_ids = [item[0] for item in data]
        sentence_ids = [item[1] for item in data]
        sentences = [item[2] for item in data]
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
  python utils.py --input_file data.tsv --mode ar --model gpt2
  python utils.py --input_file docs.tsv --mode mlm --model bert-base-uncased --format documents
  python utils.py --input_file data.tsv --mode ar --model gpt2 --output_file ./results/
  python utils.py --input_file data.tsv --output_file results.tsv --mode ar --model gpt2 \\
                   --left_context_file context.txt --top_k 10
  python utils.py --input_file data.tsv --mode mlm --model bert-base-uncased \\
                   --pll_metric within_word_l2r
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

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        print("\n❌ Error: No arguments provided. At minimum, you need:", file=sys.stderr)
        print("  --input_file <file> --mode <ar|mlm> --model <name>\n", file=sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

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
    generated_filename = '_'.join(parts + [timestamp]) + '.tsv'
    output_path = Path(args.output_file)
    if output_path.is_dir() or (not output_path.exists() and output_path.suffix == ''):
        output_path.mkdir(parents=True, exist_ok=True)
        final_output = output_path / generated_filename
        print(f"→ Output: {final_output}")
    elif args.output_file == 'simple_output.tsv':
        final_output = Path(generated_filename)
        print(f"→ Output: {final_output}")
    else:
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

