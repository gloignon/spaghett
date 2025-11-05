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
    just_attentions: bool = False
    attention_layers: Optional[List[int]] = None  # NEW: select layers (None = all)
    attention_heads: Optional[List[int]] = None   # NEW: select heads (None = all)

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
        
        # If just_attentions is True, disable all scoring features
        if self.just_attentions:
            object.__setattr__(self, 'top_k', 0)
            object.__setattr__(self, 'lookahead_n', 0)

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
        
        # NEW: validate attention selections
        if self.attention_layers is not None:
            if not all(isinstance(x, int) and x >= 0 for x in self.attention_layers):
                errors.append(f"attention_layers must be a list of non-negative integers, got {self.attention_layers}")
        if self.attention_heads is not None:
            if not all(isinstance(x, int) and x >= 0 for x in self.attention_heads):
                errors.append(f"attention_heads must be a list of non-negative integers, got {self.attention_heads}")

        # Mode-specific validation
        if self.mode == 'ar':
            # AR mode: pll_metric not applicable
            if self.pll_metric != 'original':
                errors.append(f"pll_metric is only applicable in MLM mode, got '{self.pll_metric}'")
            # Beam search requires beam_width
            if self.lookahead_strategy == 'beam' and self.beam_width < 1:
                errors.append(f"beam_width must be >= 1 when using beam search, got {self.beam_width}")

        # Warnings (don't block execution, just inform)
        if not self.just_attentions:
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

# =========================================================
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


def write_attention_output(output_file: str, attention_data: List[dict]):
    """Write attention tuples to TSV."""
    if not attention_data:
        return
    
    columns = ['doc_id', 'sentence_id', 'token_id', 'token', 'is_context', 'is_special', 'rx_token_id', 'rx_token', 'attn_score']
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter='\t')
        writer.writeheader()
        writer.writerows(attention_data)


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
    pll_metric: str = 'original',
    output_attentions: bool = False,
    just_attentions: bool = False,
    attention_layers: Optional[List[int]] = None,   # NEW
    attention_heads: Optional[List[int]] = None     # NEW
) -> Tuple[List[dict], List[dict]]:
    # Override parameters if just_attentions is True
    if just_attentions:
        output_attentions = True
        top_k = 0
        lookahead_n = 0
    
    config = ScoringConfig(
        mode=mode,
        model_name=model_name,
        top_k=top_k,
        lookahead_n=lookahead_n,
        lookahead_strategy=lookahead_strategy,
        beam_width=beam_width,
        pll_metric=pll_metric,
        just_attentions=just_attentions,
        attention_layers=attention_layers,    # NEW
        attention_heads=attention_heads       # NEW
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
    if just_attentions:
        print(f"Loading {mode.upper()} model for attention extraction: {model_name}")
    else:
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
                lookahead_strategy, beam_width, context_ids, output_attentions,
                attention_layers, attention_heads  # NEW
            )
        elif mode == 'mlm':
            model = AutoModelForMaskedLM.from_pretrained(model_name)
            if pll_metric == 'within_word_l2r':
                score_fn = lambda s: score_masked_lm_l2r(
                    s, tokenizer, model, top_k, context_ids, output_attentions,
                    attention_layers, attention_heads  # NEW
                )
            else:
                score_fn = lambda s: score_masked_lm(
                    s, tokenizer, model, top_k, context_ids, output_attentions,
                    attention_layers, attention_heads  # NEW
                )
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
    attention_results = []
    iterator = zip(doc_ids, sentence_ids, sentences)
    if progress:
        desc = "Extracting attention" if just_attentions else "Processing"
        iterator = tqdm(iterator, total=len(sentences), desc=desc)
    
    for doc_id, sent_id, sentence in iterator:
        raw_tokens, tokens, is_special_flags, surprisals, entropies, pred_cols, attn_tuples = score_fn(sentence)
        
        # Only populate results if not just_attentions
        if not just_attentions:
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
        
        # Process attention data if available
        if attn_tuples is not None:
            for token_id, token, is_context, is_special, rx_token_id, rx_token, weight in attn_tuples:
                attention_results.append({
                    'doc_id': doc_id,
                    'sentence_id': sent_id,
                    'token_id': token_id,
                    'token': token,
                    'is_context': is_context,
                    'is_special': is_special,
                    'rx_token_id': rx_token_id,
                    'rx_token': rx_token,
                    'attn_score': f'{weight:.6f}'
                })
    
    return results, attention_results


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
    pll_metric: str = 'original',
    output_attentions: bool = False,
    just_attentions: bool = False,
    attention_layers: Optional[List[int]] = None,  # NEW
    attention_heads: Optional[List[int]] = None    # NEW
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
        results, attention_results = process_sentences(
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
            pll_metric=pll_metric,
            output_attentions=output_attentions,
            just_attentions=just_attentions,
            attention_layers=attention_layers,    # NEW
            attention_heads=attention_heads       # NEW
        )
        
        # Write output only if not just_attentions
        if not just_attentions:
            write_output(output_file, results, top_k, lookahead_n, mode)
            print(f"Results written to: {output_file}")
        
        # Write attention output if requested or if just_attentions
        if (output_attentions or just_attentions) and attention_results:
            # Create attention filename
            if just_attentions:
                # For just_attentions, use the output_file directly as attention file
                attn_file = Path(output_file)
            else:
                # For normal mode with attentions, append _attention to filename
                base_path = Path(output_file)
                attn_file = base_path.parent / (base_path.stem + '_attention' + base_path.suffix)
            
            write_attention_output(str(attn_file), attention_results)
            print(f"Attention scores written to: {attn_file}")
        
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
  python utils.py --input_file data.tsv --mode ar --model gpt2
  
  # Score with attention output (last layer, all heads averaged)
  python utils.py --input_file data.tsv --mode ar --model gpt2 --output_attentions
  
  # Extract ONLY attention from specific layers and heads
  python utils.py --input_file data.tsv --mode ar --model gpt2 --just_attentions \\
                   --attention_layers 0 5 11 --attention_heads 0 3 7
  
  # Extract attention from last layer only, specific heads
  python utils.py --input_file data.tsv --mode ar --model gpt2 --just_attentions \\
                   --attention_heads 2 5
  
  # Score documents with BERT (masked LM)
  python utils.py --input_file docs.tsv --mode mlm --model bert-base-uncased --format documents
  
  # Output to folder (auto-generates filename)
  python utils.py --input_file data.tsv --mode ar --model gpt2 --output_file ./results/
  
  # With custom output and context
  python utils.py --input_file data.tsv --output_file results.tsv --mode ar --model gpt2 \\
                   --left_context_file context.txt --top_k 10
  
  # With within-word L2R scoring (MLM only)
  python utils.py --input_file data.tsv --mode mlm --model bert-base-uncased \\
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
    parser.add_argument('--output_attentions', action='store_true', 
                       help='Output attention matrices to separate TSV file (creates *_attention.tsv)')
    parser.add_argument('--just_attentions', action='store_true',
                       help='ONLY extract attention matrices (no surprisal/entropy/predictions). Faster and creates only attention TSV.')
    parser.add_argument('--attention_layers', type=int, nargs='+', default=None,
                        help='Specific layer indices to extract attention from (0-indexed). Default: last layer only')
    parser.add_argument('--attention_heads', type=int, nargs='+', default=None,
                        help='Specific attention head indices to extract (0-indexed). Default: all heads (averaged)')

    # Check if no arguments provided
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        print("\n❌ Error: No arguments provided. At minimum, you need:", file=sys.stderr)
        print("  --input_file <file> --mode <ar|mlm> --model <name>\n", file=sys.stderr)
        sys.exit(1)
    
    args = parser.parse_args()

    # Build filename parts
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_short = args.model.replace('/', '_').split('-')[0]
    
    if args.just_attentions:
        parts = [Path(args.input_file).stem, args.mode, model_short, 'attention']
        # Optional: include selection in filename
        if args.attention_layers:
            parts.append(f"L{'_'.join(map(str, args.attention_layers))}")
        if args.attention_heads:
            parts.append(f"H{'_'.join(map(str, args.attention_heads))}")
    else:
        parts = [Path(args.input_file).stem, args.mode, model_short, f'k{args.top_k}']
        if args.left_context_file:
            parts.append('extra')
        if args.mode == 'ar' and args.lookahead_n > 0:
            parts.append(f'look{args.lookahead_n}')
            if args.lookahead_strategy == 'beam':
                parts.append(f'beam{args.beam_width}')
        if args.pll_metric == 'within_word_l2r':
            parts.append('L2R')

    if args.left_context_file and args.just_attentions:
        parts.append('extra')

    generated_filename = '_'.join(parts + [timestamp]) + '.tsv'
    
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
        pll_metric=args.pll_metric,
        output_attentions=args.output_attentions,
        just_attentions=args.just_attentions,
        attention_layers=args.attention_layers,  # NEW
        attention_heads=args.attention_heads     # NEW
    )

if __name__ == "__main__":
    main()

