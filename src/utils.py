# pyright: reportGeneralTypeIssues=false
"""
Simple script for computing per-token surprisal and entropy, by sentence, with extra left context,
and exporting the top-k most probable next tokens for each scored token.
"""


import csv
import math
import os
import re
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
from tqdm import tqdm
import pandas as pd


@dataclass
class ScoringConfig:
    mode: str
    model_name: str
    pll_metric: str = 'original'
    lookahead_strategy: str = 'greedy'
    top_k: int = 3
    lookahead_n: int = 3  # Default for AR, will be adjusted in __post_init__
    beam_width: int = 3
    temperature: float = 1.0

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
        if self.temperature <= 0:
            errors.append(f"temperature must be > 0, got {self.temperature}")
        if self.mode == 'ar':
            if self.pll_metric != 'original':
                errors.append(f"pll_metric is only applicable in MLM mode, got '{self.pll_metric}'")
            if self.lookahead_strategy == 'beam' and self.beam_width < 1:
                errors.append(f"beam_width must be >= 1 when using beam search, got {self.beam_width}")
        if errors:
            bullet = "\n  • ".join(errors)
            raise ValueError(f"Invalid scoring configuration:\n  • {bullet}")

LN2 = math.log(2.0)

def setup_logger(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Create or return a shared logger for CLI processing."""
    logger = logging.getLogger("spaghett")
    if logger.handlers:
        return logger
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


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

def write_output(output_file: str, results: List[dict], top_k: int, lookahead_n: int, mode: str, top_k_cf_surprisal: bool = False, output_format: str = 'tsv'):
    columns = ['doc_id', 'sentence_id', 'token_index', 'token', 'token_decoded', 'is_special']
    # Add surprisal/entropy columns if present
    sample = results[0] if results else {}
    if 'surprisal_bits' in sample and 'entropy_bits' in sample:
        columns += ['surprisal_bits', 'entropy_bits']
    # Add layered columns if present
    layered_cols = [c for c in sample.keys() if c.startswith('layer')]
    columns += sorted(layered_cols)
    columns += [f'pred_alt_{i}' for i in range(1, top_k + 1)]
    if mode == 'ar':
        columns += [f'pred_next_{i}' for i in range(1, lookahead_n + 1)]
    df = pd.DataFrame(results)
    # Only select columns that exist in the DataFrame
    columns = [c for c in columns if c in df.columns]
    df = df[columns]
    if top_k_cf_surprisal:
        for i in range(1, top_k + 1):
            col = f'pred_alt_{i}'
            df[col] = df[col].apply(lambda val: f'{val[0]}|{val[1]:.4f}' if isinstance(val, tuple) else val)
    if output_format == 'parquet':
        df.to_parquet(output_file, index=False)
    else:
        df.to_csv(output_file, sep='\t', index=False)

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
    layers: Optional[List[int]] = None,
    temperature: float = 1.0,
    logger: Optional[logging.Logger] = None
) -> List[dict]:
    config = ScoringConfig(
        mode=mode,
        model_name=model_name,
        top_k=top_k,
        lookahead_n=lookahead_n,
        lookahead_strategy=lookahead_strategy,
        beam_width=beam_width,
        pll_metric=pll_metric,
        temperature=temperature
    )
    config.validate()
    logger = logger or logging.getLogger("spaghett")
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
            from scorer import score_autoregressive_by_layers, score_autoregressive
            model = AutoModelForCausalLM.from_pretrained(model_name)
            if layers is not None:
                score_fn = lambda s: score_autoregressive_by_layers(
                    s, left_context, tokenizer, model, layers, top_k, lookahead_n,
                    lookahead_strategy, beam_width, context_ids, temperature
                )
            else:
                score_fn = lambda s: score_autoregressive(
                    s, left_context, tokenizer, model, top_k, lookahead_n,
                    lookahead_strategy, beam_width, context_ids, temperature
                )
        elif mode == 'mlm':
            from scorer import score_masked_lm_by_layers, score_masked_lm, score_masked_lm_l2r
            model = AutoModelForMaskedLM.from_pretrained(model_name)
            if layers is not None:
                score_fn = lambda s: score_masked_lm_by_layers(
                    s, tokenizer, model, layers, top_k, context_ids, temperature
                )
            elif pll_metric == 'within_word_l2r':
                score_fn = lambda s: score_masked_lm_l2r(
                    s, tokenizer, model, top_k, context_ids, temperature
                )
            else:
                score_fn = lambda s: score_masked_lm(
                    s, tokenizer, model, top_k, context_ids, temperature
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
        if logger:
            logger.info(f"Scoring doc_id={doc_id} sentence_id={sent_id}")
        try:
            result = score_fn(sentence)
        except Exception as e:
            if logger:
                logger.error(
                    f"Failed while scoring doc_id={doc_id} sentence_id={sent_id}: {type(e).__name__}: {e}",
                    exc_info=True
                )
            raise
        if layers is not None and isinstance(result, dict):
            # Multiple layers: result is {layer_idx: ScoringResult}
            num_tokens = len(next(iter(result.values())).scored_tokens)
            for idx in range(num_tokens):
                row = {
                    'doc_id': doc_id,
                    'sentence_id': sent_id,
                    'token_index': idx + 1,
                    'token': next(iter(result.values())).raw_tokens[idx],
                    'token_decoded': next(iter(result.values())).scored_tokens[idx],
                    'is_special': next(iter(result.values())).is_special_flags[idx]
                }
                for layer_idx, layer_result in result.items():
                    row[f'layer{layer_idx}_surprisal_bits'] = '' if math.isnan(layer_result.surprisals[idx]) else f'{layer_result.surprisals[idx]:.4f}'
                    row[f'layer{layer_idx}_entropy_bits'] = '' if math.isnan(layer_result.entropies[idx]) else f'{layer_result.entropies[idx]:.4f}'
                # Use top_k preds from last layer only
                last_layer = max(result.keys())
                preds = result[last_layer].pred_columns[idx]
                for i in range(1, top_k + 1):
                    row[f'pred_alt_{i}'] = preds[i - 1] if i - 1 < len(preds) else ''
                if mode == 'ar':
                    offset = top_k
                    for i in range(1, lookahead_n + 1):
                        row[f'pred_next_{i}'] = preds[offset + i - 1] if offset + i - 1 < len(preds) else ''
                results.append(row)
        else:
            raw_tokens = result.raw_tokens
            tokens = result.scored_tokens
            is_special_flags = result.is_special_flags
            surprisals = result.surprisals
            entropies = result.entropies
            pred_cols = result.pred_columns
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
    pll_metric: str = 'original',
    layers: Optional[list] = None,
    top_k_cf_surprisal: bool = False,
    output_format: str = 'tsv',
    temperature: float = 1.0,
    log_file: str = ''
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
        if log_file:
            log_path = log_file
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_path = Path(output_file)
            log_name = f"{base_path.stem}_{ts}.log"
            log_path = str(base_path.with_name(log_name))
        os.makedirs(os.path.dirname(log_path), exist_ok=True) if os.path.dirname(log_path) else None
        # Print the log file path so users (and tests) see where logs are written
        print(f"Logging to: {log_path}")
        logger = setup_logger(log_path)
        logger.info(
            f"Starting run | mode={mode} model={model_name} "
            f"input={input_file} output={output_file} pll_metric={pll_metric} layers={layers}"
        )
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
            pll_metric=pll_metric,
            layers=layers,
            temperature=temperature,
            logger=logger
        )
        write_output(
            output_file,
            results,
            top_k,
            lookahead_n,
            mode,
            top_k_cf_surprisal=top_k_cf_surprisal,
            output_format=output_format
        )
        if 'logger' in locals():
            logger.info(f"Completed run; wrote {len(results)} rows to {output_file}")
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


