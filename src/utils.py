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
    # Only log to file, not to console, to avoid spamming console output
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


def split_sentence_by_words(sentence: str, max_words: int) -> List[str]:
    """
    Split a sentence into chunks with at most max_words words.
    """
    if max_words <= 0:
        return [sentence]

    words = sentence.split()
    if not words or len(words) <= max_words:
        return [sentence]

    chunks: List[str] = []
    current: List[str] = []

    for word in words:
        if len(current) + 1 > max_words:
            chunks.append(' '.join(current))
            current = [word]
        else:
            current.append(word)

    if current:
        chunks.append(' '.join(current))

    return chunks if chunks else [sentence]


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
    """Load input TSV and return list of (doc_id, sentence_id, sentence).
    
    Extra columns in the input file are ignored. Required columns can appear
    in any order and position. For sentences format, both 'sentence' and 'text'
    column names are accepted for the sentence data.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Input file not found: '{input_file}'\n"
            f"Please check that the file path is correct and the file exists."
        )
    
    data = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            try:
                header = next(reader)
            except StopIteration:
                raise ValueError(
                    f"Input file is empty: '{input_file}'\n"
                    f"Expected columns: doc_id, " + ("text" if format_type == 'documents' else "sentence_id, sentence (or text)")
                )
            
            # Create case-insensitive mapping of column names to indices
            header_lower = {col.lower().strip(): idx for idx, col in enumerate(header)}
            
            # Find required columns
            if format_type == 'documents':
                required_cols = {'doc_id': None, 'text': None}
                if 'doc_id' not in header_lower:
                    raise ValueError(
                        f"Missing required column 'doc_id' in '{input_file}'\n"
                        f"Available columns: {', '.join(header)}"
                    )
                if 'text' not in header_lower:
                    raise ValueError(
                        f"Missing required column 'text' in '{input_file}'\n"
                        f"Available columns: {', '.join(header)}"
                    )
                doc_id_idx = header_lower['doc_id']
                text_idx = header_lower['text']
            else:
                # Sentences format: need doc_id, sentence_id, and sentence (or text)
                if 'doc_id' not in header_lower:
                    raise ValueError(
                        f"Missing required column 'doc_id' in '{input_file}'\n"
                        f"Available columns: {', '.join(header)}"
                    )
                if 'sentence_id' not in header_lower:
                    raise ValueError(
                        f"Missing required column 'sentence_id' in '{input_file}'\n"
                        f"Available columns: {', '.join(header)}"
                    )
                # Accept either 'sentence' or 'text' for the sentence content
                sentence_idx = None
                if 'sentence' in header_lower:
                    sentence_idx = header_lower['sentence']
                elif 'text' in header_lower:
                    sentence_idx = header_lower['text']
                else:
                    raise ValueError(
                        f"Missing required column 'sentence' or 'text' in '{input_file}'\n"
                        f"Available columns: {', '.join(header)}"
                    )
                doc_id_idx = header_lower['doc_id']
                sent_id_idx = header_lower['sentence_id']
            
            row_count = 0
            for row_num, row in enumerate(reader, start=2):
                if not row or all(cell.strip() == '' for cell in row):
                    continue
                row_count += 1
                
                # Check if row has enough columns
                max_idx = max(doc_id_idx, text_idx if format_type == 'documents' else max(sent_id_idx, sentence_idx))
                if len(row) <= max_idx:
                    print(f"Warning: Skipping row {row_num}: not enough columns")
                    continue
                
                if format_type == 'documents':
                    doc_id = row[doc_id_idx].strip()
                    text = row[text_idx].strip()
                    if not doc_id or not text:
                        print(f"Warning: Skipping row {row_num}: empty field(s)")
                        continue
                    sentences = simple_sentence_split(text)
                    data.extend((doc_id, str(i), s) for i, s in enumerate(sentences, 1))
                else:
                    doc_id = row[doc_id_idx].strip()
                    sent_id = row[sent_id_idx].strip()
                    sentence = row[sentence_idx].strip()
                    if not (doc_id and sent_id and sentence):
                        print(f"Warning: Skipping row {row_num}: empty field(s)")
                        continue
                    data.append((doc_id, sent_id, sentence))
            
            if row_count == 0:
                raise ValueError(f"No data rows in '{input_file}'")
            if not data:
                raise ValueError(
                    f"No valid data in '{input_file}' ({row_count} rows were malformed)"
                )
    except PermissionError:
        raise PermissionError(f"Permission denied: '{input_file}'")
    except UnicodeDecodeError as e:
        raise ValueError(f"File '{input_file}' is not UTF-8 encoded: {e}")
    return data


def combine_short_sentences(
    rows: List[Tuple[str, str, str]],
    min_sentence_words: int = 0,
    logger: Optional[logging.Logger] = None
) -> Tuple[List[Tuple[str, str, str]], dict]:
    """
    Combine sentences shorter than min_sentence_words with the next sentence.
    Sentences are combined within the same doc_id only.

    Returns:
        (combined_rows, stats)
        stats = {"combined_sentences": int, "total_units": int}
    """
    if min_sentence_words <= 0:
        return rows, {"combined_sentences": 0, "total_units": len(rows)}

    combined: List[Tuple[str, str, str]] = []
    combine_count = 0
    i = 0
    
    while i < len(rows):
        doc_id, sent_id, sentence = rows[i]
        word_count = len(sentence.split())
        
        # If sentence is too short and not the last sentence, try to combine
        if word_count < min_sentence_words and i + 1 < len(rows):
            # Check if next sentence is in the same document
            next_doc_id = rows[i + 1][0]
            if doc_id == next_doc_id:
                # Combine with next sentence
                next_sent_id = rows[i + 1][1]
                next_sentence = rows[i + 1][2]
                combined_sentence = sentence + " " + next_sentence
                combined_sent_id = f"{sent_id}+{next_sent_id}"
                combined.append((doc_id, combined_sent_id, combined_sentence))
                combine_count += 1
                i += 2  # Skip the next sentence as it's now combined
                continue
        
        # Keep sentence as is
        combined.append((doc_id, sent_id, sentence))
        i += 1

    stats = {"combined_sentences": combine_count, "total_units": len(combined)}
    if logger and combine_count:
        logger.info(
            "Combined %s short sentence(s) (min_words=%s), resulting in %s unit(s)",
            combine_count,
            min_sentence_words,
            stats["total_units"]
        )
    return combined, stats


def split_long_inputs(
    rows: List[Tuple[str, str, str]],
    max_sentence_words: int = 0,
    logger: Optional[logging.Logger] = None
) -> Tuple[List[Tuple[str, str, str]], dict]:
    """
    Split sentences longer than the configured limits into multiple chunks.

    Returns:
        (expanded_rows, stats)
        stats = {"split_sentences": int, "total_chunks": int}
    """
    if max_sentence_words <= 0:
        return rows, {"split_sentences": 0, "total_chunks": len(rows)}

    expanded: List[Tuple[str, str, str]] = []
    split_count = 0
    for doc_id, sent_id, sentence in rows:
        chunks = split_sentence_by_words(sentence, max_sentence_words)

        if len(chunks) > 1:
            split_count += 1
        for idx, chunk in enumerate(chunks, 1):
            new_sent_id = sent_id if len(chunks) == 1 else f"{sent_id}.{idx}"
            expanded.append((doc_id, new_sent_id, chunk))

    stats = {"split_sentences": split_count, "total_chunks": len(expanded)}
    if logger and split_count:
        logger.info(
            "Split %s long sentence(s) into %s chunk(s) (max_words=%s)",
            split_count,
            stats["total_chunks"],
            max_sentence_words
        )
    return expanded, stats


def write_output(output_file: str, results: List[dict], top_k: int, lookahead_n: int, mode: str, top_k_cf_surprisal: bool = False, output_format: str = 'tsv'):
    columns = build_output_columns(infer_layers_from_sample(results), top_k, lookahead_n, mode)
    df = pd.DataFrame(results)
    # Only select columns that exist in the DataFrame
    columns = [c for c in columns if c in df.columns]
    df = df[columns]
    if top_k_cf_surprisal:
        for i in range(1, top_k + 1):
            col = f'pred_alt_{i}'
            if col in df.columns:
                df[col] = df[col].apply(format_pred_value)
    if output_format == 'parquet':
        df.to_parquet(output_file, index=False)
    else:
        df.to_csv(output_file, sep='\t', index=False)


def infer_layers_from_sample(rows: List[dict]) -> Optional[List[int]]:
    """Detect requested layers from a sample of result rows."""
    if not rows:
        return None
    sample = rows[0]
    layers = []
    for key in sample.keys():
        if key.startswith("layer") and key.endswith("_surprisal_bits"):
            try:
                layer_idx = int(key[len("layer") : -len("_surprisal_bits")])
                layers.append(layer_idx)
            except ValueError:
                continue
    return sorted(set(layers)) if layers else None


def build_output_columns(layers: Optional[List[int]], top_k: int, lookahead_n: int, mode: str) -> List[str]:
    """Construct output column order based on configuration."""
    columns = ['doc_id', 'sentence_id', 'token_index', 'token', 'token_decoded', 'is_special']
    if layers is None:
        columns += ['surprisal_bits', 'entropy_bits']
    else:
        for layer_idx in sorted(set(layers)):
            columns.append(f'layer{layer_idx}_surprisal_bits')
            columns.append(f'layer{layer_idx}_entropy_bits')
    columns += [f'pred_alt_{i}' for i in range(1, top_k + 1)]
    if mode == 'ar':
        columns += [f'pred_next_{i}' for i in range(1, lookahead_n + 1)]
    return columns


def format_pred_value(val):
    """Format a prediction entry (token, cf_surprisal) to token|value string."""
    if isinstance(val, tuple) and len(val) == 2:
        return f'{val[0]}|{val[1]:.4f}'
    return val


class IncrementalWriter:
    """
    Incrementally write scoring rows to disk to limit memory usage.
    Supports TSV and Parquet outputs.
    """

    def __init__(
        self,
        output_file: str,
        mode: str,
        layers: Optional[List[int]],
        top_k: int,
        lookahead_n: int,
        top_k_cf_surprisal: bool = False,
        output_format: str = 'tsv',
        resume: bool = False,
        existing_rows: int = 0
    ):
        self.output_file = output_file
        self.output_format = output_format
        self.top_k = top_k
        self.lookahead_n = lookahead_n
        self.mode = mode
        self.top_k_cf_surprisal = top_k_cf_surprisal
        self.columns = build_output_columns(layers, top_k, lookahead_n, mode)
        self.total_rows = existing_rows
        self.resume = resume

        self._csv_handle = None
        self._csv_writer = None
        self._parquet_writer = None
        self._parquet = None
        self._parquet_schema = None
        self._parquet_base_table = None
        self._parquet_resume = False

        if output_format == 'tsv':
            mode_flag = 'a' if resume else 'w'
            self._csv_handle = open(output_file, mode_flag, encoding='utf-8', newline='')
            self._csv_writer = csv.DictWriter(self._csv_handle, fieldnames=self.columns, delimiter='\t', extrasaction='ignore')
            # Write header unless we are resuming and the file is non-empty
            if not resume or os.path.getsize(output_file) == 0:
                self._csv_writer.writeheader()
        elif output_format == 'parquet':
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
                self._parquet = (pa, pq)
                if resume and os.path.exists(output_file):
                    # Load existing table once; we'll rewrite with existing rows then append new ones
                    self._parquet_base_table = pq.read_table(output_file)
                    self._parquet_schema = self._parquet_base_table.schema
                    self._parquet_resume = True
            except Exception as e:
                raise ImportError("pyarrow is required for incremental parquet writing") from e
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def write_rows(self, rows: List[dict]):
        if not rows:
            return
        if self.top_k_cf_surprisal and self.top_k > 0:
            for row in rows:
                for i in range(1, self.top_k + 1):
                    col = f'pred_alt_{i}'
                    if col in row:
                        row[col] = format_pred_value(row[col])
        if self.output_format == 'tsv':
            for row in rows:
                self._csv_writer.writerow(row)
        else:
            pa, pq = self._parquet
            df = pd.DataFrame(rows)
            df = df.reindex(columns=self.columns, fill_value='')
            table = pa.Table.from_pandas(df, preserve_index=False)
            if self._parquet_writer is None:
                schema = self._parquet_schema or table.schema
                self._parquet_writer = pq.ParquetWriter(self.output_file, schema)
                if self._parquet_resume and self._parquet_base_table is not None:
                    self._parquet_writer.write_table(self._parquet_base_table)
            self._parquet_writer.write_table(table)
        self.total_rows += len(rows)

    def close(self):
        if self._csv_handle:
            self._csv_handle.close()
            self._csv_handle = None
        if self._parquet_writer:
            self._parquet_writer.close()
            self._parquet_writer = None


def load_completed_docs(output_file: str, output_format: str) -> Tuple[set, int]:
    """
    Load completed doc_ids from an existing output file for resume.
    Returns (doc_id_set, row_count).
    """
    if not os.path.exists(output_file):
        return set(), 0
    if output_format == 'tsv':
        doc_ids = set()
        count = 0
        with open(output_file, encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                doc_id = row.get('doc_id')
                if doc_id:
                    doc_ids.add(doc_id)
                count += 1
        return doc_ids, count
    elif output_format == 'parquet':
        try:
            df = pd.read_parquet(output_file, columns=['doc_id'])
        except Exception:
            return set(), 0
        return set(df['doc_id'].dropna().astype(str).tolist()), len(df)
    return set(), 0

# ============================================================================
# Core processing function (I/O independent)
# ============================================================================

def process_sentences(
    sentences: List[str],
    mode: str,
    model_name: str,
    left_context: str = '',
    top_k: int = 0,
    lookahead_n: int = 3,
    lookahead_strategy: str = 'greedy',
    beam_width: int = 3,
    doc_ids: List[str] = None,
    sentence_ids: List[str] = None,
    progress: bool = True,
    pll_metric: str = 'original',
    layers: Optional[List[int]] = None,
    temperature: float = 1.0,
    logger: Optional[logging.Logger] = None,
    writer: Optional[IncrementalWriter] = None,
    flush_by_doc: bool = False,
    skip_doc_ids: Optional[set] = None,
    mlm_batch_size: int = 0
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
                    s, tokenizer, model, layers, top_k, context_ids, temperature, mlm_batch_size
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
    results: List[dict] = [] if writer is None else None
    current_doc = None
    doc_buffer: List[dict] = []

    def flush_buffer():
        nonlocal doc_buffer
        if writer and doc_buffer:
            writer.write_rows(doc_buffer)
            doc_buffer = []

    iterator = zip(doc_ids, sentence_ids, sentences)
    if progress:
        desc = "Processing"
        iterator = tqdm(iterator, total=len(sentences), desc=desc)
    for doc_id, sent_id, sentence in iterator:
        if skip_doc_ids and doc_id in skip_doc_ids:
            continue
        if writer and flush_by_doc:
            if current_doc is None:
                current_doc = doc_id
            elif doc_id != current_doc:
                flush_buffer()
                current_doc = doc_id
        # Only log to file, not to console, to avoid console spam
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
                for layer_idx, layer_result in sorted(result.items()):
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
                if writer:
                    doc_buffer.append(row)
                else:
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
                if writer:
                    doc_buffer.append(row)
                else:
                    results.append(row)

    flush_buffer()
    return results if results is not None else []

# ============================================================================

def process_from_file(
    input_file: str,
    output_file: str,
    mode: str,
    model_name: str,
    format_type: str,
    left_context_file: str = '',
    top_k: int = 0,
    lookahead_n: int = 3,
    lookahead_strategy: str = 'greedy',
    beam_width: int = 3,
    pll_metric: str = 'original',
    layers: Optional[list] = None,
    top_k_cf_surprisal: bool = False,
    output_format: str = 'tsv',
    temperature: float = 1.0,
    log_file: str = '',
    max_sentence_words: int = 0,
    min_sentence_words: int = 0,
    resume: bool = False,
    mlm_batch_size: int = 0
):
    """
    Process input TSV file and write output TSV file.
    
    This function handles I/O for CLI usage.
    Set max_sentence_words > 0 to split overly long sentences before scoring.
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
        # First combine short sentences, then split long ones
        data, _combine_stats = combine_short_sentences(data, min_sentence_words, logger)
        data, _split_stats = split_long_inputs(data, max_sentence_words, logger)
        logger.info(
            f"Starting run | mode={mode} model={model_name} "
            f"input={input_file} output={output_file} pll_metric={pll_metric} layers={layers} "
            f"max_sentence_words={max_sentence_words} min_sentence_words={min_sentence_words} sentences={len(data)}"
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
        completed_docs = set()
        existing_rows = 0
        if resume and os.path.exists(output_file):
            completed_docs, existing_rows = load_completed_docs(output_file, output_format)
            logger.info("Resuming: found %s completed doc_id(s), %s rows in %s", len(completed_docs), existing_rows, output_file)

        writer = None
        try:
            writer = IncrementalWriter(
                output_file=output_file,
                mode=mode,
                layers=layers,
                top_k=top_k,
                lookahead_n=lookahead_n,
                top_k_cf_surprisal=top_k_cf_surprisal,
                output_format=output_format,
                resume=resume and os.path.exists(output_file),
                existing_rows=existing_rows
            )
        except Exception as e:
            if resume:
                raise
            if logger:
                logger.warning("Falling back to batch write: %s", e)
            writer = None

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
            logger=logger,
            writer=writer,
            flush_by_doc=True,
            skip_doc_ids=completed_docs if completed_docs else None,
            mlm_batch_size=mlm_batch_size
        )
        if writer:
            writer.close()
            rows_written = writer.total_rows
        else:
            write_output(
                output_file,
                results,
                top_k,
                lookahead_n,
                mode,
                top_k_cf_surprisal=top_k_cf_surprisal,
                output_format=output_format
            )
            rows_written = existing_rows + len(results) if resume else len(results)

        if 'logger' in locals():
            logger.info(f"Completed run; wrote {rows_written} rows to {output_file}")
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
