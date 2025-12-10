#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Segment a TSV corpus (doc_id, text) into sentences with syntok.

Usage:
  python segment_sentences.py --in corpus.tsv --out sentences.tsv [--merge-colon-semicolon]
  # Optional knobs for safe merging:
  #   --max-prev-len 35 --max-next-len 35 --max-merged-len 60

Input  (TSV):  doc_id<TAB>text
Output (TSV):  doc_id<TAB>sentence_id<TAB>sentence
  - sentence_id restarts at 1 for each document
  - spacing preserved
"""

import argparse
import csv
import re
import sys
from pathlib import Path

try:
    import syntok.segmenter as segmenter
except ImportError as e:
    sys.stderr.write("syntok not installed. Run: pip install syntok\n")
    raise

# --- Heuristic helpers for safe merging across ';' and ':' ---

# Words = letters (incl. accents) + digits, may contain internal hyphens or apostrophes.
WORD_RE = re.compile(r"\b[\wÀ-ÖØ-öø-ÿ]+(?:[-’'][\wÀ-ÖØ-öø-ÿ]+)*\b", re.UNICODE)

# Common French “continuation” starters (feel free to extend)
CONTINUATION_WORDS = {
    "et", "mais", "ou", "donc", "or", "ni", "car",             # coord. conj.
    "cependant", "pourtant", "toutefois", "ainsi", "alors",
    "puis", "ensuite", "de", "du", "des", "par", "en", "avec",
    "autrement", "sinon", "néanmoins", "dès", "depuis"
}

def token_count(s: str) -> int:
    return len(WORD_RE.findall(s or ""))

def starts_like_continuation(s: str) -> bool:
    s2 = (s or "").lstrip()
    if not s2:
        return False
    # Lowercase first char is a cheap continuation cue (French sentences usually cap after a real stop)
    if s2[0].islower():
        return True
    m = WORD_RE.search(s2)
    if not m:
        return False
    return m.group(0).lower() in CONTINUATION_WORDS

def clause_heavy(s: str, max_commas: int = 3, max_em_dashes: int = 2) -> bool:
    s = s or ""
    return (s.count(",") >= max_commas) or (s.count("—") >= max_em_dashes)

def iter_rows(tsv_path):
    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        # Require expected columns
        required = {"doc_id", "text"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Input must have columns: {required}; got {reader.fieldnames}")
        for row in reader:
            yield row["doc_id"], row["text"]

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    # Keep content as-is but normalize line endings
    return s.replace("\r\n", "\n").replace("\r", "\n")

def syntok_sentence_split(text: str) -> list[str]:
    """
    Use syntok for sentence segmentation - more robust than regex.
    """
    if not text:
        return []
    sentences: list[str] = []
    for paragraph in segmenter.process(text):
        for sentence in paragraph:
            sentence_text = "".join(map(str, sentence)).strip()
            if sentence_text:
                sentences.append(sentence_text)
    return sentences

def should_merge(prev: str,
                 nxt: str,
                 max_prev_len: int,
                 max_next_len: int,
                 max_merged_len: int) -> bool:
    """
    Decide whether to merge prev (ending with ; or :) with nxt.
    Conditions:
      1) prev_len <= max_prev_len and next_len <= max_next_len
      2) prev_len + next_len <= max_merged_len
      3) nxt looks like a continuation (lowercase start or continuation word)
      4) neither side is clause-heavy
    """
    if not prev or not nxt:
        return False

    prev_len = token_count(prev)
    next_len = token_count(nxt)

    if prev_len > max_prev_len or next_len > max_next_len:
        return False
    if prev_len + next_len > max_merged_len:
        return False
    if clause_heavy(prev) or clause_heavy(nxt):
        return False
    if not starts_like_continuation(nxt):
        return False

    return True

def merge_colon_semicolon_sentences(sentences,
                                    max_prev_len: int = 35,
                                    max_next_len: int = 35,
                                    max_merged_len: int = 60):
    """
    Merge sentences that end with ':' or ';' with the following sentence,
    but only when the heuristic says it's safe.
    """
    if not sentences:
        return sentences

    merged = []
    i = 0
    while i < len(sentences):
        cur = sentences[i].strip()
        if i < len(sentences) - 1 and re.search(r"[;:]$", cur):
            nxt = sentences[i + 1].strip()
            if should_merge(cur, nxt, max_prev_len, max_next_len, max_merged_len):
                merged.append(f"{cur} {nxt}")
                i += 2
                continue
        merged.append(cur)
        i += 1
    return merged

def main(args):
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    # Prepare writer with header
    with open(out_path, "w", encoding="utf-8", newline="") as fo:
        writer = csv.writer(fo, delimiter="\t", lineterminator="\n")
        writer.writerow(["doc_id", "sentence_id", "sentence"])

        for doc_id, text in iter_rows(in_path):
            text = normalize_text(text)
            # Segment into sentences
            sents = syntok_sentence_split(text) if text else []

            # Apply colon/semicolon merging if requested
            if args.merge_colon_semicolon:
                sents = merge_colon_semicolon_sentences(
                    sents,
                    max_prev_len=args.max_prev_len,
                    max_next_len=args.max_next_len,
                    max_merged_len=args.max_merged_len
                )

            # Number sentences starting at 1 for each doc
            for i, sent in enumerate(sents, start=1):
                writer.writerow([doc_id, i, sent.strip()])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentence segmentation with syntok.")
    parser.add_argument("--in", dest="in_path", required=True, help="Input TSV with columns: doc_id, text")
    parser.add_argument("--out", dest="out_path", required=True, help="Output TSV path")
    parser.add_argument("--merge-colon-semicolon", action="store_true",
                        help="Merge sentences ending with ':' or ';' with the following sentence (safely).")
    parser.add_argument("--max-prev-len", type=int, default=35,
                        help="Max tokens allowed for the sentence ending with ';' or ':' to be eligible for merge.")
    parser.add_argument("--max-next-len", type=int, default=35,
                        help="Max tokens allowed for the following sentence to be eligible for merge.")
    parser.add_argument("--max-merged-len", type=int, default=60,
                        help="Max tokens allowed for the merged sentence.")
    args = parser.parse_args()
    main(args)
