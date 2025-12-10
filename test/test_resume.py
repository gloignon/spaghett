import csv
import sys
from pathlib import Path

import pytest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import IncrementalWriter, load_completed_docs, process_sentences


def test_load_completed_docs_tsv():
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    path = output_dir / "test_resume.tsv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "sentence_id", "token_index"], delimiter="\t")
        writer.writeheader()
        writer.writerow({"doc_id": "doc1", "sentence_id": "1", "token_index": 1})
        writer.writerow({"doc_id": "doc2", "sentence_id": "1", "token_index": 1})
    docs, rows = load_completed_docs(str(path), "tsv")
    assert docs == {"doc1", "doc2"}
    assert rows == 2


def test_incremental_writer_resume_tsv():
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    path = output_dir / "test_resume.tsv"
    # initial write
    w1 = IncrementalWriter(
        output_file=str(path),
        mode="ar",
        layers=None,
        top_k=0,
        lookahead_n=0,
        output_format="tsv",
    )
    w1.write_rows([
        {"doc_id": "doc1", "sentence_id": "1", "token_index": 1, "token": "a", "token_decoded": "a", "is_special": 0},
    ])
    w1.close()

    # resume append
    w2 = IncrementalWriter(
        output_file=str(path),
        mode="ar",
        layers=None,
        top_k=0,
        lookahead_n=0,
        output_format="tsv",
        resume=True,
        existing_rows=1
    )
    w2.write_rows([
        {"doc_id": "doc2", "sentence_id": "1", "token_index": 1, "token": "b", "token_decoded": "b", "is_special": 0},
    ])
    w2.close()

    with open(path, encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
    # header + 2 rows
    assert len(lines) == 3
    docs, rows = load_completed_docs(str(path), "tsv")
    assert docs == {"doc1", "doc2"}
    assert rows == 2


def test_incremental_writer_resume_parquet():
    pytest.importorskip("pyarrow")
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    path = output_dir / "test_resume.parquet"

    base_row = {
        "doc_id": "doc1",
        "sentence_id": "1",
        "token_index": 1,
        "token": "a",
        "token_decoded": "a",
        "is_special": 0,
        "surprisal_bits": "1.0",
        "entropy_bits": "2.0",
    }

    # initial write
    w1 = IncrementalWriter(
        output_file=str(path),
        mode="ar",
        layers=None,
        top_k=0,
        lookahead_n=0,
        output_format="parquet",
    )
    w1.write_rows([base_row])
    w1.close()

    docs, rows = load_completed_docs(str(path), "parquet")
    assert docs == {"doc1"}
    assert rows == 1

    # resume append
    w2 = IncrementalWriter(
        output_file=str(path),
        mode="ar",
        layers=None,
        top_k=0,
        lookahead_n=0,
        output_format="parquet",
        resume=True,
        existing_rows=1
    )
    w2.write_rows([
        dict(base_row, doc_id="doc2", token="b", token_decoded="b"),
    ])
    w2.close()

    docs, rows = load_completed_docs(str(path), "parquet")
    assert docs == {"doc1", "doc2"}
    assert rows == 2

    df = pd.read_parquet(path)
    assert len(df) == 2
