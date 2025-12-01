import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import IncrementalWriter


def test_incremental_writer_tsv(tmp_path):
    """IncrementalWriter should write header once and format pred_alt values."""
    test_output_dir = Path(__file__).parent / "output"
    out_file = test_output_dir / "test_incremental_writer.tsv"
    writer = IncrementalWriter(
        output_file=str(out_file),
        mode="ar",
        layers=None,
        top_k=1,
        lookahead_n=0,
        top_k_cf_surprisal=True,
        output_format="tsv",
    )

    rows = [
        {
            "doc_id": "d1",
            "sentence_id": "1",
            "token_index": 1,
            "token": "Hello",
            "token_decoded": "Hello",
            "is_special": 0,
            "surprisal_bits": "1.23",
            "entropy_bits": "2.34",
            "pred_alt_1": ("foo", 0.1234),
        },
        {
            "doc_id": "d2",
            "sentence_id": "1",
            "token_index": 1,
            "token": "World",
            "token_decoded": "World",
            "is_special": 0,
            "surprisal_bits": "3.21",
            "entropy_bits": "4.56",
            "pred_alt_1": ("bar", 0.9876),
        },
    ]

    writer.write_rows(rows)
    writer.close()

    assert out_file.exists()
    text = out_file.read_text(encoding="utf-8").strip().splitlines()
    # header + two rows
    assert len(text) == 3

    with out_file.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        read_rows = list(reader)

    assert reader.fieldnames[:6] == [
        "doc_id",
        "sentence_id",
        "token_index",
        "token",
        "token_decoded",
        "is_special",
    ]
    assert read_rows[0]["pred_alt_1"].count("|") == 1
    # Ensure per-row formatting happened
    assert read_rows[0]["pred_alt_1"].startswith("foo|")
    assert read_rows[1]["doc_id"] == "d2"
