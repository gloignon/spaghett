import subprocess
import sys
import os
import tempfile
from pathlib import Path
import pytest
import csv

# Ensure src is on sys.path for direct imports
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import setup_logger

def test_cli_runs_and_outputs_file():
    """Test that the CLI script runs and produces an output file."""
    # Prepare a minimal TSV input file
    input_content = "doc_id\tsentence_id\tsentence\n1\t1\tHello world.\n"
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.tsv")
        # Use a stable output directory inside the `test/` folder so tests can inspect outputs
        test_output_dir = Path(__file__).parent / "output"
        output_dir = str(test_output_dir)
        # Clean existing files in the test output folder so test is idempotent
        if test_output_dir.exists():
            for p in test_output_dir.glob("*"):
                try:
                    if p.is_file():
                        p.unlink()
                except Exception:
                    pass
        else:
            os.makedirs(output_dir, exist_ok=True)
        with open(input_path, "w", encoding="utf-8") as f:
            f.write(input_content)
        cli_path = str(Path(__file__).parent.parent / "src" / "cli.py")
        # Run the CLI
        result = subprocess.run([
            sys.executable, cli_path,
            "--input_file", input_path,
            "--output_file", output_dir,
            "--mode", "ar",
            "--model", "gpt2"
        ], capture_output=True, text=True)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        # Check that an output file was created in output_dir
        files = os.listdir(output_dir)
        assert any(f.endswith(".tsv") for f in files), "No TSV output file created by CLI"
        # Optionally, check contents
        output_files = [os.path.join(output_dir, f) for f in files if f.endswith(".tsv")]
        assert output_files, "No output TSV file found"
        with open(output_files[0], "r", encoding="utf-8") as outf:
            reader = csv.DictReader(outf, delimiter="\t")
            rows = list(reader)
            assert len(rows) > 0, "Output file is empty or missing results"
            # Check that required columns exist
            for col in ["doc_id", "sentence_id", "token"]:
                assert col in reader.fieldnames, f"Missing column: {col}"
        # Default logging should create a timestamped log file in output_dir
        log_files = list(Path(output_dir).glob("*.log"))
        assert log_files, "Expected default log file to be created"
        print(f"CLI log file created at: {log_files[0]}")
        log_text = log_files[0].read_text(encoding="utf-8")
        assert "Starting run" in log_text, "Log file missing startup entry"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
