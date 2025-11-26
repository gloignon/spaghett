import subprocess
import sys
import os
import tempfile
from pathlib import Path
import pytest
import csv

def test_cli_runs_and_outputs_file():
    """Test that the CLI script runs and produces an output file."""
    # Prepare a minimal TSV input file
    input_content = "doc_id\tsentence_id\tsentence\n1\t1\tHello world.\n"
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.tsv")
        output_dir = os.path.join(tmpdir, "output")
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

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
