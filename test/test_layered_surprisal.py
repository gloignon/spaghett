import pytest
import math
from pathlib import Path
import sys
import tempfile
import os
import subprocess
import pandas as pd


# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from scorer import (
    score_autoregressive,
    score_autoregressive_by_layers,
    score_masked_lm_by_layers
)
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM

def test_ar_layered_surprisal():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    sentence = "The cat sat on the mat."
    layers = [0, 5, 11]  # test first, middle, last layer
    results = score_autoregressive_by_layers(
        sentence=sentence,
        left_context="",
        tokenizer=tokenizer,
        model=model,
        layers=layers,
        top_k=2,
        lookahead_n=0
    )
    assert isinstance(results, dict)
    assert set(results.keys()) == set(layers)
    for layer_idx, res in results.items():
        assert len(res.surprisals) == len(res.scored_tokens)
        assert all(isinstance(s, float) for s in res.surprisals)
        assert all(s >= 0 or math.isnan(s) for s in res.surprisals)
        assert all(isinstance(e, float) for e in res.entropies)
        assert all(e >= 0 or math.isnan(e) for e in res.entropies)


def test_mlm_layered_surprisal():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    sentence = "The cat sat on the mat."
    layers = [0, 6, 11]  # test first, middle, last layer
    results = score_masked_lm_by_layers(
        sentence=sentence,
        tokenizer=tokenizer,
        model=model,
        layers=layers,
        top_k=2
    )
    assert isinstance(results, dict)
    assert set(results.keys()) == set(layers)
    for layer_idx, res in results.items():
        assert len(res.surprisals) == len(res.scored_tokens)
        assert all(isinstance(s, float) for s in res.surprisals)
        assert all(s >= 0 or math.isnan(s) for s in res.surprisals)
        assert all(isinstance(e, float) for e in res.entropies)
        assert all(e >= 0 or math.isnan(e) for e in res.entropies)

def test_mlm_layered_surprisal_chunked_equals_full():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    sentence = "The cat sat on the mat."
    layers = [0, 6, 11]
    full = score_masked_lm_by_layers(
        sentence=sentence,
        tokenizer=tokenizer,
        model=model,
        layers=layers,
        top_k=2,
        batch_size=0  # process all at once
    )
    chunked = score_masked_lm_by_layers(
        sentence=sentence,
        tokenizer=tokenizer,
        model=model,
        layers=layers,
        top_k=2,
        batch_size=1  # force tiny mini-batches
    )
    assert set(full.keys()) == set(chunked.keys()) == set(layers)
    for layer_idx in layers:
        full_res = full[layer_idx]
        chunk_res = chunked[layer_idx]
        for a, b in zip(full_res.surprisals, chunk_res.surprisals):
            if math.isnan(a):
                assert math.isnan(b)
            else:
                assert math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6)
        for a, b in zip(full_res.entropies, chunk_res.entropies):
            if math.isnan(a):
                assert math.isnan(b)
            else:
                assert math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6)


def test_mlm_all_layers_long_sentence_chunked():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    long_sentence = " ".join(["The cat sat on the mat"] * 12)  # long enough to stress batching
    num_layers = model.config.num_hidden_layers
    layers = list(range(num_layers + 1))  # include embeddings layer
    results = score_masked_lm_by_layers(
        sentence=long_sentence,
        tokenizer=tokenizer,
        model=model,
        layers=layers,
        top_k=1,
        batch_size=4  # force mini-batching for memory safety
    )
    assert set(results.keys()) == set(layers)
    seq_len = len(results[layers[-1]].surprisals)
    assert seq_len > 30  # ensure the sentence is indeed long
    for layer_idx, res in results.items():
        assert len(res.surprisals) == seq_len
        assert len(res.entropies) == seq_len
        assert all(isinstance(s, float) for s in res.surprisals)
        assert all(isinstance(e, float) for e in res.entropies)

def test_ar_all_layers():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    sentence = "The cat sat on the mat."
    num_layers = len(model.transformer.h)
    layers = list(range(num_layers + 1))  # include embeddings (0) through final block
    results = score_autoregressive_by_layers(
        sentence=sentence,
        left_context="",
        tokenizer=tokenizer,
        model=model,
        layers=layers,
        top_k=2,
        lookahead_n=0
    )
    # Default (no layers arg) should match last layer in layered scoring
    default_res = score_autoregressive(
        sentence=sentence,
        left_context="",
        tokenizer=tokenizer,
        model=model,
        top_k=2,
        lookahead_n=0
    )
    assert isinstance(results, dict)
    assert set(results.keys()) == set(layers)
    last_layer_idx = layers[-1]
    # Verify surprisal parity with default path
    for layered_surp, default_surp in zip(results[last_layer_idx].surprisals, default_res.surprisals):
        if math.isnan(layered_surp):
            assert math.isnan(default_surp)
        else:
            assert math.isclose(layered_surp, default_surp, rel_tol=1e-6, abs_tol=1e-6)
    for layer_idx, res in results.items():
        assert len(res.surprisals) == len(res.scored_tokens)
        assert all(isinstance(s, float) for s in res.surprisals)
        assert all(s >= 0 or math.isnan(s) for s in res.surprisals)
        assert all(isinstance(e, float) for e in res.entropies)
        assert all(e >= 0 or math.isnan(e) for e in res.entropies)


def test_cli_layers_all_ar():
    input_content = "doc_id\tsentence_id\tsentence\n1\t1\tHello world.\n"
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.tsv")
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)
        with open(input_path, "w", encoding="utf-8") as f:
            f.write(input_content)
        cli_path = str(Path(__file__).parent.parent / "src" / "cli.py")
        result = subprocess.run([
            sys.executable, cli_path,
            "--input_file", input_path,
            "--output_file", output_dir,
            "--mode", "ar",
            "--model", "gpt2",
            "--layers", "all"
        ], capture_output=True, text=True)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        files = os.listdir(output_dir)
        output_files = [os.path.join(output_dir, f) for f in files if f.endswith(".tsv")]
        assert output_files, "No output TSV file found"
        df = pd.read_csv(output_files[0], sep='\t')
        # Check that layer columns exist for all layers
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        num_layers = len(model.transformer.h)
        for idx in range(num_layers + 1):
            assert f"layer{idx}_surprisal_bits" in df.columns
            assert f"layer{idx}_entropy_bits" in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
