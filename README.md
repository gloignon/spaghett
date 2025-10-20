# spaghett

spaghett is a simple python tool to extract surprisal-based features from text. 

## Features
* Works with AR and masked-token models from Hugging Face
* L2R scoring is available when using a masked-token model. Results should be identical to minicons (tested in multiple situations: see test/test_minicons_compare.py).
* Extracts surprisal, entropy and the next predicted word with the highest probability (i.e. what the LLM computed would be the continuation)
* Works sentence by sentence only (for now), but you can provide a common context file for semantic continuity, it will be prepended to each sentence's context window.
* Choice of greedy algorithm or beam search to identify the most probable next word (will assemble next subtokens to reconstitute the word).
* Minimalism over performance: simple loops used in place of batching, no CUDA, nothing fancy
* Should not mess up accented characters
* Should remain robust to LLM choice, as long as it is AR (GPT-style) or masked token.
* Simple Command Line Interface, no need to modify the code (but it should be easy to do so if you need to)

## Similar work
* [minicons for python](https://github.com/kanishkamisra/minicons) also does AR and masked token models (with PLL or L2R scoring), but no entropy and no next word. It does other stuff you might need.
* [pangoling for R](https://docs.ropensci.org/pangoling/) Uses python internally. Does AR and masked too, but surprisal scores only.
* [text for R](https://cran.r-project.org/web/packages/text/index.html) Does word embeddings computations, not surprisal-based features.
* [psychformers for python](https://github.com/jmichaelov/PsychFormers) Supports different types of models, but only does surprisal scores for now.
* [lm-scorer](https://github.com/simonepri/lm-scorer) Focus is on scoring whole sentences, only surprisal scores.
  
## Installation
* You will need python, install if you don't have it already.
* Install the libraries in requirements.txt:
```bash
pip install -r requirements.txt
```

## How to use

Input is a .tsv file. Expected columns are doc_id and text (in documents mode), or doc_id, sentence_id and sentence (in sentences mode). 
The first row is expected to be the column headers.

### Command Line Interface

```bash
python src/scorer.py --input_file <file> --mode <ar|mlm> --model <model_name> [options]
```

#### Required Arguments
- `--input_file`: Path to input TSV file
- `--mode`: Model type - `ar` (autoregressive/GPT-style) or `mlm` (masked/BERT-style)
- `--model`: HuggingFace model name (e.g., `gpt2`, `bert-base-uncased`, `almanach/camembert-base`)

#### Optional Arguments
- `--output_file`: Output path (default: auto-generated with timestamp)
  - Can be a specific filename: `results.tsv`
  - Can be a folder path: `./results/` (auto-generates filename inside your folder)
- `--format`: Input format - `documents` or `sentences` (default: `sentences`)
- `--left_context_file`: Path to text file for left context (prepended to each sentence)
- `--top_k`: Number of top predictions to output (default: `3`)

#### AR Mode Options (GPT-style models)
- `--lookahead_n`: Number of continuation tokens to generate (default: `3`, use `0` to disable)
- `--lookahead_strategy`: Generation strategy - `greedy` or `beam` (default: `greedy`)
- `--beam_width`: Beam width for beam search (default: `3`, only used with `--lookahead_strategy beam`)

#### MLM Mode Options (BERT-style models)
- `--pll_metric`: Scoring variant - `original` or `within_word_l2r` (default: `original`)

### Examples

**Basic usage (AR model):**
```bash
python src/scorer.py --input_file data.tsv --mode ar --model gpt2
```

**French MLM with L2R scoring:**
```bash
python src/scorer.py --input_file in/demo_sentences.tsv --mode mlm --model cmarkea/distilcamembert-base --pll_metric within_word_l2r
```

**Documents format with context:**
```bash
python src/scorer.py --input_file docs.tsv --format documents --mode ar --model gpt2 --left_context_file context.txt
```

**Output to specific folder:**
```bash
python src/scorer.py --input_file data.tsv --mode mlm --model bert-base-uncased --output_file ./results/
```

**AR with beam search lookahead:**
```bash
python src/scorer.py --input_file data.tsv --mode ar --model gpt2 --lookahead_n 5 --lookahead_strategy beam --beam_width 3
```

### Output Format

The output TSV contains:
- `doc_id`, `sentence_id`, `token_index`: Identifiers
- `token`: Raw token (with special characters like `▁`, `Ġ`)
- `token_decoded`: Human-readable token
- `is_special`: Flag for special tokens (BOS, EOS, CLS, SEP, etc.)
- `surprisal_bits`: -log₂(p(token|context))
- `entropy_bits`: Shannon entropy of the prediction distribution
- `pred_alt_1` to `pred_alt_N`: Top-k alternative predictions (N = `--top_k`)
- `pred_next_1` to `pred_next_M`: Lookahead predictions (M = `--lookahead_n`, AR mode only)

**Note**: All special tokens are included in output. Filter using `is_special` column in post-processing if needed.

## Direct Function Calls

You can also import and use the core functions directly in Python, for instance, assuming the scorer.py script is in the same folder:

```python
from scorer import process_sentences

results = process_sentences(
    sentences=["Hello world.", "How are you?"],
    mode='ar',
    model_name='gpt2',
    top_k=5,
    lookahead_n=3,
    progress=True
)

for res in results:
    print(res)
```

## Testing

Test direct call of scoring function:
```bash
python test/test_direct_call.py
```

Compare with minicons (MLM L2R scoring):
```bash
python test/test_minicons_compare.py
```

## TODO

* Create R wrapper/package
* Add different context window modes (previous sentence, previous n words, full context, etc.)
* Performance-optimized version with batching and GPU acceleration
* Support for additional model architectures