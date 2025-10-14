spaghett is a simple python tool to extract surprisal-based features from text
* Works with AR and masked-token models from Hugging Face
* Extracts surprisal, entropy and the next predicted word with the highest probability (i.e. what the LLM computed would be the continuation)
* Works sentence by sentence only (for now), but you can provide a common context file to for semantic continuity
* Minimalism over performance: simple loops used in place of batching, no CUDA, nothing fancy
* Should not mess up accented characters
* Should remain robust to LLM choice, as long as it is AR (GPT-style) or masked token.
* Simple Command Line Interface, no need to modify the code (but it should be easy to do so if you need to)

# Similar work
* [minicons for python](https://github.com/kanishkamisra/minicons) also does AR and masked token models, but no entropy and no next word. It does an alternate type of MLM scoring and other stuff you might need.
* [pangoling for R](https://docs.ropensci.org/pangoling/) Uses python internally. Does AR and masked too, but surprisal scores only.
* [text for R](https://cran.r-project.org/web/packages/text/index.html)Does word embeddings computations, not surprisal-based features.
* [psychformers for python](https://github.com/jmichaelov/PsychFormers)Supports different types of models, but only does surprisal scores for now.
* [lm-scorer](https://github.com/simonepri/lm-scorer)Focus is on scoring whole sentences, only surprisal scores.
  
# Installation
* You will need python, install if you don't have it already.
* Install the libraries in requirements.txt

# How to use
* Input is a .tsv file. Expected columns are doc_id and text (in documents mode), or doc_id, sentence_id and sentence (in sentences mode). The first row is expected to the column headers and will be skipper.
* CLI parameters
```
    --input_file: Path to the input TSV file with documents or sentences.
    --output_file: Path to the output TSV file (default: simple_output.tsv).
    --mode: 'ar' for autoregressive (GPT-style) or 'mlm' for masked language model (BERT-style).
    --model: Name of the pre-trained model to use (e.g., 'gpt2', 'bert-base-uncased').
    --format: 'documents' or 'sentences' to specify input format.
    --left_context_file: Path to a .txt file whose contents are prepended to every sentence.
    --top_k: Number of top probable tokens to output (default: 5).
    --lookahead_n: (AR only) Number of follow tokens to generate (default: 3).
    --lookahead_strategy: (AR only) Strategy for generating follow tokens: 'greedy' or 'beam' (default: greedy).
    --beam_width: (AR only) Beam width for beam search (default: 3, only used when --lookahead_strategy=beam).
```

  * Example run (will run on the demo sentences, using the French LLM disticamembert).
```
    python -u "src\main.py" --input_file "in\demo_sentences.tsv"  --output_file "out\demo_sentences_out.tsv"  --mode mlm  --model cmarkea/distilcamembert-base  --format sentences
```

# TODO: 
* Make a wrapper for R (ideally, a package)
* Add different modes for analysis window (e.g. previous sentence, previous n words, pseudo AR for masked token model, as big as the LLM can go, etc.)
* Make another version that would have fancy batching, GPU acceleration, etc.
* Most probably next word is identified using a simple greedy algorithm, add the option to use beam search instead
