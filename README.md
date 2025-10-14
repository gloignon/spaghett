spaghett is a simple python tool to extract surprisal-based features from text
* Works with AR and masked-token models from Hugging Face
* Extracts surprisal, entropy and the next predicted word with the highest probability (i.e. what the LLM computed would be the continuation)
* Works sentence by sentence only (for now), but you can provide a common context file to for semantic continuity
* Minimalism over performance: simple loops used in place of batching, no CUDA, nothing fancy
* Should not mess up accented characters
* Should remain robust to LLM choice, as long as it is AR (GPT-style) or masked token.
* Simple Command Line Interface, no need to modify the code (but it should be easy to do so if you need to)
  
TODO: 
* Make a wrapper for R
* Add different modes for analysis window (e.g. previous sentence, previous n words, pseudo AR for masked token model, as big as the LLM can go, etc.)
* Make another version that would have fancy batching, GPU acceleration, etc.
