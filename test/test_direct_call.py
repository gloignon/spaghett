import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scorer import process_sentences

# Process sentences
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