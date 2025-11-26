import sys
import argparse
from pathlib import Path
from datetime import datetime
from utils import process_from_file

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compute per-token surprisal and entropy.",
        epilog="""
Examples:
  python cli.py --input_file data.tsv --mode ar --model gpt2
  python cli.py --input_file docs.tsv --mode mlm --model bert-base-uncased --format documents
  python cli.py --input_file data.tsv --mode ar --model gpt2 --output_file ./results/
  python cli.py --input_file data.tsv --output_file results.tsv --mode ar --model gpt2 \
               --left_context_file context.txt --top_k 10
  python cli.py --input_file data.tsv --mode mlm --model bert-base-uncased \
               --pll_metric within_word_l2r
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input_file', required=True, help='Input TSV file')
    parser.add_argument('--output_file', default='simple_output.tsv', help='Output TSV file or folder (default: auto-generated filename in current directory)')
    parser.add_argument('--mode', choices=['ar', 'mlm'], required=True,
                       help='Model mode: "ar" for autoregressive (GPT), "mlm" for masked LM (BERT)')
    parser.add_argument('--model', required=True,
                       help='HuggingFace model name (e.g., "gpt2", "bert-base-uncased")')
    parser.add_argument('--format', choices=['documents', 'sentences'], default="sentences",
                       help='Input format: "documents" (doc_id, text) or "sentences" (doc_id, sent_id, sentence)')
    parser.add_argument('--left_context_file', default='', help='File with left context (optional)')
    parser.add_argument('--top_k', type=int, default=3, help='Number of top-k predictions (default: 3)')
    parser.add_argument('--top_k_cf_surprisal', action='store_true',
        help='If set, output counterfactual surprisal for each top-k prediction (pred_alt columns will be token|surprisal)')
    parser.add_argument('--lookahead_n', type=int, default=3, help='AR: number of lookahead tokens (default: 3)')
    parser.add_argument('--lookahead_strategy', choices=['greedy', 'beam'], default='greedy',
                       help='AR: lookahead strategy - greedy or beam search (default: greedy)')
    parser.add_argument('--beam_width', type=int, default=3, help='AR: beam width for beam search (default: 3)')
    parser.add_argument('--pll_metric', choices=['original', 'within_word_l2r'],
                        default='original', help='MLM: PLL variant - "original" or "within_word_l2r" (default: original)')
    parser.add_argument('--output_format', choices=['tsv', 'parquet'], default='tsv',
        help='Output format: tsv (default) or parquet')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        print("\n❌ Error: No arguments provided. At minimum, you need:", file=sys.stderr)
        print("  --input_file <file> --mode <ar|mlm> --model <name>\n", file=sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_short = args.model.replace('/', '_').split('-')[0]
    parts = [Path(args.input_file).stem, args.mode, model_short, f'k{args.top_k}']
    if args.left_context_file:
        parts.append('extra')
    if args.mode == 'ar' and args.lookahead_n > 0:
        parts.append(f'look{args.lookahead_n}')
        if args.lookahead_strategy == 'beam':
            parts.append(f'beam{args.beam_width}')
    if args.pll_metric == 'within_word_l2r':
        parts.append('L2R')
    generated_filename = '_'.join(parts + [timestamp]) + '.tsv'
    output_path = Path(args.output_file)
    if output_path.is_dir() or (not output_path.exists() and output_path.suffix == ''):
        output_path.mkdir(parents=True, exist_ok=True)
        final_output = output_path / generated_filename
        print(f"→ Output: {final_output}")
    elif args.output_file == 'simple_output.tsv':
        final_output = Path(generated_filename)
        print(f"→ Output: {final_output}")
    else:
        final_output = output_path
        final_output.parent.mkdir(parents=True, exist_ok=True)
        print(f"→ Output: {final_output}")
    process_from_file(
        input_file=args.input_file,
        output_file=str(final_output),
        mode=args.mode,
        model_name=args.model,
        format_type=args.format,
        left_context_file=args.left_context_file,
        top_k=args.top_k,
        lookahead_n=args.lookahead_n,
        lookahead_strategy=args.lookahead_strategy,
        beam_width=args.beam_width,
        pll_metric=args.pll_metric
    )

if __name__ == "__main__":
    main()
