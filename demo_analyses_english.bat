@echo off
REM Run surprisal script on demo sentences (AR mode)
REM Two runs: one without context, one with left context
REM Operates relative to the project folder (batch file location).

cd /d "%~dp0"

if not exist "out" mkdir "out"

echo ===== English demo=====
python -u "src\cli.py" ^
  --input_file "in\demo_sentences_english.tsv" ^
  --lookahead_n 0 ^
  --top_k 1 ^
  --mode "ar" ^
  --model "gpt2" ^
  --format "sentences" ^
  --output_file "out\demo_analyses_english_no_context.tsv"

