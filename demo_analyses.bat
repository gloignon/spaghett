@echo off
REM Run surprisal script on demo sentences (AR mode)
REM Two runs: one without context, one with left context
REM Operates relative to the project folder (batch file location).

cd /d "%~dp0"

if not exist "out" mkdir "out"

echo ===== Run 1: Without left context =====
python -u "src\main.py" ^
  --input_file "in\demo_sentences.tsv" ^
  --output_file "out\demo_sentences_nocontext.tsv" ^
  --lookahead_strategy "beam" ^
  --lookahead_n 3 ^
  --top_k 3 ^
  --mode "ar" ^
  --model "lightonai/pagnol-small" ^
  --format "sentences"

if %ERRORLEVEL% NEQ 0 (
  echo Run 1 failed with exit code %ERRORLEVEL%
  exit /b %ERRORLEVEL%
)

echo.
echo ===== Run 2: With left context =====
python -u "src\main.py" ^
  --input_file "in\demo_sentences.tsv" ^
  --output_file "out\demo_sentences_withcontext.tsv" ^
  --left_context_file "in\demo_context.txt" ^
  --lookahead_strategy "beam" ^
  --lookahead_n 3 ^
  --top_k 3 ^
  --mode "ar" ^
  --model "lightonai/pagnol-small" ^
  --format "sentences"

if %ERRORLEVEL% EQU 0 (
  echo.
  echo All runs completed successfully!
  echo - No context: "out\demo_sentences_nocontext.tsv"
  echo - With context: "out\demo_sentences_withcontext.tsv"
) else (
  echo Run 2 failed with exit code %ERRORLEVEL%
)

exit /b %ERRORLEVEL%
