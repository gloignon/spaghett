@echo off
REM Run surprisal script on demo sentences (AR mode)
REM Two runs: one without context, one with left context
REM Operates relative to the project folder (batch file location).

cd /d "%~dp0"

if not exist "out" mkdir "out"

echo ===== Run 1: AR without left context (beam)=====
python -u "src\cli.py" ^
  --input_file "in\demo_sentences.tsv" ^
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
echo ===== Run 2: AR with left context (beam) =====
python -u "src\cli.py" ^
  --input_file "in\demo_sentences.tsv" ^
  --left_context_file "in\demo_context.txt" ^
  --lookahead_strategy "beam" ^
  --lookahead_n 3 ^
  --top_k 3 ^
  --mode "ar" ^
  --model "lightonai/pagnol-small" ^
  --format "sentences"

echo.
echo ===== Run 3: Masked token =====
python -u "src\cli.py" ^
  --input_file "in\demo_sentences.tsv" ^
  --left_context_file "in\demo_context.txt" ^
  --top_k 3 ^
  --mode "mlm" ^
  --model "almanach/camembert-base" ^
  --format "sentences"

if %ERRORLEVEL% EQU 0 (
  echo.
  echo All runs completed successfully!
  echo - AR, No context: "out\demo_sentences_nocontext.tsv"
  echo - AR with context: "out\demo_sentences_withcontext.tsv"
  echo - demo_sentences_mlm: "out\demo_sentences_mlm.tsv"
) else (
  echo Run 2 failed with exit code %ERRORLEVEL%
)


exit /b %ERRORLEVEL%
