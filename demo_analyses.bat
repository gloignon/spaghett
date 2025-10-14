@echo off
REM Run surprisal script on demo sentences (MLM mode, no left context)
REM Operates relative to the project folder (batch file location).

cd /d "%~dp0"

if not exist "out" mkdir "out"

python -u "src\main.py" ^
  --input_file "in\demo_sentences.tsv" ^
  --output_file "out\demo_sentences_out.tsv" ^
  --mode mlm ^
  --model cmarkea/distilcamembert-base ^
  --format sentences

if %ERRORLEVEL% EQU 0 (
  echo Finished: "out\demo_sentences_out.tsv"
) else (
  echo Script failed with exit code %ERRORLEVEL%
)

exit /b %ERRORLEVEL%
