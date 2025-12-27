@echo off
REM Single GPU training on H100
REM Usage: scripts\run_single_gpu.bat [config_override]

set CONFIG=%1
if "%CONFIG%"=="" set CONFIG=pokemon_ai/configs/h100_1gpu.yaml

echo ==================================
echo Pokemon AI - Single GPU Training
echo ==================================
echo Config: %CONFIG%
echo.

python scripts/train.py --config %CONFIG% %2 %3 %4 %5 %6 %7 %8 %9
