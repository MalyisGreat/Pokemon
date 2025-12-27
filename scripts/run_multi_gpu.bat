@echo off
REM Multi-GPU training with DeepSpeed
REM Usage: scripts\run_multi_gpu.bat [num_gpus] [config]

set NUM_GPUS=%1
if "%NUM_GPUS%"=="" set NUM_GPUS=8

set CONFIG=%2
if "%CONFIG%"=="" set CONFIG=pokemon_ai/configs/h100_8gpu.yaml

echo ==================================
echo Pokemon AI - Multi-GPU Training
echo ==================================
echo GPUs: %NUM_GPUS%
echo Config: %CONFIG%
echo.

deepspeed --num_gpus=%NUM_GPUS% scripts/train.py --config %CONFIG% %3 %4 %5 %6 %7 %8 %9
