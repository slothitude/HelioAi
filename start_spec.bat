@echo off
echo ============================================
echo  Qwen3.5-9B Server (PRIMARY)
echo  20 GPU layers, rest CPU, port 8201
echo  Speculative decoding disabled (PR #20075)
echo ============================================
echo.

set "MODEL_DIR=%~dp0models"
set "TARGET=%MODEL_DIR%\Qwen3.5-9B-Q4_K_M.gguf"
set "DRAFT=%MODEL_DIR%\Qwen3.5-2B-Q4_K_M.gguf"

if not exist "%TARGET%" (
    echo [ERROR] Target model not found: %TARGET%
    echo Run download_models.bat first.
    pause
    exit /b 1
)

echo [CONFIG]
echo   Target: 9B Q4_K_M (20 GPU layers, rest CPU)
echo   Draft:  2B Q4_K_M loaded but spec disabled (Qwen3.5 SSM/MoE)
echo   KV Cache: f16 default
echo   Context: 2048 tokens
echo   Host: 0.0.0.0 (external access)
echo   Port: 8201
echo.
echo NOTE: Speculative decoding requires llama.cpp PR #20075
echo   Once merged, add: --model-draft "%DRAFT%" -ngld 0 --draft 5
echo.

llama-server ^
  -m "%TARGET%" ^
  -ngl 20 ^
  -c 2048 ^
  -t 8 ^
  --flash-attn auto ^
  --host 0.0.0.0 ^
  --port 8201

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] llama-server failed.
    pause
    exit /b 1
)
