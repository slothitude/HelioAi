@echo off
echo ============================================
echo  Qwen3.5-9B Solo (no draft, speed baseline)
echo  Port: 8204
echo ============================================
echo.

set "MODEL_DIR=%~dp0models"
set "MODEL=%MODEL_DIR%\Qwen3.5-9B-Q4_K_M.gguf"

if not exist "%MODEL%" (
    echo [ERROR] Model not found: %MODEL%
    echo Run download_models.bat first.
    pause
    exit /b 1
)

echo [CONFIG]
echo   Model: 9B Q4_K_M (20 GPU layers, rest CPU)
echo   KV Cache: f16 default
echo   Context: 2048 tokens
echo   Host: 0.0.0.0
echo   Port: 8204
echo.
echo This is the BASELINE for speed comparison.
echo Compare tokens/sec against start_spec.bat.
echo.

llama-server ^
  -m "%MODEL%" ^
  -ngl 20 ^
  -c 2048 ^
  -t 8 ^
  --flash-attn auto ^
  --host 0.0.0.0 ^
  --port 8204

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] llama-server failed.
    pause
    exit /b 1
)
