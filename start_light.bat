@echo off
echo ============================================
echo  Qwen3.5 Light (0.8B CPU-only)
echo  Port: 8203
echo ============================================
echo.

set "MODEL_DIR=%~dp0models"
set "MODEL=%MODEL_DIR%\Qwen3.5-0.8B-Q4_K_M.gguf"

if not exist "%MODEL%" (
    echo [ERROR] Model not found: %MODEL%
    echo Run download_models.bat first.
    pause
    exit /b 1
)

echo [CONFIG]
echo   Model: 0.8B Q4_K_M (CPU-only)
echo   Context: 2048 tokens
echo   Host: 0.0.0.0
echo   Port: 8203
echo.

llama-server ^
  -m "%MODEL%" ^
  -ngl 0 ^
  -c 2048 ^
  -t 6 ^
  --host 0.0.0.0 ^
  --port 8203

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] llama-server failed.
    pause
    exit /b 1
)
