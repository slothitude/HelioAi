@echo off
setlocal enabledelayedexpansion

echo ============================================
echo  Qwen3.5 GGUF Model Downloader
echo ============================================
echo.

set "MODEL_DIR=%~dp0models"
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

REM Models from unsloth repos (case-sensitive filenames)
set MODELS[0].repo=unsloth/Qwen3.5-0.8B-GGUF
set MODELS[0].file=Qwen3.5-0.8B-Q4_K_M.gguf
set MODELS[1].repo=unsloth/Qwen3.5-2B-GGUF
set MODELS[1].file=Qwen3.5-2B-Q4_K_M.gguf
set MODELS[2].repo=unsloth/Qwen3.5-4B-GGUF
set MODELS[2].file=Qwen3.5-4B-Q4_K_M.gguf
set MODELS[3].repo=unsloth/Qwen3.5-9B-GGUF
set MODELS[3].file=Qwen3.5-9B-Q4_K_M.gguf

set SIZES[0]=~507MB
set SIZES[1]=~1.2GB
set SIZES[2]=~2.6GB
set SIZES[3]=~5.4GB

where huggingface-cli >nul 2>&1
if %ERRORLEVEL%==0 (
    echo [INFO] Using huggingface-cli
    echo.
    goto :hf_download
) else (
    echo [INFO] Using curl fallback
    echo.
    goto :curl_download
)

:hf_download
for /L %%i in (0,1,3) do (
    set "REPO=!MODELS[%%i].repo!"
    set "FILE=!MODELS[%%i].file!"
    set "SIZE=!SIZES[%%i]!"

    if exist "%MODEL_DIR%\!FILE!" (
        echo [SKIP] !FILE! already exists
    ) else (
        echo [DOWNLOAD] !FILE! !SIZE! from !REPO!
        huggingface-cli download !REPO! !FILE! --local-dir "%MODEL_DIR%" --local-dir-use-symlinks False
        if !ERRORLEVEL! neq 0 (
            echo [FALLBACK] Trying curl...
            call :curl_single "!REPO!" "!FILE!"
        )
    )
    echo.
)
goto :done

:curl_download
for /L %%i in (0,1,3) do (
    set "REPO=!MODELS[%%i].repo!"
    set "FILE=!MODELS[%%i].file!"
    set "SIZE=!SIZES[%%i]!"

    if exist "%MODEL_DIR%\!FILE!" (
        echo [SKIP] !FILE! already exists
    ) else (
        echo [DOWNLOAD] !FILE! !SIZE!
        call :curl_single "!REPO!" "!FILE!"
    )
    echo.
)
goto :done

:curl_single
set "REPO=%~1"
set "FILE=%~2"
set "URL=https://huggingface.co/!REPO!/resolve/main/!FILE!"
echo   URL: !URL!
curl -L --progress-bar -o "%MODEL_DIR%\!FILE!" "!URL!"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] curl download failed for !FILE!
)
goto :eof

:done
echo.
echo ============================================
echo  Download Summary
echo ============================================
for %%f in ("%MODEL_DIR%\*.gguf") do (
    echo   %%~nxf - %%~zf bytes
)
echo.
echo Models saved to: %MODEL_DIR%
echo Done!
pause
