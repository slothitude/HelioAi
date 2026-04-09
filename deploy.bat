@echo off
setlocal

echo ============================================
echo  Deploy Qwen3.5 Stack to lappy-server
echo ============================================
echo.

set "REMOTE=aaron@100.84.161.63"
set "REMOTE_DIR=C:/Users/aaron/hotswap"

echo [1/4] Testing SSH connection...
ssh -o ConnectTimeout=5 %REMOTE% "echo OK" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Cannot reach %REMOTE%
    echo   - Ensure OpenSSH Server is running on lappy-server
    echo   - Check network connectivity
    echo   - Try Tailscale: set REMOTE=aaron@100.84.161.63
    pause
    exit /b 1
)
echo   SSH OK

echo.
echo [2/4] Creating remote directory...
ssh %REMOTE% "if not exist %REMOTE_DIR% mkdir %REMOTE_DIR%"

echo.
echo [3/4] Copying files...
scp *.bat *.py %REMOTE%:%REMOTE_DIR%/
if %ERRORLEVEL% neq 0 (
    echo [ERROR] SCP failed
    pause
    exit /b 1
)
echo   Files copied

echo.
echo [4/4] Remote file listing:
ssh %REMOTE% "dir /b %REMOTE_DIR%\*.bat %REMOTE_DIR%\*.py 2>nul"

echo.
echo ============================================
echo  Deploy complete!
echo ============================================
echo.
echo  Next steps on lappy-server:
echo    1. cd %REMOTE_DIR%
echo    2. download_models.bat
echo    3. start_spec.bat
echo.
echo  Or run remotely:
echo    ssh %REMOTE% "cd /d %REMOTE_DIR% && download_models.bat"
echo    ssh %REMOTE% "cd /d %REMOTE_DIR% && start_spec.bat"
echo.
echo  Test from here:
echo    curl http://100.84.161.63:8201/health
echo.
pause
