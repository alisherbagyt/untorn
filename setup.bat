@echo off
echo ============================================================
echo  UNTORN Web App — Setup
echo ============================================================
echo.

echo [1/2] Installing Python web dependencies...
pip install -r requirements_web.txt
if errorlevel 1 (
    echo ERROR: Failed to install Python dependencies
    pause
    exit /b 1
)
echo.

echo [2/2] Installing Node.js frontend dependencies...
cd frontend
call npm install
if errorlevel 1 (
    echo ERROR: Failed to install Node.js dependencies
    pause
    exit /b 1
)
cd ..

echo.
echo ============================================================
echo  Setup complete!
echo  Run start_backend.bat and start_frontend.bat to launch.
echo ============================================================
pause
