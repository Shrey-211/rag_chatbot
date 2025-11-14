@echo off
REM RAG Chatbot Quick Start Script for Windows
REM This script helps you get started quickly by checking prerequisites and setting up the environment

echo ================================
echo RAG Chatbot Quick Start
echo ================================
echo.

REM Check Python
echo Step 1: Checking Prerequisites
echo --------------------------------
echo.

where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] Python is installed
    python --version
) else (
    echo [ERROR] Python is not installed
    echo Please install Python 3.11+ from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Check Node.js (optional)
where node >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] Node.js is installed
    node --version
) else (
    echo [WARNING] Node.js is not installed (optional, needed for web interface^)
    echo Install from: https://nodejs.org/
)

REM Check Ollama
where ollama >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] Ollama is installed
    ollama list
) else (
    echo [WARNING] Ollama is not installed (needed for local AI^)
    echo Install from: https://ollama.ai/
)

REM Check Tesseract
where tesseract >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] Tesseract OCR is installed
) else (
    echo [WARNING] Tesseract OCR is not installed (optional, needed for image/PDF OCR^)
    echo Install from: https://github.com/UB-Mannheim/tesseract/wiki
)

echo.
echo Step 2: Setting up Python Environment
echo ---------------------------------------
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install requirements
if exist "requirements.txt" (
    echo Installing Python dependencies (this may take a few minutes^)...
    pip install -r requirements.txt --quiet
    echo [OK] Python dependencies installed
) else (
    echo [ERROR] requirements.txt not found
    pause
    exit /b 1
)

echo.
echo Step 3: Creating Data Directories
echo ----------------------------------
echo.

if not exist "data\sample" mkdir data\sample
if not exist "data\uploads" mkdir data\uploads
if not exist "data\chroma" mkdir data\chroma

echo [OK] Data directories created

echo.
echo Step 4: Configuration Check
echo ---------------------------
echo.

if exist "config.yaml" (
    echo [OK] config.yaml found
) else (
    echo [WARNING] config.yaml not found
    if exist "config.example.yaml" (
        echo Copying config.example.yaml to config.yaml...
        copy config.example.yaml config.yaml
        echo [OK] Config file created
    )
)

echo.
echo ================================
echo Setup Complete!
echo ================================
echo.
echo Next steps:
echo.
echo 1. Download an Ollama model (if not already done^):
echo    ollama pull llama3.2:3b
echo.
echo 2. Start the backend server:
echo    venv\Scripts\activate
echo    uvicorn api.main:app --reload
echo.
echo 3. (Optional^) Start the web interface in a new command prompt:
echo    cd webapp
echo    npm install
echo    npm run dev
echo.
echo 4. Access the API documentation:
echo    http://localhost:8000/docs
echo.
echo 5. Test with a sample document:
echo    python scripts\index_documents.py data\sample\
echo    python scripts\query.py "What is RAG?"
echo.
echo For detailed instructions, see: SETUP_GUIDE.md
echo.
echo Your virtual environment is activated. Run 'deactivate' to exit it.
echo.
pause

