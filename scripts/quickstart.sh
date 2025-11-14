#!/bin/bash

# RAG Chatbot Quick Start Script
# This script helps you get started quickly by checking prerequisites and setting up the environment

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Emoji support (works on most terminals)
CHECK="✓"
CROSS="✗"
INFO="ℹ"
WARN="⚠"

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}RAG Chatbot Quick Start${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}${CHECK}${NC} $2"
    else
        echo -e "${RED}${CROSS}${NC} $2"
    fi
}

print_warning() {
    echo -e "${YELLOW}${WARN}${NC} $1"
}

print_info() {
    echo -e "${BLUE}${INFO}${NC} $1"
}

# Check prerequisites
echo -e "${BLUE}Step 1: Checking Prerequisites${NC}"
echo "--------------------------------"

# Check Python
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_status 0 "Python 3 is installed: $PYTHON_VERSION"
    
    # Check if version is 3.11+
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]); then
        print_warning "Python 3.11+ is required. You have $PYTHON_VERSION"
        echo "Please upgrade: https://www.python.org/downloads/"
    fi
else
    print_status 1 "Python 3 is not installed"
    echo "Please install Python 3.11+: https://www.python.org/downloads/"
    exit 1
fi

# Check Node.js (optional)
if command_exists node; then
    NODE_VERSION=$(node --version 2>&1)
    print_status 0 "Node.js is installed: $NODE_VERSION"
else
    print_warning "Node.js is not installed (optional, needed for web interface)"
    echo "Install from: https://nodejs.org/"
fi

# Check Ollama
if command_exists ollama; then
    print_status 0 "Ollama is installed"
    
    # Check if any models are downloaded
    if ollama list >/dev/null 2>&1; then
        print_info "Checking for downloaded models..."
        ollama list
    fi
else
    print_warning "Ollama is not installed (needed for local AI)"
    echo "Install from: https://ollama.ai/"
fi

# Check Tesseract (for OCR)
if command_exists tesseract; then
    TESSERACT_VERSION=$(tesseract --version 2>&1 | head -n1)
    print_status 0 "Tesseract OCR is installed: $TESSERACT_VERSION"
else
    print_warning "Tesseract OCR is not installed (optional, needed for image/PDF OCR)"
    echo "Install:"
    echo "  - macOS: brew install tesseract"
    echo "  - Ubuntu: sudo apt-get install tesseract-ocr"
    echo "  - Windows: https://github.com/UB-Mannheim/tesseract/wiki"
fi

# Check Poppler (for PDF processing)
if command_exists pdfinfo; then
    print_status 0 "Poppler is installed"
else
    print_warning "Poppler is not installed (optional, needed for PDF OCR)"
    echo "Install:"
    echo "  - macOS: brew install poppler"
    echo "  - Ubuntu: sudo apt-get install poppler-utils"
    echo "  - Windows: https://github.com/oschwartz10612/poppler-windows/releases/"
fi

echo ""
echo -e "${BLUE}Step 2: Setting up Python Environment${NC}"
echo "---------------------------------------"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv venv
    print_status 0 "Virtual environment created"
else
    print_status 0 "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip --quiet

# Install requirements
if [ -f "requirements.txt" ]; then
    print_info "Installing Python dependencies (this may take a few minutes)..."
    pip install -r requirements.txt --quiet
    print_status 0 "Python dependencies installed"
else
    print_status 1 "requirements.txt not found"
    exit 1
fi

echo ""
echo -e "${BLUE}Step 3: Creating Data Directories${NC}"
echo "----------------------------------"

# Create necessary directories
mkdir -p data/sample data/uploads data/chroma

print_status 0 "Data directories created"

echo ""
echo -e "${BLUE}Step 4: Configuration Check${NC}"
echo "---------------------------"

if [ -f "config.yaml" ]; then
    print_status 0 "config.yaml found"
    
    # Check if Ollama is configured
    if grep -q "provider: \"ollama\"" config.yaml; then
        print_info "LLM provider: Ollama (local)"
    elif grep -q "provider: \"openai\"" config.yaml; then
        print_info "LLM provider: OpenAI (requires API key)"
        if [ -z "$OPENAI_API_KEY" ]; then
            print_warning "OPENAI_API_KEY environment variable not set"
        fi
    fi
else
    print_status 1 "config.yaml not found"
    if [ -f "config.example.yaml" ]; then
        print_info "Copying config.example.yaml to config.yaml..."
        cp config.example.yaml config.yaml
        print_status 0 "Config file created"
    fi
fi

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Download an Ollama model (if not already done):"
echo "   ${BLUE}ollama pull llama3.2:3b${NC}"
echo ""
echo "2. Start the backend server:"
echo "   ${BLUE}source venv/bin/activate${NC}"
echo "   ${BLUE}uvicorn api.main:app --reload${NC}"
echo ""
echo "3. (Optional) Start the web interface in a new terminal:"
echo "   ${BLUE}cd webapp${NC}"
echo "   ${BLUE}npm install${NC}"
echo "   ${BLUE}npm run dev${NC}"
echo ""
echo "4. Access the API documentation:"
echo "   ${BLUE}http://localhost:8000/docs${NC}"
echo ""
echo "5. Test with a sample document:"
echo "   ${BLUE}python scripts/index_documents.py data/sample/${NC}"
echo "   ${BLUE}python scripts/query.py \"What is RAG?\"${NC}"
echo ""
echo "For detailed instructions, see: ${BLUE}SETUP_GUIDE.md${NC}"
echo ""
print_info "Your virtual environment is activated. Run 'deactivate' to exit it."
echo ""

