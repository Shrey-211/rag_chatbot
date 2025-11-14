# 🚀 Complete Setup Guide for RAG Chatbot

**Welcome!** This guide will help you set up the RAG Chatbot system from scratch. We've made it super simple - even if you're new to programming, you can follow these steps!

## 📋 Table of Contents

1. [What You'll Need](#what-youll-need)
2. [Step 1: Install Python](#step-1-install-python)
3. [Step 2: Install Node.js (for Web Interface)](#step-2-install-nodejs-for-web-interface)
4. [Step 3: Download the Project](#step-3-download-the-project)
5. [Step 4: Install Ollama (Local AI)](#step-4-install-ollama-local-ai)
6. [Step 5: Install OCR Tools](#step-5-install-ocr-tools)
7. [Step 6: Set Up Python Environment](#step-6-set-up-python-environment)
8. [Step 7: Configure the System](#step-7-configure-the-system)
9. [Step 8: Start the Backend Server](#step-8-start-the-backend-server)
10. [Step 9: Set Up Web Interface (Optional)](#step-9-set-up-web-interface-optional)
11. [Step 10: Test Everything](#step-10-test-everything)
12. [Troubleshooting](#troubleshooting)
13. [Next Steps](#next-steps)

---

## What You'll Need

Before we begin, here's what you'll need:

- **A Computer** running Windows, macOS, or Linux
- **Internet Connection** to download software
- **5-10 GB of free disk space**
- **About 30-60 minutes** of your time

Don't worry if you don't understand everything right away - just follow the steps one by one!

---

## Step 1: Install Python

Python is the programming language this project uses.

### Windows

1. **Download Python:**
   - Go to [https://www.python.org/downloads/](https://www.python.org/downloads/)
   - Click the yellow button that says "Download Python 3.11.x" (or higher version like 3.12)
   - Save the file to your computer

2. **Install Python:**
   - Double-click the downloaded file
   - ⚠️ **IMPORTANT**: Check the box that says "Add Python to PATH" at the bottom
   - Click "Install Now"
   - Wait for installation to complete
   - Click "Close"

3. **Verify Installation:**
   - Open Command Prompt (press `Win + R`, type `cmd`, press Enter)
   - Type: `python --version`
   - You should see something like: `Python 3.11.x`

### macOS

1. **Download Python:**
   - Go to [https://www.python.org/downloads/](https://www.python.org/downloads/)
   - Click "Download Python 3.11.x"
   - Save the file

2. **Install Python:**
   - Open the downloaded `.pkg` file
   - Follow the installation wizard
   - Click "Continue" and "Install"
   - Enter your password if asked

3. **Verify Installation:**
   - Open Terminal (press `Cmd + Space`, type "Terminal", press Enter)
   - Type: `python3 --version`
   - You should see: `Python 3.11.x`

### Linux (Ubuntu/Debian)

1. **Install Python:**
   ```bash
   sudo apt update
   sudo apt install python3.11 python3.11-venv python3-pip
   ```

2. **Verify Installation:**
   ```bash
   python3 --version
   ```

**✅ Python is installed!** Let's move to the next step.

---

## Step 2: Install Node.js (for Web Interface)

Node.js is needed for the web interface. If you only want to use the API, you can skip this step.

### Windows & macOS

1. **Download Node.js:**
   - Go to [https://nodejs.org/](https://nodejs.org/)
   - Download the **LTS version** (Long Term Support) - it's the recommended one
   - You need version **18** or higher

2. **Install Node.js:**
   - Open the downloaded file
   - Follow the installation wizard
   - Click "Next" → "Next" → "Install"
   - Wait for installation to complete

3. **Verify Installation:**
   - Open Command Prompt (Windows) or Terminal (macOS)
   - Type: `node --version`
   - You should see: `v18.x.x` or `v20.x.x` or higher
   - Type: `npm --version`
   - You should see: `9.x.x` or `10.x.x` or higher

### Linux (Ubuntu/Debian)

```bash
# Install Node.js 20 (recommended)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify
node --version
npm --version
```

**✅ Node.js is installed!**

---

## Step 3: Download the Project

Now let's get the project files on your computer.

### Option A: Using Git (Recommended)

If you have Git installed:

1. **Open Terminal/Command Prompt**
2. **Navigate to where you want the project:**
   ```bash
   # Windows example
   cd C:\Users\YourName\Documents
   
   # macOS/Linux example
   cd ~/Documents
   ```

3. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd rag_chatbot
   ```

### Option B: Download ZIP

If you don't have Git:

1. Go to your GitHub repository page
2. Click the green "Code" button
3. Click "Download ZIP"
4. Extract the ZIP file to a folder like:
   - Windows: `C:\Users\YourName\Documents\rag_chatbot`
   - macOS/Linux: `~/Documents/rag_chatbot`
5. Open Terminal/Command Prompt and navigate to that folder:
   ```bash
   cd path/to/rag_chatbot
   ```

**✅ Project downloaded!**

---

## Step 4: Install Ollama (Local AI)

Ollama runs AI models locally on your computer - no internet needed after setup!

### Windows

1. **Download Ollama:**
   - Go to [https://ollama.ai/download](https://ollama.ai/download)
   - Click "Download for Windows"
   - Save the file

2. **Install Ollama:**
   - Run the downloaded `.exe` file
   - Follow the installation wizard
   - Ollama will start automatically

3. **Download an AI Model:**
   - Open Command Prompt
   - Type one of these commands:
   
   ```bash
   # Small, fast model (1.3 GB) - Good for testing
   ollama pull llama3.2:1b
   
   # Better quality model (2 GB) - Recommended
   ollama pull llama3.2:3b
   
   # For vision/OCR features (7 GB)
   ollama pull llama3.2-vision:11b
   ```
   
   - Wait for download to complete (this may take a few minutes)

4. **Verify Installation:**
   ```bash
   ollama list
   ```
   You should see the model(s) you downloaded.

### macOS

1. **Download Ollama:**
   - Go to [https://ollama.ai/download](https://ollama.ai/download)
   - Click "Download for macOS"
   - Save the file

2. **Install Ollama:**
   - Open the downloaded `.dmg` file
   - Drag Ollama to Applications
   - Open Ollama from Applications

3. **Download an AI Model:**
   - Open Terminal
   - Run the same commands as Windows (see above)

### Linux

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download a model
ollama pull llama3.2:3b

# Optional: Vision model
ollama pull llama3.2-vision:11b

# Verify
ollama list
```

**✅ Ollama is ready!** Your computer can now run AI models locally.

---

## Step 5: Install OCR Tools

OCR (Optical Character Recognition) lets the system read text from images and scanned PDFs.

### Windows

#### Option A: Automated Installation (Requires Admin Rights)

1. **Open PowerShell as Administrator:**
   - Press `Win + X`
   - Click "Windows PowerShell (Admin)" or "Terminal (Admin)"
   - Click "Yes" when asked for permission

2. **Run the installation script:**
   ```powershell
   cd path\to\rag_chatbot
   .\scripts\setup_windows_ocr.ps1
   ```

3. **Follow the prompts** - the script will:
   - Download Tesseract OCR
   - Download Poppler PDF tools
   - Install them automatically
   - Update your config file

#### Option B: Manual Installation (No Admin Needed)

If you can't use admin rights, follow these steps:

1. **Download Tesseract:**
   - Go to [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
   - Download the latest Windows installer (e.g., `tesseract-ocr-w64-setup-5.x.x.exe`)
   - Install it to a location you can access (e.g., `C:\Users\YourName\Tesseract-OCR`)
   - Remember the installation path!

2. **Download Poppler:**
   - Go to [https://github.com/oschwartz10612/poppler-windows/releases/](https://github.com/oschwartz10612/poppler-windows/releases/)
   - Download the latest release (e.g., `Release-24.07.0-0.zip`)
   - Extract it to a location you can access (e.g., `C:\Users\YourName\poppler-24.07.0`)
   - Remember the path to the `Library\bin` folder!

3. **Update Configuration:**
   - Open `config.yaml` in a text editor (like Notepad)
   - Find the `ocr:` section
   - Update the paths (use double backslashes or forward slashes):
   
   ```yaml
   ocr:
     tesseract_path: "C:\\Users\\YourName\\Tesseract-OCR\\tesseract.exe"
     poppler_path: "C:\\Users\\YourName\\poppler-24.07.0\\Library\\bin"
     ocr_language: "eng"
     dpi: 300
     enabled: true
   ```

### macOS

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install OCR tools
brew install tesseract
brew install poppler

# Verify
tesseract --version
pdfinfo -v
```

### Linux (Ubuntu/Debian)

```bash
# Install OCR tools
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils

# Optional: Install additional language packs
sudo apt-get install -y tesseract-ocr-ara  # Arabic
sudo apt-get install -y tesseract-ocr-fra  # French
sudo apt-get install -y tesseract-ocr-spa  # Spanish

# Verify
tesseract --version
pdfinfo -version
```

**✅ OCR tools are installed!** Now your system can read scanned documents.

---

## ⚡ Quick Setup (Alternative)

If you want to automate steps 6-7, we have a quickstart script:

### Windows
```bash
scripts\quickstart.bat
```

### macOS/Linux
```bash
chmod +x scripts/quickstart.sh
./scripts/quickstart.sh
```

This script will automatically:
- Check all prerequisites
- Create virtual environment
- Install Python dependencies
- Create data directories
- Verify configuration

**Or follow the manual steps below:**

---

## Step 6: Set Up Python Environment

Now let's install all the Python packages the project needs.

### Create Virtual Environment

A virtual environment keeps this project's packages separate from your system.

#### Windows

```bash
# Navigate to project folder
cd path\to\rag_chatbot

# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Your prompt should now show (venv) at the beginning
```

#### macOS/Linux

```bash
# Navigate to project folder
cd ~/Documents/rag_chatbot

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Your prompt should now show (venv) at the beginning
```

### Install Python Packages

Now install all required packages:

```bash
# Upgrade pip (the package installer)
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

This will take **5-10 minutes** as it downloads and installs many packages. Don't worry if you see warnings - as long as it completes without errors, you're good!

**✅ Python environment is ready!**

---

## Step 7: Configure the System

Let's set up the configuration file.

### Basic Configuration

The project comes with a `config.yaml` file that should already be set up. Let's verify it:

1. **Open `config.yaml` in a text editor**

2. **Check these important settings:**

   ```yaml
   llm:
     provider: "ollama"  # We're using Ollama
     ollama:
       base_url: "http://localhost:11434"
       model: "llama3.2:3b"  # Make sure this matches what you downloaded
   
   embedding:
     provider: "local"  # Free, runs on your computer
     local:
       model: "all-mpnet-base-v2"
       device: "cpu"  # Change to "cuda" if you have an NVIDIA GPU
   
   ocr:
     enabled: true
     tesseract_path: ""  # Leave empty on Mac/Linux, set path on Windows
     poppler_path: ""     # Leave empty on Mac/Linux, set path on Windows
     ocr_language: "eng"  # Change to "eng+ara" for English + Arabic
     dpi: 300
   ```

3. **Update OCR paths (Windows only):**
   - If you installed OCR tools manually, update the paths as shown in Step 5
   - If you used the automated installer, the paths should already be set

4. **Save the file**

### Create Data Directories

```bash
# Windows
mkdir data\sample
mkdir data\uploads
mkdir data\chroma

# macOS/Linux
mkdir -p data/sample
mkdir -p data/uploads
mkdir -p data/chroma
```

**✅ Configuration complete!**

---

## Step 8: Start the Backend Server

Now let's start the API server!

### Make Sure Everything is Running

1. **Check Ollama is running:**
   ```bash
   ollama list
   ```
   You should see your downloaded models.

2. **Make sure your virtual environment is activated:**
   - You should see `(venv)` in your terminal prompt
   - If not, run:
     - Windows: `venv\Scripts\activate`
     - macOS/Linux: `source venv/bin/activate`

### Start the Server

```bash
# Start the FastAPI server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

You should see output like:

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using StatReload
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### Test the Server

Open your web browser and go to:

- **API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

If you see a nice interface at `/docs`, congratulations! 🎉 Your backend is running!

**Keep this terminal window open** - the server needs to stay running.

**✅ Backend server is running!**

---

## Step 9: Set Up Web Interface (Optional)

The web interface provides a nice chat UI for your RAG system.

### Open a New Terminal

**Important:** Keep the backend server running in the first terminal. Open a **new** terminal window.

### Navigate to Web App Folder

```bash
# Windows
cd path\to\rag_chatbot\webapp

# macOS/Linux
cd ~/Documents/rag_chatbot/webapp
```

### Install Dependencies

```bash
# Install Node.js packages (this takes 2-3 minutes)
npm install
```

### Configure API URL (Optional)

If your backend runs on a different port:

```bash
# Windows
copy env.example .env

# macOS/Linux
cp env.example .env
```

Edit `.env` and set:
```
VITE_API_BASE_URL=http://localhost:8000
```

### Start the Web Interface

```bash
npm run dev
```

You should see:

```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

### Open the Web Interface

Open your browser and go to: [http://localhost:5173](http://localhost:5173)

You should see a beautiful chat interface! 🎨

**✅ Web interface is running!**

---

## Step 10: Test Everything

Let's make sure everything works!

### Test 1: Upload a Document

#### Using the API (http://localhost:8000/docs):

1. Go to [http://localhost:8000/docs](http://localhost:8000/docs)
2. Find the `POST /index/file` endpoint
3. Click "Try it out"
4. Click "Choose File" and upload a PDF or text file
5. Click "Execute"
6. You should see a success response!

#### Using the Web Interface:

1. Go to [http://localhost:5173](http://localhost:5173)
2. Click the "Upload Document" button
3. Select a file
4. Wait for upload to complete
5. You should see a success message!

### Test 2: Ask a Question

#### Using the API:

1. Go to [http://localhost:8000/docs](http://localhost:8000/docs)
2. Find the `POST /query` endpoint
3. Click "Try it out"
4. Enter a query like: "What is this document about?"
5. Click "Execute"
6. You should get an AI-generated answer!

#### Using the Web Interface:

1. Type a question in the chat box
2. Press Enter
3. Watch the AI respond!

### Test 3: OCR (Optional)

If you set up OCR:

1. Find a scanned PDF or image with text
2. Upload it using either method above
3. Ask questions about it
4. The system should be able to read the text!

**✅ Everything works!** You're all set! 🎉

---

## Troubleshooting

### Problem: "python: command not found"

**Solution:**
- Windows: Make sure you checked "Add Python to PATH" during installation
- macOS/Linux: Use `python3` instead of `python`

### Problem: "Permission denied" when installing packages

**Solution:**
```bash
# Don't use sudo with virtual environment!
# Make sure venv is activated first
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Problem: Ollama not responding

**Solution:**
- Make sure Ollama is running:
  - Windows: Check system tray for Ollama icon
  - macOS: Check menu bar for Ollama icon
  - Linux: Run `ollama serve` in a terminal
- Verify model is downloaded: `ollama list`

### Problem: OCR not working

**Solution:**
- **Windows:** Make sure paths in `config.yaml` are correct
- **Mac/Linux:** Install tools: `brew install tesseract poppler` or `apt-get install tesseract-ocr poppler-utils`
- Test manually:
  ```bash
  tesseract --version
  pdfinfo -v
  ```

### Problem: "Port already in use"

**Solution:**
```bash
# Backend (change port)
uvicorn api.main:app --reload --port 8001

# Frontend (change port)
npm run dev -- --port 5174
```

### Problem: Out of memory when running models

**Solution:**
- Use smaller models:
  ```bash
  ollama pull llama3.2:1b  # Only 1.3 GB
  ```
- Update `config.yaml`:
  ```yaml
  llm:
    ollama:
      model: "llama3.2:1b"
  ```

### Problem: Slow AI responses

**Solution:**
- Use smaller models (see above)
- If you have an NVIDIA GPU, enable CUDA:
  ```yaml
  embedding:
    local:
      device: "cuda"
  ```

### Problem: Frontend can't connect to backend

**Solution:**
1. Check backend is running on port 8000
2. Check CORS settings in `config.yaml`:
   ```yaml
   api:
     cors_origins:
       - "http://localhost:5173"
       - "http://localhost:3000"
   ```

### Problem: Import errors or module not found

**Solution:**
```bash
# Make sure virtual environment is activated
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# Reinstall packages
pip install -r requirements.txt
```

### Still Having Issues?

1. **Check the logs** - Error messages usually tell you what's wrong
2. **Restart everything:**
   - Stop backend (Ctrl+C)
   - Stop frontend (Ctrl+C)
   - Restart Ollama
   - Start backend again
   - Start frontend again
3. **Create an issue** on GitHub with:
   - Your operating system
   - Python version (`python --version`)
   - Node version (`node --version`)
   - Full error message
   - What you were trying to do

---

## Next Steps

Now that everything is set up, here's what you can do:

### 1. Index Your Documents

Upload documents to build your knowledge base:
- PDF files
- Word documents (.docx)
- Text files (.txt)
- Scanned documents (with OCR)
- Images (with OCR)

### 2. Customize the System

Edit `config.yaml` to:
- Try different AI models
- Adjust chunk sizes for better retrieval
- Change OCR settings
- Modify system prompts

### 3. Use Different Models

Try other Ollama models:
```bash
# Better quality
ollama pull qwen2.5:3b
ollama pull phi3.5

# For coding questions
ollama pull deepseek-coder

# For vision/OCR
ollama pull llava:13b
```

Then update `config.yaml`:
```yaml
llm:
  ollama:
    model: "qwen2.5:3b"
```

### 4. Advanced Features

- **Multiple Languages:** Set `ocr_language: "eng+ara+fra"` for multiple languages
- **GPU Acceleration:** Set `device: "cuda"` if you have NVIDIA GPU
- **Custom Prompts:** Edit the `rag_template` in `config.yaml`
- **Docker Deployment:** Use `docker-compose up` for production

### 5. Learn More

- Read the main [README.md](README.md) for detailed documentation
- Check the `docs/` folder for guides
- Visit [http://localhost:8000/docs](http://localhost:8000/docs) for API documentation
- Explore the code in the `src/` directory

---

## Understanding What You Built

### What is RAG?

RAG (Retrieval-Augmented Generation) is like giving an AI assistant a library:

1. **Upload documents** → Store them in a database
2. **Ask questions** → AI searches the database for relevant info
3. **Get answers** → AI uses the found information to answer your question

This is better than regular ChatGPT because:
- ✅ Works with **your** documents
- ✅ More **accurate** (uses your data)
- ✅ **Private** (runs locally)
- ✅ No hallucinations (AI can't make up facts)

### System Components

- **Backend (Python + FastAPI):** Handles document processing, search, and AI
- **Frontend (React + Vite):** Provides the chat interface
- **Ollama:** Runs AI models locally on your computer
- **ChromaDB:** Stores document embeddings for fast search
- **Tesseract + Poppler:** Reads text from images and scanned PDFs

### How It Works

```
1. You upload a document
   ↓
2. System extracts text (with OCR if needed)
   ↓
3. Text is split into chunks
   ↓
4. Chunks are converted to vectors (embeddings)
   ↓
5. Vectors are stored in ChromaDB
   ↓
6. When you ask a question:
   - Question is converted to vector
   - System finds similar vectors in database
   - Relevant chunks are sent to AI
   - AI generates answer based on chunks
```

---

## System Requirements Summary

### Minimum Requirements

- **CPU:** Dual-core processor
- **RAM:** 8 GB (16 GB recommended)
- **Storage:** 10 GB free space
- **OS:** Windows 10+, macOS 12+, Ubuntu 20.04+
- **Python:** 3.11 or higher
- **Node.js:** 18 or higher

### Recommended for Better Performance

- **CPU:** Quad-core or better
- **RAM:** 16 GB or more
- **Storage:** 20 GB+ SSD
- **GPU:** NVIDIA GPU with CUDA support (optional, for faster processing)

---

## Quick Reference Commands

### Start Backend
```bash
# Activate virtual environment first!
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# Start server
uvicorn api.main:app --reload
```

### Start Frontend
```bash
cd webapp
npm run dev
```

### Ollama Commands
```bash
ollama list              # List installed models
ollama pull model-name   # Download a model
ollama rm model-name     # Remove a model
ollama serve             # Start Ollama server (Linux)
```

### Check Status
```bash
# Backend health
curl http://localhost:8000/health

# List documents
curl http://localhost:8000/documents
```

---

## 🎓 Congratulations!

You've successfully set up a production-grade RAG chatbot system! This is a complex project with many moving parts, and you did it! 🎉

**What you learned:**
- How to set up Python and Node.js
- How to use virtual environments
- How to run AI models locally
- How to set up OCR
- How to configure and run a full-stack application

**You can now:**
- Build your own AI chatbot with your documents
- Process PDFs, images, and scanned documents
- Run everything locally and privately
- Extend and customize the system

Keep learning and building! 🚀

---

## 📚 Additional Resources

- **Ollama Models:** [https://ollama.ai/library](https://ollama.ai/library)
- **Python Tutorial:** [https://docs.python.org/3/tutorial/](https://docs.python.org/3/tutorial/)
- **FastAPI Docs:** [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
- **React Tutorial:** [https://react.dev/learn](https://react.dev/learn)
- **Tesseract Languages:** [https://github.com/tesseract-ocr/tessdata](https://github.com/tesseract-ocr/tessdata)

---

## 💬 Get Help

If you get stuck:

1. Check the [Troubleshooting](#troubleshooting) section
2. Read the error message carefully
3. Search for the error on Google
4. Ask on the project's GitHub Issues page
5. Check if services are running: Ollama, Backend, Frontend

Remember: Everyone starts somewhere, and error messages are learning opportunities! Don't give up! 💪

---

**Made with ❤️ for learners and builders everywhere**

