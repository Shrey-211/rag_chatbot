# Windows OCR Setup Script for RAG Chatbot
# This script automates the installation of Tesseract and Poppler for Windows
# Run with: powershell -ExecutionPolicy Bypass -File scripts/setup_windows_ocr.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  RAG Chatbot - Windows OCR Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Continue"

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "WARNING: This script requires Administrator privileges for PATH modification." -ForegroundColor Yellow
    Write-Host "   Please run PowerShell as Administrator and try again." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "   Right-click PowerShell -> 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

# Configuration
$tesseractPath = "C:\Program Files\Tesseract-OCR"
$popplerExtractPath = "C:\Program Files"

# Try to find any poppler installation
$popplerPath = $null
$popplerBinPath = $null

# Look for any poppler-* directory
$popplerDirs = Get-ChildItem -Path $popplerExtractPath -Directory -Filter "poppler-*" -ErrorAction SilentlyContinue
if ($popplerDirs) {
    foreach ($dir in $popplerDirs) {
        $testBinPath = Join-Path $dir.FullName "Library\bin"
        if (Test-Path $testBinPath) {
            $popplerPath = $dir.FullName
            $popplerBinPath = $testBinPath
            break
        }
    }
}

# Helper Functions
function Test-InPath {
    param($Path)
    $envPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    return $envPath -like "*$Path*"
}

function Add-ToPath {
    param($PathToAdd)
    
    if (Test-InPath $PathToAdd) {
        Write-Host "   Already in PATH: $PathToAdd" -ForegroundColor Green
        return
    }
    
    Write-Host "   Adding to PATH: $PathToAdd" -ForegroundColor Yellow
    $oldPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $newPath = $oldPath + ";" + $PathToAdd
    [Environment]::SetEnvironmentVariable("Path", $newPath, "Machine")
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine")
    Write-Host "   Added to PATH successfully" -ForegroundColor Green
}

# Step 1: Install Tesseract
Write-Host "Step 1: Installing Tesseract OCR" -ForegroundColor Cyan
Write-Host ""

if (Test-Path "$tesseractPath\tesseract.exe") {
    Write-Host "   Tesseract already installed at: $tesseractPath" -ForegroundColor Green
    Add-ToPath $tesseractPath
} else {
    Write-Host "   Tesseract not found. Please install manually:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "   1. Download from: https://github.com/UB-Mannheim/tesseract/wiki" -ForegroundColor White
    Write-Host "   2. Download the latest Windows installer (e.g., tesseract-ocr-w64-setup-5.3.3.exe)" -ForegroundColor White
    Write-Host "   3. Run the installer" -ForegroundColor White
    Write-Host "   4. Install to: C:\Program Files\Tesseract-OCR" -ForegroundColor White
    Write-Host "   5. Re-run this script after installation" -ForegroundColor White
    Write-Host ""
    
    # Try to open browser to download page
    Write-Host "   Opening download page in browser..." -ForegroundColor Yellow
    try {
        Start-Process "https://github.com/UB-Mannheim/tesseract/wiki"
    } catch {
        Write-Host "   Could not open browser. Please visit manually." -ForegroundColor Yellow
    }
    
    Write-Host ""
    $response = Read-Host "   Have you installed Tesseract? (y/n)"
    
    if ($response -eq "y" -or $response -eq "Y") {
        if (Test-Path "$tesseractPath\tesseract.exe") {
            Write-Host "   Tesseract found!" -ForegroundColor Green
            Add-ToPath $tesseractPath
        } else {
            Write-Host "   Tesseract.exe not found at expected location" -ForegroundColor Red
            Write-Host "   Please ensure it's installed at: $tesseractPath" -ForegroundColor Yellow
        }
    } else {
        Write-Host "   Skipping Tesseract for now. You can run this script again later." -ForegroundColor Yellow
    }
}

Write-Host ""

# Step 2: Install Poppler
Write-Host "Step 2: Installing Poppler" -ForegroundColor Cyan
Write-Host ""

if ($popplerBinPath -and (Test-Path $popplerBinPath)) {
    Write-Host "   Poppler already installed at: $popplerPath" -ForegroundColor Green
    Add-ToPath $popplerBinPath
} else {
    Write-Host "   Poppler not found. Please install manually:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "   1. Download from: https://github.com/oschwartz10612/poppler-windows/releases/" -ForegroundColor White
    Write-Host "   2. Download: Release-23.11.0-0.zip (or latest version)" -ForegroundColor White
    Write-Host "   3. Extract to: C:\Program Files\" -ForegroundColor White
    Write-Host "   4. Ensure path is: C:\Program Files\poppler-23.11.0\Library\bin" -ForegroundColor White
    Write-Host "   5. Re-run this script after extraction" -ForegroundColor White
    Write-Host ""
    
    # Try to open browser to download page
    Write-Host "   Opening download page in browser..." -ForegroundColor Yellow
    try {
        Start-Process "https://github.com/oschwartz10612/poppler-windows/releases/"
    } catch {
        Write-Host "   Could not open browser. Please visit manually." -ForegroundColor Yellow
    }
    
    Write-Host ""
    $response = Read-Host "   Have you installed Poppler? (y/n)"
    
    if ($response -eq "y" -or $response -eq "Y") {
        # Re-scan for poppler directories
        $popplerDirs = Get-ChildItem -Path $popplerExtractPath -Directory -Filter "poppler-*" -ErrorAction SilentlyContinue
        if ($popplerDirs) {
            foreach ($dir in $popplerDirs) {
                $testBinPath = Join-Path $dir.FullName "Library\bin"
                if (Test-Path $testBinPath) {
                    $popplerPath = $dir.FullName
                    $popplerBinPath = $testBinPath
                    Write-Host "   Poppler found at: $popplerPath" -ForegroundColor Green
                    Add-ToPath $popplerBinPath
                    break
                }
            }
        }
        
        if (-not $popplerBinPath) {
            Write-Host "   Poppler not found at expected location" -ForegroundColor Red
            Write-Host "   Looking for: C:\Program Files\poppler-*\Library\bin" -ForegroundColor Yellow
            Write-Host "   Please ensure the folder structure is correct" -ForegroundColor Yellow
        }
    } else {
        Write-Host "   Skipping Poppler for now. You can run this script again later." -ForegroundColor Yellow
    }
}

Write-Host ""

# Step 3: Verify Installation
Write-Host "Step 3: Verifying Installation" -ForegroundColor Cyan
Write-Host ""

# Refresh PATH for current session
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine")

$tesseractOK = $false
$popplerOK = $false

# Test Tesseract
Write-Host "   Testing Tesseract..." -ForegroundColor Yellow
if (Test-Path "$tesseractPath\tesseract.exe") {
    try {
        $tesseractVersion = & "$tesseractPath\tesseract.exe" --version 2>&1 | Select-Object -First 1
        Write-Host "   PASS: $tesseractVersion" -ForegroundColor Green
        $tesseractOK = $true
    } catch {
        Write-Host "   FAIL: Tesseract not working" -ForegroundColor Red
    }
} else {
    Write-Host "   FAIL: Tesseract not installed" -ForegroundColor Red
}

# Test Poppler
Write-Host "   Testing Poppler..." -ForegroundColor Yellow
if ($popplerBinPath -and (Test-Path "$popplerBinPath\pdftoppm.exe")) {
    try {
        $null = & "$popplerBinPath\pdftoppm.exe" -h 2>&1
        Write-Host "   PASS: Poppler is working at $popplerPath" -ForegroundColor Green
        $popplerOK = $true
    } catch {
        Write-Host "   FAIL: Poppler not working" -ForegroundColor Red
    }
} else {
    Write-Host "   FAIL: Poppler not installed" -ForegroundColor Red
    # Try to find it one more time
    $popplerDirs = Get-ChildItem -Path "C:\Program Files" -Directory -Filter "poppler-*" -ErrorAction SilentlyContinue
    if ($popplerDirs) {
        Write-Host "   Found poppler directories:" -ForegroundColor Yellow
        foreach ($dir in $popplerDirs) {
            Write-Host "     - $($dir.FullName)" -ForegroundColor White
        }
    }
}

Write-Host ""

# Step 4: Update config.yaml
Write-Host "Step 4: Updating Configuration" -ForegroundColor Cyan
Write-Host ""

$configFile = "config.yaml"

if (Test-Path $configFile) {
    Write-Host "   Updating $configFile with OCR paths..." -ForegroundColor Yellow
    
    $content = Get-Content $configFile -Raw
    
    # Check if OCR section exists
    if ($content -match "ocr:") {
        Write-Host "   OCR section already exists in config" -ForegroundColor Green
        
        # Update paths if components are installed
        if ($tesseractOK) {
            $tesseractPathEscaped = $tesseractPath.Replace('\', '\\') + '\\tesseract.exe'
            $content = $content -replace 'tesseract_path:\s*".*"', "tesseract_path: `"$tesseractPathEscaped`""
            Write-Host "   Updated tesseract_path" -ForegroundColor Green
        }
        
        if ($popplerOK -and $popplerBinPath) {
            $popplerPathEscaped = $popplerBinPath.Replace('\', '\\')
            $content = $content -replace 'poppler_path:\s*".*"', "poppler_path: `"$popplerPathEscaped`""
            Write-Host "   Updated poppler_path to: $popplerBinPath" -ForegroundColor Green
        }
        
        Set-Content -Path $configFile -Value $content
    } else {
        Write-Host "   OCR section already configured" -ForegroundColor Green
    }
} else {
    Write-Host "   Warning: config.yaml not found" -ForegroundColor Yellow
}

Write-Host ""

# Final Summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Installation Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if ($tesseractOK -and $popplerOK) {
    Write-Host "SUCCESS: All components installed and working!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "   1. RESTART your terminal/IDE (required for PATH changes)" -ForegroundColor White
    Write-Host "   2. Restart your backend server" -ForegroundColor White
    Write-Host "   3. Upload scanned PDFs - they will work!" -ForegroundColor White
} elseif ($tesseractOK -or $popplerOK) {
    Write-Host "PARTIAL: Some components installed" -ForegroundColor Yellow
    Write-Host ""
    if (-not $tesseractOK) {
        Write-Host "   MISSING: Tesseract OCR" -ForegroundColor Red
        Write-Host "   Download: https://github.com/UB-Mannheim/tesseract/wiki" -ForegroundColor Yellow
    }
    if (-not $popplerOK) {
        Write-Host "   MISSING: Poppler" -ForegroundColor Red
        Write-Host "   Download: https://github.com/oschwartz10612/poppler-windows/releases/" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "   Run this script again after installing missing components" -ForegroundColor White
} else {
    Write-Host "NO COMPONENTS INSTALLED" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install manually:" -ForegroundColor Yellow
    Write-Host "   1. Tesseract: https://github.com/UB-Mannheim/tesseract/wiki" -ForegroundColor White
    Write-Host "   2. Poppler: https://github.com/oschwartz10612/poppler-windows/releases/" -ForegroundColor White
    Write-Host ""
    Write-Host "Then run this script again to configure PATH" -ForegroundColor White
}

Write-Host ""
Write-Host "Documentation:" -ForegroundColor Cyan
Write-Host "   - OCR_SETUP_GUIDE.md" -ForegroundColor White
Write-Host "   - docs/windows_setup_guide.md" -ForegroundColor White
Write-Host "   - scripts/setup_ocr_manual.md (for manual config)" -ForegroundColor White
Write-Host ""
