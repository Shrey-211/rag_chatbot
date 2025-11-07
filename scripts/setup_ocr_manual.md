# Manual OCR Setup (No Admin Rights Required)

If you cannot run the automated script as Administrator, follow these steps:

## Step 1: Download Tesseract

1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to a location you have access to (e.g., `C:\Users\YourName\Tesseract-OCR`)
3. Note the installation path

## Step 2: Download Poppler

1. Download from: https://github.com/oschwartz10612/poppler-windows/releases/
2. Download: `Release-23.11.0-0.zip`
3. Extract to a location you have access to (e.g., `C:\Users\YourName\poppler-23.11.0`)
4. Note the path to the `Library\bin` folder

## Step 3: Configure in config.yaml

Edit `config.yaml` and add the full paths:

```yaml
ocr:
  # Use double backslashes or forward slashes
  tesseract_path: "C:\\Users\\YourName\\Tesseract-OCR\\tesseract.exe"
  poppler_path: "C:\\Users\\YourName\\poppler-23.11.0\\Library\\bin"
  ocr_language: "eng"
  dpi: 300
  enabled: true
```

Or with forward slashes:

```yaml
ocr:
  tesseract_path: "C:/Users/YourName/Tesseract-OCR/tesseract.exe"
  poppler_path: "C:/Users/YourName/poppler-23.11.0/Library/bin"
  ocr_language: "eng"
  dpi: 300
  enabled: true
```

## Step 4: Restart Backend

Restart your backend server and it will use the configured paths!

## Step 5: Test

Upload a scanned PDF - it should now work with the configured paths.

The system will automatically use these paths instead of looking in the system PATH.

