# Windows Setup Guide for Image Processing

This guide will help you set up Tesseract OCR and Poppler on Windows for the RAG Chatbot image processing features.

## Prerequisites

- Windows 10 or later
- Administrator access for installation
- Python 3.11+ already installed

## Step 1: Install Tesseract OCR

### Download Tesseract

1. Go to: https://github.com/UB-Mannheim/tesseract/wiki
2. Download the latest installer (e.g., `tesseract-ocr-w64-setup-5.3.3.exe`)
3. Run the installer

### Installation Steps

1. **Accept the license agreement**
2. **Choose installation directory** (recommended: `C:\Program Files\Tesseract-OCR`)
3. **Select components**: Make sure to select "Additional language data" if you need non-English languages
4. **Complete the installation**

### Add Tesseract to PATH

1. Right-click "This PC" or "My Computer" → **Properties**
2. Click **Advanced system settings**
3. Click **Environment Variables**
4. Under "System variables", find **Path** and click **Edit**
5. Click **New** and add: `C:\Program Files\Tesseract-OCR`
6. Click **OK** on all dialogs

### Verify Installation

Open a new Command Prompt or PowerShell and run:

```bash
tesseract --version
```

You should see output like:
```
tesseract 5.3.3
 leptonica-1.83.1
  libgif 5.2.1 : libjpeg 8d (libjpeg-turbo 2.1.3) : libpng 1.6.40 : libtiff 4.5.1 : zlib 1.2.13 : libwebp 1.3.2 : libopenjp2 2.5.0
```

## Step 2: Install Poppler

### Download Poppler

1. Go to: https://github.com/oschwartz10612/poppler-windows/releases/
2. Download the latest release (e.g., `Release-23.11.0-0.zip`)
3. Extract the ZIP file to a permanent location (e.g., `C:\Program Files\poppler-23.11.0`)

### Add Poppler to PATH

1. Right-click "This PC" or "My Computer" → **Properties**
2. Click **Advanced system settings**
3. Click **Environment Variables**
4. Under "System variables", find **Path** and click **Edit**
5. Click **New** and add: `C:\Program Files\poppler-23.11.0\Library\bin`
   (Note: The exact path depends on where you extracted the files)
6. Click **OK** on all dialogs

### Verify Installation

Open a **new** Command Prompt or PowerShell (important: must be new for PATH to update) and run:

```bash
pdftoppm -h
```

You should see the help output for pdftoppm.

## Step 3: Install Python Dependencies

Open Command Prompt or PowerShell in your project directory:

```bash
pip install -r requirements.txt
```

This will install:
- `Pillow` - Image processing library
- `pytesseract` - Python wrapper for Tesseract
- `pdf2image` - PDF to image converter

## Step 4: Test the Installation

### Test Tesseract

Create a test script `test_ocr.py`:

```python
import pytesseract
from PIL import Image

# Test tesseract version
try:
    version = pytesseract.get_tesseract_version()
    print(f"✅ Tesseract version: {version}")
except Exception as e:
    print(f"❌ Tesseract error: {e}")
    print("Make sure Tesseract is installed and in PATH")
```

Run it:
```bash
python test_ocr.py
```

### Test Image Extraction

```python
from src.extractors.image import ImageExtractor

# Create a simple test
extractor = ImageExtractor(language="eng")

# Test with an image
try:
    result = extractor.extract("path/to/test/image.jpg")
    print(f"✅ Extracted {len(result.content)} characters")
    print(f"   Confidence: {result.metadata.get('ocr_confidence', 0):.1f}%")
except Exception as e:
    print(f"❌ Error: {e}")
```

### Test PDF OCR

```python
from src.extractors.pdf import PDFExtractor

extractor = PDFExtractor(ocr_enabled=True, dpi=300)

try:
    result = extractor.extract("path/to/scanned.pdf")
    print(f"✅ Extracted {len(result.content)} characters")
    if result.metadata.get('ocr_used'):
        print(f"   OCR was used on {result.metadata['pages_with_ocr']} pages")
except Exception as e:
    print(f"❌ Error: {e}")
```

## Troubleshooting

### Error: "tesseract is not installed or it's not in your PATH"

**Solutions:**
1. Verify Tesseract is installed: Check if `C:\Program Files\Tesseract-OCR\tesseract.exe` exists
2. Verify PATH is set correctly (see Step 1)
3. **Restart your terminal/IDE** after adding to PATH
4. As a workaround, set the path manually in your code:

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### Error: "Unable to get page count. Is poppler installed and in PATH?"

**Solutions:**
1. Verify Poppler is extracted and the `bin` folder exists
2. Verify PATH includes the `Library\bin` directory
3. **Restart your terminal/IDE** after adding to PATH
4. Try the full path in Python:

```python
import pdf2image
pdf2image.convert_from_path(
    "document.pdf",
    poppler_path=r"C:\Program Files\poppler-23.11.0\Library\bin"
)
```

### Error: "Failed loading language 'ara'"

**Solution:** Install additional language data:
1. Download from: https://github.com/tesseract-ocr/tessdata
2. Copy `.traineddata` files to: `C:\Program Files\Tesseract-OCR\tessdata\`

For example, for Arabic:
- Download `ara.traineddata`
- Copy to `C:\Program Files\Tesseract-OCR\tessdata\ara.traineddata`

### Poor OCR Quality

**Solutions:**
1. Increase DPI (use 600 instead of 300):
   ```python
   extractor = PDFExtractor(dpi=600)
   ```

2. Ensure source images are high quality (300+ DPI scans)

3. Pre-process images for better contrast:
   ```python
   from PIL import Image, ImageEnhance
   
   img = Image.open("document.jpg")
   img = img.convert('L')  # Grayscale
   enhancer = ImageEnhance.Contrast(img)
   img = enhancer.enhance(2)  # Increase contrast
   ```

### Permission Errors

If you get permission errors during installation:
1. Run Command Prompt or PowerShell **as Administrator**
2. Or install to a user directory instead of Program Files

## Additional Language Support

To add support for additional languages:

### Common Languages

Download from: https://github.com/tesseract-ocr/tessdata

1. **Arabic**: Download `ara.traineddata`
2. **Chinese Simplified**: Download `chi_sim.traineddata`
3. **French**: Download `fra.traineddata`
4. **Spanish**: Download `spa.traineddata`
5. **German**: Download `deu.traineddata`

### Installation

Copy all `.traineddata` files to:
```
C:\Program Files\Tesseract-OCR\tessdata\
```

### Usage

```python
# English + Arabic
extractor = ImageExtractor(language="eng+ara")

# English + French + Spanish
extractor = ImageExtractor(language="eng+fra+spa")
```

## Visual Studio C++ Redistributable

Some dependencies may require Visual Studio C++ Redistributable:

Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe

## Next Steps

Once everything is installed and tested:

1. Read the [Image Processing Setup Guide](image_processing_setup.md)
2. Try the examples in `examples/image_extraction_example.py`
3. Start indexing your scanned documents!

```bash
# Index scanned documents
python scripts/index_documents.py ./data/scanned_docs/

# Start the API
uvicorn api.main:app --reload

# Upload via API
curl -X POST http://localhost:8000/index/file -F "file=@scanned.jpg"
```

## Quick Reference

| Component | Installation Path | PATH Entry |
|-----------|------------------|------------|
| Tesseract | `C:\Program Files\Tesseract-OCR` | `C:\Program Files\Tesseract-OCR` |
| Poppler | `C:\Program Files\poppler-23.11.0` | `C:\Program Files\poppler-23.11.0\Library\bin` |
| Language Data | `C:\Program Files\Tesseract-OCR\tessdata` | N/A |

## Getting Help

If you encounter issues:

1. Check the main [Image Processing Setup Guide](image_processing_setup.md)
2. Verify all PATH entries are correct
3. Restart your terminal/IDE after PATH changes
4. Check Tesseract and Poppler versions are compatible
5. Review error messages carefully - they often indicate what's missing

## Performance Tips

For Windows users processing many documents:

1. **Use SSD**: Store documents on SSD for faster processing
2. **Adjust DPI**: Lower DPI (200-250) for faster processing
3. **Batch Processing**: Process multiple documents at once
4. **Resource Monitoring**: Use Task Manager to monitor CPU/RAM usage

## Security Note

When setting up in production:
- Validate uploaded files before processing
- Set size limits for uploads
- Scan uploaded files for malware
- Use a dedicated service account with limited permissions

