# Image Processing and OCR Setup Guide

This guide explains how to set up and use the image processing and OCR (Optical Character Recognition) capabilities in the RAG Chatbot system.

## Overview

The system now supports:
- **Standalone image files** (JPG, PNG, TIFF, BMP, GIF, WebP)
- **Scanned PDF documents** (PDFs containing images instead of text)
- **Mixed PDFs** (PDFs with both text and images)
- **Automatic detection** of scanned documents

## Prerequisites

### 1. Install Tesseract OCR

Tesseract is an open-source OCR engine required for extracting text from images.

#### Windows
1. Download the installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer (e.g., `tesseract-ocr-w64-setup-5.3.3.exe`)
3. During installation, note the installation path (default: `C:\Program Files\Tesseract-OCR`)
4. Add Tesseract to your system PATH:
   - Right-click "This PC" → Properties → Advanced system settings
   - Click "Environment Variables"
   - Under "System variables", find "Path" and click "Edit"
   - Click "New" and add: `C:\Program Files\Tesseract-OCR`
   - Click "OK" on all dialogs

5. Verify installation:
```bash
tesseract --version
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install libtesseract-dev
```

#### macOS
```bash
brew install tesseract
```

### 2. Install Poppler (for PDF to Image conversion)

#### Windows
1. Download from: https://github.com/oschwartz10612/poppler-windows/releases/
2. Extract to a location (e.g., `C:\Program Files\poppler-23.11.0`)
3. Add the `bin` folder to your PATH:
   - Add `C:\Program Files\poppler-23.11.0\Library\bin` to system PATH
   - Restart your terminal/IDE

4. Verify installation:
```bash
pdftoppm -h
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get install poppler-utils
```

#### macOS
```bash
brew install poppler
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

The following packages will be installed:
- `Pillow`: Image processing library
- `pytesseract`: Python wrapper for Tesseract
- `pdf2image`: PDF to image converter

## Language Support

### Installing Additional Languages

By default, Tesseract includes English (`eng`). To support other languages:

#### Windows
1. Download language data files from: https://github.com/tesseract-ocr/tessdata
2. Copy `.traineddata` files to: `C:\Program Files\Tesseract-OCR\tessdata\`

#### Linux
```bash
# Arabic
sudo apt-get install tesseract-ocr-ara

# French
sudo apt-get install tesseract-ocr-fra

# Spanish
sudo apt-get install tesseract-ocr-spa

# List all available languages
apt-cache search tesseract-ocr
```

#### Using Multiple Languages

```python
from src.extractors.image import ImageExtractor

# For English and Arabic
extractor = ImageExtractor(language="eng+ara")

# Extract text
doc = extractor.extract("path/to/image.jpg")
print(doc.content)
```

## Usage

### 1. Extracting Text from Image Files

```python
from src.extractors.image import ImageExtractor

# Create extractor
extractor = ImageExtractor(language="eng", dpi=300)

# Extract text from image
result = extractor.extract("path/to/scanned_document.jpg")

print(result.content)
print(result.metadata)
```

### 2. Extracting Text from Scanned PDFs

The PDF extractor automatically detects scanned PDFs and applies OCR:

```python
from src.extractors.pdf import PDFExtractor

# Create extractor with OCR enabled (default)
extractor = PDFExtractor(
    ocr_enabled=True,
    ocr_language="eng",
    dpi=300,
    min_text_threshold=50
)

# Extract text (OCR will be used automatically if needed)
result = extractor.extract("path/to/scanned.pdf")

# Check if OCR was used
if result.metadata.get("ocr_used"):
    print(f"OCR was used on {result.metadata['pages_with_ocr']} pages")
```

### 3. Using the Extractor Factory

The factory automatically selects the appropriate extractor:

```python
from src.extractors.base import ExtractorFactory

factory = ExtractorFactory()

# Works for images
doc1 = factory.extract("document.jpg")

# Works for scanned PDFs
doc2 = factory.extract("scanned.pdf")

# Works for regular PDFs
doc3 = factory.extract("text.pdf")
```

### 4. Indexing Images and Scanned Documents

Use the indexing script:

```bash
python scripts/index_documents.py --directory ./data/scanned_docs/
```

This will automatically:
- Detect image files (JPG, PNG, TIFF, etc.)
- Detect scanned PDFs
- Apply OCR where needed
- Index the extracted text

### 5. Via the API

Upload and index scanned documents through the API:

```bash
# Upload an image
curl -X POST "http://localhost:8000/upload" \
  -F "file=@scanned_document.jpg"

# Upload a scanned PDF
curl -X POST "http://localhost:8000/upload" \
  -F "file=@scanned_invoice.pdf"
```

The system will automatically apply OCR and index the content.

## Configuration

### PDF Extractor Settings

```python
PDFExtractor(
    ocr_enabled=True,           # Enable/disable OCR fallback
    ocr_language="eng",         # Tesseract language code
    dpi=300,                    # DPI for PDF-to-image conversion (higher = better quality, slower)
    min_text_threshold=50       # Minimum chars per page to consider it has text
)
```

### Image Extractor Settings

```python
ImageExtractor(
    language="eng",             # Tesseract language code
    dpi=None                    # Optional: force specific DPI
)
```

## Optimizing OCR Quality

### 1. Image Quality
- Use at least 300 DPI for scanning
- Higher DPI (600) for small text
- Use grayscale or black & white for text-only documents

### 2. Pre-processing
For better results, you can pre-process images:

```python
from PIL import Image, ImageEnhance
import pytesseract

# Load image
image = Image.open("document.jpg")

# Convert to grayscale
image = image.convert('L')

# Increase contrast
enhancer = ImageEnhance.Contrast(image)
image = enhancer.enhance(2)

# Apply threshold
image = image.point(lambda x: 0 if x < 128 else 255, '1')

# OCR
text = pytesseract.image_to_string(image)
```

### 3. Tesseract Configuration

```python
custom_config = r'--oem 3 --psm 6'

# OEM (OCR Engine Mode):
# 0 = Legacy engine only
# 1 = Neural nets LSTM engine only
# 2 = Legacy + LSTM engines
# 3 = Default (best available)

# PSM (Page Segmentation Mode):
# 0 = Orientation and script detection (OSD) only
# 1 = Automatic page segmentation with OSD
# 3 = Fully automatic page segmentation (default)
# 6 = Assume a single uniform block of text
# 11 = Sparse text (find as much text as possible)

text = pytesseract.image_to_string(image, config=custom_config)
```

## Troubleshooting

### Error: "tesseract is not installed or it's not in your PATH"

**Solution:**
1. Verify Tesseract is installed: `tesseract --version`
2. If not installed, follow the installation steps above
3. Ensure Tesseract is in your system PATH
4. On Windows, you may need to set the path manually:

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### Error: "Unable to get page count. Is poppler installed and in PATH?"

**Solution:**
1. Install Poppler (see prerequisites)
2. Verify installation: `pdftoppm -h`
3. Ensure Poppler's `bin` folder is in your system PATH

### Poor OCR Quality

**Solutions:**
1. Increase DPI (use 600 instead of 300)
2. Pre-process images (increase contrast, denoise)
3. Use appropriate page segmentation mode
4. Ensure correct language is set
5. Check that the original scan quality is good

### Slow Performance

**Solutions:**
1. Reduce DPI (use 200 instead of 300) for faster processing
2. Process documents in batches
3. Use multiprocessing for large document sets
4. Consider using a GPU-accelerated OCR solution for production

## Performance Benchmarks

Typical processing times (on modern CPU):

| Document Type | Pages | DPI | Time |
|--------------|-------|-----|------|
| Image (JPG) | 1 | 300 | ~2-3s |
| Image (JPG) | 1 | 600 | ~5-8s |
| Scanned PDF | 10 | 300 | ~20-30s |
| Scanned PDF | 10 | 200 | ~10-15s |

## Best Practices

1. **Scan Quality**: Use at least 300 DPI for scanning original documents
2. **File Format**: TIFF is best for archival; PNG for web; JPG for size
3. **Color Mode**: Use grayscale for text-only documents to reduce file size
4. **Batch Processing**: Process multiple documents in parallel
5. **Validation**: Check OCR confidence scores and review low-confidence extractions
6. **Backup**: Keep original scanned images alongside extracted text

## Examples

See `examples/` directory for:
- `image_extraction_example.py`: Image processing examples
- `scanned_pdf_example.py`: Scanned PDF handling
- `batch_ocr_example.py`: Batch processing multiple documents

## Additional Resources

- [Tesseract Documentation](https://tesseract-ocr.github.io/)
- [Pytesseract GitHub](https://github.com/madmaze/pytesseract)
- [pdf2image Documentation](https://pdf2image.readthedocs.io/)
- [Pillow Documentation](https://pillow.readthedocs.io/)

