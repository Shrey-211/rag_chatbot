# Image Processing Quick Start

Quick reference for using image and scanned document processing in the RAG Chatbot.

## TL;DR - Get Started in 5 Minutes

### 1. Install Dependencies

**Windows:**
```bash
# Download and install:
# - Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
# - Poppler: https://github.com/oschwartz10612/poppler-windows/releases/
# Add both to PATH
pip install -r requirements.txt
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr poppler-utils
pip install -r requirements.txt
```

**macOS:**
```bash
brew install tesseract poppler
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
tesseract --version
pdftoppm -h
```

### 3. Use It!

```bash
# Index images and scanned PDFs
python scripts/index_documents.py ./data/scanned_documents/

# Or via API
curl -X POST http://localhost:8000/index/file \
  -F "file=@scanned_invoice.jpg"
```

## Supported Formats

✅ **Image Files**: JPG, JPEG, PNG, TIFF, TIF, BMP, GIF, WebP
✅ **Scanned PDFs**: Automatically detected and processed with OCR
✅ **Mixed PDFs**: Extracts text normally, uses OCR only for image pages

## How It Works

### Automatic Detection

The system automatically:
1. Detects if a PDF has text or is scanned
2. Uses OCR only when needed
3. Handles images transparently
4. Provides confidence scores

### For Images

```python
from src.extractors.image import ImageExtractor

extractor = ImageExtractor(language="eng", dpi=300)
result = extractor.extract("document.jpg")

print(result.content)  # Extracted text
print(result.metadata['ocr_confidence'])  # Quality score
```

### For PDFs

```python
from src.extractors.pdf import PDFExtractor

extractor = PDFExtractor(
    ocr_enabled=True,      # Auto-detect scanned pages
    ocr_language="eng",    # Language for OCR
    dpi=300,               # Quality (higher = better/slower)
)

result = extractor.extract("document.pdf")
print(f"OCR used: {result.metadata['ocr_used']}")
```

### Via API

The API handles everything automatically:

```python
import requests

# Upload any supported file
files = {'file': open('scanned.jpg', 'rb')}
response = requests.post('http://localhost:8000/index/file', files=files)

print(response.json())
# {
#   "success": true,
#   "document_id": "abc123",
#   "num_chunks": 5,
#   "message": "Successfully indexed..."
# }
```

## Configuration Options

### Image Extractor

```python
ImageExtractor(
    language="eng",        # Or "eng+ara" for multiple
    dpi=None              # Optional: force specific DPI
)
```

### PDF Extractor

```python
PDFExtractor(
    ocr_enabled=True,           # Enable OCR for scanned pages
    ocr_language="eng",         # Tesseract language
    dpi=300,                    # Image conversion quality
    min_text_threshold=50       # Chars/page to skip OCR
)
```

## Common Use Cases

### 1. Index Scanned Invoices

```bash
# Put all invoices in a folder
mkdir data/invoices
cp ~/Downloads/*.jpg data/invoices/

# Index them
python scripts/index_documents.py data/invoices/

# Query them
python scripts/query.py "What was the invoice total for Company XYZ?"
```

### 2. Process Mixed Document Collections

```bash
# Works with any mix of PDFs, images, text files
python scripts/index_documents.py ./data/all_documents/
```

### 3. Multi-Language Documents

```python
from src.extractors.image import ImageExtractor

# For Arabic + English documents
extractor = ImageExtractor(language="eng+ara")
result = extractor.extract("bilingual_contract.jpg")
```

### 4. High-Quality Scans

```python
from src.extractors.pdf import PDFExtractor

# Use higher DPI for better quality (slower)
extractor = PDFExtractor(dpi=600, ocr_language="eng")
result = extractor.extract("detailed_blueprint.pdf")
```

### 5. Batch Processing

```python
from pathlib import Path
from src.extractors.base import ExtractorFactory

factory = ExtractorFactory()

for img_file in Path("data/scans").glob("*.jpg"):
    result = factory.extract(str(img_file))
    print(f"Processed {img_file.name}: {len(result.content)} chars")
```

## Performance Tips

| DPI | Quality | Speed | Use Case |
|-----|---------|-------|----------|
| 200 | Good | Fast | Quick processing, simple text |
| 300 | Great | Medium | General purpose (recommended) |
| 600 | Excellent | Slow | Small text, high accuracy needed |

**Recommendations:**
- **General documents**: 300 DPI
- **Simple invoices/receipts**: 200 DPI
- **Technical drawings**: 600 DPI
- **Large batches**: 200 DPI for speed

## Troubleshooting

### "tesseract is not installed"

```bash
# Verify installation
tesseract --version

# If not found, install it:
# Windows: See docs/windows_setup_guide.md
# Linux: sudo apt-get install tesseract-ocr
# macOS: brew install tesseract
```

### "Unable to get page count"

```bash
# Verify poppler installation
pdftoppm -h

# If not found:
# Windows: See docs/windows_setup_guide.md
# Linux: sudo apt-get install poppler-utils
# macOS: brew install poppler
```

### Poor OCR Quality

```python
# Solution 1: Increase DPI
extractor = PDFExtractor(dpi=600)

# Solution 2: Pre-process images
from PIL import Image, ImageEnhance

img = Image.open("document.jpg")
img = img.convert('L')  # Grayscale
enhancer = ImageEnhance.Contrast(img)
img = enhancer.enhance(2)
img.save("enhanced.jpg")
```

### Wrong Language Detected

```python
# Specify the correct language
extractor = ImageExtractor(language="fra")  # French

# Or multiple languages
extractor = ImageExtractor(language="eng+fra+deu")  # EN+FR+DE
```

## API Endpoints

### Upload and Index

```bash
POST /index/file
Content-Type: multipart/form-data

file: <binary file data>
```

**Example:**
```bash
curl -X POST http://localhost:8000/index/file \
  -F "file=@scanned_document.jpg"
```

**Response:**
```json
{
  "success": true,
  "document_id": "abc123def456",
  "num_chunks": 5,
  "message": "Successfully indexed file: scanned_document.jpg (5 chunks)"
}
```

### Query Indexed Documents

```bash
POST /query
Content-Type: application/json

{
  "query": "your question here",
  "top_k": 5,
  "include_sources": true
}
```

## CLI Scripts

### Index Documents

```bash
# Index a directory (recursively)
python scripts/index_documents.py ./data/documents/

# With custom chunk size
python scripts/index_documents.py ./data/documents/ --chunk-size 500
```

### Query Documents

```bash
# Interactive mode
python scripts/query.py --interactive

# Single query with sources
python scripts/query.py "What is the invoice total?" --show-sources

# Specify number of results
python scripts/query.py "Contract terms" --top-k 10
```

## Examples

Full working examples in `examples/image_extraction_example.py`:

```bash
python examples/image_extraction_example.py
```

This includes:
1. Basic image extraction
2. Scanned PDF handling
3. Factory usage
4. Multilingual OCR
5. Batch processing

## Advanced Usage

### Custom Pre-processing Pipeline

```python
from PIL import Image, ImageEnhance, ImageFilter
from src.extractors.image import ImageExtractor
import pytesseract

def preprocess_image(image_path):
    """Custom preprocessing for better OCR."""
    img = Image.open(image_path)
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)
    
    # Sharpen
    img = img.filter(ImageFilter.SHARPEN)
    
    # Apply threshold
    img = img.point(lambda x: 0 if x < 128 else 255, '1')
    
    return img

# Use preprocessed image
img = preprocess_image("document.jpg")
text = pytesseract.image_to_string(img)
```

### Confidence Filtering

```python
from src.extractors.image import ImageExtractor

extractor = ImageExtractor()
result = extractor.extract("document.jpg")

# Check confidence before indexing
if result.metadata['ocr_confidence'] < 70:
    print("⚠️  Low confidence OCR - review manually")
else:
    print("✅ High confidence - auto-indexed")
```

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from src.extractors.base import ExtractorFactory

factory = ExtractorFactory()

def process_file(file_path):
    return factory.extract(str(file_path))

# Process multiple files in parallel
files = list(Path("data/scans").glob("*.jpg"))
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_file, files))

print(f"Processed {len(results)} files")
```

## Docker Usage

The Docker image includes all OCR dependencies:

```bash
# Build
docker-compose build

# Run
docker-compose up -d

# Upload via API
curl -X POST http://localhost:8000/index/file \
  -F "file=@scanned.jpg"
```

For additional languages in Docker:

```dockerfile
# In Dockerfile, add:
RUN apt-get update && apt-get install -y \
    tesseract-ocr-ara \
    tesseract-ocr-fra \
    tesseract-ocr-spa
```

## Additional Resources

- **Full Setup Guide**: [docs/image_processing_setup.md](image_processing_setup.md)
- **Windows Setup**: [docs/windows_setup_guide.md](windows_setup_guide.md)
- **Examples**: `examples/image_extraction_example.py`
- **Tests**: `tests/test_image_extractor.py`

## Support Matrix

| OS | Tesseract | Poppler | Status |
|----|-----------|---------|--------|
| Windows 10/11 | ✅ | ✅ | Fully Supported |
| Ubuntu 20.04+ | ✅ | ✅ | Fully Supported |
| macOS 12+ | ✅ | ✅ | Fully Supported |
| Docker | ✅ | ✅ | Fully Supported |

## Getting Help

1. Check [Troubleshooting](#troubleshooting) section
2. Review [image_processing_setup.md](image_processing_setup.md)
3. Run example script: `python examples/image_extraction_example.py`
4. Check Tesseract docs: https://tesseract-ocr.github.io/

