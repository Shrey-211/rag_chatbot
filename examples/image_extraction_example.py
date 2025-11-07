"""Example script demonstrating image extraction with OCR."""

import logging
from pathlib import Path

from src.extractors.image import ImageExtractor
from src.extractors.pdf import PDFExtractor
from src.extractors.base import ExtractorFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def example_image_extraction():
    """Example: Extract text from a standalone image file."""
    print("\n" + "="*60)
    print("Example 1: Extracting text from image file")
    print("="*60)
    
    # Create image extractor
    extractor = ImageExtractor(language="eng", dpi=300)
    
    # Note: Replace with actual image path
    image_path = "data/sample/sample_document.jpg"
    
    if not Path(image_path).exists():
        print(f"\nℹ️  Create a sample image at: {image_path}")
        print("   Then run this example again.")
        return
    
    try:
        # Extract text
        result = extractor.extract(image_path)
        
        print(f"\n📄 File: {result.metadata['filename']}")
        print(f"📐 Dimensions: {result.metadata['image_width']}x{result.metadata['image_height']}")
        print(f"🎨 Format: {result.metadata['image_format']}")
        print(f"🔍 OCR Confidence: {result.metadata['ocr_confidence']:.1f}%")
        print(f"\n📝 Extracted Text:\n{'-'*60}")
        print(result.content[:500])  # First 500 chars
        if len(result.content) > 500:
            print(f"\n... ({len(result.content) - 500} more characters)")
        
    except Exception as e:
        logger.error(f"Error extracting image: {e}")


def example_scanned_pdf():
    """Example: Extract text from a scanned PDF."""
    print("\n" + "="*60)
    print("Example 2: Extracting text from scanned PDF")
    print("="*60)
    
    # Create PDF extractor with OCR enabled
    extractor = PDFExtractor(
        ocr_enabled=True,
        ocr_language="eng",
        dpi=300,
        min_text_threshold=50
    )
    
    # Note: Replace with actual PDF path
    pdf_path = "data/sample/scanned_document.pdf"
    
    if not Path(pdf_path).exists():
        print(f"\nℹ️  Create a sample scanned PDF at: {pdf_path}")
        print("   Then run this example again.")
        return
    
    try:
        # Extract text (OCR will be used automatically if needed)
        result = extractor.extract(pdf_path)
        
        print(f"\n📄 File: {result.metadata['filename']}")
        print(f"📑 Pages: {result.metadata['num_pages']}")
        print(f"🔍 OCR Enabled: {result.metadata['ocr_enabled']}")
        print(f"📸 OCR Used: {result.metadata['ocr_used']}")
        
        if result.metadata['ocr_used']:
            print(f"📄 Pages with OCR: {result.metadata['pages_with_ocr']}")
        
        print(f"\n📝 Extracted Text:\n{'-'*60}")
        print(result.content[:500])  # First 500 chars
        if len(result.content) > 500:
            print(f"\n... ({len(result.content) - 500} more characters)")
        
    except Exception as e:
        logger.error(f"Error extracting PDF: {e}")


def example_factory_usage():
    """Example: Use ExtractorFactory to handle any document type."""
    print("\n" + "="*60)
    print("Example 3: Using ExtractorFactory (automatic detection)")
    print("="*60)
    
    # Create factory
    factory = ExtractorFactory()
    
    # List of different document types
    documents = [
        "data/sample/document.jpg",
        "data/sample/document.png",
        "data/sample/scanned.pdf",
        "data/sample/regular.pdf",
        "data/sample/document.txt",
    ]
    
    for doc_path in documents:
        if not Path(doc_path).exists():
            print(f"⏭️  Skipping {doc_path} (not found)")
            continue
        
        try:
            # Factory automatically selects the right extractor
            result = factory.extract(doc_path)
            
            print(f"\n✅ {result.metadata['filename']}")
            print(f"   Extractor: {result.metadata['extractor']}")
            print(f"   Text length: {len(result.content)} characters")
            
        except Exception as e:
            logger.error(f"Error processing {doc_path}: {e}")


def example_multilingual_ocr():
    """Example: Extract text in multiple languages."""
    print("\n" + "="*60)
    print("Example 4: Multilingual OCR (English + Arabic)")
    print("="*60)
    
    # Create extractor for multiple languages
    # Note: Make sure Arabic language data is installed
    extractor = ImageExtractor(language="eng+ara")
    
    image_path = "data/sample/bilingual_document.jpg"
    
    if not Path(image_path).exists():
        print(f"\nℹ️  Create a sample bilingual image at: {image_path}")
        print("   Install Arabic language: sudo apt-get install tesseract-ocr-ara")
        return
    
    try:
        result = extractor.extract(image_path)
        
        print(f"\n📄 File: {result.metadata['filename']}")
        print(f"🌐 Languages: {result.metadata['ocr_language']}")
        print(f"\n📝 Extracted Text:\n{'-'*60}")
        print(result.content[:500])
        
    except Exception as e:
        logger.error(f"Error: {e}")
        if "ara" in str(e):
            print("\nℹ️  Arabic language data not installed.")
            print("   Install with: sudo apt-get install tesseract-ocr-ara")


def example_batch_processing():
    """Example: Process multiple images in a directory."""
    print("\n" + "="*60)
    print("Example 5: Batch processing images")
    print("="*60)
    
    directory = Path("data/sample/images")
    
    if not directory.exists():
        print(f"\nℹ️  Create sample images in: {directory}")
        return
    
    # Create extractor
    extractor = ImageExtractor(language="eng")
    
    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}
    image_files = [
        f for f in directory.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No image files found in {directory}")
        return
    
    print(f"\nFound {len(image_files)} images to process...")
    
    results = []
    for image_file in image_files:
        try:
            print(f"Processing: {image_file.name}...", end=" ")
            result = extractor.extract(str(image_file))
            results.append(result)
            print(f"✅ ({len(result.content)} chars)")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n✅ Successfully processed {len(results)}/{len(image_files)} images")
    
    # Calculate statistics
    total_chars = sum(len(r.content) for r in results)
    avg_confidence = sum(
        r.metadata.get('ocr_confidence', 0) for r in results
    ) / len(results) if results else 0
    
    print(f"📊 Total characters extracted: {total_chars:,}")
    print(f"📊 Average OCR confidence: {avg_confidence:.1f}%")


def main():
    """Run all examples."""
    print("\n🚀 RAG Chatbot - Image Extraction Examples")
    print("=" * 60)
    
    examples = [
        ("Image Extraction", example_image_extraction),
        ("Scanned PDF", example_scanned_pdf),
        ("Factory Usage", example_factory_usage),
        ("Multilingual OCR", example_multilingual_ocr),
        ("Batch Processing", example_batch_processing),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\n" + "="*60)
    choice = input("\nEnter example number (or 'all' to run all): ").strip().lower()
    
    if choice == 'all':
        for name, func in examples:
            func()
    elif choice.isdigit() and 1 <= int(choice) <= len(examples):
        _, func = examples[int(choice) - 1]
        func()
    else:
        print("Invalid choice. Running all examples...")
        for name, func in examples:
            func()
    
    print("\n" + "="*60)
    print("✅ Examples complete!")
    print("\nNext steps:")
    print("  1. Place sample images/PDFs in data/sample/")
    print("  2. Run: python scripts/index_documents.py --directory data/sample/")
    print("  3. Query your indexed documents via the API")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

