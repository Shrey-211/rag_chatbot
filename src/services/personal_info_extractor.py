"""Personal information extraction service using vision models."""

import base64
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from src.adapters.vision.base import VisionAdapter

logger = logging.getLogger(__name__)

# Check if pdf2image is available
try:
    import pdf2image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("pdf2image not available. PDF vision extraction will be limited.")


class PersonalInfoExtractor:
    """Extract personal information from documents using vision models.
    
    This service sends whole documents (PDFs, images) to vision models
    to extract structured personal information like Aadhar numbers, PAN cards,
    addresses, names, etc.
    """
    
    # Prompt for document summary (fast, focused)
    SUMMARY_PROMPT = """Analyze this document image and provide a brief summary.

Identify:
1. **Document Type** (e.g., Aadhar Card, PAN Card, Passport, Driver's License, Voter ID, Bank Statement, etc.)
2. **Person's Name** (if visible)
3. **Key Information Types** present (ID numbers, addresses, dates, etc.)

Respond in this exact format:
DOCUMENT TYPE: [type]
PERSON NAME: [name if visible, otherwise "Not visible"]
CONTAINS: [comma-separated list of information types, e.g., "ID number, address, date of birth, photo"]

Be concise. This summary helps users find the right document when searching."""

    # Prompt template for extracting personal information
    EXTRACTION_PROMPT = """Analyze this document image and extract ALL personal information present.

Look for the following types of information and list them clearly:

1. **Identity Documents:**
   - Aadhar Number (12-digit number)
   - PAN Number (10-character alphanumeric)
   - Passport Number
   - Driver's License Number
   - Voter ID
   - Any other ID numbers

2. **Personal Details:**
   - Full Name
   - Date of Birth
   - Gender
   - Father's Name / Mother's Name
   - Spouse Name

3. **Contact Information:**
   - Phone Numbers
   - Email Addresses
   - Address (permanent and current)
   - PIN Code / Zip Code

4. **Financial Information:**
   - Bank Account Numbers
   - IFSC Codes
   - Credit/Debit Card Numbers (last 4 digits)

5. **Other Information:**
   - Company/Organization Name
   - Employee ID
   - Any other relevant personal data

**IMPORTANT:** 
- Extract ONLY information that is clearly visible in the document
- Provide the exact values as shown (don't modify or format)
- If a field is not present, don't mention it
- Format your response as a structured list with clear labels

Response format:
```
DOCUMENT TYPE: [type of document if identifiable]

[Entity Type]: [Exact Value]
[Entity Type]: [Exact Value]
...
```

Now analyze the document:"""
    
    def __init__(
        self,
        vision_adapter: VisionAdapter,
        poppler_path: Optional[str] = None,
        dpi: int = 200,
    ):
        """Initialize personal info extractor.
        
        Args:
            vision_adapter: Vision model adapter
            poppler_path: Path to poppler binaries (Windows)
            dpi: DPI for PDF to image conversion
        """
        self.vision_adapter = vision_adapter
        self.poppler_path = poppler_path
        self.dpi = dpi
    
    def generate_document_summary(self, file_path: str) -> str:
        """Generate a searchable summary of the document.
        
        Args:
            file_path: Path to document file (PDF or image)
            
        Returns:
            Document summary string
        """
        path = Path(file_path)
        
        if not path.exists():
            return ""
        
        # Determine file type
        suffix = path.suffix.lower()
        
        try:
            if suffix == ".pdf":
                # Convert first page only for summary
                if not PDF2IMAGE_AVAILABLE:
                    return ""
                
                convert_kwargs = {
                    'pdf_path': file_path,
                    'dpi': 150,  # Lower DPI for faster processing
                    'fmt': 'png',
                    'first_page': 1,
                    'last_page': 1,
                }
                
                if self.poppler_path:
                    convert_kwargs['poppler_path'] = self.poppler_path
                
                images = pdf2image.convert_from_path(**convert_kwargs)
                if not images:
                    return ""
                
                # Save temp image
                temp_image_path = Path(file_path).parent / f"_temp_summary_{Path(file_path).stem}.png"
                images[0].save(temp_image_path, 'PNG')
                
                try:
                    response = self.vision_adapter.analyze_image(str(temp_image_path), self.SUMMARY_PROMPT)
                    return response.text.strip()
                finally:
                    if temp_image_path.exists():
                        temp_image_path.unlink()
                        
            elif suffix in {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}:
                response = self.vision_adapter.analyze_image(file_path, self.SUMMARY_PROMPT)
                return response.text.strip()
                
        except Exception as e:
            logger.warning(f"Could not generate document summary: {e}")
            return ""
        
        return ""
    
    def extract_from_document(self, file_path: str) -> Tuple[List[Dict[str, str]], str, str]:
        """Extract personal information and summary from a document.
        
        Args:
            file_path: Path to document file (PDF or image)
            
        Returns:
            Tuple of (list of extracted entities, raw response text, document summary)
            Each entity is a dict with keys: entity_type, entity_value, confidence, context
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # First, generate document summary (fast)
        logger.info(f"📋 Generating document summary...")
        summary = self.generate_document_summary(file_path)
        
        # Then extract detailed personal information
        # Determine file type
        suffix = path.suffix.lower()
        
        if suffix == ".pdf":
            entities, raw_response = self._extract_from_pdf(file_path)
            return entities, raw_response, summary
        elif suffix in {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}:
            entities, raw_response = self._extract_from_image(file_path)
            return entities, raw_response, summary
        else:
            logger.warning(f"Unsupported file type for vision extraction: {suffix}")
            return [], "", summary
    
    def _extract_from_image(self, image_path: str) -> Tuple[List[Dict[str, str]], str]:
        """Extract personal info from an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (extracted entities, raw response)
        """
        logger.info(f"📸 Extracting personal info from image: {Path(image_path).name}")
        
        try:
            # Analyze with vision model
            response = self.vision_adapter.analyze_image(image_path, self.EXTRACTION_PROMPT)
            
            # Parse response
            entities = self._parse_extraction_response(response.text)
            
            logger.info(f"✓ Extracted {len(entities)} personal information entities")
            
            return entities, response.text
            
        except Exception as e:
            logger.error(f"Error extracting from image: {e}")
            return [], ""
    
    def _extract_from_pdf(self, pdf_path: str) -> Tuple[List[Dict[str, str]], str]:
        """Extract personal info from a PDF by converting pages to images.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (extracted entities, raw response)
        """
        if not PDF2IMAGE_AVAILABLE:
            logger.warning("pdf2image not available. Cannot extract from PDF.")
            return [], ""
        
        logger.info(f"📄 Extracting personal info from PDF: {Path(pdf_path).name}")
        
        try:
            # Convert first page to image (most personal docs are 1-2 pages)
            logger.info(f"Converting PDF first page to image at {self.dpi} DPI...")
            
            convert_kwargs = {
                'pdf_path': pdf_path,
                'dpi': self.dpi,
                'fmt': 'png',
                'first_page': 1,
                'last_page': 1,  # Only process first page for speed
            }
            
            if self.poppler_path:
                convert_kwargs['poppler_path'] = self.poppler_path
            
            images = pdf2image.convert_from_path(**convert_kwargs)
            
            if not images:
                logger.warning("No images generated from PDF")
                return [], ""
            
            # Save first page as temporary image
            temp_image_path = Path(pdf_path).parent / f"_temp_{Path(pdf_path).stem}_page1.png"
            images[0].save(temp_image_path, 'PNG')
            
            try:
                # Extract from the image
                entities, raw_response = self._extract_from_image(str(temp_image_path))
                
                # If first page has info, optionally process second page
                if len(entities) < 3 and len(images) > 1:
                    logger.info("Processing second page for more information...")
                    # Convert second page
                    convert_kwargs['first_page'] = 2
                    convert_kwargs['last_page'] = 2
                    images2 = pdf2image.convert_from_path(**convert_kwargs)
                    
                    if images2:
                        temp_image_path2 = Path(pdf_path).parent / f"_temp_{Path(pdf_path).stem}_page2.png"
                        images2[0].save(temp_image_path2, 'PNG')
                        
                        try:
                            entities2, raw_response2 = self._extract_from_image(str(temp_image_path2))
                            entities.extend(entities2)
                            raw_response += "\n\n--- PAGE 2 ---\n" + raw_response2
                        finally:
                            # Clean up temp file
                            if temp_image_path2.exists():
                                temp_image_path2.unlink()
                
                return entities, raw_response
                
            finally:
                # Clean up temp file
                if temp_image_path.exists():
                    temp_image_path.unlink()
            
        except Exception as e:
            logger.error(f"Error extracting from PDF: {e}")
            return [], ""
    
    def _parse_extraction_response(self, response_text: str) -> List[Dict[str, str]]:
        """Parse the vision model response to extract structured entities.
        
        Args:
            response_text: Raw response from vision model
            
        Returns:
            List of entity dictionaries
        """
        entities = []
        
        if not response_text:
            return entities
        
        # Split into lines
        lines = response_text.strip().split('\n')
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if not line or line.startswith('**') or line.startswith('---'):
                continue
            
            # Try to parse entity lines (format: "Label: Value")
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    entity_type = parts[0].strip().strip('*-#').lower()
                    entity_value = parts[1].strip()
                    
                    # Skip empty values or generic headers
                    if not entity_value or entity_value.lower() in ['[type', '[exact', 'n/a', 'na', 'nil', 'none']:
                        continue
                    
                    # Clean up entity type
                    entity_type = entity_type.replace(' ', '_').replace("'", "").replace("/", "_")
                    
                    # Skip if it looks like a section header
                    if len(entity_value) > 200 or entity_type in ['document_type', 'response_format', 'important']:
                        continue
                    
                    entities.append({
                        'entity_type': entity_type,
                        'entity_value': entity_value,
                        'confidence': 'high',  # Vision models generally have good confidence
                        'context': current_section or 'general',
                    })
        
        return entities

