"""Dependency checker for OCR and document processing."""

import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class DependencyStatus:
    """Status of a dependency check."""
    
    def __init__(self, name: str, installed: bool, version: Optional[str] = None, path: Optional[str] = None):
        self.name = name
        self.installed = installed
        self.version = version
        self.path = path


class DependencyChecker:
    """Check and report on system dependencies for OCR and document processing."""
    
    @staticmethod
    def check_tesseract() -> DependencyStatus:
        """Check if Tesseract is installed and accessible.
        
        Returns:
            DependencyStatus with installation status and version
        """
        try:
            import pytesseract
            
            # Try to get version
            try:
                version = pytesseract.get_tesseract_version()
                tesseract_cmd = pytesseract.pytesseract.tesseract_cmd
                return DependencyStatus(
                    name="Tesseract OCR",
                    installed=True,
                    version=str(version),
                    path=tesseract_cmd
                )
            except Exception as e:
                # Tesseract not in PATH or not working
                logger.debug(f"Tesseract check failed: {e}")
                return DependencyStatus(name="Tesseract OCR", installed=False)
                
        except ImportError:
            return DependencyStatus(name="Tesseract OCR", installed=False)
    
    @staticmethod
    def check_poppler() -> DependencyStatus:
        """Check if Poppler is installed and accessible.
        
        Returns:
            DependencyStatus with installation status
        """
        # Check if pdftoppm (part of poppler) is available
        pdftoppm_path = shutil.which("pdftoppm")
        
        if pdftoppm_path:
            try:
                # Try to get version
                result = subprocess.run(
                    ["pdftoppm", "-v"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                version_line = result.stderr.split('\n')[0] if result.stderr else "Unknown"
                return DependencyStatus(
                    name="Poppler",
                    installed=True,
                    version=version_line,
                    path=pdftoppm_path
                )
            except Exception:
                return DependencyStatus(name="Poppler", installed=True, path=pdftoppm_path)
        
        return DependencyStatus(name="Poppler", installed=False)
    
    @staticmethod
    def check_pdf2image() -> DependencyStatus:
        """Check if pdf2image library is installed.
        
        Returns:
            DependencyStatus with installation status
        """
        try:
            import pdf2image
            return DependencyStatus(
                name="pdf2image",
                installed=True,
                version=pdf2image.__version__ if hasattr(pdf2image, '__version__') else "Unknown"
            )
        except ImportError:
            return DependencyStatus(name="pdf2image", installed=False)
    
    @staticmethod
    def check_pillow() -> DependencyStatus:
        """Check if Pillow (PIL) is installed.
        
        Returns:
            DependencyStatus with installation status
        """
        try:
            from PIL import Image
            import PIL
            return DependencyStatus(
                name="Pillow",
                installed=True,
                version=PIL.__version__
            )
        except ImportError:
            return DependencyStatus(name="Pillow", installed=False)
    
    @staticmethod
    def check_all_ocr_dependencies() -> Dict[str, DependencyStatus]:
        """Check all OCR-related dependencies.
        
        Returns:
            Dictionary of dependency name to status
        """
        return {
            "tesseract": DependencyChecker.check_tesseract(),
            "poppler": DependencyChecker.check_poppler(),
            "pdf2image": DependencyChecker.check_pdf2image(),
            "pillow": DependencyChecker.check_pillow(),
        }
    
    @staticmethod
    def get_installation_instructions() -> str:
        """Get OS-specific installation instructions for missing dependencies.
        
        Returns:
            Formatted installation instructions
        """
        system = platform.system()
        
        if system == "Windows":
            return """
╔═══════════════════════════════════════════════════════════════╗
║          Windows OCR Setup Instructions                       ║
╚═══════════════════════════════════════════════════════════════╝

🔧 AUTOMATED SETUP (Recommended):
   Run this command as Administrator:
   
   powershell -ExecutionPolicy Bypass -File scripts/setup_windows_ocr.ps1

📖 MANUAL SETUP:
   1. Install Tesseract:
      - Download from: https://github.com/UB-Mannheim/tesseract/wiki
      - Run installer and add to PATH
   
   2. Install Poppler:
      - Download from: https://github.com/oschwartz10612/poppler-windows/releases/
      - Extract and add Library/bin to PATH
   
   3. Install Python packages:
      pip install Pillow pytesseract pdf2image

📚 Full documentation: docs/windows_setup_guide.md
"""
        
        elif system == "Linux":
            return """
╔═══════════════════════════════════════════════════════════════╗
║          Linux OCR Setup Instructions                         ║
╚═══════════════════════════════════════════════════════════════╝

🔧 Ubuntu/Debian:
   sudo apt-get update
   sudo apt-get install tesseract-ocr poppler-utils
   pip install Pillow pytesseract pdf2image

🔧 Fedora/RHEL:
   sudo dnf install tesseract poppler-utils
   pip install Pillow pytesseract pdf2image

🔧 Arch Linux:
   sudo pacman -S tesseract poppler
   pip install Pillow pytesseract pdf2image
"""
        
        elif system == "Darwin":  # macOS
            return """
╔═══════════════════════════════════════════════════════════════╗
║          macOS OCR Setup Instructions                         ║
╚═══════════════════════════════════════════════════════════════╝

🔧 Using Homebrew:
   brew install tesseract poppler
   pip install Pillow pytesseract pdf2image

📚 Full documentation: docs/IMAGE_PROCESSING_QUICKSTART.md
"""
        
        else:
            return f"""
╔═══════════════════════════════════════════════════════════════╗
║          OCR Setup Instructions ({system})                    ║
╚═══════════════════════════════════════════════════════════════╝

Please install:
1. Tesseract OCR
2. Poppler (for PDF processing)
3. Python packages: pip install Pillow pytesseract pdf2image

See docs/IMAGE_PROCESSING_QUICKSTART.md for details.
"""
    
    @staticmethod
    def report_status(verbose: bool = True) -> bool:
        """Check and report status of all dependencies.
        
        Args:
            verbose: If True, print detailed status report
            
        Returns:
            True if all critical dependencies are installed
        """
        deps = DependencyChecker.check_all_ocr_dependencies()
        
        all_installed = all(dep.installed for dep in deps.values())
        
        if verbose:
            logger.info("=" * 70)
            logger.info("OCR Dependencies Status")
            logger.info("=" * 70)
            
            for name, status in deps.items():
                if status.installed:
                    version_str = f" (v{status.version})" if status.version else ""
                    path_str = f" [{status.path}]" if status.path else ""
                    logger.info(f"✅ {status.name:20} INSTALLED{version_str}{path_str}")
                else:
                    logger.warning(f"❌ {status.name:20} NOT FOUND")
            
            logger.info("=" * 70)
            
            if not all_installed:
                logger.warning("⚠️  Some OCR dependencies are missing!")
                logger.warning("   Scanned PDF and image extraction will not work.")
                logger.info("")
                logger.info(DependencyChecker.get_installation_instructions())
            else:
                logger.info("✅ All OCR dependencies are installed and ready!")
                logger.info("=" * 70)
        
        return all_installed
    
    @staticmethod
    def can_process_images() -> bool:
        """Check if system can process images with OCR.
        
        Returns:
            True if Tesseract and Pillow are available
        """
        tesseract = DependencyChecker.check_tesseract()
        pillow = DependencyChecker.check_pillow()
        return tesseract.installed and pillow.installed
    
    @staticmethod
    def can_process_scanned_pdfs() -> bool:
        """Check if system can process scanned PDFs with OCR.
        
        Returns:
            True if all required dependencies are available
        """
        deps = DependencyChecker.check_all_ocr_dependencies()
        return all(dep.installed for dep in deps.values())


def check_dependencies_on_startup():
    """Check dependencies when application starts.
    
    This function is called during application initialization to
    warn users about missing dependencies early.
    """
    checker = DependencyChecker()
    checker.report_status(verbose=True)

