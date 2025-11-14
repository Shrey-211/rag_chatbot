# Contributing to RAG Chatbot

Thank you for your interest in contributing to the RAG Chatbot project! 🎉

We welcome contributions from everyone, whether you're fixing a typo, adding a feature, or improving documentation.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Submitting Changes](#submitting-changes)
- [Style Guidelines](#style-guidelines)
- [Testing](#testing)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

Key principles:

- **Be respectful:** Treat everyone with respect and kindness
- **Be collaborative:** Work together to solve problems
- **Be patient:** Remember that we're all learning
- **Be constructive:** Provide helpful feedback and suggestions
- **Be inclusive:** Welcome and support people of all backgrounds

Please read the full [Code of Conduct](CODE_OF_CONDUCT.md) for details.

## How Can I Contribute?

### 🐛 Reporting Bugs

If you find a bug:

1. **Check if it's already reported** in [GitHub Issues](https://github.com/yourusername/rag_chatbot/issues)
2. **Create a new issue** with:
   - Clear title describing the bug
   - Steps to reproduce the problem
   - Expected behavior vs actual behavior
   - Your environment (OS, Python version, Node version)
   - Error messages or logs
   - Screenshots if relevant

### 💡 Suggesting Features

Have an idea? Great!

1. **Check existing issues** to see if it's already suggested
2. **Create a new issue** with:
   - Clear description of the feature
   - Why it would be useful
   - How it might work
   - Any examples or mockups

### 📖 Improving Documentation

Documentation improvements are always welcome!

- Fix typos or unclear explanations
- Add more examples
- Improve the setup guide
- Write tutorials
- Add code comments

### 🔧 Contributing Code

Want to write code? Awesome!

1. **Start with good first issues:** Look for issues labeled `good first issue`
2. **Comment on the issue** to let others know you're working on it
3. **Follow the development setup** below
4. **Make your changes** following our style guidelines
5. **Test thoroughly**
6. **Submit a pull request**

## Development Setup

### Prerequisites

- Python 3.11+
- Node.js 18+ (for web interface)
- Git
- Ollama (for testing)

### Setup Steps

1. **Fork the repository**
   ```bash
   # Click "Fork" button on GitHub
   ```

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR-USERNAME/rag_chatbot.git
   cd rag_chatbot
   ```

3. **Add upstream remote**
   ```bash
   git remote add upstream https://github.com/ORIGINAL-OWNER/rag_chatbot.git
   ```

4. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Activate it
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

5. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

6. **Install development tools**
   ```bash
   pip install black ruff mypy pytest pytest-cov pre-commit
   pre-commit install
   ```

7. **Set up frontend (optional)**
   ```bash
   cd webapp
   npm install
   cd ..
   ```

## Making Changes

### 1. Create a Branch

Always create a new branch for your changes:

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create and switch to a new branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

Branch naming:
- `feature/` - For new features
- `fix/` - For bug fixes
- `docs/` - For documentation changes
- `refactor/` - For code refactoring
- `test/` - For adding tests

### 2. Make Your Changes

- Write clean, readable code
- Follow the existing code style
- Add comments for complex logic
- Update documentation if needed
- Add tests for new features

### 3. Test Your Changes

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest --cov=src --cov-report=html

# Format code
black .

# Lint code
ruff check .

# Type check
mypy src/
```

### 4. Commit Your Changes

Write clear, descriptive commit messages:

```bash
# Good commit messages
git commit -m "Add support for CSV file extraction"
git commit -m "Fix OCR path handling on Windows"
git commit -m "Update setup guide with troubleshooting section"

# Bad commit messages
git commit -m "fix bug"
git commit -m "update"
git commit -m "changes"
```

**Commit message format:**
```
Add/Fix/Update/Remove: Brief description

Optional: Longer explanation if needed

Fixes #123 (if it fixes an issue)
```

## Submitting Changes

### 1. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 2. Create a Pull Request

1. Go to your fork on GitHub
2. Click "Pull Request" button
3. Fill in the PR template:
   - **Title:** Clear, descriptive title
   - **Description:** What changes you made and why
   - **Related Issues:** Link to related issues (e.g., "Fixes #123")
   - **Testing:** How you tested your changes
   - **Screenshots:** If UI changes

### 3. Respond to Feedback

- Be open to feedback and suggestions
- Make requested changes promptly
- Ask questions if something is unclear
- Update your branch if needed:
  ```bash
  git add .
  git commit -m "Address review comments"
  git push origin feature/your-feature-name
  ```

## Style Guidelines

### Python Code Style

We follow [PEP 8](https://pep8.org/) with some modifications:

```python
# Good
def extract_text_from_pdf(file_path: str, use_ocr: bool = False) -> str:
    """Extract text from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        use_ocr: Whether to use OCR for scanned PDFs
        
    Returns:
        Extracted text content
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Implementation here
    return text
```

**Key points:**
- Use type hints for function parameters and returns
- Add docstrings to all functions and classes
- Use meaningful variable names
- Keep functions focused and small
- Maximum line length: 100 characters
- Use f-strings for formatting

### TypeScript/JavaScript Style

```typescript
// Good
interface UploadResponse {
  success: boolean;
  documentId: string;
  message: string;
}

async function uploadDocument(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('/api/index/file', {
    method: 'POST',
    body: formData,
  });
  
  return response.json();
}
```

**Key points:**
- Use TypeScript for type safety
- Use async/await over promises
- Use meaningful variable names
- Add comments for complex logic

### Configuration Files

- YAML: 2 spaces for indentation
- JSON: 2 spaces for indentation
- Keep configuration well-commented

### Documentation

- Use Markdown for documentation
- Keep line length reasonable (80-100 chars)
- Use headers for organization
- Add code examples
- Include screenshots for UI features

## Testing

### Running Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_extractors.py

# Specific test
pytest tests/test_extractors.py::test_pdf_extraction

# With output
pytest tests/ -v -s

# With coverage
pytest --cov=src --cov-report=html
```

### Writing Tests

```python
import pytest
from src.extractors.pdf import PDFExtractor

def test_pdf_text_extraction():
    """Test basic PDF text extraction."""
    extractor = PDFExtractor()
    result = extractor.extract("tests/fixtures/sample.pdf")
    
    assert result.content != ""
    assert "expected text" in result.content.lower()
    assert result.metadata["num_pages"] > 0

def test_pdf_not_found():
    """Test handling of missing PDF file."""
    extractor = PDFExtractor()
    
    with pytest.raises(FileNotFoundError):
        extractor.extract("nonexistent.pdf")
```

**Test guidelines:**
- Write tests for new features
- Test both success and failure cases
- Use descriptive test names
- Use fixtures for common setup
- Mock external dependencies (API calls, file systems)

## Project Structure

Understanding the structure helps you know where to make changes:

```
rag_chatbot/
├── api/               # FastAPI application
│   ├── main.py       # Main API endpoints
│   ├── models.py     # Pydantic models
│   └── database.py   # Database operations
├── src/
│   ├── adapters/     # Pluggable adapters (LLM, embeddings, vision)
│   ├── extractors/   # Document extractors (PDF, DOCX, images)
│   ├── vectorstore/  # Vector store implementations
│   ├── retriever/    # Retrieval logic
│   ├── config/       # Configuration management
│   └── utils/        # Utilities (chunking, prompts, etc.)
├── tests/            # Test suite
├── webapp/           # React frontend
├── scripts/          # CLI tools
├── docs/             # Documentation
└── data/             # Data storage (gitignored)
```

## Common Tasks

### Adding a New Document Extractor

1. Create file in `src/extractors/`
2. Inherit from `BaseExtractor`
3. Implement `extract()` method
4. Register in `ExtractorFactory`
5. Add tests in `tests/test_extractors.py`

### Adding a New LLM Provider

1. Create file in `src/adapters/llm/`
2. Inherit from `LLMAdapter`
3. Implement required methods
4. Update `create_llm_adapter()` in `api/main.py`
5. Add tests in `tests/test_llm_adapters.py`

### Adding a New API Endpoint

1. Add endpoint in `api/main.py`
2. Create Pydantic models in `api/models.py`
3. Add tests in `tests/test_e2e.py`
4. Update API documentation in README

## Getting Help

- **Questions?** Open a [GitHub Discussion](https://github.com/yourusername/rag_chatbot/discussions)
- **Issues?** Create an [Issue](https://github.com/yourusername/rag_chatbot/issues)
- **Stuck?** Comment on your PR and ask for help

## Recognition

Contributors will be:
- Listed in the project README
- Mentioned in release notes
- Credited in commit history

Thank you for contributing! Every contribution, no matter how small, helps make this project better. 🙏

---

**Happy Coding!** 🚀

