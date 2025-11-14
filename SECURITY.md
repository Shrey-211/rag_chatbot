# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Which versions are eligible for receiving such patches depends on the CVSS v3.0 Rating:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

The RAG Chatbot team takes the security of our software seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Where to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: [INSERT YOUR SECURITY EMAIL]

### What to Include

Please include the following information in your report:

- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

This information will help us triage your report more quickly.

### What to Expect

- You should receive an acknowledgment within 48 hours
- We will investigate the issue and determine its impact and severity
- We will work to fix the issue and release a patch as soon as possible
- We will notify you when the issue has been fixed
- We will publicly disclose the issue after a patch is released

## Security Best Practices

When using RAG Chatbot, we recommend:

### General Security

1. **Keep Dependencies Updated:** Regularly update all Python and Node.js dependencies
   ```bash
   pip install --upgrade -r requirements.txt
   cd webapp && npm update
   ```

2. **Use Environment Variables:** Never hardcode API keys or secrets
   ```yaml
   # Bad
   openai:
     api_key: "sk-abc123..."
   
   # Good
   openai:
     api_key: "${OPENAI_API_KEY}"
   ```

3. **Secure File Uploads:** Only allow trusted users to upload documents
   - Implement authentication if exposing to the internet
   - Validate file types and sizes
   - Scan uploads for malware if accepting user files

4. **Network Security:** Use HTTPS in production
   ```bash
   # Use a reverse proxy like nginx or traefik
   # Never expose the API directly to the internet without TLS
   ```

### API Security

1. **CORS Configuration:** Restrict CORS origins in production
   ```yaml
   api:
     cors_origins:
       - "https://yourdomain.com"  # Don't use "*" in production
   ```

2. **Rate Limiting:** Implement rate limiting for public APIs
   ```python
   # Consider using slowapi or fastapi-limiter
   from slowapi import Limiter, _rate_limit_exceeded_handler
   from slowapi.util import get_remote_address
   
   limiter = Limiter(key_func=get_remote_address)
   app.state.limiter = limiter
   ```

3. **Input Validation:** All endpoints validate input thoroughly
   - File size limits are enforced
   - Query length is limited
   - Metadata is validated

### Data Security

1. **Document Privacy:**
   - Documents are stored locally by default
   - Vector embeddings are stored in ChromaDB/FAISS
   - Consider encrypting sensitive documents at rest

2. **Sanitize Metadata:** Don't include sensitive information in metadata
   ```python
   # Bad
   metadata = {
       "uploader": "john@company.com",
       "salary": 100000
   }
   
   # Good
   metadata = {
       "document_type": "invoice",
       "upload_date": "2025-01-01"
   }
   ```

3. **OCR Data:** Be aware that OCR extracts ALL text from images/PDFs
   - Review documents before indexing
   - Consider redacting sensitive information
   - Don't upload personal documents to shared systems

### Infrastructure Security

1. **Docker Security:**
   ```dockerfile
   # Use non-root user
   USER appuser
   
   # Keep images updated
   FROM python:3.11-slim-bookworm
   ```

2. **Environment Isolation:**
   - Use virtual environments for Python
   - Use separate environments for dev/staging/prod
   - Don't share databases between environments

3. **Logging:**
   - Don't log sensitive information
   - Monitor logs for suspicious activity
   - Rotate logs regularly

### LLM-Specific Security

1. **Prompt Injection:** Be aware of prompt injection attacks
   - System prompts are designed to resist injection
   - Context is clearly separated from queries
   - Consider implementing input filtering

2. **Data Leakage:** LLMs might expose indexed data
   - Only index documents that should be searchable
   - Implement access controls if needed
   - Don't rely on "security by obscurity"

3. **Model Selection:**
   - Use trusted models (Ollama, OpenAI)
   - Keep Ollama updated
   - Be cautious with custom/fine-tuned models

## Known Security Considerations

### Local Execution

By default, RAG Chatbot runs locally and is not exposed to the internet. This is the most secure configuration.

### File Upload

The `/index/file` endpoint accepts file uploads. In production:
- Implement authentication
- Validate file types strictly
- Set file size limits
- Consider antivirus scanning
- Use a separate storage service

### OCR

Tesseract and Poppler process potentially untrusted files:
- Keep them updated
- Run in sandboxed environments if processing untrusted documents
- Validate PDFs before processing

### Vector Store

ChromaDB/FAISS stores document embeddings:
- Embeddings can't reconstruct original text exactly
- But they can reveal document content
- Protect the data directory with appropriate file permissions

### LLM APIs

If using OpenAI or other cloud APIs:
- Your documents are sent to the API for embedding/generation
- Review the provider's privacy policy
- Use local models (Ollama) for sensitive data

## Disclosure Policy

When we receive a security bug report, we will:

1. Confirm the problem and determine affected versions
2. Audit code to find any similar problems
3. Prepare fixes for all supported versions
4. Release patches as soon as possible

## Comments on this Policy

If you have suggestions on how this process could be improved, please submit a pull request or open an issue.

## Security Updates

Security updates will be announced:
- In GitHub Security Advisories
- In release notes
- Via email to known users (if applicable)

Stay updated:
- Watch this repository for security advisories
- Subscribe to releases
- Check the changelog regularly

---

**Thank you for helping keep RAG Chatbot and its users safe!** 🔒

