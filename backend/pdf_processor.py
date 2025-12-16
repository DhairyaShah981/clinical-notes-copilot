"""
Production PDF Processor with Nanonets Async OCR Integration

This processor automatically:
1. Detects if PDF is searchable (has text layer) or scanned (image-based)
2. Uses direct text extraction for searchable PDFs (fast)
3. Uses Nanonets async OCR for scanned PDFs (high quality)
4. Returns clean markdown text ready for RAG

API: Nanonets Extraction API v1 (async)
- Submit: POST https://extraction-api.nanonets.com/api/v1/extract/async
- Status: GET https://extraction-api.nanonets.com/api/v1/extract/results/{record_id}
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from config import settings
import re
import os
import requests
import time
import logging

# Configure logger to print to console
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    force=True  # Override any existing config
)
logger = logging.getLogger(__name__)

# Nanonets API configuration
NANONETS_API_KEY = os.getenv("NANONETS_API_KEY", "7d07e046-8665-11f0-a771-3ea31997c197")
NANONETS_ASYNC_URL = "https://extraction-api.nanonets.com/api/v1/extract/async"
NANONETS_RESULTS_URL = "https://extraction-api.nanonets.com/api/v1/extract/results"

# Polling configuration
DEFAULT_POLLING_INTERVAL = 5.0   # seconds between status checks
DEFAULT_MAX_WAIT_TIME = 180.0    # seconds (3 minutes) - larger docs take longer


def clean_markdown_text(text: str) -> str:
    """Clean markdown text for better RAG performance."""
    if not text:
        return ""
    
    # Decode HTML entities
    text = text.replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&amp;', '&').replace('&quot;', '"')
    
    # Remove image tags (not useful for text search)
    text = re.sub(r'<img[^>]*>[^<]*</img>', '', text)
    text = re.sub(r'<img[^>]*/?>', '', text)
    
    # Remove excessive markdown formatting that might confuse embeddings
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # Remove headers
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold to plain
    text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic to plain
    text = re.sub(r'__([^_]+)__', r'\1', text)  # Bold to plain
    text = re.sub(r'_([^_]+)_', r'\1', text)  # Italic to plain
    
    # Clean up table markup (keep content, remove structure)
    text = re.sub(r'</?table[^>]*>', '\n', text)
    text = re.sub(r'</?thead[^>]*>', '', text)
    text = re.sub(r'</?tbody[^>]*>', '', text)
    text = re.sub(r'</?tr[^>]*>', '\n', text)
    text = re.sub(r'</?th[^>]*>', ' | ', text)
    text = re.sub(r'</?td[^>]*>', ' | ', text)
    
    # Clean up whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single
    text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)  # Leading whitespace
    text = re.sub(r'\| +\|', '|', text)  # Clean table remnants
    
    return text.strip()


def calculate_text_quality(text: str) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate quality score for extracted text (0.0 to 1.0).
    
    Checks for:
    - Real English words vs garbage characters
    - Medical terminology presence
    - Sentence structure
    - Special character ratio
    
    Returns:
        (quality_score, details_dict)
    """
    if not text or len(text) < 100:
        return 0.0, {"reason": "Text too short"}
    
    # Count real English words (3+ letters, common patterns)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    word_count = len(words)
    
    # Common English words that should appear in good text
    common_words = {'the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'with', 'on', 'by', 'at', 'from', 'as'}
    common_count = sum(1 for w in words if w in common_words)
    
    # Medical terms that indicate good clinical OCR
    medical_terms = [
        'patient', 'diagnosis', 'treatment', 'medication', 'history', 'report',
        'blood', 'test', 'result', 'clinical', 'medical', 'pathology', 'laboratory',
        'doctor', 'hospital', 'specimen', 'biopsy', 'examination', 'findings',
        'adenocarcinoma', 'carcinoma', 'tumor', 'malignant', 'benign', 'immunohistochemistry'
    ]
    medical_count = sum(1 for term in medical_terms if term in text.lower())
    
    # Check for garbled patterns (consecutive special chars or unusual sequences)
    garbled_patterns = re.findall(r'[^\w\s]{3,}|[A-Z]{10,}|\$\w+|\[\w*\]', text)
    garbled_count = len(garbled_patterns)
    
    # Special character ratio (excluding normal punctuation)
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace() and c not in '.,;:!?-()\'\"')
    special_ratio = special_chars / len(text)
    
    # Calculate score
    score = 1.0
    details = {
        "word_count": word_count,
        "common_words": common_count,
        "medical_terms": medical_count,
        "garbled_patterns": garbled_count,
        "special_char_ratio": f"{special_ratio:.1%}"
    }
    
    # Penalize low word count
    if word_count < 100:
        score -= 0.3
        details["penalty_words"] = "Low word count"
    
    # Penalize lack of common words (indicates garbled text)
    common_ratio = common_count / max(word_count, 1)
    if common_ratio < 0.05:  # Less than 5% common words
        score -= 0.3
        details["penalty_common"] = f"Low common word ratio: {common_ratio:.1%}"
    
    # Bonus for medical terms
    if medical_count >= 3:
        score += 0.2
        details["bonus_medical"] = "Medical terms found"
    elif medical_count == 0:
        score -= 0.1
        details["penalty_medical"] = "No medical terms"
    
    # Penalize garbled patterns
    if garbled_count > 10:
        penalty = min(0.4, garbled_count * 0.02)
        score -= penalty
        details["penalty_garbled"] = f"{garbled_count} garbled patterns"
    
    # Penalize high special character ratio
    if special_ratio > 0.1:
        penalty = min(0.3, special_ratio * 2)
        score -= penalty
        details["penalty_special"] = f"High special char ratio"
    
    # Clamp score
    score = max(0.0, min(1.0, score))
    details["final_score"] = score
    
    return score, details


def is_pdf_searchable(pdf_path: str, min_quality: float = 0.6) -> Tuple[bool, str, int, float]:
    """
    Detect if PDF has a QUALITY searchable text layer.
    
    Unlike simple text detection, this also checks if the text is meaningful
    (not garbage OCR from a bad scan).
    
    Returns:
        (is_quality_searchable, extracted_text, page_count, quality_score)
    """
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        total_text = ""
        pages_with_text = 0
        
        for page in doc:
            page_text = page.get_text("text")
            if page_text.strip():
                pages_with_text += 1
                total_text += page_text + "\n\n"
        
        doc.close()
        
        # Calculate text quality (not just presence)
        quality_score, quality_details = calculate_text_quality(total_text)
        
        # A "searchable" PDF needs both text AND quality
        has_text = pages_with_text > 0 and len(total_text) > 200
        has_quality = quality_score >= min_quality
        is_searchable = has_text and has_quality
        
        logger.info(f"PDF analysis: {Path(pdf_path).name}")
        logger.info(f"  Pages: {total_pages}, Pages with text: {pages_with_text}")
        logger.info(f"  Text length: {len(total_text)} chars")
        logger.info(f"  Quality score: {quality_score:.2f} (threshold: {min_quality})")
        logger.info(f"  Quality details: {quality_details}")
        logger.info(f"  Decision: {'✅ Searchable (direct extraction)' if is_searchable else '📷 Needs OCR (Nanonets)'}")
        
        return is_searchable, total_text, total_pages, quality_score
        
    except Exception as e:
        logger.error(f"Error analyzing PDF: {e}")
        return False, "", 0, 0.0


class NanonetsAsyncOCR:
    """Nanonets Async OCR client for scanned document extraction."""
    
    def __init__(
        self,
        api_key: str = NANONETS_API_KEY,
        polling_interval: float = DEFAULT_POLLING_INTERVAL,
        max_wait_time: float = DEFAULT_MAX_WAIT_TIME
    ):
        self.api_key = api_key
        self.polling_interval = polling_interval
        self.max_wait_time = max_wait_time
        self.headers = {
            "Authorization": f"Bearer {api_key}"
        }
    
    def submit_async_extraction(self, pdf_path: str) -> Tuple[bool, str, str]:
        """
        Submit PDF for async extraction.
        
        Returns:
            (success, record_id, message)
        """
        logger.info(f"🚀 Submitting to Nanonets async OCR: {Path(pdf_path).name}")
        
        try:
            with open(pdf_path, 'rb') as f:
                files = {'file': (Path(pdf_path).name, f, 'application/pdf')}
                data = {'output_format': 'markdown'}
                
                response = requests.post(
                    NANONETS_ASYNC_URL,
                    headers=self.headers,
                    files=files,
                    data=data,
                    timeout=60
                )
            
            if response.status_code not in [200, 202]:
                error_msg = f"Nanonets API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return False, "", error_msg
            
            result = response.json()
            
            record_id = result.get('record_id', '')
            success = result.get('success', False)
            message = result.get('message', '')
            status = result.get('status', '')
            
            logger.info(f"  ✅ Job submitted: record_id={record_id}, status={status}")
            
            return True, record_id, message
            
        except requests.exceptions.Timeout:
            return False, "", "Request timed out"
        except requests.exceptions.RequestException as e:
            return False, "", f"Request failed: {str(e)}"
        except Exception as e:
            return False, "", f"Unexpected error: {str(e)}"
    
    def get_extraction_status(self, record_id: str, retries: int = 3) -> Dict[str, Any]:
        """
        Check status of async extraction job with retry logic.
        
        Returns dict with: success, status, result, message, processing_time, pages_processed
        """
        last_error = None
        
        for attempt in range(retries):
            try:
                response = requests.get(
                    f"{NANONETS_RESULTS_URL}/{record_id}",
                    headers=self.headers,
                    timeout=60  # Increased timeout
                )
                
                if response.status_code == 404:
                    return {"success": False, "status": "not_found", "message": "Record not found"}
                
                if response.status_code != 200:
                    return {"success": False, "status": "error", "message": f"API error: {response.status_code}"}
                
                return response.json()
                
            except requests.exceptions.Timeout:
                last_error = "Request timed out"
                logger.warning(f"Status check timed out (attempt {attempt + 1}/{retries})")
                if attempt < retries - 1:
                    time.sleep(2)  # Brief pause before retry
                continue
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Status check failed (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2)
                continue
        
        return {"success": False, "status": "error", "message": last_error or "Max retries exceeded"}
    
    def extract_with_polling(self, pdf_path: str) -> Tuple[str, str, float]:
        """
        Submit extraction job and poll until completion.
        
        Returns:
            (markdown_text, record_id, processing_time)
        """
        # Step 1: Submit job
        success, record_id, message = self.submit_async_extraction(pdf_path)
        
        if not success:
            raise Exception(f"Failed to submit extraction job: {message}")
        
        # Step 2: Poll for completion
        start_time = time.time()
        poll_count = 0
        
        while True:
            elapsed = time.time() - start_time
            
            if elapsed > self.max_wait_time:
                raise TimeoutError(f"OCR extraction timed out after {elapsed:.1f}s (record_id: {record_id})")
            
            time.sleep(self.polling_interval)
            poll_count += 1
            
            logger.info(f"  ⏳ Polling status (attempt {poll_count}, {elapsed:.1f}s elapsed)...")
            
            status_resp = self.get_extraction_status(record_id)
            status = status_resp.get('status', 'unknown')
            
            if status == 'completed' and status_resp.get('success'):
                # Extract markdown from result
                result = status_resp.get('result', {})
                markdown_data = result.get('markdown', {})
                markdown_content = markdown_data.get('content', '')
                
                processing_time = status_resp.get('processing_time', elapsed)
                pages_processed = status_resp.get('pages_processed', 0)
                
                logger.info(f"  ✅ Extraction completed!")
                logger.info(f"     Processing time: {processing_time:.1f}s")
                logger.info(f"     Pages processed: {pages_processed}")
                logger.info(f"     Markdown length: {len(markdown_content)} chars")
                
                if not markdown_content:
                    raise Exception(f"No markdown content in result (record_id: {record_id})")
                
                return markdown_content, record_id, processing_time
            
            elif status == 'processing':
                logger.info(f"     Status: processing...")
                continue
            
            elif status == 'error' or not status_resp.get('success', True):
                error_msg = status_resp.get('message', 'Unknown error')
                raise Exception(f"Extraction failed: {error_msg} (record_id: {record_id})")
            
            else:
                logger.warning(f"     Unknown status: {status}")
                continue


class PDFProcessor:
    """
    Production PDF Processor with intelligent OCR selection.
    
    Automatically detects document type and chooses optimal extraction method:
    - Searchable PDFs → Direct text extraction (fast, free)
    - Scanned PDFs → Nanonets async OCR (slow, paid, high quality)
    """
    
    def __init__(self):
        self.text_splitter = SentenceSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        self.nanonets = NanonetsAsyncOCR()
    
    def extract_text_from_searchable_pdf(self, pdf_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extract text from searchable PDF using PyMuPDF.
        Returns (full_text, page_chunks)
        """
        try:
            doc = fitz.open(pdf_path)
            chunks = []
            full_text = ""
            
            for page_num, page in enumerate(doc):
                page_text = page.get_text("text")
                if page_text.strip():
                    chunks.append({
                        "text": page_text,
                        "metadata": {"page": page_num + 1}
                    })
                    full_text += page_text + "\n\n"
            
            doc.close()
            return full_text, chunks
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return "", []
    
    def extract_text_from_scanned_pdf(self, pdf_path: str) -> Tuple[str, List[Dict[str, Any]], str]:
        """
        Extract text from scanned PDF using Nanonets OCR.
        Returns (full_text, page_chunks, record_id)
        """
        try:
            markdown, record_id, processing_time = self.nanonets.extract_with_polling(pdf_path)
            
            # Split markdown into page-like chunks (rough approximation)
            # Nanonets doesn't preserve page boundaries in markdown, so we chunk by sections
            sections = re.split(r'\n#{1,3}\s+', markdown)
            chunks = []
            
            for i, section in enumerate(sections):
                if section.strip():
                    chunks.append({
                        "text": section.strip(),
                        "metadata": {"page": i + 1, "ocr_record_id": record_id}
                    })
            
            # If no clear sections, just use the whole text as one chunk
            if not chunks and markdown.strip():
                chunks = [{
                    "text": markdown.strip(),
                    "metadata": {"page": 1, "ocr_record_id": record_id}
                }]
            
            return markdown, chunks, record_id
            
        except Exception as e:
            logger.error(f"Nanonets OCR failed: {e}")
            raise
    
    def create_documents(self, pdf_path: str, patient_id: str = None) -> List[Document]:
        """
        Create LlamaIndex documents from PDF with intelligent OCR selection.
        
        Flow:
        1. Analyze PDF to determine if searchable (with QUALITY text) or scanned
        2. If quality searchable → direct extraction (fast, free)
        3. If scanned/poor quality → Nanonets async OCR (slow, paid, high quality)
        4. Clean and chunk text
        5. Return Document objects for indexing
        """
        filename = Path(pdf_path).name
        logger.info(f"=" * 60)
        logger.info(f"📄 Processing: {filename}")
        logger.info(f"=" * 60)
        
        # Step 1: Analyze PDF (now includes quality check)
        is_searchable, direct_text, page_count, quality_score = is_pdf_searchable(pdf_path)
        
        # Step 2: Choose extraction method based on quality analysis
        if is_searchable:
            logger.info(f"✅ PDF has quality text (score: {quality_score:.2f}) - using direct extraction")
            full_text, chunks = self.extract_text_from_searchable_pdf(pdf_path)
            ocr_method = "direct"
            ocr_record_id = None
            # Keep the existing quality score
        else:
            logger.info(f"📷 PDF needs OCR (quality: {quality_score:.2f}) - using Nanonets async OCR")
            try:
                full_text, chunks, ocr_record_id = self.extract_text_from_scanned_pdf(pdf_path)
                ocr_method = "nanonets"
                # Nanonets produces high quality output
                quality_score = 0.95
            except Exception as e:
                logger.error(f"❌ Nanonets OCR failed: {e}")
                logger.warning(f"⚠️  Falling back to direct extraction (may be poor quality)")
                full_text, chunks = self.extract_text_from_searchable_pdf(pdf_path)
                ocr_method = "direct_fallback"
                ocr_record_id = None
                # Quality remains low from initial assessment
        
        # Step 3: Create Document objects
        documents = []
        
        for chunk in chunks:
            text = chunk.get("text", "")
            page_num = chunk.get("metadata", {}).get("page", 0)
            
            # Clean text
            cleaned_text = clean_markdown_text(text)
            
            if not cleaned_text or len(cleaned_text) < 50:
                continue
            
            doc = Document(
                text=cleaned_text,
                metadata={
                    "source": filename,
                    "page_number": page_num,
                    "patient_id": patient_id or "unknown",
                    "ocr_method": ocr_method,
                    "ocr_quality": quality_score,
                    "total_pages": page_count
                }
            )
            
            if ocr_record_id:
                doc.metadata["ocr_record_id"] = ocr_record_id
            
            documents.append(doc)
        
        # Log summary
        logger.info(f"=" * 60)
        logger.info(f"📊 Extraction Summary for {filename}")
        logger.info(f"   Method: {ocr_method}")
        logger.info(f"   Quality: {quality_score:.2f}")
        logger.info(f"   Pages: {page_count}")
        logger.info(f"   Chunks created: {len(documents)}")
        logger.info(f"   Total text: {len(full_text)} chars")
        if documents:
            sample = documents[0].text[:200].replace('\n', ' ')
            logger.info(f"   Sample: {sample}...")
        logger.info(f"=" * 60)
        
        return documents


# Quick test function
def test_pdf_processor(pdf_path: str):
    """Test the PDF processor with a single file."""
    processor = PDFProcessor()
    docs = processor.create_documents(pdf_path)
    
    print(f"\n{'=' * 60}")
    print(f"TEST RESULTS")
    print(f"{'=' * 60}")
    print(f"Documents created: {len(docs)}")
    
    for i, doc in enumerate(docs[:3]):
        print(f"\n--- Document {i+1} ---")
        print(f"Page: {doc.metadata.get('page_number')}")
        print(f"OCR Method: {doc.metadata.get('ocr_method')}")
        print(f"Text preview: {doc.text[:300]}...")
    
    return docs


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_pdf_processor(sys.argv[1])
    else:
        print("Usage: python pdf_processor.py <path_to_pdf>")
