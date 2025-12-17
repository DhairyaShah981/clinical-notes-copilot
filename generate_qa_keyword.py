#!/usr/bin/env python3
"""
Generate KEYWORD-FOCUSED Q&A pairs from PDF using Groq
These questions test exact factual recall, not semantic understanding
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict
import json
import re
import PyPDF2
from groq import Groq

def extract_text_from_pdf(pdf_path: str) -> tuple[str, int]:
    """Extract text from PDF"""
    print(f"📄 Extracting text from: {pdf_path}")
    
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        
        text = ""
        for page_num, page in enumerate(pdf_reader.pages, 1):
            page_text = page.extract_text()
            text += f"\n\n--- Page {page_num} ---\n\n{page_text}"
        
        print(f"   ✓ Extracted {len(text)} characters from {num_pages} pages")
        return text, num_pages

def chunk_text(text: str, chunk_size: int = 3000) -> List[str]:
    """Split text into chunks for processing"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

def generate_keyword_questions(client: Groq, chunk: str, num_questions: int = 2) -> List[Dict]:
    """Generate KEYWORD-FOCUSED factual Q&A pairs"""
    
    prompt = f"""You are creating EXACT FACTUAL RECALL questions from a medical textbook.

Generate {num_questions} questions that require SPECIFIC FACTS from the text, not conceptual understanding.

REQUIREMENTS:
1. Questions MUST ask for:
   - Specific anatomical terms/names (e.g., "What is the name of X?")
   - Exact numbers/measurements (e.g., "What is the pH of blood?")
   - Lists of specific items (e.g., "What are the three types of X?")
   - Specific locations using proper nouns (e.g., "Where are osteocytes found?")
   - Names of specific hormones, enzymes, proteins, cells
   
2. Answers MUST be:
   - EXACT words/phrases from the text (copy verbatim)
   - Short factual statements (NOT explanations)
   - Include specific terminology from the text
   
3. AVOID:
   - "What is X?" definition questions
   - "Explain..." or "Describe..." questions
   - Questions asking about concepts or processes
   - Questions that can be paraphrased

GOOD EXAMPLES:
- "What is the pH range of normal human blood?"
- "What are the three main types of muscle tissue?"
- "What hormone is released by the thyroid gland?"
- "What is the name of the protein found in red blood cells?"
- "How many chromosomes are in a human diploid cell?"

BAD EXAMPLES (TOO SEMANTIC):
- "What is homeostasis?"
- "How does the kidney regulate blood pressure?"
- "Explain the role of mitochondria"

TEXT:
{chunk[:2000]}

Return ONLY a JSON array:
[
  {{"question": "What is the pH of...", "answer": "7.35-7.45"}},
  {{"question": "What are the three types of...", "answer": "skeletal, cardiac, and smooth"}}
]

Generate exactly {num_questions} KEYWORD-FOCUSED Q&A pairs:"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Lower temperature for more factual
            max_tokens=2000
        )
        
        content = response.choices[0].message.content.strip()
        
        # Extract JSON
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            qa_pairs = json.loads(json_match.group())
            return qa_pairs
        else:
            print(f"   ⚠️  Could not parse JSON")
            return []
            
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
        return []

def generate_qa_dataset(pdf_path: str, num_questions: int, api_key: str) -> pd.DataFrame:
    """Generate keyword-focused Q&A dataset"""
    
    client = Groq(api_key=api_key)
    full_text, num_pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(full_text, chunk_size=3000)
    print(f"   ✓ Split into {len(chunks)} chunks")
    
    questions_per_chunk = max(1, num_questions // len(chunks))
    
    print(f"\n🤖 Generating {num_questions} KEYWORD-FOCUSED Q&A pairs...")
    print(f"   Generating ~{questions_per_chunk} questions per chunk")
    
    all_qa_pairs = []
    
    for i, chunk in enumerate(chunks, 1):
        if len(all_qa_pairs) >= num_questions:
            break
        
        remaining = num_questions - len(all_qa_pairs)
        questions_to_generate = min(questions_per_chunk, remaining)
        
        print(f"\n   [{i}/{len(chunks)}] Processing chunk {i}...")
        qa_pairs = generate_keyword_questions(client, chunk, questions_to_generate)
        
        if qa_pairs:
            all_qa_pairs.extend(qa_pairs)
            print(f"      ✓ Generated {len(qa_pairs)} Q&A pairs (Total: {len(all_qa_pairs)})")
        else:
            print(f"      ⚠️  Failed for this chunk")
    
    if all_qa_pairs:
        df = pd.DataFrame(all_qa_pairs)
        df = df.head(num_questions)
        print(f"\n✅ Generated {len(df)} keyword-focused Q&A pairs")
        return df
    else:
        print(f"\n❌ Failed to generate Q&A pairs")
        return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description='Generate KEYWORD-FOCUSED Q&A pairs')
    parser.add_argument('--pdf', type=str, required=True)
    parser.add_argument('--num_questions', type=int, default=40)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--api_key', type=str)
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv('GROQ_API_KEY')
    if not api_key:
        print("❌ ERROR: Groq API key not provided!")
        sys.exit(1)
    
    print("="*60)
    print("KEYWORD-FOCUSED Q&A GENERATION (GROQ)")
    print("="*60)
    print(f"PDF: {args.pdf}")
    print(f"Target: {args.num_questions} keyword-focused Q&A pairs")
    print(f"Output: {args.output}")
    print("="*60)
    
    if not Path(args.pdf).exists():
        print(f"\n❌ PDF not found: {args.pdf}")
        sys.exit(1)
    
    df = generate_qa_dataset(args.pdf, args.num_questions, api_key)
    
    if not df.empty:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"\n✅ Saved to: {output_path}")
        print("\nSample Q&A:")
        print(f"Q: {df.iloc[0]['question']}")
        print(f"A: {df.iloc[0]['answer']}")
    else:
        print("\n❌ Failed")
        sys.exit(1)

if __name__ == "__main__":
    main()