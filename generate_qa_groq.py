#!/usr/bin/env python3
"""
Generate Q&A pairs from PDF using Groq API (FREE + FAST)

Usage:
    python generate_qa_groq.py --pdf data/anatomy_20.pdf --num_questions 10 --output data/anatomy_20_qa.csv
    python generate_qa_groq.py --pdf data/anatomy_200.pdf --num_questions 100 --output data/anatomy_200_qa.csv
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

def generate_questions_from_chunk(client: Groq, chunk: str, num_questions: int = 5) -> List[Dict]:
    """Generate Q&A pairs from a text chunk using Groq"""
    
    prompt = f"""You are an expert medical educator creating evaluation questions from a medical textbook.

Generate {num_questions} high-quality question-answer pairs from the following text. Follow these rules:

1. Questions should be:
   - Factual and specific (not vague)
   - Answerable from the text
   - Mix of difficulty levels
   - Cover different topics in the text

2. Answers should be:
   - Concise (1-3 sentences)
   - Directly from the text
   - Include specific facts/details

3. Question types to include:
   - Definition questions ("What is X?")
   - Function questions ("What does X do?")
   - Location questions ("Where is X located?")
   - Relationship questions ("How does X relate to Y?")

TEXT:
{chunk[:2000]}

Return ONLY a JSON array with this exact format:
[
  {{"question": "What is...", "answer": "..."}},
  {{"question": "Where is...", "answer": "..."}}
]

Generate exactly {num_questions} Q&A pairs:"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Fast and free
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000
        )
        
        content = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            qa_pairs = json.loads(json_match.group())
            return qa_pairs
        else:
            print(f"   ⚠️  Could not parse JSON from response")
            return []
            
    except Exception as e:
        print(f"   ⚠️  Error generating questions: {e}")
        return []

def generate_qa_dataset(pdf_path: str, num_questions: int, api_key: str) -> pd.DataFrame:
    """Generate Q&A dataset from PDF"""
    
    # Initialize Groq client
    client = Groq(api_key=api_key)
    
    # Extract text
    full_text, num_pages = extract_text_from_pdf(pdf_path)
    
    # Chunk text
    chunks = chunk_text(full_text, chunk_size=3000)
    print(f"   ✓ Split into {len(chunks)} chunks")
    
    # Calculate questions per chunk
    questions_per_chunk = max(2, num_questions // len(chunks))
    
    print(f"\n🤖 Generating {num_questions} Q&A pairs using Groq (Llama 3.1 70B)...")
    print(f"   Generating ~{questions_per_chunk} questions per chunk")
    
    all_qa_pairs = []
    
    for i, chunk in enumerate(chunks, 1):
        if len(all_qa_pairs) >= num_questions:
            break
        
        remaining = num_questions - len(all_qa_pairs)
        questions_to_generate = min(questions_per_chunk, remaining)
        
        print(f"\n   [{i}/{len(chunks)}] Processing chunk {i}...")
        qa_pairs = generate_questions_from_chunk(client, chunk, questions_to_generate)
        
        if qa_pairs:
            all_qa_pairs.extend(qa_pairs)
            print(f"      ✓ Generated {len(qa_pairs)} Q&A pairs (Total: {len(all_qa_pairs)})")
        else:
            print(f"      ⚠️  Failed to generate Q&A pairs for this chunk")
    
    # Convert to DataFrame
    if all_qa_pairs:
        df = pd.DataFrame(all_qa_pairs)
        df = df.head(num_questions)  # Ensure exact count
        print(f"\n✅ Successfully generated {len(df)} Q&A pairs")
        return df
    else:
        print(f"\n❌ Failed to generate any Q&A pairs")
        return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description='Generate Q&A pairs from PDF using Groq')
    parser.add_argument('--pdf', type=str, required=True, help='Path to PDF file')
    parser.add_argument('--num_questions', type=int, default=10, help='Number of Q&A pairs to generate')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--api_key', type=str, help='Groq API key (or set GROQ_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv('GROQ_API_KEY')
    if not api_key:
        print("❌ ERROR: Groq API key not provided!")
        print("\nOptions:")
        print("  1. Pass via --api_key argument")
        print("  2. Set GROQ_API_KEY environment variable")
        print("  3. Get free key at: https://console.groq.com")
        sys.exit(1)
    
    print("="*60)
    print("Q&A GENERATION FROM PDF (GROQ)")
    print("="*60)
    print(f"PDF: {args.pdf}")
    print(f"Target Q&A pairs: {args.num_questions}")
    print(f"Output: {args.output}")
    print(f"Model: Llama 3.1 70B (via Groq)")
    print("="*60)
    
    # Check if PDF exists
    if not Path(args.pdf).exists():
        print(f"\n❌ ERROR: PDF file not found: {args.pdf}")
        sys.exit(1)
    
    # Generate Q&A dataset
    df = generate_qa_dataset(args.pdf, args.num_questions, api_key)
    
    if not df.empty:
        # Save to CSV
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"\n✅ Saved {len(df)} Q&A pairs to: {output_path}")
        print("\nSample Q&A:")
        print(f"Q: {df.iloc[0]['question']}")
        print(f"A: {df.iloc[0]['answer']}")
    else:
        print("\n❌ Failed to generate Q&A dataset")
        sys.exit(1)

if __name__ == "__main__":
    main()

