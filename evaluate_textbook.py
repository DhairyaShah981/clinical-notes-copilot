#!/usr/bin/env python3
"""
Evaluate RAG system on medical textbooks
Compares 3 approaches: Hybrid Search, Vector Search, Keyword Search

Usage:
    python evaluate_textbook.py --pdf data/anatomy_20.pdf --qa data/anatomy_20_qa.csv --output results_20
    python evaluate_textbook.py --pdf data/anatomy_200.pdf --qa data/anatomy_200_qa.csv --output results_200
"""

import os
import sys
import pandas as pd
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import uuid

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / "backend" / ".env")

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

# Import RAG components
from vector_store import VectorStoreManager
from hybrid_search import HybridSearchEngine
from agents import ClinicalSearchAgent
from config import settings
from document_processor import DocumentProcessor

# RAGAS imports
try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError:
    print("ERROR: Missing dependencies. Install with:")
    print("pip install ragas datasets langchain langchain-openai")
    sys.exit(1)


class TextbookEvaluator:
    """Evaluate RAG on textbook with 3 different search strategies"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.vector_store = VectorStoreManager()
        self.hybrid_engine = HybridSearchEngine()
        self.document_id = None
        self.documents = None
        self.test_point_ids = []
        
    def index_textbook(self):
        """Index the textbook in Qdrant"""
        print(f"\n📚 Indexing textbook: {self.pdf_path}")
        
        # Generate unique document ID
        self.document_id = f"eval_textbook_{uuid.uuid4().hex[:8]}"
        
        # Process PDF (extracts text, chunks it)
        print("   📄 Processing PDF...")
        processor = DocumentProcessor()
        self.documents = processor.create_documents(self.pdf_path)
        
        if not self.documents:
            raise ValueError("Failed to extract documents from PDF")
        
        # Add document_id to metadata
        for doc in self.documents:
            doc.metadata['document_id'] = self.document_id
        
        print(f"   ✓ Created {len(self.documents)} chunks")
        
        # Index in Qdrant
        print("   📤 Indexing in Qdrant...")
        index, point_ids = self.vector_store.index_documents(self.documents, document_id=self.document_id)
        
        # Track for cleanup
        if point_ids:
            for pid in point_ids:
                if isinstance(pid, list):
                    self.test_point_ids.extend(pid)
                elif isinstance(pid, str):
                    self.test_point_ids.append(pid)
        
        # Index for BM25
        self.hybrid_engine.index_documents(self.documents)
        
        print(f"   ✓ Indexed {len(self.documents)} chunks to Qdrant")
        print(f"   ✓ Document ID: {self.document_id}")
        
    def query_with_strategy(self, question: str, strategy: str) -> Dict[str, Any]:
        """Query RAG with specific search strategy"""
        
        index = self.vector_store.get_index()
        
        if strategy == "hybrid":
            # Hybrid: Vector + BM25 (deterministic evaluation without the agent)
            hybrid_retriever = index.as_retriever(similarity_top_k=15)
            vector_nodes = hybrid_retriever.retrieve(question)

            filtered_vector_nodes = [
                n for n in vector_nodes
                if n.node.metadata.get("document_id") == self.document_id
            ][:15]

            hybrid_results = self.hybrid_engine.hybrid_search(
                question,
                filtered_vector_nodes,
                top_k=5
            )

            contexts = [r["text"] for r in hybrid_results]

            from openai import OpenAI
            client = OpenAI(api_key=settings.OPENAI_API_KEY)

            context_text = "\n\n".join(contexts)
            prompt = f"Based on this context, answer concisely in 1-3 sentences:\n\nContext:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"

            llm_response = client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )

            response = {
                'answer': llm_response.choices[0].message.content,
                'sources': [{'text': c} for c in contexts]
            }
            
        elif strategy == "vector":
            # Vector only: Use retriever directly
            retriever = index.as_retriever(similarity_top_k=15)
            nodes = retriever.retrieve(question)
            
            # Filter to this document
            filtered_nodes = [n for n in nodes if n.node.metadata.get("document_id") == self.document_id][:5]
            
            contexts = [n.node.text for n in filtered_nodes]
            
            # Generate answer using LLM
            from openai import OpenAI
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            
            context_text = "\n\n".join(contexts)
            prompt = f"Based on this context, answer concisely in 1-3 sentences:\n\nContext:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
            
            llm_response = client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            response = {
                'answer': llm_response.choices[0].message.content,
                'sources': [{'text': c} for c in contexts]
            }
            
        elif strategy == "keyword":
            # BM25 only
            bm25_results = self.hybrid_engine.keyword_search(question, top_k=5)
            contexts = [doc.text for doc, score in bm25_results]
            
            # Generate answer
            from openai import OpenAI
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            
            context_text = "\n\n".join(contexts)
            prompt = f"Based on this context, answer concisely in 1-3 sentences:\n\nContext:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
            
            llm_response = client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            response = {
                'answer': llm_response.choices[0].message.content,
                'sources': [{'text': c} for c in contexts]
            }
        
        # Extract contexts
        contexts = []
        if 'sources' in response:
            for source in response['sources']:
                if isinstance(source, dict) and 'text' in source:
                    contexts.append(source['text'])
        
        return {
            'answer': response.get('answer', ''),
            'contexts': contexts
        }
    
    def evaluate_strategy(self, qa_df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Evaluate RAG with specific strategy"""
        print(f"\n🔍 Evaluating {strategy.upper()} search...")
        
        results = []
        
        for i, row in qa_df.iterrows():
            question = row['question']
            ground_truth = row['answer']
            
            print(f"   [{i+1}/{len(qa_df)}] {question[:60]}...")
            
            response = self.query_with_strategy(question, strategy)
            
            results.append({
                'question': question,
                'contexts': response['contexts'],
                'answer': response['answer'],
                'ground_truth': ground_truth
            })
            
            print(f"      ✓ Retrieved {len(response['contexts'])} contexts")
        
        return pd.DataFrame(results)
    
    def cleanup(self):
        """Remove test data from Qdrant"""
        if not self.test_point_ids:
            return
        
        valid_ids = [pid for pid in self.test_point_ids if isinstance(pid, str) and len(pid) > 10]
        
        if valid_ids:
            print(f"\n🧹 Cleaning up {len(valid_ids)} test vectors...")
            try:
                from qdrant_client import models
                self.vector_store.qdrant_client.delete(
                    collection_name=settings.COLLECTION_NAME,
                    points_selector=models.PointIdsList(points=valid_ids)
                )
                print("   ✓ Cleanup complete")
            except Exception as e:
                print(f"   ⚠️  Cleanup warning: {e}")


class RAGASCalculator:
    """Calculate RAGAS metrics"""
    
    def __init__(self):
        self.evaluator_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.embeddings = OpenAIEmbeddings()
    
    def calculate(self, predictions_df: pd.DataFrame, strategy_name: str) -> Dict[str, float]:
        """Calculate RAGAS metrics for a strategy"""
        print(f"\n📊 Calculating RAGAS metrics for {strategy_name}...")
        
        eval_data = {
            'question': predictions_df['question'].tolist(),
            'contexts': predictions_df['contexts'].tolist(),
            'answer': predictions_df['answer'].tolist(),
            'ground_truth': predictions_df['ground_truth'].tolist()
        }
        
        dataset = Dataset.from_dict(eval_data)
        
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=self.evaluator_llm,
            embeddings=self.embeddings
        )
        
        def extract_score(value):
            if value is None:
                return 0.0
            if isinstance(value, (list, tuple)):
                return float(value[0]) if len(value) > 0 else 0.0
            try:
                return float(value)
            except:
                return 0.0
        
        scores = {
            'strategy': strategy_name,
            'faithfulness': extract_score(result['faithfulness']),
            'answer_relevancy': extract_score(result['answer_relevancy']),
            'context_precision': extract_score(result['context_precision']),
            'context_recall': extract_score(result['context_recall']),
        }
        
        scores['ragas_score'] = sum([
            scores['faithfulness'],
            scores['answer_relevancy'],
            scores['context_precision'],
            scores['context_recall']
        ]) / 4
        
        return scores


def generate_comparison_report(all_scores: List[Dict], output_dir: Path, pdf_name: str, num_questions: int):
    """Generate comparison report"""
    
    df = pd.DataFrame(all_scores)
    
    report = f"""# RAG Strategy Comparison - {pdf_name}

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Document:** {pdf_name}  
**Test Size:** {num_questions} questions  
**Strategies Tested:** Hybrid, Vector-Only, Keyword-Only

---

## Overall Comparison

| Strategy | Faithfulness | Answer Relevancy | Context Precision | Context Recall | **RAGAS Score** |
|----------|--------------|------------------|-------------------|----------------|-----------------|
"""
    
    for _, row in df.iterrows():
        report += f"| **{row['strategy'].title()}** | {row['faithfulness']:.3f} | {row['answer_relevancy']:.3f} | {row['context_precision']:.3f} | {row['context_recall']:.3f} | **{row['ragas_score']:.3f}** |\n"
    
    # Find best strategy
    best_strategy = df.loc[df['ragas_score'].idxmax()]
    
    report += f"""

---

## Winner: {best_strategy['strategy'].title()} Search 🏆

**RAGAS Score:** {best_strategy['ragas_score']:.3f}

### Why {best_strategy['strategy'].title()} Won:
- **Faithfulness:** {best_strategy['faithfulness']:.1%} (answers grounded in context)
- **Answer Relevancy:** {best_strategy['answer_relevancy']:.1%} (relevant to questions)
- **Context Precision:** {best_strategy['context_precision']:.1%} (relevant chunks ranked high)
- **Context Recall:** {best_strategy['context_recall']:.1%} (all relevant info retrieved)

---

## Strategy Analysis

### Hybrid Search (Vector + BM25)
- **Best for:** Balanced performance, handles both semantic and keyword queries
- **Strengths:** Combines semantic understanding with exact matching
- **Use when:** You need robust performance across different query types

### Vector-Only Search
- **Best for:** Semantic/conceptual queries
- **Strengths:** Understands synonyms and paraphrasing
- **Use when:** Queries are conceptual ("explain X", "what causes Y")

### Keyword-Only Search (BM25)
- **Best for:** Exact term matching
- **Strengths:** Fast, finds specific terms reliably
- **Use when:** Queries contain specific medical terms, codes, or names

---

**Generated by:** Textbook RAG Evaluation Pipeline
"""
    
    # Save report
    report_path = output_dir / 'comparison_report.md'
    report_path.write_text(report)
    
    # Save scores CSV
    df.to_csv(output_dir / 'strategy_scores.csv', index=False)
    
    print(f"\n✅ Saved comparison report to: {report_path}")
    print(report)


def main():
    parser = argparse.ArgumentParser(description='Evaluate RAG on medical textbook')
    parser.add_argument('--pdf', type=str, required=True, help='Path to PDF textbook')
    parser.add_argument('--qa', type=str, required=True, help='Path to Q&A CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    print("="*60)
    print("TEXTBOOK RAG EVALUATION - STRATEGY COMPARISON")
    print("="*60)
    print(f"PDF: {args.pdf}")
    print(f"Q&A: {args.qa}")
    print(f"Output: {args.output}/")
    print("="*60)
    
    # Check files exist
    if not Path(args.pdf).exists():
        print(f"\n❌ ERROR: PDF not found: {args.pdf}")
        sys.exit(1)
    
    if not Path(args.qa).exists():
        print(f"\n❌ ERROR: Q&A file not found: {args.qa}")
        sys.exit(1)
    
    # Load Q&A
    qa_df = pd.read_csv(args.qa)
    print(f"\n✓ Loaded {len(qa_df)} Q&A pairs")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator = None
    
    try:
        # Index textbook
        evaluator = TextbookEvaluator(args.pdf)
        evaluator.index_textbook()
        
        # Evaluate each strategy
        all_scores = []
        calculator = RAGASCalculator()
        
        for strategy in ['hybrid', 'vector', 'keyword']:
            # Run predictions
            predictions_df = evaluator.evaluate_strategy(qa_df, strategy)
            
            # Save predictions
            predictions_df.to_csv(output_dir / f'{strategy}_predictions.csv', index=False)
            
            # Calculate RAGAS
            scores = calculator.calculate(predictions_df, strategy)
            all_scores.append(scores)
            
            print(f"\n   ✅ {strategy.upper()}: RAGAS = {scores['ragas_score']:.3f}")
        
        # Generate comparison report
        pdf_name = Path(args.pdf).stem
        generate_comparison_report(all_scores, output_dir, pdf_name, len(qa_df))
        
        print("\n" + "="*60)
        print("✅ EVALUATION COMPLETE!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        if evaluator:
            evaluator.cleanup()


if __name__ == "__main__":
    main()

