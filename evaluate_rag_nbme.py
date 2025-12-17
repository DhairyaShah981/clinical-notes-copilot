#!/usr/bin/env python3
"""
Complete RAGAS Evaluation Script for Clinical RAG System
Uses NBME Clinical Patient Notes Dataset

This script:
1. Loads NBME dataset
2. Creates evaluation test cases
3. Writes test documents to Qdrant (like production)
4. Queries RAG WITHOUT any filtering (tests real behavior)
5. Calculates RAGAS metrics
6. Cleans up test data from Qdrant
7. Generates detailed report

Usage:
    python evaluate_rag_nbme.py --num_samples 10
"""

import os
import sys
import pandas as pd
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import ast
import uuid

# Load environment variables from backend/.env
from dotenv import load_dotenv
backend_env_path = Path(__file__).parent / "backend" / ".env"
if backend_env_path.exists():
    load_dotenv(backend_env_path)
    print(f"✓ Loaded environment variables from: {backend_env_path}")
else:
    print(f"⚠️  Warning: {backend_env_path} not found, using system environment variables")

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

# Import your RAG components
from vector_store import VectorStoreManager
from hybrid_search import HybridSearchEngine
from agents import ClinicalSearchAgent
from config import settings

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
    from llama_index.core import Document
    from llama_index.core.schema import TextNode
except ImportError:
    print("ERROR: Missing dependencies. Install with:")
    print("pip install ragas datasets langchain langchain-openai")
    sys.exit(1)


class NBMEDatasetLoader:
    """Load and prepare NBME dataset for evaluation"""
    
    def __init__(self, data_dir: str = "nbme-score-clinical-patient-notes"):
        self.data_dir = Path(data_dir)
        self.train_df = None
        self.notes_df = None
        self.features_df = None
    
    def load_data(self):
        """Load all NBME CSV files"""
        print("📂 Loading NBME dataset...")
        
        self.train_df = pd.read_csv(self.data_dir / "train.csv")
        self.notes_df = pd.read_csv(self.data_dir / "patient_notes.csv")
        self.features_df = pd.read_csv(self.data_dir / "features.csv")
        
        print(f"   ✓ Loaded {len(self.train_df)} annotations")
        print(f"   ✓ Loaded {len(self.notes_df)} patient notes")
        print(f"   ✓ Loaded {len(self.features_df)} features")
    
    def create_evaluation_dataset(self, num_samples: int = 10) -> List[Dict[str, Any]]:
        """
        Create evaluation test cases from NBME data
        Each test case uses ONE patient note with ONE question about a feature
        """
        print(f"\n📝 Creating {num_samples} evaluation test cases...")
        
        eval_cases = []
        
        # Sample directly from train.csv (which has annotations)
        sampled_annotations = self.train_df.sample(n=min(num_samples * 3, len(self.train_df)), random_state=42)
        
        for _, row in sampled_annotations.iterrows():
            if len(eval_cases) >= num_samples:
                break
            
            case_num = row['case_num']
            pn_num = row['pn_num']
            feature_num = row['feature_num']
            
            # Get patient note
            patient_note_row = self.notes_df[
                (self.notes_df['case_num'] == case_num) &
                (self.notes_df['pn_num'] == pn_num)
            ]
            
            if len(patient_note_row) == 0:
                continue
            
            patient_note = patient_note_row.iloc[0]['pn_history']
            
            # Get feature text
            feature_row = self.features_df[
                (self.features_df['case_num'] == case_num) &
                (self.features_df['feature_num'] == feature_num)
            ]
            
            if len(feature_row) == 0:
                continue
            
            feature_text = feature_row.iloc[0]['feature_text']
            
            # Parse annotation
            annotation_text = row['annotation']
            try:
                annotation_list = ast.literal_eval(annotation_text) if isinstance(annotation_text, str) else annotation_text
            except:
                annotation_list = []
            
            # Create question
            question = f"Does the patient note mention: {feature_text.replace('-', ' ')}?"
            
            # Create ground truth answer
            if annotation_list and len(annotation_list) > 0 and annotation_list[0]:
                evidence = annotation_list[0]
                ground_truth = f"Yes, the patient note mentions {feature_text.replace('-', ' ')}. Evidence: '{evidence}'"
                has_feature = True
            else:
                ground_truth = f"No, the patient note does not mention {feature_text.replace('-', ' ')}."
                has_feature = False
            
            eval_cases.append({
                'question': question,
                'ground_truth': ground_truth,
                'case_num': case_num,
                'pn_num': pn_num,
                'feature_num': feature_num,
                'feature_text': feature_text,
                'patient_note': patient_note,
                'has_feature': has_feature
            })
        
        print(f"   ✓ Created {len(eval_cases)} test cases")
        print(f"   ✓ Positive cases: {sum(1 for c in eval_cases if c['has_feature'])}")
        print(f"   ✓ Negative cases: {sum(1 for c in eval_cases if not c['has_feature'])}")
        
        if len(eval_cases) == 0:
            raise ValueError("Failed to create any test cases. Check NBME dataset format.")
        
        return eval_cases


class RAGEvaluator:
    """Run RAG pipeline - writes to Qdrant, queries without filtering"""
    
    def __init__(self):
        print("\n🔧 Initializing RAG pipeline...")
        print("   ✅ Will write test documents to Qdrant")
        print("   ✅ Will query FULL database (no filtering)")
        print("   ✅ Will cleanup test documents after evaluation")
        
        # Initialize vector store (Qdrant)
        self.vector_store = VectorStoreManager()
        
        # Track all point IDs for cleanup
        self.test_point_ids = []
        
        print("   ✓ RAG pipeline initialized")
    
    def chunk_text(self, text: str) -> List[str]:
        """Chunk text using same strategy as production"""
        chunks = []
        chunk_size = settings.CHUNK_SIZE
        chunk_overlap = settings.CHUNK_OVERLAP
        
        # Handle short texts that are smaller than chunk_size
        if len(text) <= chunk_size:
            return [text] if text.strip() else []
        
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk_text = text[i:i + chunk_size]
            if chunk_text.strip():
                chunks.append(chunk_text)
        
        return chunks if chunks else [text]  # Always return at least the full text
    
    def index_patient_note(self, patient_note: str, case_num: int, pn_num) -> tuple:
        """
        Index patient note in Qdrant (EXACTLY like production)
        Returns: (document_id, point_ids, documents)
        """
        # Generate unique document ID
        document_id = f"eval_{case_num}_{pn_num}_{uuid.uuid4().hex[:8]}"
        
        # Chunk the text
        chunks = self.chunk_text(patient_note)
        
        print(f"      📝 Note length: {len(patient_note)} chars → {len(chunks)} chunks")
        
        # Create Document objects (same as production)
        documents = []
        for idx, chunk in enumerate(chunks):
            doc = Document(
                text=chunk,
                metadata={
                    "source": f"nbme_case_{case_num}_pn_{pn_num}.txt",
                    "document_id": document_id,
                    "page_number": idx + 1,
                    "case_num": str(case_num),
                    "pn_num": str(pn_num),
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "is_eval": True  # Mark for cleanup
                }
            )
            documents.append(doc)
        
        # Index in Qdrant (EXACTLY like production)
        index, point_ids = self.vector_store.index_documents(documents)
        
        # Track point IDs for cleanup (handle various return types)
        if point_ids:
            for pid in point_ids:
                if isinstance(pid, list):
                    self.test_point_ids.extend(pid)
                elif isinstance(pid, str):
                    self.test_point_ids.append(pid)
        
        # Add to cache for BM25
        self.vector_store._documents[document_id] = documents
        
        return document_id, point_ids, documents
    
    def query_rag(self, question: str, documents: List[Document]) -> Dict[str, Any]:
        """
        Query the RAG pipeline - NO FILTERING, searches full database
        This tests the actual production behavior
        """
        # Create hybrid search engine with just this document's chunks for BM25
        doc_hybrid_engine = HybridSearchEngine()
        doc_hybrid_engine.index_documents(documents, silent=True)
        
        # Get the GLOBAL index (all documents in Qdrant)
        index = self.vector_store.get_index()
        
        # Create agent WITHOUT any filtering - tests real production behavior
        agent = ClinicalSearchAgent(
            index,
            doc_hybrid_engine,
            document_chunk_count=len(documents),
            filter_document_id=None,  # NO FILTERING
            filter_source=None  # NO FILTERING
        )
        
        # Query (no chat history)
        response = agent.query(question, chat_history=[])
        
        # Extract contexts from sources
        contexts = []
        if 'sources' in response:
            for source in response['sources']:
                if isinstance(source, dict) and 'text' in source:
                    contexts.append(source['text'])
        
        return {
            'answer': response.get('answer', ''),
            'contexts': contexts
        }
    
    def run_evaluation(self, eval_cases: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Run RAG on all evaluation cases
        Returns DataFrame with predictions
        """
        print(f"\n🚀 Running RAG predictions on {len(eval_cases)} cases...")
        print("   ✅ Writing to Qdrant (production behavior)")
        print("   ✅ Querying FULL database (no filtering)")
        
        results = []
        
        # Track indexed notes to avoid re-indexing
        indexed_notes = {}
        
        for i, case in enumerate(eval_cases, 1):
            print(f"\n   [{i}/{len(eval_cases)}] Case {case['case_num']}, Feature: {case['feature_text'][:40]}...")
            
            # Index patient note if not already indexed
            note_key = f"{case['case_num']}_{case['pn_num']}"
            
            if note_key not in indexed_notes:
                print(f"      📤 Indexing in Qdrant...")
                document_id, point_ids, documents = self.index_patient_note(
                    case['patient_note'],
                    case['case_num'],
                    case['pn_num']
                )
                indexed_notes[note_key] = (document_id, documents)
                print(f"      ✓ Indexed (doc_id: {document_id[:20]}...)")
            else:
                document_id, documents = indexed_notes[note_key]
                print(f"      ✓ Using cached index")
            
            # Query RAG (NO FILTERING - tests real production behavior)
            print(f"      🔍 Querying full database...")
            response = self.query_rag(case['question'], documents)
            
            answer_preview = response['answer'][:80].replace('\n', ' ') if response['answer'] else "No answer"
            print(f"      ✓ Answer: {answer_preview}...")
            print(f"      ✓ Retrieved {len(response['contexts'])} contexts")
            
            # Store result
            results.append({
                'question': case['question'],
                'contexts': response['contexts'],
                'answer': response['answer'],
                'ground_truth': case['ground_truth'],
                'case_num': case['case_num'],
                'feature_text': case['feature_text'],
                'has_feature': case['has_feature']
            })
        
        print(f"\n   ✓ Completed {len(results)} predictions")
        
        return pd.DataFrame(results)
    
    def cleanup(self):
        """Clean up test data from Qdrant"""
        if not self.test_point_ids:
            print("\n✅ No cleanup needed (no test points tracked)")
            return
        
        # Filter to only valid UUIDs
        valid_ids = [pid for pid in self.test_point_ids if isinstance(pid, str) and len(pid) > 10]
        
        if not valid_ids:
            print("\n✅ No valid point IDs to clean up")
            return
        
        print(f"\n🧹 Cleaning up {len(valid_ids)} test vectors from Qdrant...")
        
        try:
            from qdrant_client import models
            
            self.vector_store.qdrant_client.delete(
                collection_name=settings.COLLECTION_NAME,
                points_selector=models.PointIdsList(points=valid_ids)
            )
            
            print(f"   ✓ Deleted {len(valid_ids)} test vectors")
            print("   ✓ Qdrant is clean")
            
        except Exception as e:
            print(f"   ⚠️  Cleanup warning: {e}")
            print("   ℹ️  Test vectors may remain in Qdrant (marked with is_eval=True)")


class RAGASMetricsCalculator:
    """Calculate RAGAS metrics"""
    
    def __init__(self):
        self.evaluator_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.embeddings = OpenAIEmbeddings()
    
    def calculate_metrics(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate RAGAS metrics
        """
        print("\n📊 Calculating RAGAS metrics...")
        print("   ⏳ This will make LLM calls (may take 2-5 minutes)...")
        
        # Prepare dataset for RAGAS
        eval_data = {
            'question': predictions_df['question'].tolist(),
            'contexts': predictions_df['contexts'].tolist(),
            'answer': predictions_df['answer'].tolist(),
            'ground_truth': predictions_df['ground_truth'].tolist()
        }
        
        # Convert to Hugging Face Dataset
        dataset = Dataset.from_dict(eval_data)
        
        # Run RAGAS evaluation
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
        
        # Extract scores (handle both scalar and array results)
        def extract_score(value):
            if value is None:
                return 0.0
            if isinstance(value, (list, tuple)):
                return float(value[0]) if len(value) > 0 else 0.0
            try:
                return float(value)
            except:
                return 0.0
        
        # Access EvaluationResult using dictionary-style access
        scores = {
            'faithfulness': extract_score(result['faithfulness']),
            'answer_relevancy': extract_score(result['answer_relevancy']),
            'context_precision': extract_score(result['context_precision']),
            'context_recall': extract_score(result['context_recall']),
        }
        
        # Calculate overall RAGAS score
        scores['ragas_score'] = sum(scores.values()) / len(scores)
        
        # Get detailed results
        try:
            detailed_df = result.to_pandas()
        except:
            detailed_df = pd.DataFrame()
        
        print("\n" + "="*60)
        print("RAG EVALUATION RESULTS")
        print("="*60)
        for metric, score in scores.items():
            status = "✅" if self._check_threshold(metric, score) else "⚠️"
            print(f"{status} {metric:20s}: {score:.3f}")
        print("="*60)
        
        return {
            'scores': scores,
            'detailed_results': detailed_df
        }
    
    def _check_threshold(self, metric: str, score: float) -> bool:
        thresholds = {
            'faithfulness': 0.85,
            'answer_relevancy': 0.85,
            'context_precision': 0.80,
            'context_recall': 0.75,
            'ragas_score': 0.80
        }
        return score >= thresholds.get(metric, 0.75)


class ReportGenerator:
    """Generate evaluation report"""
    
    def generate_report(
        self,
        scores: Dict[str, float],
        detailed_df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        output_dir: Path
    ):
        """Generate comprehensive evaluation report"""
        print("\n📄 Generating evaluation report...")
        
        output_dir.mkdir(exist_ok=True)
        
        # Save files
        if not detailed_df.empty:
            detailed_df.to_csv(output_dir / 'ragas_detailed_results.csv', index=False)
        predictions_df.to_csv(output_dir / 'rag_predictions.csv', index=False)
        
        summary_df = pd.DataFrame([scores])
        summary_df.to_csv(output_dir / 'metrics_summary.csv', index=False)
        
        # Generate markdown report
        report = self._create_markdown_report(scores, predictions_df)
        report_path = output_dir / 'evaluation_report.md'
        report_path.write_text(report)
        
        print(f"   ✓ Saved to: {output_dir}/")
        print(report)
    
    def _create_markdown_report(self, scores: Dict[str, float], predictions_df: pd.DataFrame) -> str:
        def status_icon(score, threshold=0.80):
            return '✅' if score >= threshold else '⚠️'
        
        # Count contexts
        total_contexts = sum(len(c) for c in predictions_df['contexts'])
        avg_contexts = total_contexts / len(predictions_df) if len(predictions_df) > 0 else 0
        
        report = f"""# Clinical RAG System - RAGAS Evaluation Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Dataset:** NBME Clinical Patient Notes  
**Test Size:** {len(predictions_df)} samples  
**Evaluation Mode:** Full RAG Pipeline (Qdrant + No Filtering)

---

## Overall Performance

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| **Faithfulness** | {scores['faithfulness']:.3f} | >0.85 | {status_icon(scores['faithfulness'], 0.85)} |
| **Answer Relevancy** | {scores['answer_relevancy']:.3f} | >0.85 | {status_icon(scores['answer_relevancy'], 0.85)} |
| **Context Precision** | {scores['context_precision']:.3f} | >0.80 | {status_icon(scores['context_precision'], 0.80)} |
| **Context Recall** | {scores['context_recall']:.3f} | >0.75 | {status_icon(scores['context_recall'], 0.75)} |
| **RAGAS Score** | **{scores['ragas_score']:.3f}** | >0.80 | {status_icon(scores['ragas_score'], 0.80)} |

---

## Retrieval Statistics

- **Total Contexts Retrieved:** {total_contexts}
- **Average Contexts per Query:** {avg_contexts:.1f}
- **Queries with Empty Context:** {sum(1 for c in predictions_df['contexts'] if len(c) == 0)}

---

## Metric Interpretation

- **Faithfulness ({scores['faithfulness']:.1%})**: How well answers are grounded in retrieved context
- **Answer Relevancy ({scores['answer_relevancy']:.1%})**: How relevant answers are to the questions
- **Context Precision ({scores['context_precision']:.1%})**: How well relevant contexts are ranked
- **Context Recall ({scores['context_recall']:.1%})**: What fraction of relevant info was retrieved

---

## Test Configuration

- **Embedding Model:** {settings.EMBEDDING_MODEL}
- **Embedding Dimensions:** {settings.EMBEDDING_DIMENSIONS}
- **Chunk Size:** {settings.CHUNK_SIZE}
- **Chunk Overlap:** {settings.CHUNK_OVERLAP}
- **LLM:** gpt-4o-mini
- **Search Mode:** Hybrid (Vector + BM25)
- **Filtering:** None (full database search)

---

## Dataset Statistics

- **Total Test Cases:** {len(predictions_df)}
- **Positive Cases:** {sum(predictions_df['has_feature'])}
- **Negative Cases:** {sum(~predictions_df['has_feature'])}
- **Unique Cases:** {predictions_df['case_num'].nunique()}

---

**Generated by:** RAGAS Evaluation Pipeline
"""
        return report


def main():
    parser = argparse.ArgumentParser(description='Evaluate Clinical RAG System')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of test samples')
    parser.add_argument('--data_dir', type=str, default='nbme-score-clinical-patient-notes', help='NBME dataset directory')
    parser.add_argument('--output_dir', type=str, default='rag_evaluation', help='Output directory')
    parser.add_argument('--skip_cleanup', action='store_true', help='Skip Qdrant cleanup')
    
    args = parser.parse_args()
    
    print("="*60)
    print("CLINICAL RAG SYSTEM - RAGAS EVALUATION")
    print("="*60)
    print(f"Test Size: {args.num_samples} samples")
    print(f"Output: {args.output_dir}/")
    print(f"Mode: Full RAG (Qdrant write + no filtering)")
    print("="*60)
    
    # Check API keys
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ ERROR: OPENAI_API_KEY not set!")
        sys.exit(1)
    print(f"✓ OpenAI API key loaded")
    
    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        print("\n❌ ERROR: QDRANT_URL not set!")
        sys.exit(1)
    print(f"✓ Qdrant URL: {qdrant_url[:50]}...")
    
    evaluator = None
    
    try:
        # Step 1: Load dataset
        loader = NBMEDatasetLoader(args.data_dir)
        loader.load_data()
        
        # Step 2: Create test cases
        eval_cases = loader.create_evaluation_dataset(args.num_samples)
        
        # Step 3: Run RAG predictions
        evaluator = RAGEvaluator()
        predictions_df = evaluator.run_evaluation(eval_cases)
        
        # Step 4: Calculate RAGAS metrics
        calculator = RAGASMetricsCalculator()
        results = calculator.calculate_metrics(predictions_df)
        
        # Step 5: Generate report
        generator = ReportGenerator()
        generator.generate_report(
            results['scores'],
            results['detailed_results'],
            predictions_df,
            Path(args.output_dir)
        )
        
        print("\n" + "="*60)
        print("✅ EVALUATION COMPLETE!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if evaluator and not args.skip_cleanup:
            evaluator.cleanup()
        elif args.skip_cleanup:
            print("\n⚠️  Skipped cleanup (--skip_cleanup flag)")


if __name__ == "__main__":
    main()
