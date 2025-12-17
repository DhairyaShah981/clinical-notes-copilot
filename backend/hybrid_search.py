"""Hybrid Search Engine - Vector + BM25"""
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
import numpy as np
from llama_index.core import Document
from llama_index.core.schema import NodeWithScore


class HybridSearchEngine:
    def __init__(self):
        self.documents: List[Document] = []
        self.bm25 = None
        self.tokenized_corpus = []
    
    def index_documents(self, documents: List[Document], silent: bool = False):
        """Index documents for BM25
        
        Args:
            documents: List of documents to index
            silent: If True, suppress the print statement (for document-specific queries)
        """
        self.documents = documents
        self.tokenized_corpus = [
            doc.text.lower().split() for doc in documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        if not silent:
            print(f"BM25 indexed {len(documents)} documents")
    
    def keyword_search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """BM25 keyword search"""
        if self.bm25 is None:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.documents[idx], float(scores[idx])))
        return results
    
    def hybrid_search(
        self, 
        query: str, 
        vector_results: List[NodeWithScore],
        alpha: float = 0.6,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Combine vector + BM25 with RRF (Reciprocal Rank Fusion)"""
        bm25_results = self.keyword_search(query, top_k=top_k * 2)
        
        # DEBUG: Show what BM25 found
        print(f"     🔎 BM25 search found {len(bm25_results)} results for query: '{query}'")
        if len(bm25_results) > 0:
            print(f"        Top BM25 result: {bm25_results[0][0].text[:100]}...")
        
        # RRF scoring
        vector_scores = {}
        bm25_scores = {}
        all_texts = {}
        all_metadata = {}
        
        for rank, node in enumerate(vector_results):
            doc_id = hash(node.node.text[:100])
            vector_scores[doc_id] = 1 / (rank + 60)
            all_texts[doc_id] = node.node.text
            all_metadata[doc_id] = node.node.metadata
        
        for rank, (doc, score) in enumerate(bm25_results):
            doc_id = hash(doc.text[:100])
            bm25_scores[doc_id] = 1 / (rank + 60)
            all_texts[doc_id] = doc.text
            all_metadata[doc_id] = doc.metadata
        
        # Combine scores
        all_doc_ids = set(vector_scores.keys()) | set(bm25_scores.keys())
        combined_results = []
        
        for doc_id in all_doc_ids:
            v_score = vector_scores.get(doc_id, 0)
            b_score = bm25_scores.get(doc_id, 0)
            final_score = alpha * v_score + (1 - alpha) * b_score
            
            combined_results.append({
                "text": all_texts[doc_id],
                "score": final_score,
                "metadata": all_metadata[doc_id],
                "vector_score": v_score,
                "bm25_score": b_score
            })
        
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        return combined_results[:top_k]

