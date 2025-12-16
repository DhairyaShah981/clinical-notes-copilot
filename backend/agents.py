"""
Multi-Agent System with OpenAI Native Function Calling

This replaces llama-index's ReActAgent (which had API breaking changes) 
with OpenAI's native function calling - more stable and powerful.

The agent can:
1. Decide which search method to use based on the query
2. Chain multiple searches for complex queries
3. Maintain conversation context from session memory
"""
import json
import re
from typing import Dict, Any, List, Optional
from openai import OpenAI
from llama_index.core import VectorStoreIndex, Settings as LlamaSettings
from llama_index.embeddings.openai import OpenAIEmbedding
from hybrid_search import HybridSearchEngine
from config import settings


def clean_markdown_text(text: str) -> str:
    """Remove markdown formatting and clean text for display"""
    if not text:
        return ""
    
    text = re.sub(r'#{1,6}\s+', '', text)
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'`{1,3}([^`]+)`{1,3}', r'\1', text)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[#@$%&|{}\\]', '', text)
    
    return text.strip()


# Tool definitions for OpenAI function calling
CLINICAL_SEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "hybrid_search",
            "description": "🌟 PRIMARY SEARCH METHOD: Combined semantic + keyword search using Reciprocal Rank Fusion. Use this for 90% of queries - it provides the most comprehensive results by combining vector similarity with keyword matching. Best for: policy questions, benefit descriptions, treatment plans, diagnoses, any query needing complete information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query - can be natural language question or specific terms"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": "Vector similarity search only. Use ONLY for very broad conceptual queries without specific terms (e.g., 'summarize the document', 'what is the general theme'). For most queries, prefer hybrid_search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query - describe what you're looking for conceptually"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "keyword_search",
            "description": "BM25 keyword-only search. Use ONLY for exact value lookups: specific numeric values (HbA1c: 7.2), exact codes (ICD-10: E11.9), unique identifiers. For questions about topics/policies, use hybrid_search instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Exact terms to search for (numeric values, codes, identifiers)"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    }
]


class ClinicalSearchAgent:
    """
    Multi-Agent Clinical Search using OpenAI Function Calling
    
    This agent:
    1. Analyzes the user's question
    2. Decides which search tool(s) to use
    3. Executes searches and synthesizes results
    4. Maintains conversation context
    """
    
    def __init__(self, index: VectorStoreIndex, hybrid_engine: HybridSearchEngine, document_chunk_count: int = None, filter_document_id: str = None, filter_source: str = None):
        self.index = index
        self.hybrid_engine = hybrid_engine
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.document_chunk_count = document_chunk_count  # For adaptive top_k
        self.filter_document_id = filter_document_id  # For post-retrieval filtering
        self.filter_source = filter_source  # For post-retrieval filtering by filename
        
        # Set up embedding model for retrieval
        self.embed_model = OpenAIEmbedding(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY
        )
        LlamaSettings.embed_model = self.embed_model
    
    def _execute_tool(self, tool_name: str, arguments: Dict) -> Dict[str, Any]:
        """Execute a search tool and return results"""
        query = arguments.get("query", "")
        top_k = arguments.get("top_k", 5)
        
        if tool_name == "semantic_search":
            return self._semantic_search(query, top_k)
        elif tool_name == "keyword_search":
            return self._keyword_search(query, top_k)
        elif tool_name == "hybrid_search":
            return self._hybrid_search(query, top_k)
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    
    def _semantic_search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Vector similarity search with post-retrieval filtering"""
        # Retrieve more results if filtering (to compensate for filtered-out chunks)
        retrieval_k = top_k * 3 if (self.filter_document_id or self.filter_source) else top_k
        
        retriever = self.index.as_retriever(similarity_top_k=retrieval_k)
        nodes = retriever.retrieve(query)
        
        # Post-retrieval filtering by document_id or source
        if self.filter_document_id or self.filter_source:
            print(f"     🔒 Filtering {len(nodes)} results by document/source...")
            filtered_nodes = []
            for node in nodes:
                metadata = node.node.metadata
                # Check both document_id and source
                if self.filter_document_id and metadata.get("document_id") == self.filter_document_id:
                    filtered_nodes.append(node)
                elif self.filter_source and metadata.get("source") == self.filter_source:
                    filtered_nodes.append(node)
            nodes = filtered_nodes[:top_k]  # Limit to requested top_k
            print(f"     ✅ Kept {len(nodes)} results after filtering")
        
        print(f"     📊 Semantic search returned {len(nodes)} results")
        
        results = []
        for i, n in enumerate(nodes):
            cleaned = clean_markdown_text(n.node.text)
            results.append({
                "text": cleaned[:500],
                "page": n.node.metadata.get("page_number", "N/A"),
                "source": n.node.metadata.get("source", "Unknown"),
                "score": round(n.score, 4) if n.score else 0
            })
            
            # Log first result for debugging
            if i == 0:
                print(f"     🔝 Top result (score: {n.score:.4f}):")
                print(f"        Page: {n.node.metadata.get('page_number', 'N/A')}")
                print(f"        Text preview: {cleaned[:150]}...")
        
        return {
            "tool": "semantic_search",
            "query": query,
            "results": results,
            "count": len(results)
        }
    
    def _keyword_search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """BM25 keyword search"""
        results = self.hybrid_engine.keyword_search(query, top_k=top_k)
        
        print(f"     📊 Keyword search returned {len(results)} results")
        
        formatted = []
        for i, (doc, score) in enumerate(results):
            formatted.append({
                "text": clean_markdown_text(doc.text)[:500],
                "page": doc.metadata.get("page_number", "N/A"),
                "source": doc.metadata.get("source", "Unknown"),
                "score": round(score, 4)
            })
            
            # Log first result
            if i == 0:
                print(f"     🔝 Top result (BM25 score: {score:.4f}):")
                print(f"        Page: {doc.metadata.get('page_number', 'N/A')}")
                print(f"        Text preview: {clean_markdown_text(doc.text)[:150]}...")
        
        return {
            "tool": "keyword_search",
            "query": query,
            "results": formatted,
            "count": len(formatted)
        }
    
    def _hybrid_search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Combined vector + BM25 with RRF"""
        # Adaptive top_k based on document size
        if self.document_chunk_count:
            # Single-document query: retrieve more aggressively (50-70% of document)
            retrieval_k = max(top_k * 3, min(self.document_chunk_count // 2, 30))
            print(f"     📊 Document-specific: retrieving {retrieval_k}/{self.document_chunk_count} chunks")
        else:
            # Multi-document query: use conservative top_k
            retrieval_k = min(top_k * 3, 15)
            print(f"     📊 Multi-document: retrieving {retrieval_k} chunks")
        
        # Retrieve from Qdrant (no metadata filters - we'll filter post-retrieval)
        retriever = self.index.as_retriever(similarity_top_k=retrieval_k)
        vector_results = retriever.retrieve(query)
        
        # Post-retrieval filtering by document_id or source
        if self.filter_document_id or self.filter_source:
            print(f"     🔒 Filtering {len(vector_results)} vector results by document/source...")
            filtered_results = []
            for node in vector_results:
                metadata = node.node.metadata
                # Check both document_id and source
                if self.filter_document_id and metadata.get("document_id") == self.filter_document_id:
                    filtered_results.append(node)
                elif self.filter_source and metadata.get("source") == self.filter_source:
                    filtered_results.append(node)
            vector_results = filtered_results
            print(f"     ✅ Kept {len(vector_results)} vector results after filtering")
        hybrid_results = self.hybrid_engine.hybrid_search(
            query, vector_results, top_k=top_k
        )
        
        print(f"     📊 Hybrid search returned {len(hybrid_results)} results")
        
        results = []
        for i, r in enumerate(hybrid_results):
            cleaned = clean_markdown_text(r["text"])
            results.append({
                "text": cleaned[:500],
                "page": r["metadata"].get("page_number", "N/A"),
                "source": r["metadata"].get("source", "Unknown"),
                "score": round(r["score"], 4),
                "vector_score": round(r.get("vector_score", 0), 4),
                "bm25_score": round(r.get("bm25_score", 0), 4)
            })
            
            # Log first result
            if i == 0:
                print(f"     🔝 Top result (hybrid: {r['score']:.4f}, vector: {r.get('vector_score', 0):.4f}, bm25: {r.get('bm25_score', 0):.4f}):")
                print(f"        Page: {r['metadata'].get('page_number', 'N/A')}")
                print(f"        Text preview: {cleaned[:150]}...")
        
        return {
            "tool": "hybrid_search",
            "query": query,
            "results": results,
            "count": len(results)
        }
    
    def query(
        self, 
        question: str, 
        chat_history: List[Dict] = None,
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Process a user query using the multi-agent system
        
        Args:
            question: User's question
            chat_history: Previous messages for context
            max_iterations: Max tool calls to prevent infinite loops
        
        Returns:
            Dict with answer, sources, and tools_used
        """
        
        # Build messages with system prompt and history
        messages = [
            {
                "role": "system",
                "content": """You are an intelligent document search assistant.
Your job is to help users find information in their documents accurately and comprehensively.

RULES:
1. ONLY use information from the search results - never make up information
2. Always cite the source document and page number
3. If information is not found, clearly state that
4. Be thorough and provide complete answers with all relevant details
5. Quote relevant sections when answering

SEARCH STRATEGY (IMPORTANT):
- **DEFAULT: Use hybrid_search for most queries** - it combines semantic understanding + keyword matching for best results
- Use semantic_search ONLY for very broad conceptual queries without specific terms
- Use keyword_search ONLY for exact value lookups (specific codes, precise measurements, unique identifiers)

EXAMPLES:
✅ "What is the data protection policy?" → hybrid_search (needs both concept + specific terms)
✅ "What are the benefits mentioned?" → hybrid_search (comprehensive search needed)
✅ "What is the diagnosis?" → hybrid_search (medical term + context)
✅ "What medications are prescribed?" → hybrid_search (specific drugs + context)
❌ "HbA1c: 7.2" → keyword_search (exact value lookup)
❌ "ICD-10: E11.9" → keyword_search (exact code)

When in doubt, use hybrid_search - it's the most comprehensive."""
            }
        ]
        
        # Add chat history for context (last 10 messages)
        if chat_history:
            for msg in chat_history[-10:]:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        # Add current question
        messages.append({"role": "user", "content": question})
        
        # Track tools used and all search results
        tools_used = []
        all_sources = []
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            
            # Call OpenAI with tools
            response = self.client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                tools=CLINICAL_SEARCH_TOOLS,
                tool_choice="auto",
                temperature=0.1
            )
            
            assistant_message = response.choices[0].message
            
            # Check if we need to call tools
            if assistant_message.tool_calls:
                # Add assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]
                })
                
                # Execute each tool call
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    
                    print(f"  → Executing {tool_name}: {arguments.get('query', '')[:50]}...")
                    
                    # Execute the tool
                    result = self._execute_tool(tool_name, arguments)
                    tools_used.append(tool_name)
                    
                    # Collect sources
                    for r in result.get("results", []):
                        if r not in all_sources:
                            all_sources.append(r)
                    
                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
            else:
                # No more tool calls - we have the final answer
                break
        
        # Get final answer
        final_answer = assistant_message.content or "I couldn't find relevant information in the clinical notes."
        
        # Deduplicate and limit sources
        unique_sources = []
        seen_texts = set()
        for s in all_sources:
            text_key = s.get("text", "")[:100]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_sources.append(s)
        
        return {
            "answer": final_answer,
            "sources": unique_sources[:5],  # Top 5 sources
            "tools_used": list(set(tools_used)),
            "search_type": "multi_agent"
        }


class SimpleRAGEngine:
    """
    Simple RAG without agent complexity - faster for straightforward queries
    Still uses hybrid search for best results
    """
    
    def __init__(self, index: VectorStoreIndex, hybrid_engine: HybridSearchEngine):
        self.index = index
        self.hybrid_engine = hybrid_engine
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
        self.embed_model = OpenAIEmbedding(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY
        )
        LlamaSettings.embed_model = self.embed_model
    
    def query(
        self, 
        question: str, 
        use_hybrid: bool = True,
        chat_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """Query with optional hybrid search and chat history"""
        
        # Get search results
        if use_hybrid:
            retriever = self.index.as_retriever(similarity_top_k=10)
            vector_results = retriever.retrieve(question)
            hybrid_results = self.hybrid_engine.hybrid_search(
                question, vector_results, top_k=5
            )
            
            context = "\n\n".join([
                f"[Source: {r['metadata'].get('source', 'Unknown')}, Page {r['metadata'].get('page_number', 'N/A')}]\n{clean_markdown_text(r['text'])}"
                for r in hybrid_results
            ])
            search_type = "hybrid"
        else:
            retriever = self.index.as_retriever(similarity_top_k=5)
            nodes = retriever.retrieve(question)
            
            context = "\n\n".join([
                f"[Source: {n.node.metadata.get('source', 'Unknown')}, Page {n.node.metadata.get('page_number', 'N/A')}]\n{clean_markdown_text(n.node.text)}"
                for n in nodes
            ])
            search_type = "vector"
            hybrid_results = [{"text": n.node.text, "metadata": n.node.metadata} for n in nodes]
        
        # Build messages
        messages = [
            {
                "role": "system",
                "content": """You are a clinical notes search assistant for physicians.
Answer questions based ONLY on the provided clinical notes context.
Always cite the source document and page number.
If the information is not in the context, say so clearly."""
            }
        ]
        
        # Add chat history for context
        if chat_history:
            for msg in chat_history[-6:]:  # Last 6 messages
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        # Add current query with context
        messages.append({
            "role": "user",
            "content": f"""Clinical Notes Context:
{context}

Question: {question}

Please provide a detailed answer based on the clinical notes above."""
        })
        
        # Get response
        response = self.client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=messages,
            temperature=0.1
        )
        
        answer = response.choices[0].message.content
        
        # Format sources
        sources = []
        for r in hybrid_results[:5]:
            if isinstance(r, dict):
                sources.append({
                    "text": clean_markdown_text(r.get("text", ""))[:300],
                    "page": r.get("metadata", {}).get("page_number", "N/A"),
                    "source": r.get("metadata", {}).get("source", "Unknown")
                })
        
        return {
            "answer": answer,
            "sources": sources,
            "search_type": search_type,
            "tools_used": [search_type]
        }


# Backwards compatibility alias
ClinicalNotesAgent = ClinicalSearchAgent
