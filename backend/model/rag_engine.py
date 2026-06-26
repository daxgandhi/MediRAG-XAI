"""
Clinical RAG Engine
Uses ChromaDB + sentence-transformers + Gemini API
Pipeline: Question → Embedding → ChromaDB → Retrieved Evidence → Gemini → Answer
Always returns citations from retrieved documents.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, List

BASE_DIR    = Path(__file__).parent.parent
GUIDELINES  = BASE_DIR / "data" / "guidelines"
PATIENT_DOC = BASE_DIR / "data" / "patient_docs"
CHROMA_DIR  = BASE_DIR.parent / "vector_db" / "chroma_store"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION  = "medirag_knowledge"


class RAGEngine:
    def __init__(self):
        self.embed_model = None
        self.chroma      = None
        self.collection  = None
        self.gemini      = None
        self._initialized = False
        self._init()

    def _init(self):
        try:
            self._load_embedder()
            self._init_chroma()
            self._ingest_documents()
            self.provider = os.getenv("LLM_PROVIDER", "gemini").lower()
            if self.provider == "grok":
                self._init_xai()
            else:
                self._init_gemini()
            self._initialized = True
            print("[OK] RAG Engine initialized")
        except Exception as e:
            print(f"[WARN] RAG Engine init warning: {e}")

    def _load_embedder(self):
        from sentence_transformers import SentenceTransformer
        self.embed_model = SentenceTransformer(EMBED_MODEL)
        print(f"[OK] Embedder loaded: {EMBED_MODEL}")

    def _init_chroma(self):
        import chromadb
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.chroma     = client
        self.collection = client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[OK] ChromaDB ready — {self.collection.count()} chunks stored")

    def _ingest_documents(self):
        if self.collection.count() > 0:
            print(f"[INFO] {self.collection.count()} chunks already in ChromaDB, skipping ingestion.")
            return

        print("[INFO] Ingesting clinical guidelines into ChromaDB...")
        docs = self._load_all_documents()
        if not docs:
            print("[WARN] No documents found to ingest.")
            return

        texts, metas, ids = [], [], []
        for i, (text, meta) in enumerate(docs):
            chunks = self._chunk_text(text, chunk_size=400, overlap=50)
            for j, chunk in enumerate(chunks):
                texts.append(chunk)
                metas.append(meta)
                ids.append(f"doc_{i}_chunk_{j}")

        embeddings = self.embed_model.encode(texts, show_progress_bar=True).tolist()
        self.collection.add(documents=texts, embeddings=embeddings, metadatas=metas, ids=ids)
        print(f"[OK] Ingested {len(texts)} chunks from {len(docs)} documents")

    def _load_all_documents(self) -> List[tuple]:
        """Load all markdown/text documents from guidelines and patient_docs directories."""
        docs = []
        for folder in [GUIDELINES, PATIENT_DOC]:
            if not folder.exists():
                continue
            for filepath in folder.glob("**/*.md"):
                try:
                    text = filepath.read_text(encoding="utf-8")
                    meta = {
                        "source":   filepath.name,
                        "category": folder.name,
                        "path":     str(filepath),
                    }
                    docs.append((text, meta))
                except Exception as e:
                    print(f"  Warning: could not read {filepath}: {e}")
            for filepath in folder.glob("**/*.txt"):
                try:
                    text = filepath.read_text(encoding="utf-8")
                    meta = {
                        "source":   filepath.name,
                        "category": folder.name,
                        "path":     str(filepath),
                    }
                    docs.append((text, meta))
                except Exception:
                    pass
        return docs

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
        words  = text.split()
        chunks = []
        start  = 0
        while start < len(words):
            end = start + chunk_size
            chunks.append(" ".join(words[start:end]))
            start = end - overlap
        return [c for c in chunks if len(c.strip()) > 30]

    def _init_gemini(self):
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            print("[WARN] GEMINI_API_KEY not set. RAG will return retrieved context only.")
            return
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            self.gemini = genai.GenerativeModel(
                model_name,
                generation_config={"temperature": 0.2, "max_output_tokens": 1024},
                system_instruction=(
                    "You are a clinical knowledge assistant for MEDIRAG-XAI, a healthcare AI platform. "
                    "Answer ONLY using the provided clinical evidence. "
                    "Always cite the source documents. "
                    "If the provided evidence does not contain a clear answer, say so. "
                    "Never provide personal medical advice. Always recommend consulting a physician."
                ),
            )
            print(f"[OK] Gemini API connected using model: {model_name}")
        except Exception as e:
            print(f"[WARN] Gemini init error: {e}")

    def _init_xai(self):
        self.xai_key = os.getenv("XAI_API_KEY", "")
        self.xai_model = os.getenv("XAI_MODEL", "grok-2")
        if not self.xai_key or self.xai_key == "your_xai_api_key_here":
            print("[WARN] XAI_API_KEY not set. RAG will return retrieved context only.")
            return
        print(f"[OK] xAI API configured using model: {self.xai_model}")

    def _query_xai(self, question: str, context: str) -> str:
        if not hasattr(self, "xai_key") or not self.xai_key or self.xai_key == "your_xai_api_key_here":
            raise ValueError("XAI_API_KEY not set or invalid.")
        
        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.xai_key}",
            "Content-Type": "application/json"
        }
        
        system_instruction = (
            "You are a clinical knowledge assistant for MEDIRAG-XAI, a healthcare AI platform. "
            "Answer ONLY using the provided clinical evidence. "
            "Always cite the source documents. "
            "If the provided evidence does not contain a clear answer, say so. "
            "Never provide personal medical advice. Always recommend consulting a physician."
        )
        
        prompt = (
            f"Clinical Evidence:\n{context}\n\n"
            f"Patient/Doctor Question: {question}\n\n"
            "Based ONLY on the clinical evidence above, provide a comprehensive, accurate answer. "
            "Cite the source documents at the end."
        )
        
        payload = {
            "model": self.xai_model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 1024
        }
        
        import httpx
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Full RAG pipeline: embed → retrieve → generate."""
        if not self._initialized or self.collection is None:
            return self._fallback_response(question)

        # 1. Embed question
        try:
            q_embedding = self.embed_model.encode([question]).tolist()
        except Exception as e:
            return self._fallback_response(question, error=str(e))

        # 2. Retrieve from ChromaDB
        try:
            results = self.collection.query(
                query_embeddings=q_embedding,
                n_results=min(top_k, self.collection.count() or 1),
            )
            retrieved_docs  = results["documents"][0] if results["documents"] else []
            retrieved_metas = results["metadatas"][0] if results["metadatas"] else []
        except Exception as e:
            return self._fallback_response(question, error=str(e))

        if not retrieved_docs:
            return {
                "answer":  "No relevant clinical evidence found in the knowledge base for this query.",
                "sources": [],
                "chunks":  [],
                "question": question,
            }

        # 3. Format context
        context_parts = []
        sources       = []
        for doc, meta in zip(retrieved_docs, retrieved_metas):
            src = meta.get("source", "Unknown")
            context_parts.append(f"[{src}]\n{doc}")
            if src not in sources:
                sources.append(src)

        context = "\n\n---\n\n".join(context_parts)

        # 4. Generate with LLM
        answer = None
        if hasattr(self, "provider") and self.provider == "grok":
            if hasattr(self, "xai_key") and self.xai_key and self.xai_key != "your_xai_api_key_here":
                try:
                    answer = self._query_xai(question, context)
                except Exception as e:
                    answer = self._format_retrieved_answer(retrieved_docs, retrieved_metas)
                    answer += f"\n\n⚠️ Grok API error: {e}. Showing retrieved evidence only."
            else:
                answer = self._format_retrieved_answer(retrieved_docs, retrieved_metas)
                answer += "\n\n⚠️ Set XAI_API_KEY in .env for Grok AI-generated summaries."
        else:
            if self.gemini:
                try:
                    prompt = (
                        f"Clinical Evidence:\n{context}\n\n"
                        f"Patient/Doctor Question: {question}\n\n"
                        "Based ONLY on the clinical evidence above, provide a comprehensive, accurate answer. "
                        "Cite the source documents at the end."
                    )
                    response = self.gemini.generate_content(prompt)
                    answer   = response.text
                except Exception as e:
                    answer = self._format_retrieved_answer(retrieved_docs, retrieved_metas)
                    answer += f"\n\n⚠️ Gemini API error: {e}. Showing retrieved evidence only."
            else:
                answer = self._format_retrieved_answer(retrieved_docs, retrieved_metas)
                answer += "\n\n⚠️ Set GEMINI_API_KEY in .env for AI-generated summaries."

        return {
            "answer":   answer,
            "sources":  sources,
            "chunks":   [
                {"text": doc[:300], "source": meta.get("source", "N/A")}
                for doc, meta in zip(retrieved_docs, retrieved_metas)
            ],
            "question": question,
        }

    @staticmethod
    def _format_retrieved_answer(docs, metas) -> str:
        parts = ["**Retrieved Clinical Evidence:**\n"]
        for i, (doc, meta) in enumerate(zip(docs, metas), 1):
            src = meta.get("source", "Unknown")
            parts.append(f"**[{i}] Source: {src}**\n{doc[:400]}\n")
        return "\n".join(parts)

    @staticmethod
    def _fallback_response(question: str, error: str = "") -> Dict[str, Any]:
        msg = (
            "The RAG engine is not fully initialized. This may be because:\n"
            "1. No clinical documents have been added to backend/data/guidelines/\n"
            "2. ChromaDB is not configured\n"
            "3. sentence-transformers is not installed\n\n"
            "Please ensure all dependencies are installed and guideline documents exist."
        )
        if error:
            msg += f"\n\nError: {error}"
        return {
            "answer":   msg,
            "sources":  [],
            "chunks":   [],
            "question": question,
        }
