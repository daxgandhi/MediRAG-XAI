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
            self._init_llms()
            self._initialized = True
            print("[OK] RAG Engine initialized with Multi-LLM Fallback Pipeline")
        except Exception as e:
            print(f"[WARN] RAG Engine init warning: {e}")

    def _load_embedder(self):
        from sentence_transformers import SentenceTransformer
        self.embed_model = SentenceTransformer(EMBED_MODEL)
        print(f"[OK] Embedder loaded: {EMBED_MODEL}")

    def _init_chroma(self):
        """
        Initializes ChromaDB Persistent Client.
        
        NOTE ON STORAGE & RESTART BEHAVIOR:
        - In local development: Vectors persist on disk at `vector_db/chroma_store`.
        - In cloud container hosting (Render, Railway, Fly.io without persistent disks):
          Container restarts/redeploys will wipe local ephemeral disk.
          However, `_ingest_documents()` below is self-healing: if the collection is empty,
          it automatically re-embeds and indexes all guideline/patient documents on startup (~2-4 seconds).
        - Recommended for High-Scale Enterprise Production:
          Connect to a hosted cloud vector database (e.g., Pinecone, Qdrant Cloud, Chroma Cloud)
          or mount a persistent block storage volume to `vector_db/chroma_store`.
        """
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

    def _init_llms(self):
        """Initialize all available LLM providers for automatic failover."""
        # 1. Groq Init
        self.groq_key = os.getenv("GROQ_API_KEY", "").strip()
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
        self.groq_client = None
        if self.groq_key and not self.groq_key.startswith("your_"):
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=self.groq_key)
                print(f"[OK] Groq LLM initialized ({self.groq_model})")
            except Exception as e:
                print(f"[WARN] Groq init error: {e}")

        # 2. Gemini Init
        self.gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.gemini_model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
        self.gemini = None
        if self.gemini_key and not self.gemini_key.startswith("your_"):
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_key)
                self.gemini = genai.GenerativeModel(
                    self.gemini_model_name,
                    generation_config={"temperature": 0.2, "max_output_tokens": 4096},
                    system_instruction=(
                        "You are a clinical knowledge assistant for MEDIRAG-XAI, a healthcare AI platform. "
                        "Answer clearly and comprehensively using the provided clinical evidence. "
                        "Provide practical lifestyle recommendations, warnings, and when to seek medical help based on the guidelines. "
                        "Always cite the source documents. "
                        "Never provide definitive personal diagnosis. Always recommend consulting a physician."
                    ),
                )
                print(f"[OK] Gemini LLM initialized ({self.gemini_model_name})")
            except Exception as e:
                print(f"[WARN] Gemini init error: {e}")

        # 3. xAI Init (Optional)
        self.xai_key = os.getenv("XAI_API_KEY", "").strip()
        self.xai_model = os.getenv("XAI_MODEL", "grok-2").strip()

    def _query_groq(self, question: str, context: str) -> str:
        """Execute query using Groq API."""
        if not self.groq_client:
            raise ValueError("Groq client not configured or API key missing.")
        
        system_instruction = (
            "You are a clinical knowledge assistant for MEDIRAG-XAI, a healthcare AI platform. "
            "Answer clearly and comprehensively using the provided clinical evidence. "
            "Provide practical lifestyle recommendations, warnings, and when to seek medical help based on the guidelines. "
            "Always cite the source documents. "
            "Never provide definitive personal diagnosis. Always recommend consulting a physician."
        )
        prompt = (
            f"Clinical Evidence:\n{context}\n\n"
            f"Patient/Doctor Question: {question}\n\n"
            "Based on the clinical evidence above, provide a comprehensive, accurate answer. "
            "Cite the source documents at the end."
        )
        chat_completion = self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            model=self.groq_model,
            temperature=0.2,
            max_tokens=2048,
        )
        return chat_completion.choices[0].message.content

    def _query_gemini(self, question: str, context: str) -> str:
        """Execute query using Google Gemini API."""
        if not self.gemini:
            raise ValueError("Gemini client not configured or API key missing.")
        
        prompt = (
            f"Clinical Evidence:\n{context}\n\n"
            f"Patient/Doctor Question: {question}\n\n"
            "Based ONLY on the clinical evidence above, provide a comprehensive, accurate answer. "
            "Cite the source documents at the end."
        )
        response = self.gemini.generate_content(prompt)
        return response.text

    def _query_xai(self, question: str, context: str) -> str:
        """Execute query using xAI Grok API."""
        if not self.xai_key or self.xai_key.startswith("your_"):
            raise ValueError("XAI_API_KEY not configured.")
        
        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.xai_key}",
            "Content-Type": "application/json"
        }
        system_instruction = (
            "You are a clinical knowledge assistant for MEDIRAG-XAI, a healthcare AI platform. "
            "Answer ONLY using the provided clinical evidence. "
            "Always cite the source documents. "
            "Never provide personal medical advice. Always recommend consulting a physician."
        )
        prompt = (
            f"Clinical Evidence:\n{context}\n\n"
            f"Patient/Doctor Question: {question}\n\n"
            "Based on the clinical evidence above, provide a comprehensive, accurate answer. "
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

    def _generate_with_fallback(self, question: str, context: str, retrieved_docs: list, retrieved_metas: list) -> tuple[str, str]:
        """
        Execute generation across prioritized provider chain:
        Primary (Groq / Gemini based on LLM_PROVIDER) -> Fallback Provider -> Local RAG Evidence
        """
        preferred = os.getenv("LLM_PROVIDER", "groq").lower().strip()
        
        # Build ordered provider list
        if preferred == "groq":
            providers = ["groq", "gemini", "xai"]
        elif preferred == "xai":
            providers = ["xai", "groq", "gemini"]
        else:
            providers = ["gemini", "groq", "xai"]

        errors = []
        for p in providers:
            try:
                if p == "groq" and self.groq_client:
                    answer = self._query_groq(question, context)
                    return answer, "groq"
                elif p == "gemini" and self.gemini:
                    answer = self._query_gemini(question, context)
                    return answer, "gemini"
                elif p == "xai" and self.xai_key and not self.xai_key.startswith("your_"):
                    answer = self._query_xai(question, context)
                    return answer, "xai"
            except Exception as e:
                err_msg = f"{p.upper()} error: {e}"
                print(f"[WARN] Failover triggered: {err_msg}")
                errors.append(err_msg)
                continue

        # If all LLMs fail or none are configured, gracefully return retrieved clinical documents
        fallback_answer = self._format_retrieved_answer(retrieved_docs, retrieved_metas)
        if errors:
            fallback_answer += f"\n\n*(Note: LLM failover triggered due to: {'; '.join(errors)})*"
        else:
            fallback_answer += "\n\n*(Configure GROQ_API_KEY or GEMINI_API_KEY in .env for full AI generation.)*"
        return fallback_answer, "local_retrieval"

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Full RAG pipeline: embed → retrieve → multi-LLM generate with fallback."""
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
                "provider": "none",
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

        # 4. Generate with Multi-LLM Fallback
        answer, used_provider = self._generate_with_fallback(
            question, context, retrieved_docs, retrieved_metas
        )

        return {
            "answer":   answer,
            "sources":  sources,
            "chunks":   [
                {"text": doc[:300], "source": meta.get("source", "N/A")}
                for doc, meta in zip(retrieved_docs, retrieved_metas)
            ],
            "question": question,
            "provider": used_provider,
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
