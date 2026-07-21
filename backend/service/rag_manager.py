"""
RAG Manager: lightweight document store with SQLite FTS5 (fallback to LIKE search).

Features:
- Add/update documents (doc_id, content)
- Delete documents
- List current documents
- Search top-k relevant documents

Storage layout (relative to this file by default):
- rag_store/
  - rag.db (SQLite database)
  - docs/<doc_id>.txt (plain text copies for transparency/debugging)
"""
from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime
from typing import List, Dict, Optional, Tuple


class RagManager:
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.join(os.path.dirname(__file__), "rag_store")
        self.docs_dir = os.path.join(self.base_dir, "docs")
        self.db_path = os.path.join(self.base_dir, "rag.db")
        self._lock = threading.RLock()
        self._fts_enabled = False
        self._embeddings_enabled = False
        self._embedder = None  # lazy-loaded SentenceTransformer
        # Optional: ChromaDB backend
        self._use_chroma = False
        self._chroma_client = None
        self._chroma_collection = None
        self._chroma_dir = os.path.join(self.base_dir, "chroma")
        self._maybe_init_chroma()
        self._ensure_setup()

    # ---------- setup ----------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_setup(self) -> None:
        os.makedirs(self.docs_dir, exist_ok=True)
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.cursor()
                # Main table
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS docs (
                        doc_id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                # Chunk table (for retrieval units and optional vector embeddings)
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        doc_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        embedding BLOB
                    )
                    """
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)"
                )
                # Try to create FTS5 table
                try:
                    cur.execute(
                        """
                        CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
                            doc_id UNINDEXED,
                            content,
                            content='docs',
                            content_rowid='rowid'
                        )
                        """
                    )
                    # Sync trigger on insert/update/delete
                    cur.executescript(
                        """
                        CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON docs BEGIN
                            INSERT INTO docs_fts(rowid, doc_id, content)
                            VALUES (new.rowid, new.doc_id, new.content);
                        END;
                        CREATE TRIGGER IF NOT EXISTS docs_au AFTER UPDATE ON docs BEGIN
                            INSERT INTO docs_fts(docs_fts, rowid, doc_id, content)
                            VALUES('delete', old.rowid, old.doc_id, old.content);
                            INSERT INTO docs_fts(rowid, doc_id, content)
                            VALUES (new.rowid, new.doc_id, new.content);
                        END;
                        CREATE TRIGGER IF NOT EXISTS docs_ad AFTER DELETE ON docs BEGIN
                            INSERT INTO docs_fts(docs_fts, rowid, doc_id, content)
                            VALUES('delete', old.rowid, old.doc_id, old.content);
                        END;
                        """
                    )
                    self._fts_enabled = True
                except sqlite3.OperationalError:
                    # FTS5 not available
                    self._fts_enabled = False
                conn.commit()
            finally:
                conn.close()

    # ---------- chroma setup ----------
    def _maybe_init_chroma(self) -> None:
        # Enable when env RAG_BACKEND=chroma or RAG_USE_CHROMA=true
        backend = os.environ.get("RAG_BACKEND", "").lower()
        use_flag = os.environ.get("RAG_USE_CHROMA", "false").lower() in {"1", "true", "yes"}
        if backend != "chroma" and not use_flag:
            return
        try:
            import chromadb  # type: ignore
            from chromadb.utils import embedding_functions as ef  # type: ignore
            os.makedirs(self._chroma_dir, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(path=self._chroma_dir)
            # Try to use a sentence-transformer embedding function inside Chroma
            model_name = os.environ.get("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            try:
                st_embed = ef.SentenceTransformerEmbeddingFunction(model_name=model_name)
            except Exception:
                st_embed = None
            self._chroma_collection = self._chroma_client.get_or_create_collection(
                name="rag_docs",
                embedding_function=st_embed,
                metadata={"hnsw:space": "cosine"}
            )
            self._use_chroma = True
        except Exception:
            # Keep fallback
            self._use_chroma = False

    # ---------- embeddings setup ----------
    def _ensure_embedder(self):
        if self._embedder is not None:
            return
        try:
            # Lazy import to avoid hard dependency if user doesn't need embeddings
            from sentence_transformers import SentenceTransformer  # type: ignore
            # Small, widely available model; can be replaced via env if needed
            model_name = os.environ.get("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            self._embedder = SentenceTransformer(model_name)
            self._embeddings_enabled = True
        except Exception:
            self._embedder = None
            self._embeddings_enabled = False

    def _embed_texts(self, texts: List[str]):
        self._ensure_embedder()
        if not self._embeddings_enabled or self._embedder is None:
            return None
        try:
            import numpy as np  # type: ignore
            vecs = self._embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            # Ensure float32
            if vecs.dtype != np.float32:
                vecs = vecs.astype(np.float32)
            return vecs
        except Exception:
            return None

    @staticmethod
    def _split_into_chunks(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
        text = text.strip()
        if not text:
            return []
        words = text.split()
        chunks = []
        current = []
        current_len = 0
        for w in words:
            current.append(w)
            current_len += len(w) + 1
            if current_len >= chunk_size:
                chunks.append(" ".join(current).strip())
                # start next chunk with overlap
                if overlap > 0:
                    overlap_words = " ".join(current)[-overlap:].split()
                    current = overlap_words
                    current_len = len(" ".join(current))
                else:
                    current = []
                    current_len = 0
        if current:
            chunks.append(" ".join(current).strip())
        return chunks

    # ---------- CRUD ----------
    def list_documents(self) -> List[Dict]:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.cursor()
                rows = cur.execute(
                    "SELECT doc_id, length(content) AS size, created_at, updated_at FROM docs ORDER BY doc_id"
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def add_document(self, content: str, doc_id: Optional[str] = None) -> Dict:
        doc_id = doc_id or self._gen_doc_id()
        now = datetime.utcnow().isoformat() + "Z"
        with self._lock:
            if self._use_chroma and self._chroma_collection is not None:
                # In Chroma, we store chunks as separate documents with metadata doc_id
                chunks = self._split_into_chunks(content)
                ids = [f"{doc_id}::{i}" for i in range(len(chunks))]
                metadatas = [{"doc_id": doc_id, "created_at": now, "chunk": i} for i in range(len(chunks))]
                try:
                    # If collection has embedding function, we can upsert directly
                    self._chroma_collection.upsert(ids=ids, metadatas=metadatas, documents=chunks)
                except TypeError:
                    # Some versions require 'add' or don't support upsert consistently
                    try:
                        self._chroma_collection.add(ids=ids, metadatas=metadatas, documents=chunks)
                    except Exception:
                        # Try manual embeddings
                        embeddings = self._embed_texts(chunks)
                        if embeddings is not None:
                            self._chroma_collection.upsert(ids=ids, metadatas=metadatas, documents=chunks, embeddings=embeddings.tolist())
                        else:
                            raise
                # Also write copy for transparency
                path = os.path.join(self.docs_dir, f"{doc_id}.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                return {"doc_id": doc_id, "created_at": now}
            # Fallback: SQLite path
            conn = self._connect()
            try:
                cur = conn.cursor()
                # Upsert
                cur.execute(
                    """
                    INSERT INTO docs(doc_id, content, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(doc_id) DO UPDATE SET
                        content=excluded.content,
                        updated_at=excluded.updated_at
                    """,
                    (doc_id, content, now, now),
                )
                # Rebuild chunks for this doc
                cur.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
                chunks = self._split_into_chunks(content)
                # Compute embeddings if possible
                embeddings = self._embed_texts(chunks)
                if embeddings is not None:
                    # Store with embeddings
                    import numpy as np  # type: ignore
                    for i, ch in enumerate(chunks):
                        vec: np.ndarray = embeddings[i]
                        cur.execute(
                            "INSERT INTO chunks(doc_id, content, created_at, embedding) VALUES (?, ?, ?, ?)",
                            (doc_id, ch, now, vec.tobytes()),
                        )
                else:
                    # Store without embeddings
                    for ch in chunks:
                        cur.execute(
                            "INSERT INTO chunks(doc_id, content, created_at, embedding) VALUES (?, ?, ?, NULL)",
                            (doc_id, ch, now),
                        )
                conn.commit()
                # Write copy to docs/
                path = os.path.join(self.docs_dir, f"{doc_id}.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                return {"doc_id": doc_id, "created_at": now}
            finally:
                conn.close()

    def delete_document(self, doc_id: str) -> Dict:
        with self._lock:
            if self._use_chroma and self._chroma_collection is not None:
                try:
                    self._chroma_collection.delete(where={"doc_id": doc_id})
                except Exception:
                    pass
                path = os.path.join(self.docs_dir, f"{doc_id}.txt")
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass
                return {"deleted": True, "doc_id": doc_id}
            # Fallback
            conn = self._connect()
            try:
                cur = conn.cursor()
                cur.execute("DELETE FROM docs WHERE doc_id = ?", (doc_id,))
                cur.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
                conn.commit()
                # Remove file copy
                path = os.path.join(self.docs_dir, f"{doc_id}.txt")
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass
                return {"deleted": True, "doc_id": doc_id}
            finally:
                conn.close()

    # ---------- Search ----------
    def search(self, query: str, k: int = 3) -> List[Dict]:
        k = max(1, min(k, 50))
        with self._lock:
            # Chroma path
            if self._use_chroma and self._chroma_collection is not None:
                try:
                    res = self._chroma_collection.query(query_texts=[query], n_results=k, include=["documents", "metadatas", "distances"])
                    docs = (res.get("documents") or [[]])[0]
                    metas = (res.get("metadatas") or [[]])[0]
                    dists = (res.get("distances") or [[]])[0]
                    out = []
                    for doc, meta, dist in zip(docs, metas, dists):
                        doc_id = meta.get("doc_id") if isinstance(meta, dict) else None
                        score = 1.0 - float(dist) if dist is not None else None
                        snippet = (doc or "")[:400]
                        out.append({"doc_id": doc_id, "score": score, "snippet": snippet})
                    return out
                except Exception:
                    # fall back to local store if query fails
                    pass
            # Fallback: SQLite path
            conn = self._connect()
            try:
                cur = conn.cursor()
                # Preferred: vector similarity over chunks if embeddings exist
                has_any_embedding = cur.execute(
                    "SELECT EXISTS(SELECT 1 FROM chunks WHERE embedding IS NOT NULL)"
                ).fetchone()[0]
                if has_any_embedding:
                    qvecs = self._embed_texts([query])
                    if qvecs is not None:
                        try:
                            import numpy as np  # type: ignore
                            qv = qvecs[0]
                            rows = cur.execute(
                                "SELECT chunk_id, doc_id, content, embedding FROM chunks WHERE embedding IS NOT NULL"
                            ).fetchall()
                            if not rows:
                                # fall back if no embeddings stored yet
                                raise RuntimeError("no-embeddings")
                            # Build matrices
                            emb_list = []
                            meta: List[Tuple[int, str, str]] = []
                            for r in rows:
                                emb = np.frombuffer(r["embedding"], dtype=np.float32)
                                emb_list.append(emb)
                                meta.append((r["chunk_id"], r["doc_id"], r["content"]))
                            M = np.vstack(emb_list)
                            # Cosine similarity (qv and M are normalized already)
                            sims = M @ qv
                            top_idx = np.argsort(-sims)[:k]
                            out = []
                            for idx in top_idx:
                                chunk_id, doc_id, content = meta[idx]
                                score = float(sims[idx])
                                snippet = content[:400]
                                out.append({
                                    "doc_id": doc_id,
                                    "chunk_id": chunk_id,
                                    "score": score,
                                    "snippet": snippet,
                                })
                            return out
                        except Exception:
                            # Any failure -> fallback to text search
                            pass

                # Next: FTS5 over full documents
                if self._fts_enabled:
                    try:
                        sql = (
                            "SELECT d.doc_id AS doc_id, bm25(docs_fts) AS score, "
                            "snippet(docs_fts, 1, '[', ']', ' … ', 10) AS snippet, d.content AS content "
                            "FROM docs_fts JOIN docs d ON docs_fts.rowid = d.rowid "
                            "WHERE docs_fts MATCH ? ORDER BY score LIMIT ?"
                        )
                        rows = cur.execute(sql, (query, k)).fetchall()
                        return [dict(r) for r in rows]
                    except sqlite3.OperationalError:
                        pass

                # Fallback: naive LIKE search on full documents
                pattern = f"%{query}%"
                rows = cur.execute(
                    "SELECT doc_id, content FROM docs WHERE content LIKE ? LIMIT ?",
                    (pattern, k),
                ).fetchall()
                tmp = []
                for r in rows:
                    content = r["content"]
                    idx = content.lower().find(query.lower())
                    start = max(idx - 60, 0) if idx != -1 else 0
                    end = min(start + 200, len(content))
                    snippet = content[start:end]
                    tmp.append({
                        "doc_id": r["doc_id"],
                        "score": None,
                        "snippet": snippet,
                        "content": None,
                    })
                return tmp
            finally:
                conn.close()

    # ---------- utils ----------
    def _gen_doc_id(self) -> str:
        return datetime.utcnow().strftime("%Y%m%d%H%M%S%f")


# Singleton instance
rag_manager = RagManager()
