"""
RAG Pipeline — the core learning module of this project.

What you'll learn:
  1. Document chunking: splitting long policy docs into smaller searchable pieces
  2. Embeddings: converting text chunks into numeric vectors (sentence-transformers)
  3. Vector store: storing and searching vectors in ChromaDB
  4. Retrieval: given a query, find the most relevant policy chunks
  5. Augmented generation: pass retrieved chunks to LLM as context
"""
import os
import glob
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from app.config import CHROMA_PATH, POLICIES_PATH, EMBEDDING_MODEL


_model = None
_client = None
_collection = None


def get_embedding_model() -> SentenceTransformer:
    """Load embedding model once and cache it (lazy loading)."""
    global _model
    if _model is None:
        print(f"[RAG] Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_chroma_collection():
    """Get or create ChromaDB collection."""
    global _client, _collection
    if _collection is None:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _client.get_or_create_collection(
            name="insurance_policies",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def chunk_document(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """
    Split a document into overlapping chunks.

    Why overlap? So that important context near chunk boundaries isn't lost.
    chunk_size=400 words is a common sweet spot — enough context, not too noisy.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap  # slide forward with overlap
    return chunks


def load_and_index_policies(force_reload: bool = False) -> int:
    """
    Load all .txt policy documents, chunk them, embed, and store in ChromaDB.
    Returns number of chunks indexed.

    This is the INDEXING phase of RAG (done once, or when docs change).
    """
    collection = get_chroma_collection()

    # Skip if already indexed (unless forced)
    if not force_reload and collection.count() > 0:
        print(f"[RAG] Knowledge base already indexed: {collection.count()} chunks")
        return collection.count()

    model = get_embedding_model()
    policy_files = glob.glob(os.path.join(POLICIES_PATH, "*.txt"))

    if not policy_files:
        print("[RAG] No policy documents found in", POLICIES_PATH)
        return 0

    all_chunks = []
    all_ids = []
    all_metadata = []

    for filepath in policy_files:
        filename = os.path.basename(filepath)
        with open(filepath, "r") as f:
            text = f.read()

        chunks = chunk_document(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{filename}__chunk_{i}")
            all_metadata.append({"source": filename, "chunk_index": i})

    # Embed all chunks at once (batched for efficiency)
    print(f"[RAG] Embedding {len(all_chunks)} chunks from {len(policy_files)} documents...")
    embeddings = model.encode(all_chunks, show_progress_bar=True).tolist()

    # Clear existing and re-index
    if force_reload and collection.count() > 0:
        collection.delete(ids=collection.get()["ids"])

    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        ids=all_ids,
        metadatas=all_metadata,
    )

    print(f"[RAG] Indexed {len(all_chunks)} chunks into ChromaDB")
    return len(all_chunks)


def retrieve_relevant_policies(query: str, n_results: int = 4) -> list[dict]:
    """
    Given a claim query, retrieve the most relevant policy chunks.

    This is the RETRIEVAL phase of RAG.
    Steps:
      1. Embed the query using the same model as the documents
      2. Compute cosine similarity between query vector and all stored vectors
      3. Return top-k most similar chunks with their source and distance score
    """
    collection = get_chroma_collection()

    if collection.count() == 0:
        print("[RAG] Knowledge base is empty. Run load_and_index_policies() first.")
        return []

    model = get_embedding_model()

    # Embed the query
    query_embedding = model.encode([query]).tolist()

    # Search ChromaDB for nearest neighbors
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    for i in range(len(results["documents"][0])):
        retrieved.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "similarity_score": round(1 - results["distances"][0][i], 3),
        })

    return retrieved


def get_kb_stats() -> dict:
    """Return knowledge base statistics."""
    collection = get_chroma_collection()
    count = collection.count()
    sources = set()
    if count > 0:
        items = collection.get(include=["metadatas"])
        for m in items["metadatas"]:
            sources.add(m.get("source", "unknown"))
    return {"total_chunks": count, "documents": len(sources), "sources": list(sources)}
