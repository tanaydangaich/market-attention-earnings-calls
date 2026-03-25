"""
RAG index builder over earnings call transcripts.
Uses LlamaIndex + ChromaDB (local, no infra needed) + OpenAI embeddings.

Run once to build the index:
  from src.rag.indexer import build_index
  build_index()

The index persists to chroma_db/ and is reloaded by the Streamlit app.
Cost: ~$1 for 5,200 transcripts with text-embedding-3-small.
"""

from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR, CHROMA_DIR, OPENAI_API_KEY, EMBEDDING_MODEL, LLM_MODEL


def build_index(force: bool = False) -> None:
    """
    Build ChromaDB vector index from transcripts.parquet.
    Skips if index already exists unless force=True.
    """
    try:
        import chromadb
        from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.llms.openai import OpenAI
    except ImportError:
        raise ImportError(
            "pip install llama-index llama-index-vector-stores-chroma "
            "llama-index-embeddings-openai llama-index-llms-openai chromadb"
        )

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection("transcripts")

    if collection.count() > 0 and not force:
        print(f"Index already contains {collection.count()} chunks. Pass force=True to rebuild.")
        return

    transcripts_path = PROCESSED_DIR / "transcripts.parquet"
    if not transcripts_path.exists():
        raise FileNotFoundError("transcripts.parquet not found — run pipeline first")

    df = pd.read_parquet(transcripts_path)
    print(f"Building index over {len(df):,} transcripts...")

    # configure LlamaIndex
    Settings.embed_model = OpenAIEmbedding(
        model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY
    )
    Settings.llm = OpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY)

    # build documents — one per transcript with rich metadata
    documents = []
    for _, row in df.iterrows():
        text = str(row.get("full_text", ""))
        if not text.strip():
            continue

        doc = Document(
            text=text,
            metadata={
                "company_name": str(row.get("company_name", "")),
                "cik": str(row.get("cik", "")),
                "ticker": str(row.get("ticker", "")),
                "call_date": str(row.get("call_date", "")),
                "fiscal_year": str(row.get("fiscal_year", "")),
                "fiscal_quarter": str(row.get("fiscal_quarter", "")),
                "lm_tone": str(row.get("full_lm_tone", "")),
            },
        )
        documents.append(doc)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    print(f"Index built: {collection.count()} chunks in {CHROMA_DIR}")


if __name__ == "__main__":
    build_index(force=False)
