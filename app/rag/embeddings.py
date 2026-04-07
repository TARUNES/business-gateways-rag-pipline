import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_model = SentenceTransformer("all-MiniLM-L6-v2")


def build_index(chunks: list[str]) -> faiss.IndexFlatL2:
    logger.info("Building FAISS index from document chunks...")
    embeddings = _model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype(np.float32)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    logger.info(f"FAISS index ready {index.ntotal} vectors stored")
    return index


def embed_query(question: str) -> np.ndarray:
    vector = _model.encode([question], convert_to_numpy=True, show_progress_bar=False)
    return vector.astype(np.float32)
