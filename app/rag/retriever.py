import logging
import faiss
from app.rag.embeddings import embed_query

logger = logging.getLogger(__name__)


def find_relevant_chunks(
    question: str, 
    index: faiss.IndexFlatL2, 
    chunks: list[str], 
    k: int = 10,
    threshold: float = 1.1
) -> list[tuple[str, float]]:
    query_vector = embed_query(question)
    distances, indices = index.search(query_vector, k=min(k, len(chunks)))
    
    results = []
    for i in range(len(indices[0])):
        idx = int(indices[0][i])
        dist = float(distances[0][i])
        
        if idx != -1 and dist <= threshold:
            results.append((chunks[idx], dist))
            
    logger.info(f"Retrieved {len(results)} chunks below threshold {threshold}")
    return results
