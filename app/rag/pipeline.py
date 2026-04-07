import logging
import faiss
from app.config import settings
from app.rag.loader import load_document, split_into_chunks
from app.rag.embeddings import build_index
from app.rag.retriever import find_relevant_chunks
from app.llm.ollama import ask_ollama

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self):
        self._index: faiss.IndexFlatL2 | None = None
        self._chunks: list[str] = []

    def build(self) -> None:
        text = load_document(settings.document_path)
        self._chunks = split_into_chunks(text, settings.chunk_size)
        self._index = build_index(self._chunks)
        logger.info("RAG pipeline ready")

    async def answer(self, question: str) -> tuple[str, float]:
        if self._index is None:
            raise RuntimeError("Pipeline not initialised – call build() first")
        
        # Retrieve chunks with distances
        results = find_relevant_chunks(question, self._index, self._chunks, k=10, threshold=1.1)
        
        if not results:
            logger.info(f"No relevant chunks found for: {question!r}")
            return "I don't have that information. The provided context does not cover this question.", 0.0
            
        # Extract chunks for the LLM
        filtered_chunks = [r[0] for r in results]
        
        # Calculate confidence score based on the best (minimum) distance
        # Standard L2 distance normalization: 1 / (1 + distance)
        best_distance = results[0][1]
        confidence_score = round(1.0 / (1.0 + best_distance), 2)
        
        logger.info(f"Answering using {len(filtered_chunks)} chunks (confidence: {confidence_score})")
        
        answer = await ask_ollama(question, filtered_chunks)
        return answer, confidence_score


pipeline = RAGPipeline()
