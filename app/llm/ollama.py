import logging
import httpx
from app.config import settings

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
You are an expert assistant for Business Gateway International (BGI) and the JSRS system.
Your task is to answer the user's question using ONLY the provided context snippets.

INSTRUCTIONS:
1. Provide a direct, clear, and concise answer.
2. DO NOT mention "snippets", "context", or "source" in your response.
3. Use the context below to synthesize the answer.
4. If the information is not present, say "I don't have that information."
5. Do not use any outside knowledge.

Context Snippets:
{context}

Question: {question}

Answer:"""


async def ask_ollama(question: str, chunks: list[str]) -> str:
    # Format chunks into a single string
    context_text = ""
    for i, chunk in enumerate(chunks):
        context_text += f"\n--- Snippet {i+1} ---\n{chunk}\n"

    prompt = _PROMPT_TEMPLATE.format(context=context_text, question=question)
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
    }

    url = f"{settings.ollama_url}/api/generate"

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(url, json=payload)

        if response.status_code == 404:
            raise ValueError(
                f"Model '{settings.ollama_model}' not found in Ollama."
            )

        response.raise_for_status()
        
        result = response.json()
        answer = result.get("response", "").strip()
        logger.debug(f"Ollama response: {answer[:80]}...")
        return answer

    except httpx.ConnectError:
        raise ConnectionError(
            f"Could not connect to Ollama at {settings.ollama_url}."
        )
