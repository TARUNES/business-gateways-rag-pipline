import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
    document_path: str = os.getenv("DOCUMENT_PATH", "data/New FAQ AI..pdf")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "400"))


settings = Settings()
