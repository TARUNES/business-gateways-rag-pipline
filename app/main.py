import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.routes import router
from app.rag.pipeline import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("pypdf").setLevel(logging.ERROR) 
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up – loading document and building index...")
    pipeline.build()
    logger.info("Server ready...")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Business Gateway RAG API",
    description="Answer questions about JSRS using a simple RAG pipeline powered by Ollama",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}


def run():
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
