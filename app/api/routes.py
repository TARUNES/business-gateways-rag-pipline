import logging
from fastapi import APIRouter, HTTPException
from app.models import QuestionRequest, AnswerResponse
from app.rag.pipeline import pipeline

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest) -> AnswerResponse:
    if not request.question.strip(): 
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        answer, _ = await pipeline.answer(request.question)
        return AnswerResponse(answer=answer)
    except (ValueError, ConnectionError) as e:
        logger.error(str(e))
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate an answer")
