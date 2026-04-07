"""Request and response schemas for the API layer."""
from pydantic import BaseModel


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
