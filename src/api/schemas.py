from pydantic import BaseModel
from typing import Optional, List


class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class RAGResponse(BaseModel):
    answer: str
    context: Optional[List[str]] = None
    system: str
