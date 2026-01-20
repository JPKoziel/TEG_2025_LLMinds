from fastapi import FastAPI, HTTPException
from src.api.schemas import QuestionRequest, RAGResponse
from src.api.services import graph_rag_service, naive_rag_service

app = FastAPI(
    title="LLMinds API",
    description="GraphRAG vs Naive RAG",
    version="1.0.0"
)

@app.post("/rag/graph", response_model=RAGResponse)
def graph_rag_endpoint(req: QuestionRequest):
    try:
        result = graph_rag_service(
            question=req.question
        )
        return {
            "answer": result["answer"],
            "context": result.get("context"),
            "system": "graph_rag"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/naive", response_model=RAGResponse)
def graph_rag_endpoint(req: QuestionRequest):
    try:
        result = naive_rag_service(
            question=req.question
        )
        return {
            "answer": result["answer"],
            "context": result.get("context"),
            "system": "naive_rag"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
