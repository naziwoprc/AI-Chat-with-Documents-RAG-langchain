from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ragEngine import RAGEngine
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(
    title="AI Chat with Documents API",
    version="1.2.0",
    description="Production-ready RAG API with confidence gating and reranking.",
)


# Request model
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


# Response model
class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]


engine = RAGEngine(
    pdf_path="Data/liver_disease.pdf",
    persist_dir="./chroma_db_langchain",
    api_key=os.getenv("NVIDIA_API_KEY"),
)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: ChatRequest):
    try:
        result = engine.answer(request.question)

        if "Sources:" in result:
            answer_text, sources_test = result.split("Sources:")
            sources = [
                line.strip("- ").strip()
                for line in sources_test.strip().split("\n")
                if line.strip()
            ]
        else:
            answer_text = result
            sources = []
        return AnswerResponse(answer=answer_text.strip(), source=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
