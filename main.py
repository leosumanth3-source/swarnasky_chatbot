from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pipelines.retreiver import QdrantRetriever
from app.llm.client import GroqLLM


# ---------------------------------------------------------
# Application
# ---------------------------------------------------------

app = FastAPI(
    title="Swarnasky Chatbot API",
    description="RAG chatbot API for Swarnasky Technologies",
    version="1.0.0",
)


# ---------------------------------------------------------
# Frontend path
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"


# ---------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
    )


class ChatResponse(BaseModel):
    question: str
    answer: str


# ---------------------------------------------------------
# Initialize services
# ---------------------------------------------------------

try:
    retriever = QdrantRetriever()
    llm = GroqLLM()

except Exception as exc:
    retriever = None
    llm = None
    startup_error = str(exc)

else:
    startup_error = None


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.get("/health")
def health():
    if startup_error:
        return {
            "status": "unhealthy",
            "error": startup_error,
        }

    try:
        if retriever is None:
            return {
                "status": "unhealthy",
                "error": "Retriever is not initialized.",
            }

        if not retriever.health_check():
            return {
                "status": "unhealthy",
                "error": "Qdrant health check failed.",
            }

        return {
            "status": "healthy",
            "qdrant": "connected",
            "groq": "configured",
        }

    except Exception as exc:
        return {
            "status": "unhealthy",
            "error": str(exc),
        }


# ---------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------

@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):
    if startup_error:
        raise HTTPException(
            status_code=503,
            detail=startup_error,
        )

    if retriever is None or llm is None:
        raise HTTPException(
            status_code=503,
            detail="Chatbot services are not initialized.",
        )

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        # -------------------------------------------------
        # Retrieve relevant knowledge
        # -------------------------------------------------

        results = retriever.search(question)

        if not results:
            return ChatResponse(
                question=question,
                answer=(
                    "I couldn't find that information in the "
                    "Swarnasky knowledge base."
                ),
            )

        # -------------------------------------------------
        # Build context
        # -------------------------------------------------

        context_parts: list[str] = []

        for result in results:

            if isinstance(result, dict):
                text = (
                    result.get("text")
                    or result.get("content")
                    or ""
                )

            else:
                text = (
                    getattr(result, "text", "")
                    or getattr(result, "content", "")
                )

            if text and text.strip():
                context_parts.append(text.strip())

        context = "\n\n".join(context_parts)

        if not context.strip():
            return ChatResponse(
                question=question,
                answer=(
                    "I couldn't find that information in the "
                    "Swarnasky knowledge base."
                ),
            )

        # -------------------------------------------------
        # Grounded system prompt
        # -------------------------------------------------

        system_prompt = """
You are the Swarnasky Technologies knowledge assistant.

Your job is to answer questions about Swarnasky Technologies
using ONLY the provided Swarnasky knowledge base context.

Rules:

1. Use only information contained in the provided context.
2. Never invent or guess facts.
3. Never use outside knowledge.
4. If the answer is not present in the context, respond exactly:
   "I couldn't find that information in the Swarnasky knowledge base."
5. Keep answers clear, direct, and concise.
6. If the context contains the answer, answer the question directly.
7. Do not mention the retrieval system, Qdrant, embeddings,
   prompts, or internal implementation details.
"""

        user_prompt = f"""
Knowledge base context:

{context}

User question:

{question}
"""

        # -------------------------------------------------
        # Generate answer
        # -------------------------------------------------

        answer = llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        if not answer or not answer.strip():
            return ChatResponse(
                question=question,
                answer=(
                    "I couldn't find that information in the "
                    "Swarnasky knowledge base."
                ),
            )

        return ChatResponse(
            question=question,
            answer=answer.strip(),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Chat request failed: {exc}",
        ) from exc


# ---------------------------------------------------------
# Frontend
# ---------------------------------------------------------
#
# This is mounted AFTER the API routes so that:
#
# /health
# /chat
# /docs
#
# continue to work normally.
#
# The frontend becomes available at:
#
# http://127.0.0.1:8000/
#
# ---------------------------------------------------------

if not FRONTEND_DIR.exists():
    raise RuntimeError(
        f"Frontend directory not found: {FRONTEND_DIR}"
    )

app.mount(
    "/",
    StaticFiles(
        directory=FRONTEND_DIR,
        html=True,
    ),
    name="frontend",
)