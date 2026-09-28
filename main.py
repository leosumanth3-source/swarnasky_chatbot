from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pipelines.retreiver import QdrantRetriever
from app.llm.client import GroqLLM


app = FastAPI(
    title="Swarnasky Chatbot API",
    description="RAG chatbot API for Swarnasky Technologies",
    version="1.0.0",
)


# ---------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


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

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if startup_error:
        raise HTTPException(
            status_code=503,
            detail=startup_error,
        )

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        # Retrieve relevant knowledge
        results = retriever.search(question)

        # No relevant information found
        if not results:
            return ChatResponse(
                question=question,
                answer=(
                    "I couldn't find that information in the "
                    "Swarnasky knowledge base."
                ),
            )

        # Build context from retrieved chunks
        context_parts = []

        for result in results:
            if isinstance(result, dict):
                text = result.get("text") or result.get("content") or ""
            else:
                text = getattr(result, "text", "") or getattr(
                    result, "content", ""
                )

            if text:
                context_parts.append(text)

        context = "\n\n".join(context_parts)

        if not context.strip():
            return ChatResponse(
                question=question,
                answer=(
                    "I couldn't find that information in the "
                    "Swarnasky knowledge base."
                ),
            )

        # Grounded system prompt
        system_prompt = """
You are the Swarnasky Technologies knowledge assistant.

Answer the user's question using ONLY the provided Swarnasky
knowledge base context.

Rules:
1. Do not invent facts.
2. Do not use outside knowledge.
3. If the answer is not present in the context, say:
   "I couldn't find that information in the Swarnasky knowledge base."
4. Keep answers clear and concise.
5. When the context contains the answer, answer directly.
"""

        user_prompt = f"""
Knowledge base context:

{context}

User question:
{question}
"""

        answer = llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        return ChatResponse(
            question=question,
            answer=answer,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Chat request failed: {exc}",
        ) from exc