from __future__ import annotations

import os
import time
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pipelines.retreiver import QdrantRetriever
from app.llm.client import GroqLLM
from fastapi.middleware.cors import CORSMiddleware

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Swarnasky Chatbot API",
    description="RAG chatbot API for Swarnasky Technologies",
    version="1.2.0",
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5500,http://127.0.0.1:5500"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
    )


class ChatResponse(BaseModel):
    question: str
    answer: str


# ============================================================
# GLOBAL REQUEST TIMING MIDDLEWARE
# ============================================================

@app.middleware("http")
async def request_timing_middleware(
    request: Request,
    call_next,
):
    """
    Measures the COMPLETE FastAPI request lifecycle.

    This is intentionally separate from the /chat internal
    timings so we can identify any time gap outside the
    chat() function itself.
    """

    request_id = uuid4().hex[:8]

    request_start = time.perf_counter()

    print()
    print("=" * 80)
    print(
        f"[HTTP {request_id}] ENTER"
        f" method={request.method}"
        f" path={request.url.path}"
    )

    try:
        response = await call_next(request)

    except Exception as exc:
        total_time = time.perf_counter() - request_start

        print(
            f"[HTTP {request_id}] EXCEPTION"
            f" total={total_time:.3f}s"
            f" error={exc}"
        )

        print("=" * 80)

        raise

    total_time = time.perf_counter() - request_start

    print(
        f"[HTTP {request_id}] RESPONSE"
        f" status={response.status_code}"
        f" total={total_time:.3f}s"
    )

    print("=" * 80)

    return response


# ============================================================
# STARTUP
# ============================================================

startup_start = time.perf_counter()

try:
    print("[STARTUP] Initializing Qdrant retriever...")

    retriever_start = time.perf_counter()

    retriever = QdrantRetriever()

    retriever_init_time = time.perf_counter() - retriever_start

    print(
        f"[STARTUP] Retriever initialized in "
        f"{retriever_init_time:.3f}s"
    )

    print("[STARTUP] Initializing Groq client...")

    llm_start = time.perf_counter()

    llm = GroqLLM()

    llm_init_time = time.perf_counter() - llm_start

    print(
        f"[STARTUP] Groq client initialized in "
        f"{llm_init_time:.3f}s"
    )

except Exception as exc:
    retriever = None
    llm = None
    startup_error = str(exc)

else:
    startup_error = None


startup_time = time.perf_counter() - startup_start

print(
    f"[STARTUP] Total initialization time: "
    f"{startup_time:.3f}s"
)

if startup_error:
    print(
        f"[STARTUP ERROR] "
        f"{startup_error}"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    health_start = time.perf_counter()

    print("[HEALTH] Handler entered")

    if startup_error:
        return {
            "status": "unhealthy",
            "error": startup_error,
        }

    if retriever is None:
        return {
            "status": "unhealthy",
            "error": "Retriever is not initialized.",
        }

    try:

        qdrant_start = time.perf_counter()

        qdrant_ok = retriever.health_check()

        qdrant_time = time.perf_counter() - qdrant_start

        total_time = time.perf_counter() - health_start

        print(
            f"[HEALTH] "
            f"qdrant={qdrant_time:.3f}s "
            f"total={total_time:.3f}s"
        )

        if not qdrant_ok:
            return {
                "status": "unhealthy",
                "error": "Qdrant health check failed.",
            }

        return {
            "status": "healthy",
            "qdrant": "connected",
            "groq": "configured",
            "response_time_seconds": round(
                total_time,
                3,
            ),
        }

    except Exception as exc:

        total_time = time.perf_counter() - health_start

        print(
            f"[HEALTH ERROR] "
            f"total={total_time:.3f}s "
            f"error={exc}"
        )

        return {
            "status": "unhealthy",
            "error": str(exc),
        }


# ============================================================
# CHAT
# ============================================================

@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):

    handler_start = time.perf_counter()

    print()
    print("-" * 80)
    print("[CHAT] HANDLER ENTERED")

    # --------------------------------------------------------
    # STARTUP CHECK
    # --------------------------------------------------------

    startup_check_start = time.perf_counter()

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

    startup_check_time = (
        time.perf_counter()
        - startup_check_start
    )

    print(
        f"[CHAT TIMING] Startup check: "
        f"{startup_check_time:.6f}s"
    )

    # --------------------------------------------------------
    # VALIDATE QUESTION
    # --------------------------------------------------------

    validation_start = time.perf_counter()

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    validation_time = (
        time.perf_counter()
        - validation_start
    )

    print(
        f"[CHAT TIMING] Validation: "
        f"{validation_time:.6f}s"
    )

    print(
        f"[CHAT] Question: {question}"
    )

    # --------------------------------------------------------
    # STEP 1: RETRIEVAL
    # --------------------------------------------------------

    print("[CHAT] Starting retrieval...")

    retrieval_start = time.perf_counter()

    try:

        results = retriever.search(
            question
        )

    except Exception as exc:

        retrieval_time = (
            time.perf_counter()
            - retrieval_start
        )

        print(
            f"[RETRIEVAL ERROR] "
            f"time={retrieval_time:.3f}s "
            f"error={exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Retrieval failed: {exc}",
        ) from exc

    retrieval_time = (
        time.perf_counter()
        - retrieval_start
    )

    result_count = (
        len(results)
        if results
        else 0
    )

    print(
        f"[CHAT TIMING] Retrieval: "
        f"{retrieval_time:.3f}s"
    )

    print(
        f"[RETRIEVAL] Results: "
        f"{result_count}"
    )

    # --------------------------------------------------------
    # NO RESULTS
    # --------------------------------------------------------

    if not results:

        total_time = (
            time.perf_counter()
            - handler_start
        )

        print(
            f"[CHAT TIMING] Handler total: "
            f"{total_time:.3f}s"
        )

        print("[CHAT] Returning no-results response")
        print("-" * 80)

        return ChatResponse(
            question=question,
            answer=(
                "I couldn't find that information in "
                "the Swarnasky knowledge base."
            ),
        )

    # --------------------------------------------------------
    # STEP 2: BUILD CONTEXT
    # --------------------------------------------------------

    print("[CHAT] Building context...")

    context_start = time.perf_counter()

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
                or getattr(
                    result,
                    "content",
                    "",
                )
            )

        if text and text.strip():
            context_parts.append(
                text.strip()
            )

    context = "\n\n".join(
        context_parts
    )

    context_time = (
        time.perf_counter()
        - context_start
    )

    context_chars = len(context)
    context_words = len(
        context.split()
    )

    print(
        f"[CHAT TIMING] Context: "
        f"{context_time:.3f}s"
    )

    print(
        f"[CONTEXT] "
        f"chunks={len(context_parts)} "
        f"chars={context_chars} "
        f"words={context_words}"
    )

    # --------------------------------------------------------
    # NO USABLE CONTEXT
    # --------------------------------------------------------

    if not context.strip():

        total_time = (
            time.perf_counter()
            - handler_start
        )

        print(
            f"[CHAT TIMING] Handler total: "
            f"{total_time:.3f}s"
        )

        print(
            "[CHAT] Returning no-context response"
        )

        print("-" * 80)

        return ChatResponse(
            question=question,
            answer=(
                "I couldn't find that information in "
                "the Swarnasky knowledge base."
            ),
        )

    # --------------------------------------------------------
    # STEP 3: SYSTEM PROMPT
    # --------------------------------------------------------

    prompt_start = time.perf_counter()

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
""".strip()

    user_prompt = f"""
Knowledge base context:

{context}

User question:

{question}
""".strip()

    prompt_time = (
        time.perf_counter()
        - prompt_start
    )

    print(
        f"[CHAT TIMING] Prompt construction: "
        f"{prompt_time:.6f}s"
    )

    # --------------------------------------------------------
    # STEP 4: GROQ
    # --------------------------------------------------------

    print("[CHAT] Starting Groq request...")

    llm_start = time.perf_counter()

    try:

        answer = llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    except Exception as exc:

        llm_time = (
            time.perf_counter()
            - llm_start
        )

        handler_time = (
            time.perf_counter()
            - handler_start
        )

        print(
            f"[LLM ERROR] "
            f"time={llm_time:.3f}s "
            f"error={exc}"
        )

        print(
            f"[CHAT TIMING] Handler before error: "
            f"{handler_time:.3f}s"
        )

        print("-" * 80)

        raise HTTPException(
            status_code=500,
            detail=f"LLM request failed: {exc}",
        ) from exc

    llm_time = (
        time.perf_counter()
        - llm_start
    )

    print(
        f"[CHAT TIMING] Groq: "
        f"{llm_time:.3f}s"
    )

    # --------------------------------------------------------
    # EMPTY RESPONSE
    # --------------------------------------------------------

    if not answer or not answer.strip():

        handler_time = (
            time.perf_counter()
            - handler_start
        )

        print(
            f"[CHAT TIMING] Handler total: "
            f"{handler_time:.3f}s"
        )

        print(
            "[CHAT] Returning empty-answer fallback"
        )

        print("-" * 80)

        return ChatResponse(
            question=question,
            answer=(
                "I couldn't find that information in "
                "the Swarnasky knowledge base."
            ),
        )

    # --------------------------------------------------------
    # RESPONSE CONSTRUCTION
    # --------------------------------------------------------

    response_start = time.perf_counter()

    final_answer = answer.strip()

    response = ChatResponse(
        question=question,
        answer=final_answer,
    )

    response_time = (
        time.perf_counter()
        - response_start
    )

    # --------------------------------------------------------
    # FINAL TIMING
    # --------------------------------------------------------

    handler_total = (
        time.perf_counter()
        - handler_start
    )

    print(
        f"[CHAT TIMING] Response construction: "
        f"{response_time:.6f}s"
    )

    print(
        f"[CHAT TIMING] Handler total: "
        f"{handler_total:.3f}s"
    )

    print(
        f"[CHAT BREAKDOWN] "
        f"retrieval={retrieval_time:.3f}s | "
        f"context={context_time:.3f}s | "
        f"prompt={prompt_time:.6f}s | "
        f"groq={llm_time:.3f}s | "
        f"response={response_time:.6f}s"
    )

    print(
        "[CHAT] HANDLER RETURNING"
    )

    print("-" * 80)

    return response


# ============================================================
# FRONTEND
# ============================================================

if not FRONTEND_DIR.exists():
    raise RuntimeError(
        f"Frontend directory not found: "
        f"{FRONTEND_DIR}"
    )


app.mount(
    "/",
    StaticFiles(
        directory=FRONTEND_DIR,
        html=True,
    ),
    name="frontend",
)