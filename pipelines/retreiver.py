"""
Swarnasky Production Retriever

Responsibilities:
- Embed user queries locally
- Retrieve candidates from Qdrant
- Detect broad service-intent queries
- Add deterministic service-page coverage
- Apply semantic + lexical reranking
- Remove exact duplicates
- Enforce document diversity
- Apply MMR using stored vectors
- Return clean retrieval results
- Provide a CLI for testing

Embedding model:
    BAAI/bge-small-en-v1.5

Vector size:
    384

Qdrant collection:
    swarnasky_knowledge
"""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)

from sentence_transformers import SentenceTransformer


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(".env")


QDRANT_URL = os.getenv("QDRANT_URL", "").strip().strip('"').strip("'")
QDRANT_API_KEY = (
    os.getenv("QDRANT_API_KEY", "")
    .strip()
    .strip('"')
    .strip("'")
)

COLLECTION_NAME = "swarnasky_knowledge"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384


# ============================================================
# RETRIEVAL SETTINGS
# ============================================================

DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_K = 20

# Higher = prioritize semantic relevance.
# Lower = prioritize diversity.
MMR_LAMBDA = 0.80

# Normal queries can have up to two chunks from one document.
DEFAULT_MAX_PER_DOCUMENT = 2

# Broad service queries intentionally use one chunk per
# service page to maximize coverage.
SERVICE_MAX_PER_DOCUMENT = 1


# ============================================================
# SWARNASKY SERVICE URLS
# ============================================================

SERVICES_URL = (
    "https://www.swarnasky.com/services"
)

SERVICE_URLS = [
    "https://www.swarnasky.com/services",
    "https://www.swarnasky.com/services/claims-workflow-automation",
    "https://www.swarnasky.com/services/cloud-core-modernization",
    "https://www.swarnasky.com/services/connected-risk-solutions",
    "https://www.swarnasky.com/services/digital-insurance-platforms",
    "https://www.swarnasky.com/services/immersive-insurance-experiences",
    "https://www.swarnasky.com/services/insurance-intelligence",
]

NORMALIZED_SERVICE_URLS = {
    url.lower().rstrip("/")
    for url in SERVICE_URLS
}

NORMALIZED_SERVICES_URL = (
    SERVICES_URL.lower().rstrip("/")
)


# ============================================================
# SERVICE INTENT TERMS
# ============================================================

SERVICE_QUERY_PATTERNS = [
    r"\bwhat services\b",
    r"\bwhich services\b",
    r"\bservices do you\b",
    r"\bservices does\b",
    r"\bwhat solutions\b",
    r"\bwhich solutions\b",
    r"\bsolutions do you\b",
    r"\bwhat do you offer\b",
    r"\bwhat does .* offer\b",
    r"\bwhat can .* help\b",
    r"\bwhat capabilities\b",
    r"\bwhat offerings\b",
    r"\bofferings\b",
    r"\bservices offered\b",
    r"\bsolutions offered\b",
    r"\bservice portfolio\b",
    r"\bservice offering\b",
]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for lexical comparison.

    This does NOT modify the original text stored in results.
    """

    if not text:
        return ""

    text = text.lower()

    # Keep alphanumeric characters and spaces.
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize(text: str) -> set[str]:
    """
    Return normalized lexical tokens.
    """

    normalized = normalize_text(text)

    if not normalized:
        return set()

    return set(normalized.split())


# ============================================================
# QUERY INTENT
# ============================================================

def is_broad_service_query(query: str) -> bool:
    """
    Detect broad questions asking what Swarnasky offers.

    Examples:
        What services does Swarnasky provide?
        What solutions do you offer?
        What can Swarnasky help with?
    """

    normalized = normalize_text(query)

    if not normalized:
        return False

    for pattern in SERVICE_QUERY_PATTERNS:
        if re.search(pattern, normalized):
            return True

    # Additional lightweight service-intent fallback.
    service_terms = {
        "service",
        "services",
        "solution",
        "solutions",
        "offering",
        "offerings",
        "capabilities",
    }

    tokens = set(normalized.split())

    if tokens.intersection(service_terms):
        broad_phrases = {
            "what",
            "which",
            "offer",
            "provide",
            "provides",
            "do",
            "does",
        }

        if tokens.intersection(broad_phrases):
            return True

    return False


# ============================================================
# SERVICE DOCUMENT DETECTION
# ============================================================

def is_service_document(
    result: dict[str, Any],
) -> bool:
    """
    Return True only for known Swarnasky service URLs.
    """

    url = str(
        result.get("url", "")
    ).lower().rstrip("/")

    return url in NORMALIZED_SERVICE_URLS


# ============================================================
# LEXICAL RELEVANCE
# ============================================================

def lexical_overlap(
    query: str,
    result: dict[str, Any],
) -> float:
    """
    Calculate lightweight lexical overlap between the query
    and the result's title/section/text.

    This supplements semantic similarity.
    """

    query_tokens = tokenize(query)

    if not query_tokens:
        return 0.0

    searchable_text = " ".join(
        [
            str(result.get("title", "")),
            str(result.get("section", "")),
            str(result.get("text", "")),
        ]
    )

    result_tokens = tokenize(searchable_text)

    if not result_tokens:
        return 0.0

    overlap = query_tokens.intersection(result_tokens)

    return len(overlap) / len(query_tokens)


# ============================================================
# RERANKING
# ============================================================

def calculate_rerank_score(
    query: str,
    result: dict[str, Any],
    broad_service_query: bool = False,
) -> float:
    """
    Calculate final relevance score.

    Components:
    - semantic similarity
    - lexical overlap
    - service-specific intent boosts
    """

    vector_score = float(
        result.get("score", 0.0) or 0.0
    )

    lexical_score = lexical_overlap(
        query,
        result,
    )

    # Base score.
    score = (
        vector_score * 0.75
        + lexical_score * 0.25
    )

    url = str(
        result.get("url", "")
    ).lower().rstrip("/")

    title = normalize_text(
        str(result.get("title", ""))
    )

    # --------------------------------------------------------
    # Broad service-query boosts
    # --------------------------------------------------------

    if broad_service_query:

        # Main services overview.
        if url == NORMALIZED_SERVICES_URL:
            score += 0.15

        # Individual service pages.
        elif url in NORMALIZED_SERVICE_URLS:
            score += 0.08

        # Helpful title signals.
        if (
            "services" in title
            or "solutions" in title
        ):
            score += 0.03

    # --------------------------------------------------------
    # General service-document boost
    # --------------------------------------------------------

    if is_service_document(result):
        score += 0.02

    return float(score)


# ============================================================
# RESULT DEDUPLICATION
# ============================================================

def remove_exact_duplicates(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Remove duplicate chunks using chunk_id first and text second.
    """

    seen_chunk_ids: set[str] = set()
    seen_text: set[str] = set()

    unique_results: list[dict[str, Any]] = []

    for result in results:

        chunk_id = str(
            result.get("chunk_id", "")
        ).strip()

        text = normalize_text(
            str(result.get("text", ""))
        )

        # Prefer stable chunk ID.
        if chunk_id:
            if chunk_id in seen_chunk_ids:
                continue

            seen_chunk_ids.add(chunk_id)

        # Also prevent duplicated text.
        if text:
            if text in seen_text:
                continue

            seen_text.add(text)

        unique_results.append(result)

    return unique_results


# ============================================================
# QDRANT RETRIEVER
# ============================================================

class QdrantRetriever:
    """
    Production-oriented Qdrant retriever.
    """

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        model_name: str = EMBEDDING_MODEL,
        top_k: int = DEFAULT_TOP_K,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        mmr_lambda: float = MMR_LAMBDA,
        max_per_document: int = DEFAULT_MAX_PER_DOCUMENT,
    ) -> None:

        if not QDRANT_URL:
            raise RuntimeError(
                "QDRANT_URL is missing from .env"
            )

        if not QDRANT_API_KEY:
            raise RuntimeError(
                "QDRANT_API_KEY is missing from .env"
            )

        if not (
            QDRANT_URL.startswith("https://")
            or QDRANT_URL.startswith("http://")
        ):
            raise RuntimeError(
                "QDRANT_URL must start with "
                "http:// or https://"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        if candidate_k < top_k:
            raise ValueError(
                "candidate_k must be >= top_k"
            )

        if not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError(
                "mmr_lambda must be between 0 and 1"
            )

        self.collection_name = collection_name
        self.model_name = model_name
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.mmr_lambda = mmr_lambda
        self.max_per_document = max_per_document

        print(
            f"Loading embedding model: "
            f"{self.model_name}"
        )

        self.model = SentenceTransformer(
            self.model_name
        )

        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )

    # ========================================================
    # HEALTH
    # ========================================================

    def health_check(self) -> bool:
        """
        Verify Qdrant connectivity and collection existence.
        """

        try:
            info = self.client.get_collection(
                self.collection_name
            )

            points_count = getattr(
                info,
                "points_count",
                None,
            )

            print(
                "Qdrant server      : OK"
            )

            print(
                f"Collection          : "
                f"{self.collection_name}"
            )

            if points_count is not None:
                print(
                    f"Points              : "
                    f"{points_count}"
                )

            return True

        except Exception as exc:

            print(
                f"Qdrant health check failed: "
                f"{exc}"
            )

            return False

    # ========================================================
    # QUERY EMBEDDING
    # ========================================================

    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        """
        Generate normalized query embedding.
        """

        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        vector = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        vector = np.asarray(
            vector,
            dtype=np.float32,
        )

        if vector.ndim != 1:
            raise RuntimeError(
                "Query embedding has unexpected shape: "
                f"{vector.shape}"
            )

        if vector.shape[0] != EMBEDDING_DIMENSION:
            raise RuntimeError(
                "Embedding dimension mismatch. "
                f"Expected {EMBEDDING_DIMENSION}, "
                f"got {vector.shape[0]}"
            )

        return vector.tolist()

    # ========================================================
    # POINT FORMATTING
    # ========================================================

    def format_point(
        self,
        point: Any,
    ) -> dict[str, Any]:
        """
        Convert a Qdrant point/record into a stable result dict.

        Handles both:
        - ScoredPoint from query_points()
        - Record from scroll()
        """

        payload = (
            point.payload
            if getattr(point, "payload", None)
            else {}
        )

        score = getattr(
            point,
            "score",
            0.0,
        )

        if score is None:
            score = 0.0

        vector = getattr(
            point,
            "vector",
            None,
        )

        return {
            "id": str(
                getattr(point, "id", "")
            ),

            "score": float(score),

            "chunk_id": str(
                payload.get("chunk_id", "")
            ),

            "document_id": str(
                payload.get("document_id", "")
            ),

            "chunk_hash": str(
                payload.get("chunk_hash", "")
            ),

            "title": str(
                payload.get("title", "")
            ),

            "url": str(
                payload.get("url", "")
            ),

            "canonical_url": str(
                payload.get("canonical_url", "")
            ),

            "page_type": str(
                payload.get("page_type", "")
            ),

            "section": str(
                payload.get("section", "")
            ),

            "section_index": payload.get(
                "section_index"
            ),

            "section_level": payload.get(
                "section_level"
            ),

            "chunk_index": payload.get(
                "chunk_index"
            ),

            "section_chunk_index": payload.get(
                "section_chunk_index"
            ),

            "text": str(
                payload.get("text", "")
            ),

            "word_count": payload.get(
                "word_count",
                0,
            ),

            "embedding_model": str(
                payload.get(
                    "embedding_model",
                    "",
                )
            ),

            "embedding_version": str(
                payload.get(
                    "embedding_version",
                    "",
                )
            ),

            # Used internally by MMR.
            "_vector": vector,
        }

    # ========================================================
    # SEMANTIC SEARCH
    # ========================================================

    def semantic_search(
        self,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """
        Perform semantic vector search with detailed timing.
        """

        import time

        total_start = time.perf_counter()

        # --------------------------------------------------------
        # Query embedding
        # --------------------------------------------------------

        start = time.perf_counter()

        query_vector = self.embed_query(
            query
        )

        embedding_time = (
            time.perf_counter() - start
        )

        print(
            f"[RETRIEVER TIMING] "
            f"Query embedding: {embedding_time:.3f}s"
        )

        # --------------------------------------------------------
        # Qdrant request
        # --------------------------------------------------------

        start = time.perf_counter()

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=True,
        )

        qdrant_time = (
            time.perf_counter() - start
        )

        print(
            f"[RETRIEVER TIMING] "
            f"Qdrant query: {qdrant_time:.3f}s"
        )

        # --------------------------------------------------------
        # Format results
        # --------------------------------------------------------

        start = time.perf_counter()

        points = getattr(
            response,
            "points",
            response,
        )

        results: list[dict[str, Any]] = []

        for point in points:
            result = self.format_point(point)
            results.append(result)

        formatting_time = (
            time.perf_counter() - start
        )

        total_time = (
            time.perf_counter() - total_start
        )

        print(
            f"[RETRIEVER TIMING] "
            f"Formatting: {formatting_time:.3f}s"
        )

        print(
            f"[RETRIEVER TIMING] "
            f"Semantic search total: {total_time:.3f}s"
        )

        return results
    # ========================================================
    # SERVICE COVERAGE
    # ========================================================

    def get_services_anchor(
        self,
        limit_per_service: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Retrieve representative chunks from every known
        service page.

        This deliberately uses URL metadata rather than
        semantic similarity.

        Broad service queries need coverage across the
        service portfolio.
        """

        results: list[dict[str, Any]] = []

        for service_url in SERVICE_URLS:

            service_filter = Filter(
                must=[
                    FieldCondition(
                        key="url",
                        match=MatchValue(
                            value=service_url
                        ),
                    )
                ]
            )

            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=service_filter,
                limit=limit_per_service,
                with_payload=True,
                with_vectors=True,
            )

            for point in points:

                result = self.format_point(
                    point
                )

                # Scroll() returns Record objects,
                # not ScoredPoint objects.
                result["score"] = 0.0

                results.append(result)

        return results

    # ========================================================
    # SERVICE REPRESENTATIVE SELECTION
    # ========================================================

    def select_service_representatives(
        self,
        results: list[dict[str, Any]],
        query: str,
        max_per_service: int = 1,
    ) -> list[dict[str, Any]]:
        """
        For broad service questions, select the strongest
        chunk from each service page.

        This prevents one service page from dominating
        the final answer.
        """

        grouped: dict[
            str,
            list[dict[str, Any]]
        ] = {}

        for result in results:

            url = str(
                result.get("url", "")
            ).lower().rstrip("/")

            if url not in NORMALIZED_SERVICE_URLS:
                continue

            grouped.setdefault(
                url,
                [],
            ).append(result)

        selected: list[dict[str, Any]] = []

        for service_url in SERVICE_URLS:

            normalized_url = (
                service_url.lower().rstrip("/")
            )

            service_results = grouped.get(
                normalized_url,
                [],
            )

            if not service_results:
                continue

            service_results.sort(
                key=lambda item: float(
                    item.get(
                        "rerank_score",
                        item.get(
                            "score",
                            0.0,
                        ),
                    )
                    or 0.0
                ),
                reverse=True,
            )

            selected.extend(
                service_results[
                    :max_per_service
                ]
            )

        return selected

    # ========================================================
    # DOCUMENT DIVERSITY
    # ========================================================

    def enforce_document_limit(
        self,
        results: list[dict[str, Any]],
        max_per_document: int,
    ) -> list[dict[str, Any]]:
        """
        Limit the number of results from each document.
        """

        counts: dict[str, int] = {}

        selected: list[dict[str, Any]] = []

        for result in results:

            document_id = str(
                result.get(
                    "document_id",
                    "",
                )
            )

            # Fallback to URL when document_id is missing.
            if not document_id:
                document_id = str(
                    result.get(
                        "url",
                        "",
                    )
                )

            current_count = counts.get(
                document_id,
                0,
            )

            if current_count >= max_per_document:
                continue

            counts[document_id] = (
                current_count + 1
            )

            selected.append(result)

        return selected

    # ========================================================
    # COSINE SIMILARITY
    # ========================================================

    @staticmethod
    def cosine_similarity(
        vector_a: Any,
        vector_b: Any,
    ) -> float:
        """
        Calculate cosine similarity between vectors.
        """

        if vector_a is None or vector_b is None:
            return 0.0

        a = np.asarray(
            vector_a,
            dtype=np.float32,
        )

        b = np.asarray(
            vector_b,
            dtype=np.float32,
        )

        if a.ndim != 1 or b.ndim != 1:
            return 0.0

        if a.shape != b.shape:
            return 0.0

        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)

        if a_norm == 0.0 or b_norm == 0.0:
            return 0.0

        return float(
            np.dot(a, b)
            / (a_norm * b_norm)
        )

    # ========================================================
    # MMR
    # ========================================================

    def mmr_select(
        self,
        results: list[dict[str, Any]],
        top_k: int,
        max_per_document: int,
    ) -> list[dict[str, Any]]:
        """
        Maximal Marginal Relevance selection.

        Balances:
            relevance
            +
            diversity
        """

        if not results:
            return []

        candidates = list(results)

        selected: list[dict[str, Any]] = []

        document_counts: dict[str, int] = {}

        while candidates and len(selected) < top_k:

            best_candidate = None
            best_mmr_score = float("-inf")

            for candidate in candidates:

                document_id = str(
                    candidate.get(
                        "document_id",
                        "",
                    )
                )

                if not document_id:
                    document_id = str(
                        candidate.get(
                            "url",
                            "",
                        )
                    )

                if (
                    document_counts.get(
                        document_id,
                        0,
                    )
                    >= max_per_document
                ):
                    continue

                relevance = float(
                    candidate.get(
                        "rerank_score",
                        candidate.get(
                            "score",
                            0.0,
                        ),
                    )
                    or 0.0
                )

                if not selected:

                    diversity_penalty = 0.0

                else:

                    similarities = [
                        self.cosine_similarity(
                            candidate.get(
                                "_vector"
                            ),
                            chosen.get(
                                "_vector"
                            ),
                        )
                        for chosen in selected
                    ]

                    diversity_penalty = max(
                        similarities
                    ) if similarities else 0.0

                mmr_score = (
                    self.mmr_lambda
                    * relevance
                    - (
                        1.0
                        - self.mmr_lambda
                    )
                    * diversity_penalty
                )

                if (
                    mmr_score
                    > best_mmr_score
                ):
                    best_mmr_score = (
                        mmr_score
                    )

                    best_candidate = (
                        candidate
                    )

            if best_candidate is None:
                break

            document_id = str(
                best_candidate.get(
                    "document_id",
                    "",
                )
            )

            if not document_id:
                document_id = str(
                    best_candidate.get(
                        "url",
                        "",
                    )
                )

            document_counts[
                document_id
            ] = (
                document_counts.get(
                    document_id,
                    0,
                )
                + 1
            )

            best_candidate[
                "mmr_score"
            ] = float(
                best_mmr_score
            )

            selected.append(
                best_candidate
            )

            candidates.remove(
                best_candidate
            )

        return selected

    # ========================================================
    # MAIN SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int | None = None,
        candidate_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Complete retrieval pipeline.
        """

        query = query.strip()

        if not query:
            return []

        final_top_k = (
            top_k
            if top_k is not None
            else self.top_k
        )

        final_candidate_k = (
            candidate_k
            if candidate_k is not None
            else self.candidate_k
        )

        broad_service_query = (
            is_broad_service_query(
                query
            )
        )

        # ----------------------------------------------------
        # 1. Semantic retrieval
        # ----------------------------------------------------

        semantic_results = (
            self.semantic_search(
                query=query,
                limit=final_candidate_k,
            )
        )

        # ----------------------------------------------------
        # 2. Deterministic service coverage
        # ----------------------------------------------------

        if broad_service_query:

            service_results = (
                self.get_services_anchor(
                    limit_per_service=3
                )
            )

            combined_results = (
                semantic_results
                + service_results
            )

        else:

            combined_results = (
                semantic_results
            )

        # ----------------------------------------------------
        # 3. Exact deduplication
        # ----------------------------------------------------

        combined_results = (
            remove_exact_duplicates(
                combined_results
            )
        )

        # ----------------------------------------------------
        # 4. Intent-aware filtering
        # ----------------------------------------------------

        if broad_service_query:

            service_results = [
                result
                for result in combined_results
                if is_service_document(
                    result
                )
            ]

            # If service retrieval worked, stay inside
            # the service knowledge area.
            if service_results:
                combined_results = (
                    service_results
                )

        # ----------------------------------------------------
        # 5. Reranking
        # ----------------------------------------------------

        for result in combined_results:

            result[
                "rerank_score"
            ] = calculate_rerank_score(
                query=query,
                result=result,
                broad_service_query=(
                    broad_service_query
                ),
            )

        combined_results.sort(
            key=lambda result: float(
                result.get(
                    "rerank_score",
                    0.0,
                )
                or 0.0
            ),
            reverse=True,
        )

        # ----------------------------------------------------
        # 6. Broad service coverage
        # ----------------------------------------------------

        if broad_service_query:

            # Select at most one representative
            # chunk from every service page.
            coverage_results = (
                self.select_service_representatives(
                    combined_results,
                    query=query,
                    max_per_service=1,
                )
            )

            # Coverage results are already diverse.
            # Sort them by rerank relevance.
            coverage_results.sort(
                key=lambda result: float(
                    result.get(
                        "rerank_score",
                        0.0,
                    )
                    or 0.0
                ),
                reverse=True,
            )

            # Use MMR only after coverage has been
            # established.
            final_results = (
                self.mmr_select(
                    results=coverage_results,
                    top_k=final_top_k,
                    max_per_document=(
                        SERVICE_MAX_PER_DOCUMENT
                    ),
                )
            )

        else:

            # ------------------------------------------------
            # Normal query retrieval
            # ------------------------------------------------

            diverse_candidates = (
                self.enforce_document_limit(
                    results=combined_results,
                    max_per_document=(
                        self.max_per_document
                    ),
                )
            )

            final_results = (
                self.mmr_select(
                    results=diverse_candidates,
                    top_k=final_top_k,
                    max_per_document=(
                        self.max_per_document
                    ),
                )
            )

        # ----------------------------------------------------
        # 7. Remove internal vectors
        # ----------------------------------------------------

        for result in final_results:
            result.pop(
                "_vector",
                None,
            )

        return final_results


# ============================================================
# CLI DISPLAY
# ============================================================

def print_results(
    query: str,
    results: list[dict[str, Any]],
) -> None:

    print()
    print("=" * 80)
    print("RETRIEVAL RESULTS")
    print("=" * 80)

    print(
        f"Query       : {query}"
    )

    print(
        f"Results     : {len(results)}"
    )

    print()

    if not results:

        print(
            "No relevant results found."
        )

        print("=" * 80)

        return

    for index, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"Rank {index}"
        )

        print(
            f"Vector score: "
            f"{float(result.get('score', 0.0)):.4f}"
        )

        print(
            f"Rerank score: "
            f"{float(result.get('rerank_score', 0.0)):.4f}"
        )

        print(
            f"MMR score   : "
            f"{float(result.get('mmr_score', 0.0)):.4f}"
        )

        print(
            f"Title: "
            f"{result.get('title', '')}"
        )

        print(
            f"Section: "
            f"{result.get('section', '')}"
        )

        print(
            f"URL: "
            f"{result.get('url', '')}"
        )

        print(
            "Text:"
        )

        print(
            result.get(
                "text",
                "",
            )
        )

        print()
        print("-" * 80)
        print()


# ============================================================
# CLI
# ============================================================

def main() -> None:

    print()
    print("=" * 80)
    print("SWARNASKY QDRANT RETRIEVER")
    print("=" * 80)
    print()

    retriever = QdrantRetriever()

    if not retriever.health_check():

        raise SystemExit(
            "Qdrant health check failed."
        )

    print()

    while True:

        try:

            query = input(
                "Enter your question "
                "(or 'exit'): "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):

            print()
            break

        if not query:
            continue

        if query.lower() in {
            "exit",
            "quit",
            "q",
        }:
            break

        try:

            results = retriever.search(
                query=query,
                top_k=DEFAULT_TOP_K,
                candidate_k=DEFAULT_CANDIDATE_K,
            )

            print_results(
                query,
                results,
            )

        except Exception as exc:

            print()
            print(
                "Retrieval failed:"
            )

            print(
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            print()


if __name__ == "__main__":
    main()