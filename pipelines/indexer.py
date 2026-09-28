
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv(".env")

COLLECTION_NAME = "swarnasky_knowledge"

EMBEDDINGS_PATH = Path(
    "data/processed/embeddings.jsonl"
)

VECTOR_SIZE = 384
DISTANCE = Distance.COSINE

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_VERSION = "1"

BATCH_SIZE = 32

# IMPORTANT:
# True = completely rebuild the collection from the current
# embeddings file.
#
# This is what we want right now because the old collection
# contains 83 old points and the new pipeline contains 122.
REBUILD_COLLECTION = True


# Stable namespace for deterministic point IDs
POINT_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "https://swarnasky.com/qdrant/swarnasky_knowledge",
)


# ============================================================
# ENVIRONMENT
# ============================================================

def get_qdrant_client() -> QdrantClient:

    qdrant_url = os.getenv(
        "QDRANT_URL",
        ""
    ).strip()

    qdrant_api_key = os.getenv(
        "QDRANT_API_KEY",
        ""
    ).strip()

    if not qdrant_url:
        raise RuntimeError(
            "QDRANT_URL is missing from .env"
        )

    if not qdrant_api_key:
        raise RuntimeError(
            "QDRANT_API_KEY is missing from .env"
        )

    print(
        f"Qdrant URL          : {qdrant_url}"
    )

    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
    )

    return client


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings() -> list[dict[str, Any]]:

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: "
            f"{EMBEDDINGS_PATH}"
        )

    records: list[dict[str, Any]] = []

    with EMBEDDINGS_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON on line "
                    f"{line_number}: {exc}"
                ) from exc

            records.append(record)

    return records


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_embeddings(
    records: list[dict[str, Any]],
) -> None:

    if not records:
        raise RuntimeError(
            "No embeddings found."
        )

    for index, record in enumerate(
        records,
        start=1,
    ):

        if "embedding" not in record:
            raise RuntimeError(
                f"Record {index} is missing "
                f"'embedding'."
            )

        embedding = record["embedding"]

        if not isinstance(
            embedding,
            list,
        ):
            raise RuntimeError(
                f"Record {index} embedding "
                f"is not a list."
            )

        if len(embedding) != VECTOR_SIZE:
            raise RuntimeError(
                f"Record {index} has "
                f"{len(embedding)} dimensions. "
                f"Expected {VECTOR_SIZE}."
            )

        if not record.get("chunk_id"):
            raise RuntimeError(
                f"Record {index} is missing "
                f"'chunk_id'."
            )

        if not record.get("chunk_hash"):
            raise RuntimeError(
                f"Record {index} is missing "
                f"'chunk_hash'."
            )

        if not record.get("text"):
            raise RuntimeError(
                f"Record {index} is missing "
                f"'text'."
            )


# ============================================================
# STABLE POINT ID
# ============================================================

def build_point_id(
    record: dict[str, Any],
) -> str:

    """
    Generate a deterministic UUID for each chunk.

    chunk_id is preferred because it is already stable.
    """

    chunk_id = str(
        record.get("chunk_id")
        or record.get("chunk_hash")
        or ""
    ).strip()

    if not chunk_id:
        raise RuntimeError(
            "Cannot create point ID: "
            "missing chunk_id/chunk_hash."
        )

    return str(
        uuid.uuid5(
            POINT_NAMESPACE,
            chunk_id,
        )
    )


# ============================================================
# PAYLOAD
# ============================================================

def build_payload(
    record: dict[str, Any],
) -> dict[str, Any]:

    return {
        "document_id": record.get(
            "document_id",
            "",
        ),

        "chunk_id": record.get(
            "chunk_id",
            "",
        ),

        "chunk_hash": record.get(
            "chunk_hash",
            "",
        ),

        "content_hash": record.get(
            "content_hash",
            "",
        ),

        "chunker_version": record.get(
            "chunker_version",
            "",
        ),

        "url": record.get(
            "url",
            "",
        ),

        "canonical_url": record.get(
            "canonical_url",
            "",
        ),

        "title": record.get(
            "title",
            "",
        ),

        "page_type": record.get(
            "page_type",
            "general",
        ),

        "section": record.get(
            "section",
            "",
        ),

        "section_index": record.get(
            "section_index",
            0,
        ),

        "section_level": record.get(
            "section_level",
            0,
        ),

        "chunk_index": record.get(
            "chunk_index",
            0,
        ),

        "section_chunk_index": record.get(
            "section_chunk_index",
            0,
        ),

        "word_count": record.get(
            "word_count",
            0,
        ),

        "text": record.get(
            "text",
            "",
        ),

        "embedding_model": EMBEDDING_MODEL,

        "embedding_version": EMBEDDING_VERSION,
    }


# ============================================================
# CREATE COLLECTION
# ============================================================

def create_collection(
    client: QdrantClient,
) -> None:

    exists = client.collection_exists(
        COLLECTION_NAME
    )

    if exists:

        print(
            f"Collection already exists: "
            f"{COLLECTION_NAME}"
        )

        if REBUILD_COLLECTION:

            print(
                "Rebuild mode enabled."
            )

            print(
                f"Deleting existing collection: "
                f"{COLLECTION_NAME}"
            )

            client.delete_collection(
                COLLECTION_NAME
            )

            print(
                "Old collection deleted."
            )

        else:

            print(
                "Keeping existing collection."
            )

    if not client.collection_exists(
        COLLECTION_NAME
    ):

        print(
            f"Creating collection: "
            f"{COLLECTION_NAME}"
        )

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=DISTANCE,
            ),
        )

        print(
            "Collection created successfully."
        )

    collection_info = client.get_collection(
        COLLECTION_NAME
    )

    actual_vectors = (
        collection_info.config.params.vectors
    )

    if isinstance(
        actual_vectors,
        dict,
    ):
        # Named-vector configuration.
        # This project uses a single unnamed vector,
        # so this should normally not happen.
        raise RuntimeError(
            "Unexpected named-vector configuration "
            "in Qdrant collection."
        )

    actual_size = actual_vectors.size
    actual_distance = actual_vectors.distance

    if actual_size != VECTOR_SIZE:
        raise RuntimeError(
            f"Collection dimension mismatch: "
            f"{actual_size} != {VECTOR_SIZE}"
        )

    if actual_distance != DISTANCE:
        raise RuntimeError(
            f"Collection distance mismatch: "
            f"{actual_distance} != {DISTANCE}"
        )

    print(
        f"Collection configuration verified: "
        f"{actual_size} dimensions / "
        f"{actual_distance}"
    )


# ============================================================
# INDEX BATCH
# ============================================================

def index_batch(
    client: QdrantClient,
    records: list[dict[str, Any]],
) -> None:

    points: list[PointStruct] = []

    for record in records:

        point_id = build_point_id(
            record
        )

        payload = build_payload(
            record
        )

        vector = record["embedding"]

        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
        )

    if not points:
        return

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )


# ============================================================
# INDEX ALL EMBEDDINGS
# ============================================================

def index_embeddings(
    client: QdrantClient,
    records: list[dict[str, Any]],
) -> None:

    total = len(records)

    for start in range(
        0,
        total,
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            total,
        )

        batch = records[start:end]

        index_batch(
            client,
            batch,
        )

        print(
            f"Indexed {end}/{total} embeddings"
        )


# ============================================================
# VERIFICATION
# ============================================================

def verify_collection(
    client: QdrantClient,
    expected_count: int,
) -> None:

    print()
    print(
        "QDRANT VERIFICATION"
    )
    print(
        "-" * 40
    )

    collection_info = client.get_collection(
        COLLECTION_NAME
    )

    actual_count = (
        collection_info.points_count
    )

    print(
        f"Expected points : {expected_count}"
    )

    print(
        f"Actual points   : {actual_count}"
    )

    if actual_count != expected_count:

        raise RuntimeError(
            "Qdrant point count does not "
            "match the embedding input count."
        )

    # Verify at least one point can be retrieved
    sample = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=1,
        with_payload=True,
        with_vectors=True,
    )

    points = sample[0]

    if not points:
        raise RuntimeError(
            "Collection contains no readable points."
        )

    point = points[0]

    vector = point.vector

    if isinstance(
        vector,
        dict,
    ):
        raise RuntimeError(
            "Unexpected named vector returned."
        )

    if len(vector) != VECTOR_SIZE:
        raise RuntimeError(
            f"Stored vector has "
            f"{len(vector)} dimensions; "
            f"expected {VECTOR_SIZE}."
        )

    print(
        "Vector verification : OK"
    )

    print(
        "Point readability   : OK"
    )

    print()
    print(
        "Qdrant collection verification passed."
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print(
        "=" * 70
    )
    print(
        "SWARNASKY QDRANT INDEXING"
    )
    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Load embeddings
    # --------------------------------------------------------

    records = load_embeddings()

    print(
        f"Embeddings loaded : {len(records)}"
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_embeddings(
        records
    )

    print(
        "Input validation  : OK"
    )

    # --------------------------------------------------------
    # Connect
    # --------------------------------------------------------

    client = get_qdrant_client()

    # Test connection
    client.get_collections()

    print(
        "Qdrant connection  : OK"
    )

    # --------------------------------------------------------
    # Collection
    # --------------------------------------------------------

    create_collection(
        client
    )

    # --------------------------------------------------------
    # Index
    # --------------------------------------------------------

    index_embeddings(
        client,
        records,
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    verify_collection(
        client,
        expected_count=len(records),
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "QDRANT INDEXING COMPLETED"
    )

    print(
        f"Collection         : {COLLECTION_NAME}"
    )

    print(
        f"Points indexed     : {len(records)}"
    )

    print(
        f"Vector dimensions  : {VECTOR_SIZE}"
    )

    print(
        f"Distance           : {DISTANCE}"
    )

    print(
        f"Embedding model    : {EMBEDDING_MODEL}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
