"""
Swarnasky Local Embedding Pipeline

Embedding model:
    BAAI/bge-small-en-v1.5

Dimensions:
    384

No API key required.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path(
    "data/processed/chunks.jsonl"
)

OUTPUT_PATH = Path(
    "data/processed/embeddings.jsonl"
)

EMBEDDING_MODEL = (
    "BAAI/bge-small-en-v1.5"
)

EMBEDDING_VERSION = "1.0"

EXPECTED_DIMENSION = 384

BATCH_SIZE = 32


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Input not found: "
            f"{INPUT_PATH}"
        )

    print()
    print(
        "=" * 70
    )

    print(
        "LOADING EMBEDDING MODEL"
    )

    print(
        f"Model: "
        f"{EMBEDDING_MODEL}"
    )

    print(
        "=" * 70
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    chunks = []

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON on "
                    f"line {line_number}: "
                    f"{exc}"
                )

            required_fields = [
                "chunk_id",
                "document_id",
                "chunk_hash",
                "text",
            ]

            for field in required_fields:

                if not record.get(
                    field
                ):

                    raise ValueError(
                        f"Missing required "
                        f"field '{field}' "
                        f"on line "
                        f"{line_number}"
                    )

            chunks.append(
                record
            )

    if not chunks:

        raise RuntimeError(
            "No chunks found."
        )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        f"Chunks loaded: "
        f"{len(texts)}"
    )

    print(
        "Generating embeddings..."
    )

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim != 2:

        raise RuntimeError(
            "Unexpected embedding shape: "
            f"{embeddings.shape}"
        )

    if embeddings.shape[0] != len(
        chunks
    ):

        raise RuntimeError(
            "Embedding count does not "
            "match chunk count."
        )

    if embeddings.shape[1] != (
        EXPECTED_DIMENSION
    ):

        raise RuntimeError(
            "Embedding dimension mismatch. "
            f"Expected "
            f"{EXPECTED_DIMENSION}, "
            f"got "
            f"{embeddings.shape[1]}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = OUTPUT_PATH.with_suffix(
        ".tmp"
    )

    with temp_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for chunk, embedding in zip(
            chunks,
            embeddings,
        ):

            vector = embedding.tolist()

            if not vector:
                raise RuntimeError(
                    "Empty embedding."
                )

            output_record = {
                **chunk,

                "embedding": vector,

                "embedding_model": (
                    EMBEDDING_MODEL
                ),

                "embedding_version": (
                    EMBEDDING_VERSION
                ),

                "embedding_dimensions": (
                    EXPECTED_DIMENSION
                ),
            }

            file.write(
                json.dumps(
                    output_record,
                    ensure_ascii=False,
                )
            )

            file.write("\n")

    temp_path.replace(
        OUTPUT_PATH
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    with OUTPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        output_records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if len(output_records) != len(
        chunks
    ):

        raise RuntimeError(
            "Output record count mismatch."
        )

    ids = [
        record["chunk_id"]
        for record in output_records
    ]

    if len(ids) != len(set(ids)):

        raise RuntimeError(
            "Duplicate chunk IDs detected."
        )

    for record in output_records:

        vector = np.asarray(
            record["embedding"],
            dtype=np.float32,
        )

        if vector.shape != (
            EXPECTED_DIMENSION,
        ):

            raise RuntimeError(
                "Invalid vector dimension."
            )

        if not np.isfinite(
            vector
        ).all():

            raise RuntimeError(
                "Non-finite embedding value."
            )

    print()
    print(
        "=" * 70
    )

    print(
        "EMBEDDING PIPELINE COMPLETED"
    )

    print(
        f"Chunks processed   : "
        f"{len(chunks)}"
    )

    print(
        f"Output records     : "
        f"{len(output_records)}"
    )

    print(
        f"Embedding model    : "
        f"{EMBEDDING_MODEL}"
    )

    print(
        f"Dimensions         : "
        f"{EXPECTED_DIMENSION}"
    )

    print(
        f"Output             : "
        f"{OUTPUT_PATH.resolve()}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()