"""Parquet serialization and provenance hashing utilities."""
from __future__ import annotations

import hashlib
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import MODEL_FEATURE_COLUMNS


def _dataframe_hash(df: pd.DataFrame) -> str:
    """Compute a stable SHA-256 digest of a DataFrame's content for provenance tracking.

    Only hashes head/tail samples + column names + shape.  Two DataFrames
    that differ only in the middle can produce the same hash.  This is
    intentionally fast (not cryptographically exhaustive) — it exists for
    provenance logging, not correctness assertions.
    """
    h = hashlib.sha256()
    # Hash column names (sorted for stability across column order)
    h.update(",".join(sorted(df.columns)).encode())
    # Hash shape
    h.update(f"{df.shape}".encode())
    # Hash a sample of the actual data (full hash is too slow for 200k+ rows)
    sample = df.head(500).to_csv(index=False).encode()
    h.update(sample)
    tail = df.tail(100).to_csv(index=False).encode()
    h.update(tail)
    return h.hexdigest()


def _save_parquet_with_metadata(
    df: pd.DataFrame,
    path: str | os.PathLike[str],
    *,
    source_hash: str,
    holdout: str,
    n_competitions: int,
) -> None:
    """Save a DataFrame as parquet with provenance metadata in the file footer."""
    table = pa.Table.from_pandas(df)
    existing_meta = table.schema.metadata or {}
    extra = {
        b"prs.source_hash": source_hash.encode(),
        b"prs.holdout": holdout.encode(),
        b"prs.n_competitions": str(n_competitions).encode(),
        b"prs.n_events": str(len(df)).encode(),
        b"prs.features": ",".join(MODEL_FEATURE_COLUMNS).encode(),
    }
    merged = {**existing_meta, **extra}
    table = table.replace_schema_metadata(merged)
    pq.write_table(table, str(path))
