"""FAISS retrieval over the medical knowledge base.

FAISS is a library for fast nearest-neighbor search over vectors. This
project uses IndexFlatL2: it stores every document vector and, for a query
vector, finds the documents with the smallest L2 (Euclidean) distance.

A smaller L2 distance means the query vector is closer to that document
vector under this search setup. It is not a probability, not a medical
confidence score, and it does not prove that the passage is correct or
relevant to a particular image.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np

from src.embeddings import SentenceTransformerEncoder, TextEncoder

REQUIRED_FIELDS = ("id", "topic", "text")


class KnowledgeBaseError(ValueError):
    """Raised when the knowledge base cannot be used."""


class RetrievalError(ValueError):
    """Raised when a retrieval request cannot be completed."""


@dataclass
class RetrievedDocument:
    id: str
    topic: str
    text: str
    source_title: str | None
    source_url: str | None
    source_type: str | None
    publication_year: int | str | None
    license: str | None
    status: str | None
    score_l2_distance: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "text": self.text,
            "source_title": self.source_title,
            "source_url": self.source_url,
            "source_type": self.source_type,
            "publication_year": self.publication_year,
            "license": self.license,
            "status": self.status,
            "score_l2_distance": self.score_l2_distance,
            "rank": self.rank,
        }


def load_knowledge_base(path: Path | str) -> list[dict[str, Any]]:
    """Load and validate the JSON knowledge base."""
    kb_path = Path(path)
    if not kb_path.exists():
        raise KnowledgeBaseError(f"Knowledge base not found: {kb_path}")
    try:
        with kb_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise KnowledgeBaseError(f"Knowledge base is not valid JSON: {exc}") from exc

    if not isinstance(data, list):
        raise KnowledgeBaseError("Knowledge base must be a JSON list of documents.")
    if not data:
        raise KnowledgeBaseError("Knowledge base is empty.")

    documents = [validate_document(item, index) for index, item in enumerate(data)]
    ids = [doc["id"] for doc in documents]
    duplicates = {doc_id for doc_id in ids if ids.count(doc_id) > 1}
    if duplicates:
        raise KnowledgeBaseError(f"Duplicate document ids: {sorted(duplicates)}")
    return documents


def validate_document(item: Any, index: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise KnowledgeBaseError(f"Document at index {index} is not an object.")
    for field in REQUIRED_FIELDS:
        value = item.get(field)
        if value is None or str(value).strip() == "":
            raise KnowledgeBaseError(f"Document at index {index} is missing required field '{field}'.")
    return item


class Retriever:
    """Embed documents, store them in FAISS, and retrieve top-k passages."""

    def __init__(
        self,
        documents: Sequence[dict[str, Any]],
        encoder: TextEncoder,
        embeddings: np.ndarray | None = None,
    ):
        if not documents:
            raise RetrievalError("Cannot build a retriever from an empty document list.")
        self.documents = [validate_document(doc, i) for i, doc in enumerate(documents)]
        self.encoder = encoder
        if embeddings is None:
            texts = [str(doc["text"]) for doc in self.documents]
            embeddings = encoder.encode_texts(texts)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        if self.embeddings.ndim != 2:
            raise RetrievalError("Embeddings must be a 2-D array of shape (n_docs, dimension).")
        if self.embeddings.shape[0] != len(self.documents):
            raise RetrievalError("Number of embeddings must match number of documents.")
        if self.embeddings.shape[1] < 1:
            raise RetrievalError("Embedding dimension must be at least 1.")
        self.dimension = int(self.embeddings.shape[1])
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.embeddings)

    @classmethod
    def from_knowledge_base(
        cls,
        path: Path | str,
        encoder: TextEncoder | None = None,
        embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> "Retriever":
        documents = load_knowledge_base(path)
        if encoder is None:
            encoder = SentenceTransformerEncoder(embedding_model_id)
        return cls(documents, encoder)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedDocument]:
        if query is None or str(query).strip() == "":
            raise RetrievalError("Query must be a non-empty string.")
        if top_k < 1:
            raise RetrievalError("top_k must be >= 1.")
        if self.index.ntotal == 0:
            raise RetrievalError("The FAISS index contains no documents.")

        k = min(int(top_k), int(self.index.ntotal))
        query_vector = self.encoder.encode_texts([query]).astype(np.float32)
        if query_vector.shape[1] != self.dimension:
            raise RetrievalError(
                f"Query embedding dimension {query_vector.shape[1]} does not match "
                f"index dimension {self.dimension}."
            )
        distances, indices = self.index.search(query_vector, k)
        results: list[RetrievedDocument] = []
        for rank in range(k):
            doc_index = int(indices[0][rank])
            if doc_index < 0:
                continue
            document = self.documents[doc_index]
            results.append(
                RetrievedDocument(
                    id=str(document["id"]),
                    topic=str(document.get("topic") or ""),
                    text=str(document.get("text") or ""),
                    source_title=document.get("source_title"),
                    source_url=document.get("source_url"),
                    source_type=document.get("source_type"),
                    publication_year=document.get("publication_year"),
                    license=document.get("license"),
                    status=document.get("status"),
                    score_l2_distance=float(distances[0][rank]),
                    rank=rank + 1,
                )
            )
        return results

    def save(self, directory: Path | str) -> None:
        """Save FAISS index, embeddings, and a copy of the documents."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out / "index.faiss"))
        np.save(out / "embeddings.npy", self.embeddings)
        with (out / "documents.json").open("w", encoding="utf-8") as handle:
            json.dump(self.documents, handle, indent=2, ensure_ascii=False)
        meta = {"dimension": self.dimension, "n_documents": len(self.documents)}
        with (out / "meta.json").open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)

    @classmethod
    def load(cls, directory: Path | str, encoder: TextEncoder) -> "Retriever":
        directory = Path(directory)
        documents = load_knowledge_base(directory / "documents.json")
        embedding_path = directory / "embeddings.npy"
        if embedding_path.exists():
            embeddings = np.load(embedding_path).astype(np.float32)
            retriever = cls(documents, encoder, embeddings=embeddings)
        else:
            index = faiss.read_index(str(directory / "index.faiss"))
            embeddings = np.zeros((index.ntotal, index.d), dtype=np.float32)
            retriever = cls(documents, encoder, embeddings=embeddings)
            retriever.index = index
        return retriever
