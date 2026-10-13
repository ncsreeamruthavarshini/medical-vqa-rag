import json
from pathlib import Path

import numpy as np
import pytest

from src.embeddings import HashOverlapEncoder
from src.retrieval import KnowledgeBaseError, RetrievalError, Retriever, load_knowledge_base, validate_document


def _docs():
    return [
        {"id": "d1", "topic": "kidneys", "text": "The kidneys sit in the posterior abdomen beside the spine."},
        {"id": "d2", "topic": "ventricles", "text": "The lateral ventricles are cavities in the brain."},
        {"id": "d3", "topic": "chest", "text": "A chest radiograph shows lungs heart and diaphragm."},
    ]


def test_load_valid_project_knowledge_base():
    docs = load_knowledge_base("data/knowledge_base/medical_knowledge.json")
    assert len(docs) >= 10
    assert docs[0]["id"] == "kb_001"


def test_empty_knowledge_base(tmp_path: Path):
    path = tmp_path / "empty.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(KnowledgeBaseError, match="empty"):
        load_knowledge_base(path)


def test_malformed_json(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(KnowledgeBaseError):
        load_knowledge_base(path)


def test_missing_fields():
    with pytest.raises(KnowledgeBaseError, match="text"):
        validate_document({"id": "x", "topic": "t"}, 0)


def test_duplicate_ids(tmp_path: Path):
    path = tmp_path / "dup.json"
    path.write_text(
        json.dumps(
            [
                {"id": "a", "topic": "t", "text": "one"},
                {"id": "a", "topic": "t", "text": "two"},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(KnowledgeBaseError, match="Duplicate"):
        load_knowledge_base(path)


def test_embedding_dimension_matches_index():
    encoder = HashOverlapEncoder(dimension=16)
    retriever = Retriever(_docs(), encoder)
    assert retriever.dimension == 16
    assert retriever.index.d == 16
    assert retriever.embeddings.shape == (3, 16)
    assert retriever.embeddings.dtype == np.float32


def test_top_k_larger_than_corpus():
    retriever = Retriever(_docs(), HashOverlapEncoder())
    hits = retriever.retrieve("Where are the kidneys located?", top_k=50)
    assert len(hits) == 3
    assert [doc.rank for doc in hits] == [1, 2, 3]
    assert all(hasattr(doc, "score_l2_distance") for doc in hits)
    payload = hits[0].to_dict()
    assert payload["id"]
    assert "text" in payload
    assert "topic" in payload


def test_kidney_query_ranks_kidney_document_first():
    retriever = Retriever(_docs(), HashOverlapEncoder())
    hits = retriever.retrieve("Where are the kidneys located?", top_k=1)
    assert hits[0].id == "d1"


def test_empty_query_rejected():
    retriever = Retriever(_docs(), HashOverlapEncoder())
    with pytest.raises(RetrievalError):
        retriever.retrieve("   ", top_k=3)


def test_save_and_load_index(tmp_path: Path):
    encoder = HashOverlapEncoder()
    retriever = Retriever(_docs(), encoder)
    retriever.save(tmp_path)
    loaded = Retriever.load(tmp_path, encoder)
    assert loaded.dimension == retriever.dimension
    assert loaded.retrieve("kidneys", top_k=1)[0].id == retriever.retrieve("kidneys", top_k=1)[0].id
