#!/usr/bin/env python3
"""Retrieve passages for sample questions and, if labels exist, report Recall@k."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.metrics import recall_at_k
from src.retrieval import Retriever


def main() -> None:
    config = load_config()
    kb_path = config["_resolved_paths"]["knowledge_base"]
    queries_path = config["_resolved_paths"]["retrieval_eval_queries"]
    top_k = config["retrieval"]["top_k"]

    print("Building retriever...")
    retriever = Retriever.from_knowledge_base(
        kb_path,
        embedding_model_id=config["model"]["embedding_id"],
    )
    print(f"Documents: {len(retriever.documents)} | Embedding dimension: {retriever.dimension}")
    print("L2 distance is a closeness score in vector space, not a medical probability.\n")

    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    retrieved_ids = []
    relevant_ids = []
    for item in queries:
        query = item["query"]
        hits = retriever.retrieve(query, top_k=top_k)
        print("========================================")
        print("Query:", query)
        print("Human topical labels (informal):", item.get("relevant_ids"))
        for doc in hits:
            print(f"  rank {doc.rank}: {doc.id} | {doc.topic} | L2={doc.score_l2_distance:.4f}")
            print(f"    {doc.text}")
            print(f"    source={doc.source_title} | status={doc.status}")
        retrieved_ids.append([doc.id for doc in hits])
        relevant_ids.append(item.get("relevant_ids") or [])

    metrics = recall_at_k(retrieved_ids, relevant_ids, k=top_k)
    print("\nInformal Recall@k on manually labeled topical queries:")
    print(metrics)
    print("This is not a published retrieval benchmark and should not be reported as scientific accuracy.")


if __name__ == "__main__":
    main()
