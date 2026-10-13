#!/usr/bin/env python3
"""Validate the knowledge base and optionally rebuild a saved FAISS index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.retrieval import Retriever, load_knowledge_base


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate KB and rebuild the vector index")
    parser.add_argument("--save-index", action="store_true", help="Write index files under data/processed/")
    args = parser.parse_args()

    config = load_config()
    kb_path = config["_resolved_paths"]["knowledge_base"]
    documents = load_knowledge_base(kb_path)
    print(f"Valid knowledge base: {kb_path}")
    print(f"Documents: {len(documents)}")
    preliminary = sum(1 for doc in documents if doc.get("status") == "preliminary")
    recorded = sum(1 for doc in documents if doc.get("status") == "source_recorded")
    print(f"Preliminary / unverified: {preliminary}")
    print(f"Source recorded: {recorded}")

    retriever = Retriever.from_knowledge_base(
        kb_path,
        embedding_model_id=config["model"]["embedding_id"],
    )
    print(f"FAISS index built. ntotal={retriever.index.ntotal} dim={retriever.dimension}")

    if args.save_index:
        out = config["_resolved_paths"]["processed_dir"] / "faiss_index"
        retriever.save(out)
        print(f"Saved index to {out}")


if __name__ == "__main__":
    main()
