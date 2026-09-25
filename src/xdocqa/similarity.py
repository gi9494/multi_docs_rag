"""Step 5 - Similarity: link chunks of DIFFERENT documents that talk about the same thing.

Reads the chunks from data/chunks/ and writes into Neo4j:
  - an `embedding` on every Chunk (the vector that represents its meaning), plus a vector index;
  - (:Chunk)-[:SIMILAR_TO {score}]->(:Chunk) between chunks of different documents.
Also writes:
  data/embeddings/embeddings.npz            the vectors (reused on the next run if nothing changed)
  docs/images/similarity_distribution.png   how similar cross-document chunks are, and the threshold
  data/embeddings/report.json               numbers for the README

How the links are chosen:
  1. every chunk gets an embedding; similarity between two chunks = cosine of their vectors (0..1);
  2. only pairs from DIFFERENT documents are considered: chunks of the same book always look
     alike (same author, same words), and we want bridges between books;
  3. a pair becomes a link if
       - it is among the TOP_K most similar chunks of at least one of the two, and
       - its score is in the top (100 - percentile)% of ALL cross-document pairs.
     The threshold comes from the data, not from a fixed number like 0.85: every embedding
     model has its own scale.

Usage:
    python -m xdocqa.similarity
    python -m xdocqa.similarity --percentile 95 --top-k 5     # fewer, stronger links
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()

TOP_K = 10            # candidates per chunk
PERCENTILE = 90       # keep links in the top 10% of all cross-document similarities
EMBED_BATCH = 32


# ------------------------------------------------------------------ embeddings

def load_chunks(corpus_file: Path, chunks_dir: Path) -> list[dict]:
    corpus = yaml.safe_load(corpus_file.read_text())["documents"]
    chunks = []
    for doc in corpus:
        path = chunks_dir / f"{doc}.jsonl"
        if path.exists():
            chunks += [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return chunks


def embed(texts: list[str], model: str) -> np.ndarray:
    import litellm
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        print(f"  embedding chunks {i + 1}-{min(i + EMBED_BATCH, len(texts))} of {len(texts)}", end="\r", flush=True)
        r = litellm.embedding(model=model, input=texts[i:i + EMBED_BATCH])
        vectors += [d["embedding"] for d in r.data]
    print()
    v = np.array(vectors, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)    # unit length: dot product = cosine


def get_embeddings(chunks: list[dict], model: str, cache: Path) -> np.ndarray:
    """Compute the embeddings, or reuse the saved ones if chunks and model are the same."""
    ids = [c["chunk_id"] for c in chunks]
    if cache.exists():
        saved = np.load(cache, allow_pickle=False)
        if list(saved["ids"]) == ids and str(saved["model"]) == model:
            print(f"  reusing saved embeddings ({cache})")
            return saved["vectors"]
    vectors = embed([c["text"] for c in chunks], model)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, ids=np.array(ids), model=np.array(model), vectors=vectors)
    return vectors


# ------------------------------------------------------------------ links

def find_links(chunks: list[dict], vectors: np.ndarray, top_k: int, percentile: float):
    """Return (links, threshold, all cross-document scores)."""
    docs = np.array([c["doc"] for c in chunks])
    sim = vectors @ vectors.T
    other_doc = docs[:, None] != docs[None, :]              # True where the two chunks are from different books

    cross = sim[np.triu(other_doc, k=1)]                     # every cross-document pair, once
    threshold = float(np.percentile(cross, percentile))

    candidates = set()
    for i in range(len(chunks)):
        scores = np.where(other_doc[i], sim[i], -np.inf)
        for j in np.argsort(-scores)[:top_k]:
            candidates.add((min(i, j), max(i, j)))

    links = [{"a": chunks[i]["chunk_id"], "b": chunks[j]["chunk_id"], "score": round(float(sim[i, j]), 4),
              "authors": tuple(sorted((chunks[i]["author"], chunks[j]["author"])))}
             for i, j in sorted(candidates) if sim[i, j] >= threshold]
    return links, threshold, cross


def plot(cross: np.ndarray, threshold: float, percentile: float, model: str, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(cross.min(), cross.max(), 61)          # same bins for both, so the bars line up
    ax.hist(cross, bins=bins, color="#9aa5b1", label="all pairs of chunks from different books")
    ax.hist(cross[cross >= threshold], bins=bins, color="#2f6fdd", label="kept as SIMILAR_TO candidates")
    ax.legend(frameon=False, loc="upper left")
    ax.axvline(threshold, color="#2f6fdd", linestyle="--")
    ax.text(threshold, ax.get_ylim()[1] * 0.9, f"  threshold {threshold:.2f}\n  (percentile {percentile:g})",
            color="#2f6fdd", va="top")
    ax.set_xlabel(f"cosine similarity between chunks of different books ({model})")
    ax.set_ylabel("pairs of chunks")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)


# ------------------------------------------------------------------ Neo4j

def write(chunks: list[dict], vectors: np.ndarray, links: list[dict]) -> None:
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(os.environ["NEO4J_URI"],
                                  auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))
    with driver.session() as s:
        s.run("""
            UNWIND $rows AS r
            MATCH (c:Chunk {id: r.id}) SET c.embedding = r.embedding
        """, rows=[{"id": c["chunk_id"], "embedding": v.tolist()} for c, v in zip(chunks, vectors)])
        s.run(f"""
            CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS FOR (c:Chunk) ON c.embedding
            OPTIONS {{indexConfig: {{`vector.dimensions`: {vectors.shape[1]},
                                    `vector.similarity_function`: 'cosine'}}}}
        """)
        s.run("MATCH ()-[r:SIMILAR_TO]->() DELETE r")      # rebuild the links from scratch
        s.run("""
            UNWIND $rows AS r
            MATCH (a:Chunk {id: r.a}), (b:Chunk {id: r.b})
            MERGE (a)-[l:SIMILAR_TO]->(b) SET l.score = r.score
        """, rows=[{k: l[k] for k in ("a", "b", "score")} for l in links])
    driver.close()


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="corpus.yaml")
    ap.add_argument("--chunks", default="data/chunks")
    ap.add_argument("--model", default=os.getenv("EMBEDDING_MODEL"))
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--percentile", type=float, default=PERCENTILE)
    ap.add_argument("--no-neo4j", action="store_true", help="compute and plot only, do not write to Neo4j")
    args = ap.parse_args()
    if not args.model:
        raise SystemExit("No embedding model: set EMBEDDING_MODEL in .env")

    chunks = load_chunks(Path(args.corpus), Path(args.chunks))
    if not chunks:
        raise SystemExit("No chunks: run xdocqa.chunking first")
    print(f"{len(chunks)} chunks from {len({c['doc'] for c in chunks})} documents")

    vectors = get_embeddings(chunks, args.model, Path("data/embeddings/embeddings.npz"))
    links, threshold, cross = find_links(chunks, vectors, args.top_k, args.percentile)

    linked = {x for l in links for x in (l["a"], l["b"])}
    pairs = Counter(" - ".join(l["authors"]) for l in links)
    report = {
        "model": args.model, "top_k": args.top_k, "percentile": args.percentile,
        "threshold": round(threshold, 4),
        "cross_similarity": {"min": round(float(cross.min()), 3), "median": round(float(np.median(cross)), 3),
                             "max": round(float(cross.max()), 3)},
        "links": len(links), "chunks_with_links": len(linked), "chunks_without_links": len(chunks) - len(linked),
        "links_per_author_pair": dict(pairs.most_common()),
    }
    print(f"cross-document similarity min/median/max: {report['cross_similarity']}")
    print(f"threshold (percentile {args.percentile:g}): {threshold:.3f}")
    print(f"links: {len(links)} | chunks with at least one link: {len(linked)}/{len(chunks)}")
    for pair, n in pairs.most_common():
        print(f"  {pair:22s} {n}")
    strongest = max(links, key=lambda l: l["score"], default=None)
    if strongest:
        print(f"strongest link: {strongest['a']} <-> {strongest['b']} ({strongest['score']})")

    Path("data/embeddings/report.json").write_text(json.dumps(report, indent=2))
    plot(cross, threshold, args.percentile, args.model, Path("docs/images/similarity_distribution.png"))
    print("plot -> docs/images/similarity_distribution.png")

    if not args.no_neo4j:
        write(chunks, vectors, links)
        print("written to Neo4j: embeddings, vector index 'chunk_embedding', SIMILAR_TO links")


if __name__ == "__main__":
    main()