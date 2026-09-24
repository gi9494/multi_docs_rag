"""Step 4 - Graph: load the documents, their sections and their chunks into Neo4j.

Reads data/chunks/<name>.jsonl (step 3) and corpus.yaml, and builds this graph:

    (:Document)-[:HAS_SECTION]->(:Section)-[:HAS_SECTION]->(:Section) ... -[:HAS_CHUNK]->(:Chunk)
    (:Chunk)-[:NEXT]->(:Chunk)                         reading order inside a document

    Document  name, author, title
    Section   id, title, depth, doc, author, start_page, end_page, summary (from PageIndex)
    Chunk     id, text, words, doc, author, section, part, parts, start_page, end_page

The database is emptied and rebuilt at every run, so the graph always matches data/chunks/.

Usage:
    docker compose up -d          # Neo4j must be running
    python -m xdocqa.graph
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()  # NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


# ------------------------------------------------------------------ build the rows (no database needed)

def section_id(doc: str, path_ids: list[str]) -> str:
    """A section is identified by its document and the ids of the tree nodes above it,
    e.g. 'kant1785::0004/0006'. Titles alone are not enough: Hume has several "PART I."."""
    return f"{doc}::{'/'.join(path_ids)}"


def build_rows(doc: str, info: dict, chunks: list[dict]) -> dict[str, list[dict]]:
    """Turn the chunks of one document into the rows to write: sections, chunks, links."""
    sections: dict[str, dict] = {}
    for c in chunks:
        path, ids = c["path"], c["path_ids"]
        for depth in range(1, len(path) + 1):          # every level of the path is a section
            sid = section_id(doc, ids[:depth])
            s = sections.setdefault(sid, {
                "id": sid, "title": path[depth - 1], "depth": depth, "doc": doc, "author": info["author"],
                "parent": section_id(doc, ids[:depth - 1]) if depth > 1 else None,
                "start_page": c["start_page"], "end_page": c["end_page"], "summary": None,
            })
            s["start_page"] = min(s["start_page"], c["start_page"])
            s["end_page"] = max(s["end_page"], c["end_page"])
        own = sections[section_id(doc, ids)]
        own["summary"] = own["summary"] or c.get("section_summary")

    chunk_rows = [{
        "id": c["chunk_id"], "section_id": section_id(doc, c["path_ids"]),
        "props": {"text": c["text"], "words": c["words"], "doc": doc, "author": info["author"],
                  "section": c["section"], "part": c["part"], "parts": c["parts"],
                  "start_page": c["start_page"], "end_page": c["end_page"]},
    } for c in chunks]

    next_rows = [{"a": a["chunk_id"], "b": b["chunk_id"]} for a, b in zip(chunks, chunks[1:])]
    return {"sections": list(sections.values()), "chunks": chunk_rows, "next": next_rows}


# ------------------------------------------------------------------ write to Neo4j

SCHEMA = [
    "CREATE CONSTRAINT document_name IF NOT EXISTS FOR (d:Document) REQUIRE d.name IS UNIQUE",
    "CREATE CONSTRAINT section_id IF NOT EXISTS FOR (s:Section) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
]


def write(session, doc: str, info: dict, rows: dict) -> None:
    session.run("MERGE (d:Document {name: $name}) SET d.author = $author, d.title = $title",
                name=doc, author=info["author"], title=info["title"])
    session.run("""
        UNWIND $rows AS r
        MERGE (s:Section {id: r.id})
        SET s.title = r.title, s.depth = r.depth, s.doc = r.doc, s.author = r.author,
            s.start_page = r.start_page, s.end_page = r.end_page, s.summary = r.summary
    """, rows=rows["sections"])
    session.run("""
        UNWIND $rows AS r
        MATCH (s:Section {id: r.id}), (d:Document {name: r.doc})
        WITH s, d, r WHERE r.parent IS NULL
        MERGE (d)-[:HAS_SECTION]->(s)
    """, rows=rows["sections"])
    session.run("""
        UNWIND $rows AS r
        MATCH (s:Section {id: r.id}), (p:Section {id: r.parent})
        MERGE (p)-[:HAS_SECTION]->(s)
    """, rows=[r for r in rows["sections"] if r["parent"]])
    session.run("""
        UNWIND $rows AS r
        MERGE (c:Chunk {id: r.id}) SET c += r.props
        WITH c, r
        MATCH (s:Section {id: r.section_id})
        MERGE (s)-[:HAS_CHUNK]->(c)
    """, rows=rows["chunks"])
    session.run("""
        UNWIND $rows AS r
        MATCH (a:Chunk {id: r.a}), (b:Chunk {id: r.b})
        MERGE (a)-[:NEXT]->(b)
    """, rows=rows["next"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="corpus.yaml")
    ap.add_argument("--chunks", default="data/chunks")
    args = ap.parse_args()

    from neo4j import GraphDatabase
    corpus = yaml.safe_load(Path(args.corpus).read_text())["documents"]
    driver = GraphDatabase.driver(os.environ["NEO4J_URI"],
                                  auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))
    driver.verify_connectivity()

    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")        # start from an empty database
        for q in SCHEMA:
            session.run(q)
        for doc, info in corpus.items():
            path = Path(args.chunks) / f"{doc}.jsonl"
            if not path.exists():
                print(f"{doc}: no chunks, skipped (run xdocqa.chunking first)")
                continue
            chunks = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            rows = build_rows(doc, info, chunks)
            write(session, doc, info, rows)
            print(f"{doc}: {len(rows['sections'])} sections, {len(rows['chunks'])} chunks")

        counts = session.run("""
            MATCH (n) WITH labels(n)[0] AS label, count(*) AS n RETURN label, n ORDER BY label
        """).data()
        print("\nin Neo4j:", {r["label"]: r["n"] for r in counts})
    driver.close()
    print("open http://localhost:7474 to look at the graph")


if __name__ == "__main__":
    main()