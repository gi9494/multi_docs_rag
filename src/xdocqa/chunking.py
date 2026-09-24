"""Step 3 - Cleaning and semantic chunking: from the PageIndex trees to the final chunks.

For every tree in data/trees/ this module produces:
  data/chunks/<name>.jsonl    one line per chunk: text, pages, the section it belongs to
  data/chunks/report.json     statistics per document

How it works, per document:
  1. CLEAN THE TREE
     - keep only the author's pages (ranges in corpus.yaml) and drop excluded sections
       (e.g. "Endnotes", listed in corpus.yaml under "exclude");
     - merge duplicated nodes (a node with one child covering exactly the same pages);
  2. TEXT OF EACH SECTION
     - the text is read from the converted PDF, keeping only the body text: lines printed smaller
       than the main text (footnotes), lines repeated at the top/bottom of most pages (running
       heads) and lines that are just a page number are dropped;
     - the text of the author's pages is joined into one string;
     - each section starts exactly where its title appears in the text (not at the start of
       its page), so two sections never share text;
     - a section = its leaves, plus the text a chapter has before its first sub-section.
  3. SEMANTIC CHUNKING
     - each section is split into sentences; each sentence (with its neighbours) gets an embedding;
     - where the meaning changes a lot between two consecutive sentences, the topic changes;
     - the section is cut only at the biggest changes of the WHOLE corpus (top percentile),
       so a section about one topic stays whole, however long it is;
     - no piece shorter than MIN_CHUNK_WORDS; pieces longer than MAX_CHUNK_WORDS (limit of the
       embedding model) are cut at their biggest change.

Usage:
    python -m xdocqa.chunking
    python -m xdocqa.chunking --percentile 90      # more cuts (smaller chunks)
    python -m xdocqa.chunking --only kant1785 jsmill-utilitarianism
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import re
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()

MIN_CHUNK_WORDS = 150      # about half a page: shorter pieces carry too little context
MAX_CHUNK_WORDS = 1500     # safety cap, well below the input limit of the embedding model
BREAK_PERCENTILE = 95      # cut at the top 5% biggest changes of meaning in the corpus
WINDOW = 1                 # each sentence is embedded together with 1 sentence before and after
EMBED_BATCH = 64
SMALL_FONT = 0.9           # lines printed smaller than 90% of the main text size are footnotes
EDGE_LINES = 2             # running heads/feet are searched in the first/last 2 lines of each page
REPEATED = 0.3             # a line found at the edge of > 30% of the pages is a running head
MAX_NOTE_WORDS = 120       # a bracketed note longer than this is treated as an unclosed "["


# ------------------------------------------------------------------ 1. clean the tree

def walk(nodes: list[dict], depth: int = 0, path: tuple = (), ids: tuple = ()):
    """Yield (node, depth, titles from the root, node ids from the root) in reading order.
    Ids are needed because titles repeat (Hume has several sections called "PART I.")."""
    for n in nodes:
        p, i = (*path, n["title"].strip()), (*ids, n.get("node_id") or n["title"].strip())
        yield n, depth, p, i
        yield from walk(n.get("nodes") or [], depth + 1, p, i)


def collapse_duplicates(nodes: list[dict]) -> list[dict]:
    """A node whose only child covers the same pages is the same section twice: keep one."""
    out = []
    for n in nodes:
        kids = collapse_duplicates(n.get("nodes") or [])
        while len(kids) == 1 and (kids[0]["start_index"], kids[0]["end_index"]) == (n["start_index"], n["end_index"]):
            kids = collapse_duplicates(kids[0].get("nodes") or [])
        out.append({**n, "nodes": kids})
    return out


def drop_sections(nodes: list[dict], excluded: list[str]) -> list[dict]:
    """Drop the sections whose title starts with one of the excluded names (e.g. "Endnotes")."""
    names = [e.lower() for e in excluded]
    return [{**n, "nodes": drop_sections(n.get("nodes") or [], excluded)} for n in nodes
            if not n["title"].strip().lower().startswith(tuple(names))] if names else nodes


def keep_pages(nodes: list[dict], first: int, last: int) -> list[dict]:
    """Drop the nodes outside the author's pages, clip the ones that cross the border."""
    out = []
    for n in nodes:
        if n["end_index"] < first or n["start_index"] > last:
            continue
        out.append({**n, "start_index": max(n["start_index"], first), "end_index": min(n["end_index"], last),
                    "nodes": keep_pages(n.get("nodes") or [], first, last)})
    return out


# ------------------------------------------------------------------ 2. text of each section

def read_lines(pdf_path: Path, first: int, last: int) -> dict[int, list[tuple[float, str]]]:
    """Page number -> list of (font size, text) for every line, in reading order."""
    import pymupdf
    pages = {}
    with pymupdf.open(pdf_path) as doc:
        for pno in range(first, min(last, doc.page_count) + 1):
            lines = []
            for block in doc[pno - 1].get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    spans = [sp for sp in line["spans"] if sp["text"].strip()]
                    if not spans:
                        continue
                    chars = sum(len(sp["text"]) for sp in spans)
                    size = sum(sp["size"] * len(sp["text"]) for sp in spans) / chars
                    lines.append((round(size * 2) / 2, "".join(sp["text"] for sp in line["spans"]).strip()))
            pages[pno] = lines
    return pages


def body_text(pages: dict[int, list[tuple[float, str]]]) -> tuple[dict[int, str], dict]:
    """Keep only the main text of every page. Returns page -> text, and what was dropped."""
    sizes = Counter()
    for lines in pages.values():
        for size, text in lines:
            sizes[size] += len(text)
    body = sizes.most_common(1)[0][0] if sizes else 0          # the size used for most characters

    norm = lambda t: re.sub(r"\d+", "#", t.lower()).strip()
    edges = Counter()
    for lines in pages.values():
        edge = lines[:EDGE_LINES] + lines[-EDGE_LINES:]
        edges.update({norm(t) for _, t in edge})
    running = {t for t, c in edges.items() if c > REPEATED * len(pages)}

    dropped = Counter()
    out = {}
    for pno, lines in pages.items():
        kept = []
        for i, (size, text) in enumerate(lines):
            at_edge = i < EDGE_LINES or i >= len(lines) - EDGE_LINES
            if size < SMALL_FONT * body:
                dropped["small_font"] += 1                     # footnotes
            elif at_edge and norm(text) in running:
                dropped["running_head"] += 1                   # "On the Genealogy of Morality 20"
            elif re.fullmatch(r"[\W\d]*|[ivxlcdm]+", text.lower()):
                dropped["page_number"] += 1                    # "75", "– 12 –", "xxxvi"
            else:
                kept.append(text)
        out[pno] = "\n".join(kept)
    return out, {"body_font_size": body, **dropped}


def load_text(pdf_path: Path, first: int, last: int) -> tuple[str, list[int], list[int], dict]:
    """Join the body text of the author's pages.
    Returns text, page numbers, the offset where each page starts, and what was dropped."""
    texts, dropped = body_text(read_lines(pdf_path, first, last))
    parts, numbers, starts, pos = [], [], [], 0
    for pno in sorted(texts):
        numbers.append(pno)
        starts.append(pos)
        parts.append(texts[pno] + "\n")
        pos += len(parts[-1])
    return "".join(parts), numbers, starts, dropped


def title_pattern(title: str, max_words: int | None = None, line_start: bool = True):
    words = re.findall(r"\w+", title)[:max_words]
    if not words:
        return None
    body = r"[\W_]+".join(re.escape(w) for w in words)
    prefix = r"^[ \t]*" if line_start else r"(?<!\w)"
    return re.compile(prefix + body + r"(?!\w)", re.IGNORECASE | re.MULTILINE)  # "Chapter 1" != "Chapter 10"


def find_title(text: str, title: str, lo: int, hi: int) -> int | None:
    for pat in (title_pattern(title), title_pattern(title, line_start=False), title_pattern(title, 3)):
        if pat is not None and (m := pat.search(text, lo, hi)):
            return m.start()
    return None


def sections(tree_nodes: list[dict], text: str, numbers: list[int], starts: list[int]) -> list[dict]:
    """Find where every node starts in the text; return the pieces of text that belong to one node only."""
    def offset_of_page(p, end=False):
        i = bisect.bisect_left(numbers, p)
        if end:
            return starts[i + 1] if i + 1 < len(starts) else len(text)
        return starts[min(i, len(starts) - 1)]

    flat = list(walk(tree_nodes))
    cursor, anchors = 0, []
    for node, depth, path, ids in flat:
        lo = max(cursor, offset_of_page(node["start_index"]))
        hi = offset_of_page(node["end_index"], end=True)
        found = find_title(text, node["title"], lo, hi) if lo < hi else None
        cursor = found if found is not None else lo
        anchors.append(cursor)

    out = []
    for i, (node, depth, path, ids) in enumerate(flat):
        end = anchors[i + 1] if i + 1 < len(flat) else len(text)
        if not text[anchors[i]:end].strip():
            continue
        out.append({"node_id": node.get("node_id"), "title": node["title"].strip(), "path": list(path), "path_ids": list(ids),
                    "is_leaf": not node.get("nodes"), "summary": node.get("summary"),
                    "start": anchors[i], "end": end})
    return out


# ------------------------------------------------------------------ 3. semantic chunking

SENTENCE_END = re.compile(r"(?<=[.!?;:])[\"'”’)]*\s+(?=[\"'“‘(]?[A-Z0-9])")


def clean(raw: str) -> str:
    flat = re.sub(r"(\w)-\n(\w)", r"\1\2", raw)           # re-join hyphenated words
    return re.sub(r"\s*\n\s*", " ", flat).strip()          # page lines -> one flow of text


def sentences(text: str, base: int) -> list[tuple[str, int, int]]:
    """Split into sentences, keeping where each one starts and ends in the document text.

    Never split inside brackets: footnotes like "[Cicero, De Off. lib. i. cap. 6.]" contain
    full stops but are not sentences, and cutting there would start a chunk with "6.]".
    """
    out, lo = [], 0
    for m in list(SENTENCE_END.finditer(text)) + [None]:
        hi = m.start() if m else len(text)
        if (m and text.count("[", lo, hi) > text.count("]", lo, hi)
                and len(text[lo:hi].split()) < MAX_NOTE_WORDS):
            continue                                        # inside [ ... ]: keep going
            # (but not forever: a "[" that the PDF never closes must not swallow a whole section)
        s = clean(text[lo:hi])
        if s:
            out.append((s, base + lo, base + hi))
        lo = m.end() if m else lo
    return out


def embed(texts: list[str], model: str) -> np.ndarray:
    import litellm
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        r = litellm.embedding(model=model, input=texts[i:i + EMBED_BATCH])
        vectors += [d["embedding"] for d in r.data]
    v = np.array(vectors, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)    # unit length: dot product = cosine


def distances(sents: list[str], model: str) -> np.ndarray:
    """Change of meaning between sentence i and i+1 (0 = same meaning, 2 = opposite)."""
    if len(sents) < 2:
        return np.zeros(0)
    windows = [" ".join(sents[max(0, i - WINDOW): i + WINDOW + 1]) for i in range(len(sents))]
    v = embed(windows, model)
    return 1 - np.sum(v[:-1] * v[1:], axis=1)


def cut(sents: list[str], dist: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Cut after sentence i when dist[i] > threshold, respecting min and max chunk size."""
    words = [len(s.split()) for s in sents]

    def split(lo: int, hi: int) -> list[tuple[int, int]]:  # sentences lo..hi-1
        n = sum(words[lo:hi])
        cands = [i for i in range(lo, hi - 1) if dist[i] > threshold
                 and sum(words[lo:i + 1]) >= MIN_CHUNK_WORDS and sum(words[i + 1:hi]) >= MIN_CHUNK_WORDS]
        if not cands and n > MAX_CHUNK_WORDS:              # too long for the embedding model:
            cands = [i for i in range(lo, hi - 1)          # cut at its biggest change anyway
                     if sum(words[lo:i + 1]) >= MIN_CHUNK_WORDS and sum(words[i + 1:hi]) >= MIN_CHUNK_WORDS]
        if not cands:
            return [(lo, hi)]
        best = max(cands, key=lambda i: dist[i])           # biggest change first, then recurse
        return split(lo, best + 1) + split(best + 1, hi)

    return split(0, len(sents))  # (first sentence, last sentence + 1) of every piece


TITLE_ONLY_WORDS = 30  # a "section" this short is just a heading ("SECTION II. OF BENEVOLENCE.")


def glue_small(chunks: list[dict]) -> list[dict]:
    """Chunks shorter than the minimum are glued to a neighbour:
    - a heading alone always goes with the text that FOLLOWS it;
    - a short piece of text goes with the next chunk of the same chapter,
      or else with the previous one of the same chapter."""
    def join(first: dict, second: dict, keep: dict) -> dict:
        other = second if keep is first else first
        keep = dict(keep)
        keep["text"] = first["text"] + " " + second["text"]
        keep["words"] = first["words"] + second["words"]
        keep["start"], keep["end"] = first["start"], second["end"]
        keep["also_contains"] = keep.get("also_contains", []) + [other["section"]]
        return keep

    def small(c):
        return c["words"] < MIN_CHUNK_WORDS and c["parts"] == 1

    out, pending = [], None
    for c in chunks:
        if pending is not None:
            if pending["words"] < TITLE_ONLY_WORDS or c["path"][0] == pending["path"][0]:
                c = join(pending, c, keep=c)                 # glue to the next chunk
            elif out and out[-1]["path"][0] == pending["path"][0]:
                out[-1] = join(out[-1], pending, keep=out[-1])  # or to the previous one
            else:
                out.append(pending)
            pending = None
        if small(c):
            pending = c
        else:
            out.append(c)
    if pending is not None:                                  # small piece at the very end
        if out:
            out[-1] = join(out[-1], pending, keep=out[-1])
        else:
            out.append(pending)
    return out


# ------------------------------------------------------------------ main

def page_of(offset: int, numbers: list[int], starts: list[int]) -> int:
    return numbers[max(bisect.bisect_right(starts, offset) - 1, 0)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="corpus.yaml")
    ap.add_argument("--trees", default="data/trees")
    ap.add_argument("--converted", default="data/converted")
    ap.add_argument("--out", default="data/chunks")
    ap.add_argument("--model", default=os.getenv("EMBEDDING_MODEL"),
                    help="LiteLLM embedding model; default: EMBEDDING_MODEL in .env")
    ap.add_argument("--percentile", type=float, default=BREAK_PERCENTILE)
    ap.add_argument("--only", nargs="*", help="process only these documents (names as in corpus.yaml)")
    args = ap.parse_args()
    if not args.model:
        raise SystemExit("No embedding model: set EMBEDDING_MODEL in .env (see .env.example)")

    corpus = yaml.safe_load(Path(args.corpus).read_text())["documents"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # pass 1: sections and sentence distances of every document
    docs = {}
    for name, info in corpus.items():
        if args.only and name not in args.only:
            continue
        tree_file, pdf_file = Path(args.trees) / f"{name}.json", Path(args.converted) / f"{name}.pdf"
        if not tree_file.exists() or not pdf_file.exists():
            print(f"{name}: missing tree or converted PDF, skipped")
            continue
        first, last = info["pages"]
        tree = json.loads(tree_file.read_text())
        nodes = keep_pages(collapse_duplicates(drop_sections(tree["structure"], info.get("exclude", []))),
                           first, last)
        text, numbers, starts, dropped = load_text(pdf_file, first, last)
        print(f"{name}: dropped lines {dropped}")
        secs = sections(nodes, text, numbers, starts)
        print(f"{name}: {len(secs)} sections, embedding sentences...", flush=True)
        for s in secs:
            s["sentences"] = sentences(text[s["start"]:s["end"]], s["start"])
            s["dist"] = distances([t for t, _, _ in s["sentences"]], args.model)
        docs[name] = (info, secs, numbers, starts, dropped)

    # the threshold comes from the data: the top (100 - percentile)% changes of the whole corpus
    all_d = np.concatenate([s["dist"] for _, secs, _, _, _ in docs.values() for s in secs if len(s["dist"])])
    threshold = float(np.percentile(all_d, args.percentile))
    print(f"\nchange-of-meaning threshold (percentile {args.percentile:g} of {len(all_d)} values): {threshold:.3f}")

    # pass 2: cut and write
    reports = []
    for name, (info, secs, numbers, starts, dropped) in docs.items():
        chunks, n_split = [], 0
        for s in secs:
            pieces = cut([t for t, _, _ in s["sentences"]], s["dist"], threshold) if s["sentences"] else []
            n_split += len(pieces) > 1
            for k, (a, b) in enumerate(pieces, start=1):
                sents = s["sentences"][a:b]
                body = " ".join(t for t, _, _ in sents)
                chunks.append({
                    "doc": name, "author": info["author"], "doc_title": info["title"],
                    "section_id": s["node_id"], "section": s["title"], "path": s["path"], "path_ids": s["path_ids"],
                    "part": k, "parts": len(pieces), "text": body, "words": len(body.split()),
                    "start": sents[0][1], "end": sents[-1][2], "section_summary": s["summary"],
                })
        merged = glue_small(chunks)
        for i, c in enumerate(merged):
            c["chunk_id"] = f"{info['author'].lower()}-{i:04d}"
            c["start_page"] = page_of(c.pop("start"), numbers, starts)
            c["end_page"] = page_of(c.pop("end") - 1, numbers, starts)

        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for c in merged:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        w = sorted(c["words"] for c in merged) or [0]
        r = {"document": name, "sections": len(secs), "chunks": len(merged), "sections_split": n_split,
             "dropped_lines": dropped,
             "words_per_chunk": {"min": w[0], "median": int(statistics.median(w)), "max": w[-1]}}
        reports.append(r)
        print(f"{name}: {r['sections']} sections -> {r['chunks']} chunks ({n_split} sections split) "
              f"| words per chunk min/median/max {w[0]}/{r['words_per_chunk']['median']}/{w[-1]}")

    (out / "report.json").write_text(json.dumps({"threshold": threshold, "percentile": args.percentile,
                                                  "documents": reports}, indent=2))
    print(f"\nchunks -> {out}/")


if __name__ == "__main__":
    main()