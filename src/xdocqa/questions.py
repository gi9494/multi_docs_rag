"""Step 7 - Questions: generate questions that need MORE THAN ONE BOOK to be answered.

For every cross-document community of step 6:
  1. choose PAIRS of authors, so that every author of the community is used
     (e.g. Hume-Mill, then Nietzsche-Hume...), one question per pair;
  2. for each pair, take the 2 chunks of each author closest to the centre of the community
     (the most "on topic"): 4 short passages, not 8 long ones, so a small model does not mix them up;
  3. ask the LLM for one question on the two authors, its answer and the passages used;
  4. check every question: its sources must exist and come from BOTH authors, and it must not refer to
     "passage P1". If not, it is rejected (and counted, with the reason).

Writes:
  data/questions/raw/<community>.json   the answers of the LLM for each community (a re-run skips these)
  data/questions/questions.jsonl        the accepted questions, one per line, with their sources
  data/questions/questions.md           the same, easy to read
  data/questions/report.json            numbers for the README

Usage:
    python -m xdocqa.questions
    python -m xdocqa.questions --only 25          # one community (e.g. to try the prompt)
    python -m xdocqa.questions --force            # ask the LLM again, also for communities already done
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()

PER_AUTHOR = 2            # passages of each author in one question
MAX_PASSAGE_WORDS = 700   # longer chunks are cut
MAX_QUESTIONS = 3         # questions (= author pairs) per community
ATTEMPTS = 2              # ask again if the answer is not valid JSON
CONTEXT = 16384           # context window asked to Ollama: the prompt must never be cut

PROMPT = """You are building an exam on moral philosophy. Below are passages by {a} and by {b} \
on the same topic.

{passages}

Task: write ONE question that can only be answered by knowing what BOTH {a} and {b} say \
(how they differ, where they agree, or how their views combine), and its answer.

Rules:
- Each passage starts with its author in capitals. When you say what {a} thinks, use ONLY the passages \
of {a}; when you say what {b} thinks, use ONLY the passages of {b}. Never move an idea from one author \
to the other.
- Use only what the passages say, nothing else you know.
- The question must make sense to someone who has not seen the passages: name the authors, \
never write "passage P1" or "the text above".
- The answer: 3-6 sentences, citing the passages like [P1].
- "type": "comparison" if the answer is mainly about differences, "agreement" if mainly about what they \
share, "synthesis" if it combines them.
- "topic": the topic of the passages in max 6 words.

Answer with JSON only:
{{"topic": "...", "question": "...", "answer": "...", "type": "comparison", "sources": ["P1", "P3"]}}"""


# ------------------------------------------------------------------ inputs

def load_inputs(corpus_file: Path, chunks_dir: Path, embeddings: Path):
    corpus = yaml.safe_load(corpus_file.read_text())["documents"]
    chunks = {}
    for doc in corpus:
        path = chunks_dir / f"{doc}.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    c = json.loads(line)
                    c["title"] = corpus[doc]["title"]
                    chunks[c["chunk_id"]] = c
    saved = np.load(embeddings, allow_pickle=False)
    vectors = dict(zip([str(i) for i in saved["ids"]], saved["vectors"]))
    return chunks, vectors


def central_by_author(member_ids: list[str], chunks: dict, vectors: dict) -> dict[str, list[dict]]:
    """For each author, their chunks of the community, the most central first."""
    ids = [i for i in member_ids if i in chunks and i in vectors]
    centre = np.mean([vectors[i] for i in ids], axis=0)
    by_author: dict[str, list[dict]] = {}
    for i in sorted(ids, key=lambda i: -float(vectors[i] @ centre)):
        by_author.setdefault(chunks[i]["author"], []).append(chunks[i])
    return by_author


def choose_pairs(by_author: dict[str, list[dict]], n: int) -> list[tuple[str, str]]:
    """Pairs of authors: first the ones that bring in an author not used yet, then the ones with more text."""
    pairs = list(itertools.combinations(sorted(by_author), 2))
    chosen, used = [], set()
    while pairs and len(chosen) < n:
        best = max(pairs, key=lambda p: (len(set(p) - used), len(by_author[p[0]]) + len(by_author[p[1]])))
        chosen.append(best)
        used |= set(best)
        pairs.remove(best)
    return chosen


def format_passages(passages: list[dict]) -> str:
    blocks = []
    for n, c in enumerate(passages, 1):
        words = c["text"].split()
        text = " ".join(words[:MAX_PASSAGE_WORDS]) + (" [...]" if len(words) > MAX_PASSAGE_WORDS else "")
        blocks.append(f"[P{n}] {c['author'].upper()} - {c['title']}, {c['section']} "
                      f"(pp. {c['start_page']}-{c['end_page']})\n{text}")
    return "\n\n".join(blocks)


# ------------------------------------------------------------------ LLM

def parse_json(text: str) -> dict | None:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) and "question" in data else None


def ask(prompt: str, model: str) -> tuple[dict | None, str]:
    import litellm
    litellm.drop_params = True                          # ignore options a provider does not know
    extra = {"num_ctx": CONTEXT} if model.startswith("ollama") else {}
    raw = ""
    for _ in range(ATTEMPTS):
        r = litellm.completion(model=model, messages=[{"role": "user", "content": prompt}],
                               temperature=0.3, response_format={"type": "json_object"},
                               timeout=float(os.getenv("LLM_TIMEOUT", 1800)), **extra)
        raw = r.choices[0].message.content or ""
        data = parse_json(raw)
        if data:
            return data, raw
    return None, raw


# ------------------------------------------------------------------ checks

def check(q: dict, passages: list[dict], pair: list[str]) -> tuple[dict | None, str | None]:
    """Return (question with its real sources, None) or (None, reason why it is rejected)."""
    if not str(q.get("question", "")).strip() or not str(q.get("answer", "")).strip():
        return None, "empty question or answer"
    if re.search(r"\bP\d+\b|passage", q["question"], flags=re.I):
        return None, "question refers to the passages"
    labels = [str(s).strip().strip("[]") for s in q.get("sources", [])]
    numbers = sorted({int(s[1:]) for s in labels if re.fullmatch(r"P\d+", s) and 1 <= int(s[1:]) <= len(passages)})
    if not numbers:
        return None, "no valid sources"
    used = [passages[n - 1] for n in numbers]
    authors = sorted({c["author"] for c in used})
    if authors != sorted(pair):
        return None, "sources do not cover both authors"

    def cite(m):                  # "[P2]" in the answer -> "[Kant, p. 9]", readable without the passages
        n = int(m.group(1))
        return f"[{passages[n - 1]['author']}, p. {passages[n - 1]['start_page']}]" if 1 <= n <= len(passages) else ""
    return {
        "question": q["question"].strip(),
        "answer": re.sub(r"\[?\bP(\d+)\b\]?", cite, q["answer"].strip()),
        "type": q.get("type") if q.get("type") in ("comparison", "agreement", "synthesis") else "other",
        "authors": authors,
        "sources": [{"chunk_id": c["chunk_id"], "author": c["author"], "title": c["title"],
                     "section": c["section"], "pages": [c["start_page"], c["end_page"]]} for c in used],
    }, None


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="corpus.yaml")
    ap.add_argument("--chunks", default="data/chunks")
    ap.add_argument("--communities", default="data/communities/communities.json")
    ap.add_argument("--embeddings", default="data/embeddings/embeddings.npz")
    ap.add_argument("--out", default="data/questions")
    ap.add_argument("--model", default=os.getenv("LLM_MODEL"))
    ap.add_argument("--only", type=int, help="one community id")
    ap.add_argument("--force", action="store_true", help="ask the LLM again for communities already done")
    args = ap.parse_args()
    if not args.model:
        raise SystemExit("No model: set LLM_MODEL in .env")

    chunks, vectors = load_inputs(Path(args.corpus), Path(args.chunks), Path(args.embeddings))
    all_groups = [g for g in json.loads(Path(args.communities).read_text())["communities"] if g["cross_document"]]
    todo = [g for g in all_groups if args.only is None or g["id"] == args.only]
    if not todo:
        raise SystemExit(f"No cross-document community with id {args.only}")
    out, raw_dir = Path(args.out), Path(args.out) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. ask the LLM: one call per pair of authors (communities already done are skipped)
    for k, g in enumerate(todo, 1):
        cache = raw_dir / f"{g['id']}.json"
        if cache.exists() and "calls" in json.loads(cache.read_text()) and not args.force:
            continue
        by_author = central_by_author([m["id"] for m in g["members"]], chunks, vectors)
        calls = []
        for a, b in choose_pairs(by_author, MAX_QUESTIONS):
            passages = by_author[a][:PER_AUTHOR] + by_author[b][:PER_AUTHOR]
            print(f"[{k}/{len(todo)}] community #{g['id']}: {a} - {b} ({len(passages)} passages)...", flush=True)
            data, raw = ask(PROMPT.format(a=a, b=b, passages=format_passages(passages)), args.model)
            if data is None:
                print("   the answer was not valid JSON")
            calls.append({"authors": [a, b], "passages": [p["chunk_id"] for p in passages],
                          "parsed": data, "raw": raw})
        cache.write_text(json.dumps({"community": g["id"], "model": args.model, "calls": calls},
                                    indent=2, ensure_ascii=False))

    # 2. check all the answers and write the results (also for the communities of earlier runs)
    accepted, rejected, topics = [], Counter(), {}
    for g in all_groups:
        cache = raw_dir / f"{g['id']}.json"
        saved = json.loads(cache.read_text()) if cache.exists() else {}
        if "calls" not in saved:
            continue
        for call in saved["calls"]:
            if call["parsed"] is None:
                rejected["answer not valid JSON"] += 1
                continue
            passages = [chunks[i] for i in call["passages"] if i in chunks]
            topics.setdefault(g["id"], str(call["parsed"].get("topic", "")).strip())
            ok, reason = check(call["parsed"], passages, call["authors"])
            if ok:
                n = sum(1 for q in accepted if q["community"] == g["id"]) + 1
                accepted.append({"id": f"c{g['id']}-q{n}", "community": g["id"], "topic": topics[g["id"]],
                                 **ok, "model": saved["model"]})
            else:
                rejected[reason] += 1

    with (out / "questions.jsonl").open("w", encoding="utf-8") as f:
        for q in accepted:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    md = ["# Generated questions\n"]
    for q in accepted:
        src = "; ".join(f"{s['author']}, {s['section']} (pp. {s['pages'][0]}-{s['pages'][1]})" for s in q["sources"])
        md += [f"## {q['id']} · {q['topic']} · {q['type']}\n", f"**Q:** {q['question']}\n",
               f"**A:** {q['answer']}\n", f"*Sources:* {src}\n"]
    (out / "questions.md").write_text("\n".join(md), encoding="utf-8")

    report = {
        "model": args.model, "communities_asked": len(topics),
        "accepted": len(accepted), "rejected": sum(rejected.values()), "rejected_reasons": dict(rejected),
        "types": dict(Counter(q["type"] for q in accepted)),
        "author_pairs": dict(Counter(" - ".join(q["authors"]) for q in accepted).most_common()),
        "topics": topics,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\ncommunities with an answer: {len(topics)} | questions accepted: {len(accepted)} | "
          f"rejected: {sum(rejected.values())} {dict(rejected) if rejected else ''}")
    print(f"types: {report['types']}")
    print(f"author pairs: {report['author_pairs']}")
    print(f"\nquestions -> {out / 'questions.jsonl'} and {out / 'questions.md'}")


if __name__ == "__main__":
    main()