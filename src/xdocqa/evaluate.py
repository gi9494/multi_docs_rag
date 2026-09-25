"""Step 8 - Evaluation: is every answer really written in its sources?

For every question of step 7 a second LLM (the "judge", JUDGE_MODEL in .env, by default the same as
LLM_MODEL) reads the answer and ONLY the passages it cites, and:
  1. splits the answer into claims, one idea each ("Mill: justice is linked to a personal right");
  2. gives every claim a verdict:
       supported      the passage of that author says it
       not_supported  the passages do not say it (invented, or taken from another author)
       contradicted   the passage of that author says the opposite
     and copies the sentence of the passage that proves it (evidence);
  3. says if the question really needs BOTH authors to be answered.

The evidence is then looked for in the passage text (no LLM): a "supported" claim whose evidence is
not really in the passage is counted as not_supported. So the judge cannot just say "yes".

Writes:
  data/evaluation/raw/<question id>.json   the judge's answer for each question (a re-run skips these)
  data/evaluation/evaluation.jsonl         one line per question: claims, verdicts, scores
  data/evaluation/report.json              numbers for the README
  data/evaluation/final_questions.jsonl    the questions that pass: all claims supported + need both books
  data/evaluation/final_questions.md       the same, easy to read

Usage:
    python -m xdocqa.evaluate
    python -m xdocqa.evaluate --only c25-q2      # one question
    python -m xdocqa.evaluate --force            # judge again
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

from xdocqa.questions import CONTEXT, format_passages, load_inputs

load_dotenv()

ATTEMPTS = 2              # ask again if the answer is not valid JSON


def ask_judge(prompt: str, model: str) -> tuple[dict | None, str]:
    """Call the judge, asking for JSON; try again if the answer has no "claims"."""
    import litellm
    litellm.drop_params = True
    extra = {"num_ctx": CONTEXT} if model.startswith("ollama") else {}
    raw = ""
    for _ in range(ATTEMPTS):
        r = litellm.completion(model=model, messages=[{"role": "user", "content": prompt}],
                               temperature=0, response_format={"type": "json_object"},
                               timeout=float(os.getenv("LLM_TIMEOUT", 1800)), **extra)
        raw = r.choices[0].message.content or ""
        text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
        start, end = text.find("{"), text.rfind("}")
        try:
            data = json.loads(text[start:end + 1]) if start >= 0 else None
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and isinstance(data.get("claims"), list):
            return data, raw
    return None, raw

VERDICTS = ("supported", "not_supported", "contradicted")

PROMPT = """You are checking an exam answer against its sources. Be strict.

SOURCES
{passages}

QUESTION
{question}

ANSWER
{answer}

Task:
1. Split the ANSWER into claims: each claim is one idea attributed to one author.
2. For each claim give a verdict, using ONLY the SOURCES of THAT author:
   - "supported": a source of that author says it (same meaning, even with other words);
   - "not_supported": no source of that author says it (it may be invented, or said by the other author);
   - "contradicted": a source of that author says the opposite.
   Copy as "evidence" the exact sentence of the source that decides the verdict (empty if there is none).
3. "needs_both": true if the QUESTION cannot be answered without the sources of both authors.

Answer with JSON only:
{{"claims": [{{"claim": "...", "author": "...", "verdict": "supported", "evidence": "..."}}], "needs_both": true}}"""


def squash(text: str) -> str:
    """Lowercase letters and digits only: the evidence is found even if spaces, quotes or hyphens differ."""
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKC", text).lower())


def score(judged: dict, sources: list[dict]) -> dict:
    texts = {squash(c["text"]) for c in sources}
    claims = []
    for c in judged.get("claims", []):
        if not isinstance(c, dict) or c.get("verdict") not in VERDICTS:
            continue
        # the evidence may skip words with "..."; every piece must be in the text of one source
        pieces = [squash(p) for p in re.split(r"\.\.\.|…", str(c.get("evidence", "")))]
        pieces = [p for p in pieces if len(p) >= 20]
        found = bool(pieces) and any(all(p in t for p in pieces) for t in texts)
        verdict = c["verdict"]
        if verdict == "supported" and not found:
            verdict = "not_supported"           # the judge said yes, but could not show where
        claims.append({"claim": c.get("claim", ""), "author": c.get("author", ""), "verdict": verdict,
                       "judge_verdict": c["verdict"], "evidence": c.get("evidence", ""), "evidence_found": found})
    n = len(claims)
    counts = Counter(c["verdict"] for c in claims)
    return {"claims": claims, "n_claims": n, "support": round(counts["supported"] / n, 3) if n else 0.0,
            "grounded": n > 0 and counts["supported"] == n, "contradicted": counts["contradicted"] > 0,
            "needs_both": bool(judged.get("needs_both"))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="corpus.yaml")
    ap.add_argument("--chunks", default="data/chunks")
    ap.add_argument("--embeddings", default="data/embeddings/embeddings.npz")
    ap.add_argument("--questions", default="data/questions/questions.jsonl")
    ap.add_argument("--out", default="data/evaluation")
    ap.add_argument("--model", default=os.getenv("JUDGE_MODEL") or os.getenv("LLM_MODEL"))
    ap.add_argument("--only", help="one question id, e.g. c25-q2")
    ap.add_argument("--force", action="store_true", help="judge again the questions already judged")
    args = ap.parse_args()
    if not args.model:
        raise SystemExit("No model: set JUDGE_MODEL or LLM_MODEL in .env")

    chunks, _ = load_inputs(Path(args.corpus), Path(args.chunks), Path(args.embeddings))
    questions = [json.loads(l) for l in Path(args.questions).read_text(encoding="utf-8").splitlines() if l.strip()]
    todo = [q for q in questions if args.only is None or q["id"] == args.only]
    if not todo:
        raise SystemExit("No questions to judge: run xdocqa.questions first (or check --only)")
    out, raw_dir = Path(args.out), Path(args.out) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. ask the judge (questions already judged are skipped)
    for k, q in enumerate(todo, 1):
        cache = raw_dir / f"{q['id']}.json"
        if cache.exists() and not args.force:
            continue
        sources = [chunks[s["chunk_id"]] for s in q["sources"]]
        print(f"[{k}/{len(todo)}] judging {q['id']} ({' - '.join(q['authors'])})...", flush=True)
        data, raw = ask_judge(PROMPT.format(passages=format_passages(sources), question=q["question"],
                                            answer=q["answer"]), args.model)
        cache.write_text(json.dumps({"id": q["id"], "model": args.model, "parsed": data, "raw": raw},
                                    indent=2, ensure_ascii=False))
        if data is None:
            print("   the answer was not valid JSON")

    # 2. score all the judged questions
    rows, failed = [], 0
    for q in questions:
        cache = raw_dir / f"{q['id']}.json"
        if not cache.exists():
            continue
        saved = json.loads(cache.read_text())
        if saved["parsed"] is None:
            failed += 1
            continue
        s = score(saved["parsed"], [chunks[x["chunk_id"]] for x in q["sources"]])
        rows.append({"id": q["id"], "community": q["community"], "authors": q["authors"], "type": q["type"],
                     "question": q["question"], **s, "judge": saved["model"]})

    with (out / "evaluation.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 3. keep the good questions: every claim supported, and the question needs both books
    good_ids = {r["id"] for r in rows if r["grounded"] and r["needs_both"]}
    final = [q for q in questions if q["id"] in good_ids]
    with (out / "final_questions.jsonl").open("w", encoding="utf-8") as f:
        for q in final:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    md = ["# Final questions (checked by the judge)\n"]
    for q in final:
        src = "; ".join(f"{s['author']}, {s['section']} (pp. {s['pages'][0]}-{s['pages'][1]})" for s in q["sources"])
        md += [f"## {q['id']} · {q['topic']} · {q['type']}\n", f"**Q:** {q['question']}\n",
               f"**A:** {q['answer']}\n", f"*Sources:* {src}\n"]
    (out / "final_questions.md").write_text("\n".join(md), encoding="utf-8")

    n = len(rows)
    pct =lambda x: round(100 * x / n, 1) if n else 0.0
    claims = [c for r in rows for c in r["claims"]]
    by_author = defaultdict(Counter)
    for c in claims:
        by_author[c["author"]][c["verdict"]] += 1
    report = {
        "judge": args.model, "questions_judged": n, "judge_failed": failed,
        "fully_grounded_pct": pct(sum(r["grounded"] for r in rows)),
        "mean_support": round(sum(r["support"] for r in rows) / n, 3) if n else 0.0,
        "with_contradiction_pct": pct(sum(r["contradicted"] for r in rows)),
        "needs_both_authors_pct": pct(sum(r["needs_both"] for r in rows)),
        "claims": dict(Counter(c["verdict"] for c in claims)),
        "judge_yes_without_evidence": sum(1 for c in claims if c["judge_verdict"] == "supported" and not c["evidence_found"]),
        "claims_by_author": {a: dict(v) for a, v in sorted(by_author.items())},
        "final_questions": len(final),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\njudge: {args.model} | questions judged: {n} (failed: {failed})")
    print(f"fully grounded: {report['fully_grounded_pct']}% | mean support: {report['mean_support']} | "
          f"with a contradiction: {report['with_contradiction_pct']}% | need both authors: {report['needs_both_authors_pct']}%")
    print(f"claims: {report['claims']} | judge said 'supported' without real evidence: {report['judge_yes_without_evidence']}")
    for r in sorted(rows, key=lambda r: r["support"])[:3]:
        print(f"\n  [{r['id']}] support {r['support']}: {r['question'][:90]}")
        for c in r["claims"]:
            if c["verdict"] != "supported":
                print(f"     {c['verdict']:>13} | {c['author']}: {c['claim'][:90]}")
    print(f"\nkept: {len(final)} of {n} questions -> {out / 'final_questions.md'}")
    print(f"evaluation -> {out / 'evaluation.jsonl'}, {out / 'report.json'}")


if __name__ == "__main__":
    main()