"""Document structure: PageIndex builds the table-of-contents tree of every converted PDF,
with a short LLM summary of each section.

For every PDF in data/converted/ this module produces:
  data/trees/<name>.json     the tree: sections with title, first and last page, summary, sub-sections
  data/trees/report.json     a summary per document

How it works:
  - documents are processed ONE AT A TIME;
  - "flash" mode first: the tree is built from the PDF's bookmarks and layout, then the LLM
    writes the summaries;
  - if flash finds no real structure, "standard" mode: the LLM reads the pages, rebuilds the
    headings, then writes the summaries;
  - the summaries are checked: small models sometimes answer in JSON, cut the answer or
    refuse; those are cleaned, and the unusable ones are redone once for that section only;
  - the tree is saved and its first levels are printed, to check it by eye.

Trees are cached: a document that already has a tree is skipped (use --force to rebuild).

Usage:
    python -m xdocqa.structure
    python -m xdocqa.structure --only kant1785 --force
    python -m xdocqa.structure --mode standard
    python -m xdocqa.structure --no-summary      # tree only, no LLM (flash mode)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads the settings in the .env file of the project folder (LLM_MODEL, API keys...)

MIN_LEAVES = 3  # fewer leaves than this = flash found no real structure
# how many summaries PageIndex may ask the LLM at the same time (flash mode).
# Keep it low with a local model: Ollama answers one request at a time anyway.
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "2"))
# seconds before a single LLM call is abandoned. LiteLLM's default (600) is too short for a
# large local model reading many pages at once.
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "1800"))


# ------------------------------------------------------------------ progress

class LLMProgress:
    """Count the LLM calls made by PageIndex and print them live, with the first errors.

    PageIndex calls the LLM through LiteLLM, which lets us register callbacks:
    functions that LiteLLM runs after every successful or failed call.
    """

    def __init__(self):
        import litellm
        litellm.request_timeout = LLM_TIMEOUT  # applies to every call PageIndex makes
        self.reset()
        litellm.success_callback.append(self.on_success)
        litellm.failure_callback.append(self.on_failure)

    def reset(self):
        self.ok, self.failed, self.errors, self.t0 = 0, 0, set(), time.time()

    def show(self):
        print(f"  LLM calls: {self.ok} done, {self.failed} failed | {time.time() - self.t0:.0f}s elapsed",
              end="\r", flush=True)

    def on_success(self, kwargs, response, start, end):
        self.ok += 1
        self.show()

    def on_failure(self, kwargs, response, start, end):
        self.failed += 1
        error = str(kwargs.get("exception", "unknown error"))[:300]
        if error not in self.errors:  # print each different error once, not a thousand times
            self.errors.add(error)
            print(f"\n  LLM call failed: {error}")
        self.show()


# ------------------------------------------------------------------ building the tree

def build_flash(pdf: Path, model: str | None, summary: bool) -> dict:
    from pageindex import page_index_flash
    return page_index_flash(
        str(pdf),
        use_embedded_toc=True,   # use the PDF bookmarks when they look trustworthy
        summary=summary,         # one short LLM summary per section
        summary_model=model,
        summary_concurrency=LLM_CONCURRENCY,
        optimize=False,          # keep the author's structure as it is: we size the chunks ourselves later
    )


def build_standard(pdf: Path, model: str, summary: bool) -> dict:
    from pageindex import page_index
    return page_index(
        str(pdf),
        model=model,
        if_add_node_id="yes",
        if_add_node_summary="yes" if summary else "no",
        if_add_node_text="no",   # the text of each section comes from our own data/pages files
    )


def build_tree(pdf: Path, mode: str, model: str | None, summary: bool) -> dict:
    t0 = time.time()
    tree, used = None, mode
    if mode == "flash":
        print(f"  building the tree (flash){', then the summaries' if summary else ', no LLM'}...")
        tree = build_flash(pdf, model, summary)
        if tree.get("toc_source") in ("pages", "unreadable") or len(leaves(tree["structure"])) < MIN_LEAVES:
            print(f"\n  flash found no real structure (source: {tree.get('toc_source')}), trying standard mode")
            tree, used = None, "standard"
    if tree is None:
        print("  building the tree (standard: the LLM reads the pages)...")
        tree = build_standard(pdf, model, summary)
    print()  # end the live progress line
    tree["_meta"] = {"mode": used, "model": model if (summary or used == "standard") else None,
                     "summaries": summary, "seconds": round(time.time() - t0, 1)}
    return tree


# ------------------------------------------------------------------ cleaning the summaries

REPAIR_MAX_CHARS = 12_000  # text sent to the LLM when a single summary has to be redone
REPAIR_PROMPT = """Summarise this section of "{doc}" (section: "{title}") in 2-4 sentences.
Reply with plain text only: no JSON, no bullet points, no introduction like "Here is a summary".

Text:
{text}"""


def clean_summary(text: str | None) -> str | None:
    """Turn what the LLM returned into plain text, or None if nothing usable is left.

    Small models sometimes answer in JSON ({"summary": ...} or {"points": [...]}),
    wrap the answer in ```json fences, cut it half-way, or refuse ("Please provide the text").
    """
    if not text:
        return None
    t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text.strip()).strip()  # drop ```json fences

    if t.startswith(("{", "[")):
        try:
            data = json.loads(t)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and isinstance(data.get("summary"), str):
            t = data["summary"]
        elif isinstance(data, dict) and isinstance(data.get("points"), list):
            t = " ".join(str(p).rstrip(".") + "." for p in data["points"])
        elif isinstance(data, list):
            t = " ".join(str(p).rstrip(".") + "." for p in data)
        else:  # broken JSON (cut half-way, badly closed): salvage what we can
            m = re.search(r'"summary"\s*:\s*"(.+)', t, re.DOTALL)
            if m:  # a "summary" field: take its text, drop the broken ending
                t = re.sub(r'["”\s}\]]+$', "", m.group(1))
            else:  # otherwise keep the sentences that are inside quotes
                sentences = re.findall(r'"([^"]{30,})"', t)
                t = " ".join(s.rstrip(".") + "." for s in sentences)

    t = t.strip()
    if len(t) < 40 or t.lower().startswith(("please provide", "i need the")):
        return None
    return t


def load_pages(path: Path) -> dict[int, str]:
    """Page number -> text, from the output of step 1."""
    pages = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            pages[row["page"]] = row["text"]
    return pages


def resummarize(node: dict, doc: str, pages: dict[int, str], model: str) -> str | None:
    import litellm
    text = "\n".join(pages.get(p, "") for p in range(node["start_index"], node["end_index"] + 1))
    if len(text.split()) < 30:  # nothing to summarise (title page, blank page...)
        return None
    prompt = REPAIR_PROMPT.format(doc=doc, title=node["title"], text=text[:REPAIR_MAX_CHARS])
    try:
        r = litellm.completion(model=model, messages=[{"role": "user", "content": prompt}], temperature=0)
        return clean_summary(r.choices[0].message.content)
    except Exception as e:
        print(f"\n  could not redo the summary of '{node['title']}': {e}")
        return None


def repair_summaries(tree: dict, pages: dict[int, str] | None, model: str | None) -> dict:
    """Clean every summary; redo once, with the LLM, the ones that are unusable. Returns counts."""
    doc = tree.get("doc_title") or tree.get("doc_name", "")
    counts = {"ok": 0, "cleaned": 0, "redone": 0, "missing": 0}
    for node, _ in walk(tree["structure"]):
        if "summary" not in node:
            continue
        original = node["summary"]
        fixed = clean_summary(original)
        if fixed is None and pages is not None and model:
            fixed = resummarize(node, doc, pages, model)
            counts["redone" if fixed else "missing"] += 1
        elif fixed is None:
            counts["missing"] += 1
        else:
            counts["ok" if fixed == (original or "").strip() else "cleaned"] += 1
        node["summary"] = fixed
    return counts


# ------------------------------------------------------------------ looking at the tree

def walk(nodes: list[dict], depth: int = 0):
    """Yield (node, depth) for every node, in reading order."""
    for n in nodes:
        yield n, depth
        yield from walk(n.get("nodes") or [], depth + 1)


def leaves(nodes: list[dict]) -> list[dict]:
    return [n for n, _ in walk(nodes) if not n.get("nodes")]


def describe(name: str, tree: dict) -> dict:
    nodes = tree["structure"]
    leaf_pages = [n["end_index"] - n["start_index"] + 1 for n in leaves(nodes)] or [0]
    all_nodes = [n for n, _ in walk(nodes)]
    return {
        "document": name,
        "mode": tree["_meta"]["mode"],
        "source": tree.get("toc_source") or "llm",
        "sections": len(all_nodes),
        "leaves": len(leaves(nodes)),
        "depth": max((d for _, d in walk(nodes)), default=0) + 1,
        "leaf_pages": {"min": min(leaf_pages), "median": statistics.median(leaf_pages),
                       "max": max(leaf_pages)},
        "summaries": sum(1 for n in all_nodes if n.get("summary")),
        "seconds": tree["_meta"]["seconds"],
    }


def print_outline(tree: dict, max_depth: int = 2, max_lines: int = 25) -> None:
    lines = [f"  {'   ' * d}- {n['title']}  (pp. {n['start_index']}-{n['end_index']})"
             for n, d in walk(tree["structure"]) if d < max_depth]
    print("\n".join(lines[:max_lines]))
    if len(lines) > max_lines:
        print(f"  ... {len(lines) - max_lines} more")


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--converted", default="data/converted")
    ap.add_argument("--pages", default="data/pages", help="page texts from step 1 (to redo broken summaries)")
    ap.add_argument("--out", default="data/trees")
    ap.add_argument("--mode", choices=["flash", "standard"], default="flash")
    ap.add_argument("--model", default=os.getenv("LLM_MODEL"),
                    help="LiteLLM model name; default: LLM_MODEL in .env")
    ap.add_argument("--only", nargs="*", help="process only these documents (file names without .pdf)")
    ap.add_argument("--force", action="store_true", help="rebuild trees that already exist")
    ap.add_argument("--no-summary", action="store_true", help="tree only, no summaries")
    args = ap.parse_args()
    summary = not args.no_summary
    if not args.model and (summary or args.mode == "standard"):
        raise SystemExit("No model configured: set LLM_MODEL in .env (see .env.example) or pass --model")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(Path(args.converted).glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs in {args.converted}/: run xdocqa.pdf_conversion first")

    progress = LLMProgress() if args.model else None

    reports = []
    for pdf in pdfs:  # one document at a time
        if args.only and pdf.stem not in args.only:
            continue
        dest = out / f"{pdf.stem}.json"
        print(f"\n{pdf.name}")

        if dest.exists() and not args.force:
            print("  tree already exists, loading it (use --force to rebuild)")
            tree = json.loads(dest.read_text())
        else:
            if progress:
                progress.reset()
            try:
                tree = build_tree(pdf, args.mode, args.model, summary)
            except Exception as e:  # one failing document should not stop the others
                print(f"\n  FAILED: {type(e).__name__}: {e}")
                continue
            dest.write_text(json.dumps(tree, indent=2, ensure_ascii=False))

        # check the summaries (also on trees built before: they get repaired too)
        if any("summary" in n for n, _ in walk(tree["structure"])):
            pages_file = Path(args.pages) / f"{pdf.stem}.jsonl"
            pages = load_pages(pages_file) if pages_file.exists() else None
            c = repair_summaries(tree, pages, args.model)
            print(f"\n  summaries: {c['ok']} ok, {c['cleaned']} cleaned, {c['redone']} redone, {c['missing']} missing")
            dest.write_text(json.dumps(tree, indent=2, ensure_ascii=False))

        r = describe(pdf.stem, tree)
        reports.append(r)
        lp = r["leaf_pages"]
        print(f"  mode {r['mode']} (source: {r['source']}) | sections {r['sections']} | leaves {r['leaves']} "
              f"| depth {r['depth']} | pages per leaf min/median/max {lp['min']}/{lp['median']}/{lp['max']} "
              f"| summaries {r['summaries']} | {r['seconds']}s")
        print_outline(tree)

    (out / "report.json").write_text(json.dumps(reports, indent=2))
    print(f"\nreport -> {out / 'report.json'}")


if __name__ == "__main__":
    main()