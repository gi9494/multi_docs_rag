"""PDF conversion: turn any PDF (born-digital, scanned, or mixed) into reliable text.

For every PDF in data/raw/ this module produces:
  data/converted/<name>.pdf    the same PDF, where every page now has a reliable text layer
  data/pages/<name>.jsonl      one line per page: its text, where the text came from, a quality score
  data/converted/report.json   a summary per document

How it works, page by page:
  1. read the text layer that is already in the PDF (if any)
  2. score it: what share of its words are real words of the language?
  3. decide the route:
       - "native": the existing text is good -> the page is copied untouched
       - "ocr":    no text on a scanned page, or text of low quality -> the page is
                   rendered as an image, read with OCR (RapidOCR), and rebuilt as
                   image + invisible text layer
     if a page had text and OCR does not score better, the original text is kept
  4. re-read every page from the converted PDF and score it again

Usage:
    python -m xdocqa.pdf_conversion
    python -m xdocqa.pdf_conversion --dict-lang en --quality-min 0.85
"""

from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import pymupdf  # reads, renders and writes PDFs
import numpy as np
from wordfreq import zipf_frequency  # word frequency lists, used here as a dictionary

# ------------------------------------------------------------------ settings

MIN_WORDS = 20          # below this, a page counts as "without text"
QUALITY_MIN = 0.85      # below this share of real words, the text is considered bad
IMAGE_COVERAGE = 0.5    # an image covering > 50% of the page suggests a scan
OCR_DPI = 300           # resolution used to render pages for OCR
JPEG_QUALITY = 75       # compression of the page image stored in the converted PDF

WORD_RE = re.compile(r"[^\W\d_]{2,}")  # letter-only tokens of 2+ characters, any alphabet


# ------------------------------------------------------------------ quality score

def words(text: str) -> list[str]:
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)  # re-join "vir-\ntue" -> "virtue"
    return [w.lower() for w in WORD_RE.findall(text)]


@lru_cache(maxsize=200_000)
def is_known(word: str, lang: str) -> bool:
    return zipf_frequency(word, lang) > 0


def text_quality(text: str, lang: str) -> tuple[float, int]:
    """Share of tokens that are real words of `lang`, plus the number of tokens."""
    toks = words(text)
    if not toks:
        return 0.0, 0
    known = sum(is_known(w, lang) for w in toks)
    return known / len(toks), len(toks)


# ------------------------------------------------------------------ routing

def image_coverage(page: pymupdf.Page) -> float:
    """Largest share of the page covered by a single image (0 = no images, 1 = full page)."""
    page_area = page.rect.width * page.rect.height
    best = 0.0
    for img in page.get_image_info():
        x0, y0, x1, y1 = img["bbox"]
        best = max(best, (x1 - x0) * (y1 - y0) / page_area)
    return min(best, 1.0)


def decide_route(n_words: int, quality: float, coverage: float,
                 quality_min: float = QUALITY_MIN) -> tuple[str, str]:
    """Return (route, reason). Kept free of PDF objects so it is easy to test."""
    if n_words < MIN_WORDS:
        if coverage >= IMAGE_COVERAGE:
            return "ocr", "scan_without_text"
        return "native", "sparse_page"          # title page, blank page, short epigraph...
    if quality < quality_min:
        return "ocr", "low_quality_text"
    return "native", "good_text"


# ------------------------------------------------------------------ OCR engine

def load_ocr_engine():
    """Return a function image -> [(box, text, confidence), ...] in reading order.

    Two packages provide RapidOCR; pyproject.toml installs the right one for your Python:
      - rapidocr-onnxruntime (Python <= 3.12): models are shipped inside the package
      - rapidocr             (Python >= 3.13): models are downloaded on first use
    """
    try:
        from rapidocr_onnxruntime import RapidOCR
        engine = RapidOCR()

        def run(img):
            result, _ = engine(img)
            return [(box, txt, float(score)) for box, txt, score in (result or [])]
    except ImportError:
        from rapidocr import RapidOCR
        engine = RapidOCR()

        def run(img):
            out = engine(img)
            if out.boxes is None:
                return []
            return list(zip(out.boxes.tolist(), out.txts, map(float, out.scores)))
    return run


def render(page: pymupdf.Page) -> pymupdf.Pixmap:
    return page.get_pixmap(dpi=OCR_DPI, colorspace=pymupdf.csRGB, alpha=False)


def to_array(pix: pymupdf.Pixmap) -> np.ndarray:
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return img[:, :, ::-1].copy()  # RGB -> BGR, the channel order OCR libraries expect


def ocr_page(src_page: pymupdf.Page, ocr) -> tuple[pymupdf.Pixmap, list]:
    """Render the page as an image and read it. Returns (image, lines)."""
    pix = render(src_page)
    return pix, ocr(to_array(pix))


def add_ocr_page(out: pymupdf.Document, src_page: pymupdf.Page, pix: pymupdf.Pixmap, lines: list) -> None:
    """Append to `out` a copy of src_page made of: page image + invisible OCR text.

    The old text layer is dropped entirely (it was missing or bad).
    """
    page = out.new_page(width=src_page.rect.width, height=src_page.rect.height)
    page.insert_image(page.rect, stream=pix.tobytes("jpeg", jpg_quality=JPEG_QUALITY))

    scale = 72 / OCR_DPI  # pixels -> PDF points
    for box, text, _ in lines:
        if not text.strip():
            continue
        xs = [p[0] * scale for p in box]
        ys = [p[1] * scale for p in box]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        height = y1 - y0
        # font size: as tall as the line, but never wider than the box
        width_at_1pt = pymupdf.get_text_length(text, fontname="helv", fontsize=1)
        size = min(height * 0.8, (x1 - x0) / width_at_1pt) if width_at_1pt else height * 0.8
        page.insert_text((x0, y1 - 0.2 * height), text,
                         fontname="helv", fontsize=max(size, 1), render_mode=3)  # 3 = invisible


# ------------------------------------------------------------------ main

def open_pdf(path: Path) -> pymupdf.Document:
    """Open a PDF and fail with a clear message if it cannot be used."""
    doc = pymupdf.open(path)
    if doc.needs_pass:
        raise ValueError("the PDF is password-protected")
    if doc.page_count == 0:
        head = path.read_bytes()[:16]
        raise ValueError(
            f"0 readable pages (size {path.stat().st_size / 1e6:.1f} MB, first bytes {head!r}). "
            "A real PDF starts with b'%PDF': otherwise the file is not a PDF, "
            "or it is damaged. Open it in Preview, or download it again."
        )
    return doc


def analyse(src: pymupdf.Document, lang: str, quality_min: float) -> list[dict]:
    rows = []
    for i, page in enumerate(src, start=1):
        q, n = text_quality(page.get_text(), lang)
        cov = image_coverage(page)
        route, reason = decide_route(n, q, cov, quality_min)
        rows.append({"page": i, "route": route, "reason": reason,
                     "quality_before": round(q, 3), "words_before": n,
                     "image_coverage": round(cov, 2)})
    return rows


def convert(pdf_path: Path, out_pdf: Path, out_pages: Path,
            dict_lang: str, quality_min: float, get_ocr) -> dict:
    src = open_pdf(pdf_path)
    rows = analyse(src, dict_lang, quality_min)

    out = pymupdf.open()
    for row, page in zip(rows, src):
        if row["route"] == "ocr":
            print(f"  OCR page {row['page']}/{len(rows)}", end="\r")
            pix, lines = ocr_page(page, get_ocr())
            ocr_q, _ = text_quality("\n".join(t for _, t, _ in lines), dict_lang)
            row["ocr_quality"] = round(ocr_q, 3)
            row["ocr_confidence"] = round(sum(s for *_, s in lines) / len(lines), 3) if lines else 0.0

            # the page had text, and OCR did not read it better: keep the original.
            # typical case: a correct quotation in Latin or German, which only *looks* bad
            if row["reason"] == "low_quality_text" and ocr_q <= row["quality_before"]:
                row["route"], row["reason"] = "native", "ocr_not_better"
            else:
                add_ocr_page(out, page, pix, lines)
                continue
        out.insert_pdf(src, from_page=row["page"] - 1, to_page=row["page"] - 1)
    out.set_toc(src.get_toc())  # keep the bookmarks: page numbers did not change
    out.save(out_pdf, garbage=3, deflate=True)
    out.close()
    src.close()

    # re-read everything from the converted PDF: this is the text the rest of the pipeline uses
    with pymupdf.open(out_pdf) as doc, out_pages.open("w", encoding="utf-8") as f:
        for row, page in zip(rows, doc):
            text = page.get_text()
            q, n = text_quality(text, dict_lang)
            row.update({"text": text, "quality": round(q, 3), "words": n})
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return summarize(pdf_path.stem, rows)


def summarize(name: str, rows: list[dict]) -> dict:
    ocr = [r for r in rows if r["route"] == "ocr"]
    scored = [r for r in rows if r["words"] >= MIN_WORDS]
    reasons: dict[str, int] = {}
    for r in rows:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    return {
        "document": name,
        "pages": len(rows),
        "native_pages": len(rows) - len(ocr),
        "ocr_pages": len(ocr),
        "reasons": reasons,
        "mean_quality": round(sum(r["quality"] for r in scored) / max(len(scored), 1), 3),
        # low score but original kept (OCR was not better): often foreign-language passages
        "kept_original": [r["page"] for r in rows if r["reason"] == "ocr_not_better"],
        "still_low_quality": [r["page"] for r in scored if r["quality"] < QUALITY_MIN],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data")
    ap.add_argument("--dict-lang", default="en", help="wordfreq language for the quality score")
    ap.add_argument("--quality-min", type=float, default=QUALITY_MIN)
    args = ap.parse_args()

    out_pdf_dir = Path(args.out) / "converted"
    out_pages_dir = Path(args.out) / "pages"
    out_pdf_dir.mkdir(parents=True, exist_ok=True)
    out_pages_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(Path(args.raw).glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {args.raw}/")

    engine = None

    def get_ocr():  # the OCR model is loaded only if some page actually needs it
        nonlocal engine
        if engine is None:
            engine = load_ocr_engine()
        return engine

    reports = []
    for pdf in pdfs:
        print(f"\n{pdf.name}")
        try:
            r = convert(pdf, out_pdf_dir / pdf.name, out_pages_dir / f"{pdf.stem}.jsonl",
                        args.dict_lang, args.quality_min, get_ocr)
        except Exception as e:  # one broken PDF should not stop the others
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue
        reports.append(r)
        print(f"  pages {r['pages']} | native {r['native_pages']} | ocr {r['ocr_pages']} "
              f"| mean quality {r['mean_quality']}")
        print(f"  reasons: {r['reasons']}")
        if r["still_low_quality"]:
            print(f"  still low quality: {r['still_low_quality'][:15]}")
        if r["kept_original"]:
            print(f"  low score, original kept (OCR not better): {r['kept_original'][:15]}")

    (out_pdf_dir / "report.json").write_text(json.dumps(reports, indent=2))
    print(f"\nreport -> {out_pdf_dir / 'report.json'}")


if __name__ == "__main__":
    main()