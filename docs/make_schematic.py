"""Generate docs/architecture.pdf: a three-page schematic of grounded-rag.

    pip install reportlab && python docs/make_schematic.py

Page 1: the request path (indexing + query). Page 2: the evaluation and CI loop.
Page 3: a component reference table mapping each part to its source file.
"""

from __future__ import annotations

import math
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

OUT = Path(__file__).with_name("architecture.pdf")
W, H = landscape(A4)
M = 28  # page margin

INK = colors.HexColor("#1f2933")
MUTED = colors.HexColor("#52606d")
LINE = colors.HexColor("#7b8794")

# One colour family per layer; the legend on page 1 explains them.
LAYERS = {
    "data": (colors.HexColor("#f0f4f8"), colors.HexColor("#829ab1")),
    "retrieval": (colors.HexColor("#e3f2fd"), colors.HexColor("#1e6fb8")),
    "generation": (colors.HexColor("#f3e8fd"), colors.HexColor("#7b3fb5")),
    "safety": (colors.HexColor("#fff4e5"), colors.HexColor("#c26a00")),
    "eval": (colors.HexColor("#e6f6ec"), colors.HexColor("#2f8a4c")),
    "infra": (colors.HexColor("#eef0f2"), colors.HexColor("#616e7c")),
}


def header(c: canvas.Canvas, title: str, subtitle: str, page: int) -> None:
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(M, H - M - 12, title)
    c.setFont("Helvetica", 10)
    c.setFillColor(MUTED)
    c.drawString(M, H - M - 30, subtitle)
    c.setFont("Helvetica", 8)
    footer = f"grounded-rag · github.com/dulminda/grounded-rag · page {page}/3"
    c.drawRightString(W - M, M - 12, footer)
    c.setStrokeColor(colors.HexColor("#d9e2ec"))
    c.setLineWidth(0.8)
    c.line(M, H - M - 40, W - M, H - M - 40)


def band(c: canvas.Canvas, x: float, y: float, w: float, h: float, label: str) -> None:
    """A dashed container grouping a stage of the system."""
    c.setStrokeColor(colors.HexColor("#bcccdc"))
    c.setDash(3, 3)
    c.setLineWidth(0.8)
    c.roundRect(x, y, w, h, 8, stroke=1, fill=0)
    c.setDash()
    c.setFillColor(MUTED)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x + 8, y + h - 13, label.upper())


def box(c: canvas.Canvas, x: float, y: float, w: float, h: float, title: str,
        lines: list[str], layer: str, mono: str | None = None) -> tuple[float, float, float, float]:
    fill, stroke = LAYERS[layer]
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(1.2)
    c.roundRect(x, y, w, h, 6, stroke=1, fill=1)
    c.setFillColor(stroke)
    c.rect(x, y + h - 4, w, 4, stroke=0, fill=1)  # coloured top edge
    ty = y + h - 17
    c.setFillColor(INK)
    for t in simpleSplit(title, "Helvetica-Bold", 9, w - 12):
        c.setFont("Helvetica-Bold", 9)
        c.drawString(x + 6, ty, t)
        ty -= 11
    c.setFont("Helvetica", 7.3)
    c.setFillColor(colors.HexColor("#323f4b"))
    for line in lines:
        if not line:
            ty -= 4  # paragraph gap
            continue
        for t in simpleSplit(line, "Helvetica", 7.3, w - 12):
            c.drawString(x + 6, ty, t)
            ty -= 9
    # Several paths are separated by spaces; use " | " to keep a multi-word command on one line.
    mono_lines = (mono.split(" | ") if " | " in mono else mono.split()) if mono else []
    c.setFont("Courier", 5.8)
    c.setFillColor(stroke)
    for i, t in enumerate(reversed(mono_lines)):
        c.drawString(x + 6, y + 6 + i * 7.5, t)
    last_baseline = ty + 9
    if last_baseline < y + 6 + 7.5 * len(mono_lines) + 3:
        raise ValueError(f"text overflows box {title!r}")
    return x, y, w, h


def arrow(c: canvas.Canvas, points: list[tuple[float, float]], label: str = "",
          color=LINE, label_pos: int = 0, dashed: bool = False) -> None:
    """Polyline with an arrowhead at the last point; label sits near segment `label_pos`."""
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(1.1)
    if dashed:
        c.setDash(2, 2)
    p = c.beginPath()
    p.moveTo(*points[0])
    for pt in points[1:]:
        p.lineTo(*pt)
    c.drawPath(p, stroke=1, fill=0)
    c.setDash()
    (x1, y1), (x2, y2) = points[-2], points[-1]
    a = math.atan2(y2 - y1, x2 - x1)
    s = 5.5
    head = c.beginPath()
    head.moveTo(x2, y2)
    head.lineTo(x2 - s * math.cos(a - 0.4), y2 - s * math.sin(a - 0.4))
    head.lineTo(x2 - s * math.cos(a + 0.4), y2 - s * math.sin(a + 0.4))
    head.close()
    c.drawPath(head, stroke=0, fill=1)
    if label:
        (lx1, ly1), (lx2, ly2) = points[label_pos], points[label_pos + 1]
        c.setFont("Helvetica-Oblique", 6.8)
        c.setFillColor(MUTED)
        c.drawCentredString((lx1 + lx2) / 2, (ly1 + ly2) / 2 + 3, label)


def right(b):  # midpoint of a box's right edge
    return b[0] + b[2], b[1] + b[3] / 2


def left(b):
    return b[0], b[1] + b[3] / 2


def top(b, f: float = 0.5):
    return b[0] + b[2] * f, b[1] + b[3]


def bottom(b, f: float = 0.5):
    return b[0] + b[2] * f, b[1]


def chain(c, boxes, labels=()):
    labels = list(labels) + [""] * len(boxes)
    for a, b, lab in zip(boxes, boxes[1:], labels, strict=False):
        arrow(c, [right(a), left(b)], lab)


# --------------------------------------------------------------------------------------------
def page_architecture(c: canvas.Canvas) -> None:
    header(c, "grounded-rag: how the components fit together",
           "Indexing runs once per corpus. The query path runs on every request. One YAML config "
           "controls every stage; a tracer records each one.", 1)

    # ---- Indexing band -----------------------------------------------------------------------
    band(c, M, 382, W - 2 * M, 128, "1 · Indexing (offline, once per corpus)")
    bw, bh, y = 118, 82, 398
    docs = box(c, M + 10, y, bw, bh, "Documents",
               ["SciFact: 5,183 abstracts", "or your own .md / .txt / .jsonl"], "data",
               "ingest/loaders.py")
    chunk = box(c, M + 150, y, bw, bh, "Chunker",
                ["Sentence-aware splitting", "never cuts a sentence", "optional overlap"],
                "data", "ingest/chunking.py")
    bm25i = box(c, M + 300, y + 44, 150, 40, "BM25 index",
                ["sparse term-weight matrix"], "retrieval")
    densei = box(c, M + 300, y - 2, 150, 40, "Dense index",
                 ["LSA (offline) or bge-small vectors"], "retrieval")
    saved = box(c, M + 482, y, 128, bh, "Saved index",
                ["Everything needed to search,", "plus the config that built it"], "infra",
                "indexes/*.pkl")
    cfg = box(c, W - M - 140, y, 128, bh, "Pipeline config",
              ["One YAML = one system", "chunking, retrieval,", "fusion, rerank, model",
               "typos fail loudly"], "infra", "configs/*.yaml")
    chain(c, [docs, chunk], ["load"])
    jx = M + 284  # junction between the chunker and the two indexes
    arrow(c, [right(chunk), (jx, right(chunk)[1]), (jx, left(bm25i)[1]), left(bm25i)])
    arrow(c, [(jx, right(chunk)[1]), (jx, left(densei)[1]), left(densei)])
    kx = M + 466
    arrow(c, [right(bm25i), (kx, right(bm25i)[1]), (kx, right(saved)[1]), left(saved)])
    arrow(c, [right(densei), (kx, right(densei)[1]), (kx, right(saved)[1])])
    arrow(c, [left(cfg), right(saved)], "settings", dashed=True)

    # ---- Query band ----------------------------------------------------------------------------
    band(c, M, 150, W - 2 * M, 222, "2 · Query path (every request)")
    y, bh = 178, 150
    bw, gap = 88, 9.5
    xs = [M + 10 + i * (bw + gap) for i in range(8)]
    entry = box(c, xs[0], y, bw, bh, "Entry point",
                ["CLI:", "grounded search", "grounded ask", "", "HTTP API:", "/v1/search",
                 "/v1/answer", "/v1/verify"], "infra", "cli.py api/app.py")
    retr = box(c, xs[1], y, bw, bh, "Retrievers",
               ["BM25: exact word", "matches, ~0.2 ms", "", "Dense: matches", "meaning, ~3 ms",
                "", "each returns its top 100"], "retrieval", "retrieval/bm25.py dense.py")
    fuse = box(c, xs[2], y, bw, bh, "Fusion",
               ["Merges the two ranked lists", "", "RRF: by rank",
                "Linear: weighted, normalised scores", "", "weight tuned on train"],
               "retrieval", "retrieval/hybrid.py")
    rerank = box(c, xs[3], y, bw, bh, "Reranker (optional)",
                 ["Cross-encoder rescores the top 50 by reading query and passage together",
                  "", "keeps top 5"], "retrieval", "retrieval/rerank.py")
    prompt = box(c, xs[4], y, bw, bh, "Prompt builder",
                 ["Numbers the 5 sources", "Escapes them inside <source> tags", "",
                  "Sources are marked as untrusted data"], "safety", "generation/prompts.py")
    claude = box(c, xs[5], y, bw, bh, "Claude",
                 ["Answers only from the sources", "", "Output forced into a JSON schema:",
                  "answer, citation ids,", "insufficient-context flag"], "generation",
                 "generation/llm.py")
    valid = box(c, xs[6], y, bw, bh, "Citation validator",
                ["Drops citation ids that don't exist",
                 "No valid citation -> abstain",
                 "Uncited SUPPORT or CONTRADICT -> NOT_ENOUGH_INFO"], "safety",
                "generation/answerer.py")
    resp = box(c, xs[7], y, bw, bh, "Response",
               ["Answer or verdict", "Validated citations", "Ranked sources", "",
                "Trace: time per stage", "Tokens and $ cost"], "infra", "pipeline.py")
    chain(c, [entry, retr, fuse, rerank, prompt, claude, valid, resp])

    # index feeds the retrievers
    arrow(c, [bottom(saved, 0.5), (bottom(saved)[0], 354), (top(retr)[0], 354), top(retr)],
          "loaded at startup", label_pos=1, dashed=True)

    # ---- Cross-cutting --------------------------------------------------------------------
    y = 62
    box(c, M, y, 250, 72, "Tracer (cross-cutting)",
                ["Every stage runs inside a span: start, duration and attributes "
                 "(result counts, token usage, USD cost). Returned with each response."],
                "infra", "tracing.py")
    box(c, M + 262, y, 250, 72, "Failure handling",
              ["Model refusal -> HTTP 422 (server-side fallback model tried first).",
               "API error, truncation or schema mismatch -> HTTP 502.",
               "No credentials -> search still works, generation returns 503."],
              "safety", "generation/llm.py api/app.py")

    # legend
    lx, ly = M + 540, y + 58
    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(INK)
    c.drawString(lx, ly + 6, "Colour key")
    names = [("data", "Data and ingestion"), ("retrieval", "Search"),
             ("generation", "LLM generation"), ("safety", "Guardrails and validation"),
             ("infra", "Interfaces and plumbing"), ("eval", "Evaluation (page 2)")]
    for i, (k, n) in enumerate(names):
        cx = lx + (i % 2) * 125
        cy = ly - 12 - (i // 2) * 15
        fill, stroke = LAYERS[k]
        c.setFillColor(fill)
        c.setStrokeColor(stroke)
        c.roundRect(cx, cy - 2, 14, 10, 2, stroke=1, fill=1)
        c.setFillColor(INK)
        c.setFont("Helvetica", 7.5)
        c.drawString(cx + 19, cy, n)


# --------------------------------------------------------------------------------------------
def page_evaluation(c: canvas.Canvas) -> None:
    header(c, "grounded-rag: how quality is measured and protected",
           "Settings are tuned on the training claims and reported on separate dev claims. CI "
           "re-runs the benchmark on every push and blocks regressions.", 2)

    box(c, M, 268, 150, 220, "SciFact benchmark",
               ["Allen Institute for AI, 2020", "", "5,183 paper abstracts", "",
                "Train: 809 claims", "used only for tuning", "",
                "Dev: 300 claims", "used only for reporting", "",
                "Expert labels per claim:", "relevant abstracts (qrels)",
                "SUPPORT / CONTRADICT / NEI", "evidence sentences"], "data",
               "eval/datasets.py")

    # --- retrieval lane
    band(c, M + 168, 348, 600, 140, "A · Retrieval quality (free, no API key)")
    tune = box(c, M + 180, 362, 108, 100, "Tune on train",
               ["Grid search:", "BM25 k1 and b", "fusion method and weight", "",
                "best values -> configs"], "eval", "eval/tune.py")
    cfgs = box(c, M + 300, 362, 100, 100, "Systems under test",
               ["One YAML per variant:", "bm25, bm25_tuned,", "rm3, chunk100,", "dense_lsa,",
                "hybrid rrf / tuned"], "infra", "configs/*.yaml")
    run = box(c, M + 412, 362, 116, 100, "Evaluate on dev",
              ["Each of 300 claims:", "search, then compare the ranking with the expert labels",
               "", "nDCG@10, MRR@10,", "recall@10 / @100, latency"], "eval",
              "eval/runner.py metrics.py")
    stats = box(c, M + 540, 362, 108, 100, "Statistics",
                ["95% bootstrap CI per metric", "",
                 "Paired bootstrap test: same claims, two systems -> is the gap real?"],
                "eval", "eval/metrics.py")
    rep = box(c, M + 660, 362, 96, 100, "Reports",
              ["Markdown table", "+ per-query JSON", "", "--baseline picks", "the comparison"],
              "infra", "reports/")
    arrow(c, [(M + 150, left(tune)[1]), left(tune)], "train", label_pos=0)
    chain(c, [tune, cfgs, run, stats, rep])
    arrow(c, [(M + 150, 376), (M + 160, 376), (M + 160, 353), (run[0] + 58, 353),
              bottom(run)], "dev claims", label_pos=2)

    # --- end-to-end lane
    band(c, M + 168, 222, 600, 118, "B · End-to-end answer quality (needs ANTHROPIC_API_KEY)")
    ver = box(c, M + 180, 234, 190, 80, "Claim verification on dev",
              ["Retrieve -> Claude -> verdict, scored against the expert labels. "
               "No AI judge is needed."], "generation", "grounded eval verify | eval/runner.py")
    vmet = box(c, M + 382, 234, 186, 80, "What it reports",
               ["Verdict accuracy and macro-F1 vs the majority-class baseline (0.413)",
                "Citation precision and recall vs gold evidence",
                "Evidence-in-context rate, $/claim, latency"], "eval", "eval/runner.py")
    box(c, M + 580, 234, 176, 80, "Your own documents",
              ["No expert labels? Claude writes test questions (synth), then a judge checks "
               "each answer claim by claim (faithfulness)."], "generation",
              "eval/synth.py judge.py")
    arrow(c, [(M + 150, left(ver)[1]), left(ver)], "labels", label_pos=0)
    chain(c, [ver, vmet])

    # --- CI lane
    band(c, M, 62, W - 2 * M, 150, "C · Continuous integration (GitHub Actions, every push and PR)")
    unit = box(c, M + 12, 78, 170, 104, "1. Unit tests",
               ["57 pytest tests on Python 3.11 and 3.12, plus lint",
                "8-document fixture corpus",
                "Fake LLM and mock HTTP server: free, repeatable, no key"], "infra", "tests/")
    bench = box(c, M + 200, 78, 170, 104, "2. Re-run benchmark",
                ["Download SciFact", "Evaluate bm25, bm25_tuned and hybrid_lsa_tuned on dev",
                 "Table added to the job summary"], "eval", ".github/workflows/ci.yml")
    gate = box(c, M + 388, 78, 170, 104, "3. Quality gate",
               ["Compares each metric with its floor:",
                "bm25_tuned nDCG@10 >= 0.655, recall@100 >= 0.89, ...",
                "Raise floors when quality improves"], "safety", "eval/thresholds.yaml")
    chain(c, [unit, bench, gate], ["pass", "scores"])
    findings_strip(c)
    # pass / fail outcomes
    px = M + 590
    for i, (txt, col) in enumerate([("PASS: change can merge", "#2f8a4c"),
                                    ("FAIL: metric below floor, merge blocked", "#c0392b")]):
        yy = 150 - i * 44
        c.setFillColor(colors.HexColor(col))
        c.roundRect(px, yy - 8, 168, 26, 5, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 8)
        c.drawCentredString(px + 84, yy + 2, txt)
        arrow(c, [right(gate), (px - 12, right(gate)[1]), (px - 12, yy + 5), (px, yy + 5)])


def findings_strip(c: canvas.Canvas) -> None:
    """Key results on page 2, above the lanes."""
    x, y, w, h = M + 168, 494, W - 2 * M - 168, 26
    c.setFillColor(colors.HexColor("#fffbea"))
    c.setStrokeColor(colors.HexColor("#d9a400"))
    c.setLineWidth(0.8)
    c.roundRect(x, y, w, h, 5, stroke=1, fill=1)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 7.6)
    c.drawString(x + 8, y + 15, "Measured on dev (nDCG@10):")
    c.setFont("Helvetica", 7.6)
    c.drawString(x + 130, y + 15,
                 "BM25 0.662 vs published 0.665 (reproduced)  ·  naive blend -0.084 (p<0.001)  ·  "
                 "tuned system +0.012 (p=0.023)")
    c.drawString(x + 130, y + 5,
                 "...of which the blend alone, vs tuned BM25: +0.006 (p=0.105, not significant). "
                 "Separate paired tests isolate each change.")


# --------------------------------------------------------------------------------------------
ROWS = [
    ("Component", "File", "What it does", "Key design decision"),
    ("Loaders", "ingest/loaders.py", "Reads JSONL or folders of .md/.txt into documents",
     "Same Document type for SciFact and your own files"),
    ("Chunker", "ingest/chunking.py", "Splits documents into passages of up to N words",
     "Never splits a sentence, so citations map back to whole sentences"),
    ("BM25", "retrieval/bm25.py", "Keyword search over a precomputed sparse matrix",
     "Written from scratch; reproduces the BEIR baseline (0.662 vs 0.665)"),
    ("Dense retriever", "retrieval/dense.py", "Finds passages with similar meaning",
     "LSA works offline for CI; bge-small is one install away"),
    ("Fusion", "retrieval/hybrid.py", "Merges the BM25 and dense result lists",
     "RRF needs no score calibration; weights are tuned on train only"),
    ("Reranker", "retrieval/rerank.py", "Re-scores the top 50 with a cross-encoder",
     "Optional: it costs latency, and the benchmark shows when it pays off"),
    ("Prompt builder", "generation/prompts.py", "Formats numbered, escaped sources",
     "Retrieved text is untrusted: fenced, escaped, declared as data"),
    ("Claude client", "generation/llm.py", "Calls Claude with a JSON-schema output",
     "Structured output; refusal fallback; cost from real token usage"),
    ("Citation validator", "generation/answerer.py", "Checks every cited source id",
     "Never trusts the model: invalid ids dropped, uncited verdicts downgraded"),
    ("Pipeline + tracer", "pipeline.py, tracing.py", "Runs the stages and times each one",
     "Every response carries latency, tokens and $"),
    ("API and CLI", "api/app.py, cli.py", "HTTP endpoints and terminal commands",
     "Search works without an API key; generation degrades to 503"),
    ("Metrics and stats", "eval/metrics.py", "nDCG, MRR, recall, CIs, paired tests",
     "Paired tests over the same queries detect small real differences"),
    ("Runners and tuning", "eval/runner.py, tune.py", "Benchmark runs and grid search",
     "Tune on train, report on dev: no peeking at the test answers"),
    ("Judge and synth", "eval/judge.py, synth.py", "Eval for corpora without labels",
     "AI grading AI: hand-check ~50 judgements before trusting it"),
    ("Quality gate", "eval/gate.py, thresholds.yaml", "Fails CI when a metric drops",
     "Floors only go up, unless there is a written reason"),
]


def page_reference(c: canvas.Canvas) -> None:
    header(c, "grounded-rag: component reference",
           "Where each part lives and why it is built that way. Paths are relative to "
           "src/grounded/ unless shown otherwise.", 3)
    cell = ParagraphStyle("cell", fontName="Helvetica", fontSize=7.8, leading=9.6, textColor=INK)
    head = ParagraphStyle("head", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
    code = ParagraphStyle("code", parent=cell, fontName="Courier", fontSize=7.2)
    data = []
    for i, row in enumerate(ROWS):
        if i == 0:
            data.append([Paragraph(v, head) for v in row])
        else:
            data.append([Paragraph(f"<b>{row[0]}</b>", cell), Paragraph(row[1], code),
                         Paragraph(row[2], cell), Paragraph(row[3], cell)])
    t = Table(data, colWidths=[110, 150, 230, 296])
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334e68")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d9e2ec")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    layer_of = ["data", "data", "retrieval", "retrieval", "retrieval", "retrieval", "safety",
                "generation", "safety", "infra", "infra", "eval", "eval", "eval", "safety"]
    for r, layer in enumerate(layer_of, start=1):
        style.append(("BACKGROUND", (0, r), (0, r), LAYERS[layer][0]))
        style.append(("LINEBEFORE", (0, r), (0, r), 3, LAYERS[layer][1]))
    t.setStyle(TableStyle(style))
    _, th = t.wrapOn(c, W - 2 * M, H)
    t.drawOn(c, M, H - M - 52 - th)
    walkthrough(c, H - M - 52 - th - 22)


STEPS = [
    ("Entry", "<font face='Courier'>grounded ask \"Does vitamin D reduce fracture risk?\"</font> "
     "loads the saved index and the config that built it."),
    ("Retrieve", "BM25 and the dense retriever each return their top 100 passages. Fusion merges "
     "them into one ranked list, and the top 5 are kept (after reranking, if enabled)."),
    ("Prompt", "The 5 passages are numbered [1]-[5], escaped and wrapped in &lt;source&gt; tags, "
     "with an instruction to treat them as data and answer only from them."),
    ("Generate", "Claude returns JSON that must match the schema: the answer text with [n] "
     "markers, the list of cited numbers, and an insufficient-context flag."),
    ("Validate", "Cited numbers are checked against the 5 sources. Invalid ones are dropped, and "
     "an answer left with no valid citation is marked as an abstention."),
    ("Respond", "The answer, resolved citations (document id, title, snippet), the ranked sources "
     "and a trace with time, tokens and $ per stage come back as one JSON object."),
]


def walkthrough(c: canvas.Canvas, y_top: float) -> None:
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(INK)
    c.drawString(M, y_top, "Life of one request")
    style = ParagraphStyle("step", fontName="Helvetica", fontSize=8, leading=10, textColor=INK)
    col_w = (W - 2 * M - 20) / 3
    for i, (name, text) in enumerate(STEPS):
        col, row = i % 3, i // 3
        x = M + col * (col_w + 10)
        y = y_top - 16 - row * 62
        c.setFillColor(colors.HexColor("#334e68"))
        c.circle(x + 9, y - 8, 8, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawCentredString(x + 9, y - 11, str(i + 1))
        p = Paragraph(f"<b>{name}.</b> {text}", style)
        _, ph = p.wrapOn(c, col_w - 26, 60)
        p.drawOn(c, x + 24, y - ph + 2)


def main() -> None:
    c = canvas.Canvas(str(OUT), pagesize=(W, H))
    c.setTitle("grounded-rag architecture")
    c.setAuthor("dulminda")
    c.setSubject("Components of grounded-rag and how they interact")
    page_architecture(c)
    c.showPage()
    page_evaluation(c)
    c.showPage()
    page_reference(c)
    c.showPage()
    c.save()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
