"""Command-line interface: ``grounded --help``."""

from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

from grounded.config import PipelineConfig

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
eval_app = typer.Typer(no_args_is_help=True, help="Evaluate, tune and gate pipelines.")
app.add_typer(eval_app, name="eval")

DEFAULT_SCIFACT = Path("data/scifact")


def _load(dataset: str, data_dir: Path, split: str, corpus: Path | None, examples: Path | None):
    from grounded.eval.datasets import (
        load_jsonl_examples,
        load_scifact_corpus,
        load_scifact_examples,
    )
    from grounded.ingest.loaders import load_corpus

    if dataset == "scifact":
        return load_scifact_corpus(data_dir), load_scifact_examples(data_dir, split)
    if dataset == "custom":
        if corpus is None or examples is None:
            raise typer.BadParameter("--dataset custom needs --corpus and --examples")
        return load_corpus(corpus), load_jsonl_examples(examples)
    raise typer.BadParameter(f"unknown dataset {dataset!r}")


def _claude(config: PipelineConfig):
    from grounded.generation.llm import ClaudeClient

    return ClaudeClient(model=config.generation.model, effort=config.generation.effort)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


@app.command()
def data(dest: Path = typer.Option(DEFAULT_SCIFACT, help="Where to put SciFact.")) -> None:
    """Download the SciFact benchmark (5,183 abstracts, 1,109 labelled claims)."""
    from grounded.eval.datasets import download_scifact

    typer.echo(f"SciFact ready at {download_scifact(dest)}")


@app.command()
def index(
    config: Path = typer.Option(..., help="Pipeline YAML."),
    out: Path = typer.Option(..., help="Where to write the index (.pkl)."),
    corpus: Path | None = typer.Option(None, help="JSONL file or directory of .md/.txt."),
    scifact: Path | None = typer.Option(None, help="Index the SciFact corpus at this path."),
) -> None:
    """Build and save a retrieval index."""
    from grounded.eval.datasets import load_scifact_corpus
    from grounded.ingest.loaders import load_corpus
    from grounded.pipeline import RAGPipeline

    if (corpus is None) == (scifact is None):
        raise typer.BadParameter("pass exactly one of --corpus or --scifact")
    docs = load_scifact_corpus(scifact) if scifact else load_corpus(corpus)
    pipeline = RAGPipeline.from_documents(PipelineConfig.from_yaml(config), docs)
    pipeline.save_index(out)
    typer.echo(f"Indexed {len(docs)} documents ({len(pipeline.retriever.chunks)} chunks) -> {out}")


@app.command()
def search(
    query: str, index: Path = typer.Option(...), k: int = typer.Option(5, "-k", "--k")
) -> None:
    """Retrieve the top-k chunks for a query (no LLM)."""
    from grounded.pipeline import RAGPipeline

    for i, r in enumerate(RAGPipeline.load_index(index).retrieve(query, k), start=1):
        typer.echo(f"{i:>2}. [{r.score:.3f}] {r.chunk.title[:90]}  ({r.chunk.id})")


@app.command()
def ask(question: str, index: Path = typer.Option(...), verify: bool = False) -> None:
    """Answer a question (or verify a claim with --verify), with citations."""
    from grounded.pipeline import RAGPipeline

    base = RAGPipeline.load_index(index)
    pipeline = RAGPipeline(base.config, base.retriever, base.reranker, _claude(base.config))
    resp = pipeline.verify(question) if verify else pipeline.answer(question)
    typer.echo(json.dumps(resp.to_dict(), indent=2, default=str))


@app.command()
def serve(index: Path = typer.Option(...), host: str = "127.0.0.1", port: int = 8000) -> None:
    """Serve the HTTP API."""
    import os

    import uvicorn

    os.environ["GROUNDED_INDEX"] = str(index)
    uvicorn.run("grounded.api.app:create_app", factory=True, host=host, port=port)


@app.command()
def synth(
    config: Path = typer.Option(...),
    corpus: Path = typer.Option(...),
    out: Path = typer.Option(...),
    n: int = 100,
) -> None:
    """Generate a synthetic eval set (question + source doc) from your own corpus."""
    from grounded.eval.synth import synthesize_examples
    from grounded.ingest.chunking import chunk_corpus
    from grounded.ingest.loaders import load_corpus

    cfg = PipelineConfig.from_yaml(config)
    chunks = chunk_corpus(load_corpus(corpus), cfg.chunking.max_words)
    written = synthesize_examples(chunks, _claude(cfg), n, out)
    typer.echo(f"Wrote {written} examples -> {out}")


@eval_app.command("retrieval")
def eval_retrieval(
    config: list[Path] = typer.Option(..., help="One or more pipeline YAMLs (first = baseline)."),
    dataset: str = "scifact",
    data_dir: Path = DEFAULT_SCIFACT,
    split: str = "dev",
    corpus: Path | None = None,
    examples: Path | None = None,
    out: Path = Path("reports/retrieval.json"),
    report: Path | None = typer.Option(None, help="Also write a markdown table here."),
) -> None:
    """Compare retrieval configs: nDCG/MRR/recall with CIs and paired significance tests."""
    from grounded.eval.report import retrieval_table
    from grounded.eval.runner import run_retrieval_eval

    docs, exs = _load(dataset, data_dir, split, corpus, examples)
    runs = []
    for path in config:
        cfg = PipelineConfig.from_yaml(path)
        typer.echo(f"→ {cfg.name}")
        runs.append(run_retrieval_eval(cfg, docs, exs).to_dict())
    _write_json(out, runs)
    table = retrieval_table(runs)
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(f"## Retrieval: {dataset} / {split} ({len(exs)} queries)\n\n{table}\n")
    typer.echo(table)


@eval_app.command("verify")
def eval_verify(
    config: Path = typer.Option(...),
    data_dir: Path = DEFAULT_SCIFACT,
    split: str = "dev",
    limit: int | None = typer.Option(None, help="Evaluate only the first N claims (cost cap)."),
    workers: int = 8,
    out: Path = Path("reports/verify.json"),
) -> None:
    """End-to-end SciFact claim verification: verdict accuracy, macro-F1, citation quality."""
    from grounded.eval.report import verify_table
    from grounded.eval.runner import run_verify_eval
    from grounded.pipeline import RAGPipeline

    cfg = PipelineConfig.from_yaml(config)
    docs, exs = _load("scifact", data_dir, split, None, None)
    exs = exs[:limit] if limit else exs
    pipeline = RAGPipeline.from_documents(cfg, docs, _claude(cfg))

    def progress(done: int, spent: float) -> None:
        if done % 25 == 0 or done == len(exs):
            typer.echo(f"  {done}/{len(exs)} claims, ${spent:.3f} spent")

    run = run_verify_eval(pipeline, exs, workers=workers, on_progress=progress).to_dict()
    _write_json(out, [run])
    typer.echo(verify_table([run]))


@eval_app.command("qa")
def eval_qa(
    config: Path = typer.Option(...),
    corpus: Path = typer.Option(...),
    examples: Path = typer.Option(...),
    limit: int | None = None,
    workers: int = 8,
    out: Path = Path("reports/qa.json"),
) -> None:
    """Free-form QA on your own corpus, scored by an LLM faithfulness judge."""
    from grounded.eval.judge import FaithfulnessJudge
    from grounded.eval.runner import run_qa_eval
    from grounded.pipeline import RAGPipeline

    cfg = PipelineConfig.from_yaml(config)
    docs, exs = _load("custom", DEFAULT_SCIFACT, "dev", corpus, examples)
    exs = exs[:limit] if limit else exs
    llm = _claude(cfg)
    run = run_qa_eval(RAGPipeline.from_documents(cfg, docs, llm), exs, FaithfulnessJudge(llm),
                      workers=workers).to_dict()
    _write_json(out, [run])
    for name, m in run["metrics"].items():
        typer.echo(f"{name:>20}: {m['mean']:.3f}  [{m['ci_low']:.3f}, {m['ci_high']:.3f}]")
    typer.echo(f"{'total cost':>20}: ${run['extra']['total_cost_usd']:.3f}")


@eval_app.command("tune")
def eval_tune(
    data_dir: Path = DEFAULT_SCIFACT,
    split: str = typer.Option("train", help="Tune on train; report on dev."),
    out: Path = Path("reports/tuning.json"),
) -> None:
    """Grid-search BM25 (k1, b) and hybrid fusion weights on the training split."""
    from grounded.eval.tune import tune_bm25, tune_fusion
    from grounded.ingest.chunking import chunk_corpus
    from grounded.retrieval.bm25 import BM25Retriever
    from grounded.retrieval.dense import DenseRetriever, LSAEmbedder

    docs, exs = _load("scifact", data_dir, split, None, None)
    chunks = chunk_corpus(docs, max_words=10_000)  # SciFact abstracts: one chunk per doc
    bm25_rows = tune_bm25(chunks, exs)
    best = bm25_rows[0]
    typer.echo(f"BM25 best on {split}: k1={best['k1']} b={best['b']} ndcg@10={best['ndcg@10']:.4f}")
    lexical = BM25Retriever(chunks, k1=best["k1"], b=best["b"])
    fusion_rows = tune_fusion(lexical, DenseRetriever(chunks, LSAEmbedder()), exs)
    for row in fusion_rows[:5]:
        typer.echo(f"  fusion {row['method']:<6} dense_w={row['dense_weight']:.1f} "
                   f"ndcg@10={row['ndcg@10']:.4f}")
    _write_json(out, {"split": split, "bm25": bm25_rows, "fusion": fusion_rows})


@eval_app.command("gate")
def eval_gate(
    results: list[Path] = typer.Option(..., help="Result JSON files from `grounded eval`."),
    thresholds: Path = typer.Option(..., help="YAML of metric floors."),
) -> None:
    """Exit non-zero if any tracked metric is below its floor. Use in CI."""
    from grounded.eval.gate import check

    runs = [r for path in results for r in json.loads(path.read_text())]
    failures = check(runs, yaml.safe_load(thresholds.read_text()))
    if failures:
        for f in failures:
            typer.secho(f"FAIL {f}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.secho("Quality gate passed.", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
