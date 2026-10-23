from pathlib import Path

import pytest

from grounded.config import PipelineConfig
from grounded.eval.judge import FaithfulnessJudge
from grounded.eval.runner import run_qa_eval, run_retrieval_eval, run_verify_eval
from grounded.pipeline import RAGPipeline
from tests.conftest import FakeLLM

CONFIGS = sorted((Path(__file__).parents[1] / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_shipped_configs_validate(path):
    cfg = PipelineConfig.from_yaml(path)
    assert cfg.name == path.stem


def test_config_rejects_typos():
    with pytest.raises(ValueError):
        PipelineConfig.model_validate({"retrieval": {"methd": "bm25"}})


def test_answer_end_to_end(pipeline):
    resp = pipeline.answer("How does aspirin reduce platelet aggregation?")
    d = resp.to_dict()
    assert d["sources"][0]["doc_id"] == "aspirin"
    assert d["answer"]["citations"][0]["doc_id"] == "aspirin"
    assert [s["name"] for s in d["trace"]["spans"]] == ["retrieve", "generate"]
    assert d["trace"]["spans"][1]["attributes"]["cost_usd"] == pytest.approx(0.001)
    assert len(d["sources"]) == 3  # generation.context_chunks


def test_retrieval_only_pipeline_refuses_generation(docs, bm25_config):
    p = RAGPipeline.from_documents(bm25_config, docs)
    assert p.retrieve("statins", 1)[0].doc_id == "statins"
    with pytest.raises(RuntimeError):
        p.answer("statins?")


def test_index_roundtrip(tmp_path, pipeline):
    path = tmp_path / "idx.pkl"
    pipeline.save_index(path)
    loaded = RAGPipeline.load_index(path)
    q = "What guides Cas9 to its DNA target?"
    before = [r.chunk.id for r in pipeline.retrieve(q, 3)]
    assert [r.chunk.id for r in loaded.retrieve(q, 3)] == before


def test_run_retrieval_eval(docs, examples, bm25_config):
    run = run_retrieval_eval(bm25_config, docs, examples)
    assert run.n_queries == len(examples)
    assert run.metrics["recall@10"]["mean"] == 1.0
    assert len(run.per_query["ndcg@10"]) == len(examples)
    assert run.latency_ms["p95"] >= run.latency_ms["p50"]


def test_run_verify_eval(docs, examples, bm25_config):
    p = RAGPipeline.from_documents(bm25_config, docs, FakeLLM(citations=[1], verdict="SUPPORT"))
    run = run_verify_eval(p, examples, workers=2)
    # 4 of 6 gold labels are SUPPORT.
    assert run.metrics["accuracy"]["mean"] == pytest.approx(4 / 6)
    assert run.extra["majority_class"] == "SUPPORT"
    assert run.extra["confusion"]["CONTRADICT"]["SUPPORT"] == 1
    assert run.extra["total_cost_usd"] == pytest.approx(0.006)


def test_run_verify_eval_counts_refusals_as_errors(docs, examples, bm25_config):
    p = RAGPipeline.from_documents(bm25_config, docs, FakeLLM(refuse=True))
    run = run_verify_eval(p, examples, workers=2)
    assert run.n_queries == 0 and run.extra["errors"] == len(examples)


def test_run_qa_eval_with_judge(docs, examples, bm25_config):
    llm = FakeLLM(citations=[1])
    run = run_qa_eval(RAGPipeline.from_documents(bm25_config, docs, llm), examples,
                      FaithfulnessJudge(llm), workers=2)
    assert run.metrics["faithfulness"]["mean"] == pytest.approx(0.5)
    assert run.metrics["context_hit"]["mean"] == 1.0
