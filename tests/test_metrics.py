import pytest

from grounded.eval.gate import check
from grounded.eval.metrics import (
    bootstrap_ci,
    macro_f1,
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap_test,
    recall_at_k,
)

REL = frozenset({"a", "c"})


def test_rank_metrics():
    ranked = ["b", "a", "d", "c"]
    assert recall_at_k(ranked, REL, 2) == 0.5
    assert recall_at_k(ranked, REL, 4) == 1.0
    assert mrr_at_k(ranked, REL, 10) == 0.5
    assert mrr_at_k(["x"], REL, 10) == 0.0
    ideal = 1 + 1 / 1.5849625
    assert ndcg_at_k(ranked, REL, 10) == pytest.approx((1 / 1.5849625 + 1 / 2.3219281) / ideal)
    assert ndcg_at_k(["a", "c"], REL, 10) == pytest.approx(1.0)


def test_macro_f1():
    gold = ["S", "S", "C", "N"]
    pred = ["S", "C", "C", "N"]
    # S: p=1 r=.5 f=.667; C: p=.5 r=1 f=.667; N: f=1
    assert macro_f1(gold, pred, ["S", "C", "N"]) == pytest.approx((2 / 3 + 2 / 3 + 1) / 3)


def test_bootstrap_ci_brackets_mean():
    values = [0, 1] * 100
    lo, hi = bootstrap_ci(values)
    assert lo < 0.5 < hi
    assert hi - lo < 0.2


def test_paired_test():
    base = [0.5] * 50 + [0.6] * 50
    assert paired_bootstrap_test(base, base) == (0.0, 1.0)
    better = [x + 0.1 for x in base]
    delta, p = paired_bootstrap_test(base, better)
    assert delta == pytest.approx(0.1)
    assert p < 0.01
    noisy = [x + (0.05 if i % 2 else -0.05) for i, x in enumerate(base)]
    assert paired_bootstrap_test(base, noisy)[1] > 0.5


def test_gate():
    results = [{"kind": "retrieval", "name": "bm25", "metrics": {"ndcg@10": {"mean": 0.66}}}]
    assert check(results, {"retrieval": {"bm25": {"ndcg@10": 0.6}}}) == []
    failures = check(results, {"retrieval": {"bm25": {"ndcg@10": 0.7, "recall@100": 0.9}}})
    assert [f.metric for f in failures] == ["ndcg@10", "recall@100"]
    missing = check(results, {"retrieval": {"other": {"ndcg@10": 0.1}}})
    assert missing[0].value is None
