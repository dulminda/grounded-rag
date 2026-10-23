"""CI quality gate: fail the build when a tracked metric drops below its floor.

thresholds.yaml::

    retrieval:            # result kind
      bm25-tuned:         # system name
        ndcg@10: 0.64     # minimum acceptable mean
        recall@100: 0.90
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GateFailure:
    system: str
    metric: str
    value: float | None
    floor: float

    def __str__(self) -> str:
        got = "missing" if self.value is None else f"{self.value:.4f}"
        return f"{self.system}: {self.metric} = {got} (floor {self.floor:.4f})"


def check(results: list[dict[str, Any]], thresholds: dict[str, Any]) -> list[GateFailure]:
    failures: list[GateFailure] = []
    for kind, systems in thresholds.items():
        for system, floors in systems.items():
            run = next((r for r in results if r["kind"] == kind and r["name"] == system), None)
            for metric, floor in floors.items():
                value = None if run is None else run["metrics"].get(metric, {}).get("mean")
                if value is None or value < floor:
                    failures.append(GateFailure(system, metric, value, float(floor)))
    return failures
