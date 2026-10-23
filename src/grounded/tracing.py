"""Minimal span tracer: per-stage latency and attributes for every request.

Spans are plain dicts so they can be returned in API responses, written to JSONL, or mapped
onto OpenTelemetry spans by an exporter without changing call sites.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from typing import Any


class Trace:
    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or uuid.uuid4().hex
        self.spans: list[dict[str, Any]] = []
        self._t0 = time.perf_counter()

    @contextmanager
    def span(self, name: str, **attributes: Any):
        start = time.perf_counter()
        record: dict[str, Any] = {"name": name, "attributes": dict(attributes)}
        try:
            yield record["attributes"]
        except Exception as e:
            record["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            record["start_ms"] = round((start - self._t0) * 1000, 3)
            record["duration_ms"] = round((time.perf_counter() - start) * 1000, 3)
            self.spans.append(record)

    def duration(self, name: str) -> float:
        return sum(s["duration_ms"] for s in self.spans if s["name"] == name)

    def to_dict(self) -> dict[str, Any]:
        return {"trace_id": self.trace_id, "spans": self.spans}
