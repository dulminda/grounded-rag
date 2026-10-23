"""ClaudeClient against a local mock of the Messages API: request shape and response handling."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from grounded.generation.llm import ClaudeClient, LLMError, LLMRefusal, cost_usd
from grounded.generation.schemas import ClaimVerdict


class _Server:
    def __init__(self):
        self.requests: list[dict] = []
        self.reply: dict = {}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                outer.requests.append(
                    {"path": self.path, "beta": self.headers.get("anthropic-beta"), "body": body}
                )
                out = json.dumps(outer.reply).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"


def _message(text=None, stop_reason="end_turn"):
    content = [{"type": "thinking", "thinking": "", "signature": "sig"}]
    if text is not None:
        content.append({"type": "text", "text": text})
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": content, "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": 1000, "output_tokens": 100}}


@pytest.fixture
def server(monkeypatch):
    s = _Server()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", s.url)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    yield s
    s.httpd.shutdown()


VALID = json.dumps({"verdict": "CONTRADICT", "rationale": "r", "citations": [2]})


def test_request_shape_and_parsing(server):
    server.reply = _message(VALID)
    result = ClaudeClient(effort="low", max_retries=0).generate("sys", "prompt", ClaimVerdict)
    req = server.requests[0]
    assert req["beta"] == "server-side-fallback-2026-07-01"
    assert req["body"]["fallbacks"] == "default"
    assert req["body"]["model"] == "claude-opus-5-5"
    assert req["body"]["output_config"]["effort"] == "low"
    assert req["body"]["output_config"]["format"]["schema"]["additionalProperties"] is False
    assert "thinking" not in req["body"]  # adaptive by default on this model
    assert result.parsed == ClaimVerdict(verdict="CONTRADICT", rationale="r", citations=[2])
    assert result.usage.cost_usd == pytest.approx(cost_usd("claude-opus-5-5", 1000, 100))


def test_fallback_can_be_disabled(server):
    server.reply = _message(VALID)
    ClaudeClient(fallback=False, max_retries=0).generate("s", "p", ClaimVerdict)
    assert server.requests[0]["beta"] is None and "fallbacks" not in server.requests[0]["body"]


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        (_message(None, stop_reason="refusal"), LLMRefusal),
        (_message(VALID, stop_reason="max_tokens"), LLMError),
        (_message('{"verdict": "MAYBE"}'), LLMError),
        (_message(None), LLMError),
    ],
)
def test_bad_responses_raise(server, reply, error):
    server.reply = reply
    with pytest.raises(error):
        ClaudeClient(max_retries=0).generate("s", "p", ClaimVerdict)
