"""Tests for Ollama reasoning control on the OpenAI-compat /v1 surface.

Covers:
- _is_ollama_openai_compat_url: URL classification (local host + /v1 path)
- reasoning_effort (NOT think) is the knob /v1 honors, resolved per model
- the parameter is withheld for non-thinking models and non-Ollama endpoints
- the /api/show probe URL follows the configured endpoint
- the two surfaces' value vocabularies do not leak into each other

Vocabularies below were read back from a 0.33.3 daemon's own 400 responses:
  native /api/chat  think            true|false|low|medium|high|max
  compat /v1        reasoning_effort minimal|low|medium|high|xhigh|ultra|max|none
`think` on /v1 is silently dropped -- byte-identical output to sending an
unknown parameter -- which is why it is no longer sent there.
"""
import asyncio
import json

from src import llm_core


# ---------------------------------------------------------------------------
# Fake HTTP client — captures the outgoing payload without network I/O
# ---------------------------------------------------------------------------

class _FakeResp:
    status_code = 200

    async def aiter_lines(self):
        # Yield a minimal done event so stream_llm exits cleanly
        yield json.dumps({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]})
        yield "data: [DONE]"

    async def aread(self):
        return b""


class _FakeStreamCtx:
    def __init__(self, captured):
        self._captured = captured

    async def __aenter__(self):
        return _FakeResp()

    async def __aexit__(self, *a):
        return False


class _FakeClient:
    """Minimal stand-in for httpx.AsyncClient that captures request payload."""

    def __init__(self):
        self.captured_payload = {}

    def stream(self, method, url, **kw):
        self.captured_payload = kw.get("json") or {}
        return _FakeStreamCtx(self.captured_payload)


def _capture_payload(monkeypatch, url, model, *, thinking=None, tools=None):
    """Run stream_llm, intercept the HTTP payload, and return it.

    ``thinking`` stubs the /api/show capability probe so no test needs a live
    daemon: True/False for a daemon that answered, None for one that could
    not be reached (which is what exercises the name-pattern fallback).
    """
    client = _FakeClient()
    monkeypatch.setattr(llm_core, "_get_http_client", lambda: client)
    monkeypatch.setattr(llm_core, "_is_host_dead", lambda u: False)
    monkeypatch.setattr(llm_core, "note_model_activity", lambda *a, **k: None)
    monkeypatch.setattr(llm_core, "_clear_host_dead", lambda *a, **k: None)
    monkeypatch.setattr(llm_core, "get_context_length", lambda u, m: 32768)
    monkeypatch.setattr(llm_core, "_ollama_native_thinking", lambda u, m: thinking)

    async def run():
        return [c async for c in llm_core.stream_llm(
            url, model, [{"role": "user", "content": "hi"}], tools=tools,
        )]

    asyncio.run(run())
    return client.captured_payload


# ---------------------------------------------------------------------------
# _is_ollama_openai_compat_url — pure function, no I/O
# ---------------------------------------------------------------------------

class TestIsOllamaOpenAICompatUrl:
    """Unit tests for the URL classifier that gates think-suppression."""

    # Positive cases — should be True
    def test_default_port_v1_root(self):
        assert llm_core._is_ollama_openai_compat_url("http://127.0.0.1:11434/v1")

    def test_default_port_chat_completions(self):
        assert llm_core._is_ollama_openai_compat_url("http://127.0.0.1:11434/v1/chat/completions")

    def test_localhost_default_port(self):
        assert llm_core._is_ollama_openai_compat_url("http://localhost:11434/v1")

    def test_localhost_default_port_with_path(self):
        assert llm_core._is_ollama_openai_compat_url("http://localhost:11434/v1/chat/completions")

    def test_loopback_ipv6(self):
        # IPv6 addresses in URLs require square brackets per RFC 3986
        assert llm_core._is_ollama_openai_compat_url("http://[::1]:11434/v1")

    def test_any_local_non_default_port(self):
        """Localhost on a non-default port (custom OLLAMA_HOST) must also match."""
        assert llm_core._is_ollama_openai_compat_url("http://127.0.0.1:11435/v1")

    def test_localhost_non_default_port(self):
        assert llm_core._is_ollama_openai_compat_url("http://localhost:8080/v1/chat/completions")

    def test_zero_dot_zero_host(self):
        assert llm_core._is_ollama_openai_compat_url("http://0.0.0.0:11434/v1")

    # Negative cases — should be False
    def test_openai_api_v1(self):
        """Real OpenAI endpoint must never match, even though path is /v1."""
        assert not llm_core._is_ollama_openai_compat_url("https://api.openai.com/v1")

    def test_openai_chat_completions(self):
        assert not llm_core._is_ollama_openai_compat_url("https://api.openai.com/v1/chat/completions")

    def test_ollama_native_api_path(self):
        """The native /api path is a different surface and must not match /v1."""
        assert not llm_core._is_ollama_openai_compat_url("http://localhost:11434/api")

    def test_ollama_native_api_chat(self):
        assert not llm_core._is_ollama_openai_compat_url("http://localhost:11434/api/chat")

    def test_remote_openrouter(self):
        assert not llm_core._is_ollama_openai_compat_url("https://openrouter.ai/api/v1")

    def test_empty_string(self):
        assert not llm_core._is_ollama_openai_compat_url("")

    def test_none_like_empty(self):
        assert not llm_core._is_ollama_openai_compat_url(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Payload injection — reasoning_effort, resolved per model
# ---------------------------------------------------------------------------

V1 = "http://127.0.0.1:11434/v1/chat/completions"
TOOLS = [{"type": "function", "function": {"name": "get_time", "parameters": {}}}]


class TestCompatReasoningEffort:
    """Assert what actually reaches the wire on Ollama's /v1 surface."""

    def test_think_is_never_sent_on_v1(self, monkeypatch):
        """Regression: /v1 silently drops `think`, so sending it is a no-op
        that reads like working suppression. It must not be sent at all."""
        for thinking in (True, False, None):
            payload = _capture_payload(monkeypatch, V1, "qwen3:14b", thinking=thinking)
            assert "think" not in payload

    def test_override_level_is_sent(self, monkeypatch):
        """A per-model override is the only way to pick a level, and it must
        survive to the wire on the surface that accepts it."""
        payload = _capture_payload(monkeypatch, V1, "muse-glimmer:30b", thinking=True)
        assert payload.get("reasoning_effort") == "medium"

    def test_no_effort_for_non_thinking_model(self, monkeypatch):
        """A model the daemon says does not reason gets no parameter — an
        out-of-vocabulary or unexpected value is a 400, not a no-op."""
        payload = _capture_payload(monkeypatch, V1, "llama3.2:3b", thinking=False)
        assert "reasoning_effort" not in payload

    def test_no_effort_when_capability_known_and_no_override(self, monkeypatch):
        """Thinking-capable but unconfigured: omit and let the model default
        stand, since the compat surface has no bool to say "on"."""
        payload = _capture_payload(monkeypatch, V1, "qwen3:14b", thinking=True)
        assert "reasoning_effort" not in payload

    def test_nothing_leaks_to_a_real_openai_endpoint(self, monkeypatch):
        """The URL guard is what matters, not the model name."""
        payload = _capture_payload(
            monkeypatch, "https://api.openai.com/v1/chat/completions",
            "muse-glimmer:30b", thinking=True,
        )
        assert "reasoning_effort" not in payload
        assert "think" not in payload

    def test_non_default_port_still_matches(self, monkeypatch):
        """Custom-port localhost Ollama (OLLAMA_HOST=0.0.0.0:11435)."""
        payload = _capture_payload(
            monkeypatch, "http://127.0.0.1:11435/v1/chat/completions",
            "muse-glimmer:30b", thinking=True,
        )
        assert payload.get("reasoning_effort") == "medium"

    def test_remote_daemon_over_tailscale(self, monkeypatch):
        """The daemon is routinely not localhost; port 11434 is the tell."""
        payload = _capture_payload(
            monkeypatch, "http://100.64.0.1:11434/v1/chat/completions",
            "muse-glimmer:30b", thinking=True,
        )
        assert payload.get("reasoning_effort") == "medium"


class TestCompatToolSuppression:
    """Suppression fires only when reasoning would land inline and eat the call."""

    def test_suppressed_when_probe_failed_and_tools_present(self, monkeypatch):
        payload = _capture_payload(
            monkeypatch, V1, "qwen3:14b", thinking=None, tools=TOOLS)
        assert payload.get("reasoning_effort") == "none"

    def test_not_suppressed_without_tools(self, monkeypatch):
        payload = _capture_payload(monkeypatch, V1, "qwen3:14b", thinking=None)
        assert "reasoning_effort" not in payload

    def test_not_suppressed_when_reasoning_is_out_of_band(self, monkeypatch):
        """A reported thinking capability means reasoning streams in its own
        `reasoning` field, which cannot swallow a tool call."""
        payload = _capture_payload(
            monkeypatch, V1, "qwen3:14b", thinking=True, tools=TOOLS)
        assert "reasoning_effort" not in payload

    def test_harmony_model_keeps_its_level_with_tools(self, monkeypatch):
        """gpt-oss reasoning arrives on the analysis channel, already demuxed."""
        payload = _capture_payload(
            monkeypatch, V1, "gpt-oss:20b", thinking=None, tools=TOOLS)
        assert payload.get("reasoning_effort") == "medium"

    def test_off_clamps_to_floor_for_a_model_with_no_off_switch(self, monkeypatch):
        """Measured: gpt-oss returns MORE reasoning for "none" than for "low"."""
        monkeypatch.setitem(llm_core._OLLAMA_THINK_OVERRIDES, "gpt-oss", False)
        payload = _capture_payload(monkeypatch, V1, "gpt-oss:20b", thinking=True)
        assert payload.get("reasoning_effort") == "low"


class TestShowProbeUrl:
    """The probe must follow the configured endpoint, not localhost."""

    def test_v1_root_rewrites_to_api(self):
        assert llm_core._ollama_show_url("http://100.64.0.1:11434/v1") == \
            "http://100.64.0.1:11434/api/show"

    def test_v1_chat_completions_rewrites_to_api(self):
        assert llm_core._ollama_show_url(
            "http://100.64.0.1:11434/v1/chat/completions") == \
            "http://100.64.0.1:11434/api/show"

    def test_native_url_keeps_api_root(self):
        assert llm_core._ollama_show_url("http://host:11434/api/chat") == \
            "http://host:11434/api/show"

    def test_bare_host_gets_api(self):
        assert llm_core._ollama_show_url("http://host:11434") == \
            "http://host:11434/api/show"

    def test_proxy_subpath_is_preserved(self):
        """/api sits beside /v1 at the same mount point, not under it."""
        assert llm_core._ollama_show_url("https://gw.example/ollama/v1") == \
            "https://gw.example/ollama/api/show"


class TestVocabularySplit:
    """The two surfaces disagree; neither may send the other's values."""

    def test_native_rejects_compat_only_levels(self):
        assert llm_core._normalize_think_value("xhigh") is None
        assert llm_core._normalize_think_value("none") is None
        assert llm_core._normalize_think_value("ultra") is None

    def test_native_accepts_bools_and_its_own_levels(self):
        assert llm_core._normalize_think_value(False) is False
        assert llm_core._normalize_think_value(True) is True
        assert llm_core._normalize_think_value("max") == "max"

    def test_compat_accepts_its_own_levels(self):
        assert llm_core._normalize_reasoning_effort("xhigh") == "xhigh"
        assert llm_core._normalize_reasoning_effort("minimal") == "minimal"

    def test_compat_maps_bools(self):
        """No bools on the wire: False means the off level, True means omit."""
        assert llm_core._normalize_reasoning_effort(False) == "none"
        assert llm_core._normalize_reasoning_effort(True) is None

    def test_both_reject_garbage(self):
        assert llm_core._normalize_think_value("bogus") is None
        assert llm_core._normalize_reasoning_effort("bogus") is None

    def test_native_payload_drops_an_unsendable_override(self):
        """An override valid only on /v1 must not reach the native body."""
        payload = llm_core._build_ollama_payload(
            "m", [{"role": "user", "content": "hi"}], 0.7, 100, think="xhigh")
        assert "think" not in payload


class TestCallerRequestedEffort:
    """A caller can state what reasoning its workload is worth."""

    def _payload(self, monkeypatch, model, requested, thinking=True):
        captured = {}

        def fake_post(url, h, **kw):
            captured.update(kw.get("json") or {})
            raise RuntimeError("stop after payload build")

        monkeypatch.setattr(llm_core, "_ollama_native_thinking", lambda u, m: thinking)
        monkeypatch.setattr(llm_core, "httpx_post_kimi_aware", fake_post)
        monkeypatch.setattr(llm_core, "note_model_activity", lambda *a, **k: None)
        monkeypatch.setattr(llm_core, "_get_cached_response", lambda k: None)
        try:
            llm_core.llm_call(V1, model, [{"role": "user", "content": "hi"}],
                              reasoning_effort=requested)
        except Exception:
            pass
        return captured

    def test_requested_outranks_the_override_map(self, monkeypatch):
        """muse-glimmer is configured "medium"; the caller asks for none."""
        payload = self._payload(monkeypatch, "muse-glimmer:30b", "none")
        assert payload.get("reasoning_effort") == "none"

    def test_requested_outranks_the_capability_probe(self, monkeypatch):
        """A thinking-capable model with no override would omit the parameter."""
        payload = self._payload(monkeypatch, "qwen3:14b", "none")
        assert payload.get("reasoning_effort") == "none"

    def test_requested_still_clamps_for_a_model_with_no_off_switch(self, monkeypatch):
        """Asking gpt-oss for off gets the floor, not a value it ignores."""
        payload = self._payload(monkeypatch, "gpt-oss:20b", "none")
        assert payload.get("reasoning_effort") == "low"

    def test_unusable_value_falls_back_to_normal_resolution(self, monkeypatch):
        """A bad level must not reach the wire — it 400s the whole request."""
        payload = self._payload(monkeypatch, "muse-glimmer:30b", "bogus")
        assert payload.get("reasoning_effort") == "medium"

    def test_omitted_keeps_resolver_behaviour(self, monkeypatch):
        payload = self._payload(monkeypatch, "muse-glimmer:30b", None)
        assert payload.get("reasoning_effort") == "medium"
