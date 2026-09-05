"""Tests for the shared chat/non-chat model-name heuristic.

The embedding check used to be prefix-anchored in every copy of this list, so
vendor-prefixed embedding names (`nomic-embed-text`, `mxbai-embed-large`,
`snowflake-arctic-embed`) classified as chat-capable and got auto-picked as a
session's default model.
"""
import pytest

from src.model_name_heuristics import first_chat_model, is_chat_model


# ── Embedding models: the infix regression this module exists for ──

EMBEDDING_IDS = [
    # Ollama-style tags — the names that reached the composer model picker.
    "nomic-embed-text:v1.5",
    "nomic-embed-text",
    "mxbai-embed-large",
    "mxbai-embed-large:335m",
    "snowflake-arctic-embed",
    "snowflake-arctic-embed2:568m",
    "bge-m3",
    "all-minilm",
    "granite-embedding:278m",
    "embeddinggemma",
    # OpenAI-style, bare and vendor-prefixed.
    "text-embedding-3-small",
    "text-embedding-ada-002",
    "openai/text-embedding-3-large",
    # Other providers' retrieval models.
    "snowflake/arctic-embed-l",
    "nvidia/nv-embed-v1",
    "nvidia/nv-embedqa-e5-v5",
    "jina-embeddings-v3",
    "voyage-3-large",
    "cohere/embed-english-v3.0",
    "qwen3-embedding:8b",
]


@pytest.mark.parametrize("model_id", EMBEDDING_IDS)
def test_embedding_models_are_not_chat(model_id):
    assert is_chat_model(model_id) is False


@pytest.mark.parametrize("model_id", [
    "bge-reranker-v2-m3",
    "jina-reranker-v2",
    "mixedbread-ai/mxbai-rerank-large-v1",
])
def test_reranker_models_are_not_chat(model_id):
    assert is_chat_model(model_id) is False


# ── Chat models must survive the wider infix matching ──

@pytest.mark.parametrize("model_id", [
    # The five models on the endpoint that surfaced this bug.
    "gemma4:12b", "gpt-oss:20b", "muse-glimmer:30b", "qwen3.8:27b",
    # General chat ids, bare and vendor-prefixed.
    "gpt-4o", "gpt-4o-mini", "claude-sonnet-4", "llama-3.3-70b",
    "deepseek-chat", "gemini-2.0-flash", "o3",
    "google/gemma-2b-it", "bigcode/starcoder2-15b-instruct",
    "gpt-4o-audio-preview",
    # Near-misses for the infix tokens: these must NOT be swept up.
    "eclipse-7b",     # contains "clip", but not "clip-"
    "voyager-2-8b",   # starts with "voyage", but not "voyage-"
])
def test_chat_models_are_chat(model_id):
    assert is_chat_model(model_id) is True


@pytest.mark.parametrize("model_id", [
    "dall-e-3", "openai/dall-e-3",
    "tts-1", "gpt-4o-mini-tts",
    "whisper-1", "openai/whisper-large-v3",
    "gpt-image-1", "sora-1",
    "text-moderation-latest", "omni-moderation-latest",
    "stable-diffusion-xl",
    "gpt-audio", "gpt-3.5-turbo-instruct",
    "gpt-4o-realtime-preview",
])
def test_other_non_chat_models(model_id):
    assert is_chat_model(model_id) is False


@pytest.mark.parametrize("bad", [None, 123, 4.5, ["x"], {"a": 1}])
def test_non_string_id_is_treated_as_chat(bad):
    # Defensive boundary: a non-compliant upstream can yield a non-string
    # model id; it must not crash on .lower() (treated as chat-capable).
    assert is_chat_model(bad) is True


# ── first_chat_model ──

class TestFirstChatModel:
    def test_skips_leading_embedding_model(self):
        # The live ordering from the Ollama endpoint in the bug report.
        models = ["nomic-embed-text:v1.5", "muse-glimmer:30b", "qwen3.8:27b",
                  "gpt-oss:20b", "gemma4:12b"]
        assert first_chat_model(models) == "muse-glimmer:30b"

    def test_skips_embedding_and_tts(self):
        models = ["text-embedding-ada-002", "whisper-large-v3", "gpt-4o"]
        assert first_chat_model(models) == "gpt-4o"

    def test_falls_back_to_first_when_all_non_chat(self):
        assert first_chat_model(["whisper-large-v3"]) == "whisper-large-v3"

    @pytest.mark.parametrize("models", [[], None])
    def test_empty_returns_default(self, models):
        assert first_chat_model(models) is None
        assert first_chat_model(models, default="") == ""

    def test_non_string_entries_are_treated_as_chat(self):
        # Mirrors is_chat_model's defensive boundary: a non-string id counts as
        # chat-capable rather than crashing, so it wins the first-match scan.
        assert first_chat_model([None, "gpt-4o"]) is None
        assert first_chat_model(["text-embedding-3-small", "gpt-4o"]) == "gpt-4o"
