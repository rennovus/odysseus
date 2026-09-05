# src/model_name_heuristics.py
"""Single source of truth for "can this model id hold a conversation?".

This is a NAME heuristic: it guesses a model's family from its ID alone, for
the paths that have nothing else to go on — a bare list of IDs returned by
``/v1/models`` or ``/api/tags``. It is deliberately separate from
``src.model_capabilities``, which normalizes authoritative capability metadata
and explicitly does not infer capabilities from a bare model ID.

Four copies of this classification used to live in ``routes/model_routes.py``,
``src/endpoint_resolver.py``, ``routes/session_routes.py`` and
``routes/research/research_routes.py``, diverging in both matching strategy
(startswith vs. substring) and contents.

The embedding check was the costly divergence: every copy anchored it, so a
vendor- or family-prefixed name — ``nomic-embed-text``, ``mxbai-embed-large``,
``snowflake-arctic-embed``, ``openai/text-embedding-3-large`` — classified as
chat-capable. Endpoints that list an embedding model first then had it picked
as the default chat model. The embed/rerank tokens are matched anywhere in the
id here.

Deliberately dependency-free (stdlib only) so every caller can import it
without any risk of a circular import.
"""

from typing import Any, Iterable, Optional

# Matched ANYWHERE in the id. Reserved for tokens that no chat model carries,
# so vendor prefixes (`openai/`, `nvidia/`) and family prefixes (`nomic-`,
# `mxbai-`) can't smuggle a non-chat model past the check.
_NON_CHAT_INFIXES = (
    # Embedding / retrieval. "embed" subsumes text-embedding, embedding,
    # nomic-embed-text, mxbai-embed-large, arctic-embed, nv-embedqa, etc.
    "embed", "rerank", "minilm", "bge", "-nemoretriever",
    # Speech / audio.
    "whisper", "-transcribe", "tts-", "-tts",
    # Image / video.
    "dall-e", "stable-diffusion", "ai-synthetic-video",
    # Safety / classification / scoring.
    "moderation", "content-safety", "-safety", "-reward", "llama-guard",
    "topic-control", "calibration", "gliner",
    # Legacy completions-only families.
    "davinci", "babbage",
    # Non-conversational vision / multimodal encoders.
    "nvclip", "clip-", "kosmos", "fuyu", "deplot", "vila", "neva",
    # Misc. non-chat endpoints.
    "-realtime", "-codex", "codex-", "riva", "-parse", "cosmos-reason2",
)

# Matched only at the START of the id. Reserved for tokens short or common
# enough that an infix match would sweep up real chat models — "clip" would
# catch "eclipse-7b", "voyage" would catch "voyager-2-8b".
_NON_CHAT_PREFIXES = (
    "clip", "sora", "voyage-", "gpt-image", "chatgpt-image",
    # gpt-audio, gpt-audio-mini etc. — but NOT gpt-4o-audio-preview, which is
    # a chat model.
    "gpt-audio",
    "gpt-3.5-turbo-instruct",  # legacy OpenAI completions model
)


def is_chat_model(model_id: Any) -> bool:
    """Return True if the model ID looks like a chat/completions-capable model."""
    if not isinstance(model_id, str):
        # Non-compliant upstreams can return non-string IDs (e.g. int/None);
        # treat them as chat-capable rather than crashing on .lower().
        return True
    mid = model_id.lower()
    if any(mid.startswith(prefix) for prefix in _NON_CHAT_PREFIXES):
        return False
    return not any(substr in mid for substr in _NON_CHAT_INFIXES)


def first_chat_model(models: Optional[Iterable[Any]], default: Any = None) -> Any:
    """First model that isn't an embedding/tts/etc.

    Falls back to the first entry when every model is non-chat (an
    embeddings-only endpoint should still resolve to something rather than
    fail), and to ``default`` when there are no models at all.
    """
    models = list(models or [])
    for m in models:
        if is_chat_model(m):
            return m
    return models[0] if models else default
