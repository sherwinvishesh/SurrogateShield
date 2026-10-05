"""Audit I16 / E3 — RAG (J8): document surrogates restore in answers, the
document map is shared with the chat so the same name gets the same
surrogate, chromadb telemetry is off, large documents are detected in
segments and fail closed, and ``rag --forget`` removes chunks and mappings.

Real chromadb in a temp dir; the embedder, detector and provider are stubs.
"""

import re

import pytest

import surrogateshield.core.storage.shadow_map as store
from util import DetectedEntity

NAMES = ("Sarah Mitchell", "Tom Okafor", "Ann Lee")


def fake_cascade(text, skip_values=None, skip_location_entities=False, **kw):
    skip = skip_values or set()
    ents = [DetectedEntity(text=m.group(), start=m.start(), end=m.end(), type="PERSON",
                           score=0.9, source="fake")
            for name in NAMES for m in re.finditer(re.escape(name), text)
            if m.group() not in skip]
    return ents, []


def embed(text):
    v = [0.0] * 26
    for ch in text.lower():
        if "a" <= ch <= "z":
            v[ord(ch) - 97] += 1.0
    return [x + 1e-3 for x in v]


@pytest.fixture
def env(tmp_path, monkeypatch):
    import config
    import pipeline
    from chatbot import providers
    from chatbot.chat import ClaudeChat
    from chatbot.rag import RAGStore
    from util import Conversation

    monkeypatch.setenv("SURROGATESHIELD_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(store, "_secret_cache", {})
    monkeypatch.setattr(config, "SHADOWMAP_DIR", None)
    monkeypatch.setattr(config, "DEVICE_KEY_PATH", None)
    monkeypatch.setattr(pipeline.sentinel_layer, "run_cascade", fake_cascade)
    monkeypatch.setattr("chatbot.chat.load_settings", lambda: {"llm_provider": "claude"})
    monkeypatch.setattr("settings_manager.load_settings",
                        lambda: {"llm_provider": "claude", "detailed_view": False})

    sent = []

    def provider(payload, system):            # quotes the retrieved excerpt back
        sent.append(payload[-1]["content"])
        return "From the documents: " + payload[-1]["content"]
    monkeypatch.setattr(providers, "build", lambda p: provider)

    rag = RAGStore(tmp_path / "rag", embedder=embed)
    chat = ClaudeChat(Conversation(id="rag-conv"))
    return pipeline, rag, chat, sent


def test_I16_telemetry_off_and_private_dir(env, tmp_path):
    _, rag, _, _ = env
    assert rag._client.get_settings().anonymized_telemetry is False
    assert (tmp_path / "rag").stat().st_mode & 0o777 == 0o700


def test_I16_document_surrogates_restore_in_answers(env):
    pipeline, rag, chat, sent = env
    pipeline.anonymise_for_rag("Sarah Mitchell signed the lease with Tom Okafor.", rag,
                               source="lease.txt")
    assert "Sarah Mitchell" not in "".join(rag.texts())
    p = pipeline.Pipeline(chat=chat, rag=rag)
    restored, _, _ = p.process_turn("Who signed the lease?", interactive=False)
    assert "Sarah Mitchell" in restored and "Tom Okafor" in restored
    assert "Sarah Mitchell" not in "".join(sent)                       # nothing real was sent


def test_I16_query_and_index_share_the_surrogate(env):
    pipeline, rag, chat, sent = env
    _, rag_shadow = pipeline.anonymise_for_rag("Sarah Mitchell owes rent.", rag, source="a.txt")
    doc_surrogate = rag_shadow.lookup_original("Sarah Mitchell")
    p = pipeline.Pipeline(chat=chat, rag=rag)
    _, _, smap = p.process_turn("What does Sarah Mitchell owe?", interactive=False)
    assert smap["Sarah Mitchell"] == doc_surrogate
    assert sent[-1].endswith(f"What does {doc_surrogate} owe?")
    # a second document reuses the surrogate too
    _, rag_shadow = pipeline.anonymise_for_rag("Sarah Mitchell moved out.", rag, source="b.txt")
    assert rag_shadow.lookup_original("Sarah Mitchell") == doc_surrogate


def test_I16_rag_prefix_not_stored_in_history(env):
    pipeline, rag, chat, _ = env
    pipeline.anonymise_for_rag("Tom Okafor paid the deposit.", rag, source="d.txt")
    pipeline.Pipeline(chat=chat, rag=rag).process_turn("Deposit?", interactive=False)
    assert all("Document excerpt" not in m.content for m in chat.conversation.api_messages
               if m.role == "user")


def test_I16_document_surrogate_avoids_chat_surrogates(env, monkeypatch):
    pipeline, rag, chat, _ = env
    p = pipeline.Pipeline(chat=chat, rag=None)
    _, _, smap = p.process_turn("Ann Lee called.", interactive=False)
    from storage.logic import surrogates_in_use
    assert smap["Ann Lee"] in surrogates_in_use()
    seen = {}
    real = pipeline.MimicGen.generate_all

    def spy(self, ents, **kw):
        seen["used"] = set(self.used_surrogates)
        return real(self, ents, **kw)
    monkeypatch.setattr(pipeline.MimicGen, "generate_all", spy)
    pipeline.anonymise_for_rag("Tom Okafor wrote.", rag)
    assert smap["Ann Lee"] in seen["used"]


def test_I16_large_document_segmented_exactly():
    import pipeline
    text = ("Sarah Mitchell paid.\n\n" * 3000) + "x" * 45_000
    parts = pipeline.split_segments(text, 20_000)
    assert "".join(parts) == text and all(len(p) <= 20_000 for p in parts)
    assert all(p.endswith("\n") for p in parts[:2])                    # cut at a break


def test_I16_detection_failure_indexes_nothing(env, monkeypatch):
    pipeline, rag, _, _ = env
    from surrogateshield.core.errors import DetectorUnavailable

    def boom(text, **kw):
        raise DetectorUnavailable("model missing")
    monkeypatch.setattr(pipeline.sentinel_layer, "run_cascade", boom)
    with pytest.raises(DetectorUnavailable):
        pipeline.anonymise_for_rag("Sarah Mitchell.", rag)
    assert rag.document_count() == 0


def test_I16_forget_removes_chunks_and_orphan_mappings(env):
    pipeline, rag, _, _ = env
    pipeline.anonymise_for_rag("Sarah Mitchell and Ann Lee met.", rag, source="a.txt")
    pipeline.anonymise_for_rag("Ann Lee left.", rag, source="b.txt")
    removed, erased = pipeline.forget_rag_document("a.txt", rag)
    assert removed == 1 and erased == 1                    # Sarah gone, Ann still in b.txt
    from storage.logic import ShadowMap
    originals = set(ShadowMap("rag_global").all_mappings().values())
    assert originals == {"Ann Lee"}
    assert set(row["source"] for row in rag.documents().values()) == {"b.txt"}
    assert pipeline.forget_rag_document("a.txt", rag) == (0, 0)


def test_I16_dead_add_rag_document_removed():
    import pipeline
    assert not hasattr(pipeline.Pipeline, "add_rag_document")
