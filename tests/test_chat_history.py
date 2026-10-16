"""Audit I18 — chat history bookkeeping (J8): turns are recorded only after a
successful call, the display history holds the real text, and the RAG
context goes into one request only. Model-free: the provider is a fake."""

import pytest

from chatbot import providers
from chatbot.chat import ClaudeChat
from util import Conversation


class FakeProvider:
    def __init__(self):
        self.payloads = []
        self.fail = False

    def __call__(self, payload, system):
        self.payloads.append([dict(m) for m in payload])
        if self.fail:
            raise providers.EmptyResponse("refused")
        return f"reply {len(self.payloads)}"


@pytest.fixture
def chat(monkeypatch):
    fake = FakeProvider()
    monkeypatch.setattr(providers, "build", lambda provider: fake)
    monkeypatch.setattr("chatbot.chat.load_settings", lambda: {"llm_provider": "claude"})
    c = ClaudeChat(Conversation(id="hist-1"))
    c.fake = fake
    return c


def test_I18_failed_call_leaves_no_dangling_turn(chat):
    chat.send("Hi Zoe", display_message="Hi Ann")
    chat.fake.fail = True
    with pytest.raises(providers.ProviderError):
        chat.send("Second", display_message="Second")
    assert [m.content for m in chat.conversation.api_messages] == ["Hi Zoe", "reply 1"]
    assert len(chat.conversation.messages) == 2
    chat.fake.fail = False
    chat.send("Third")
    roles = [m["role"] for m in chat.fake.payloads[-1]]
    assert roles == ["user", "assistant", "user"]              # no two user turns in a row


def test_I18_display_history_holds_real_text(chat):
    chat.send("Hi Zoe Park", display_message="Hi Ann Lee")
    chat.update_last_assistant_message("Hello Ann Lee")
    assert [m.content for m in chat.conversation.messages] == ["Hi Ann Lee", "Hello Ann Lee"]
    assert [m.content for m in chat.conversation.api_messages] == ["Hi Zoe Park", "reply 1"]
    assert "Ann" not in str(chat.fake.payloads)


def test_I18_rag_prefix_only_in_its_own_request(chat):
    chat.send("Q1", context_prefix="[DOC] excerpt\n")
    chat.send("Q2")
    assert chat.fake.payloads[0][-1]["content"] == "[DOC] excerpt\nQ1"
    assert "[DOC]" not in str(chat.fake.payloads[1])
    assert all("[DOC]" not in m.content for m in
               chat.conversation.api_messages + chat.conversation.messages)


def test_I18_pipeline_passes_real_text_and_prefix_separately():
    import inspect
    import pipeline
    src = inspect.getsource(pipeline.Pipeline.process_turn)
    assert "display_message=user_message" in src and "context_prefix=context_prefix" in src
    assert "sanitised = context_prefix + sanitised" not in src
