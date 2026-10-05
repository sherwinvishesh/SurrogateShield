# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
chatbot/chat.py — Claude API Conversation Handler

Manages multi-turn conversations with the Claude API.
Maintains conversation history as a list of {role, content} dicts
and persists conversations to <conversations_dir>/<conv_id>.json (sealed).

This module is intentionally isolated — it has NO imports from
detection, generation, storage, or reconstruction. The pipeline.py
module connects everything. chatbot.chat only talks to the API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from chatbot import providers
from settings_manager import load_settings
from storage.logic import (
    KIND_TRANSCRIPT,
    CorruptStoreError,
    StorageError,
    _derive_key,
    _legacy_key,
    conversations_dir,
    seal,
    unseal,
    validate_id,
    write_private,
)
from util import Conversation, ConversationMessage, get_logger, new_conversation_id

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# System prompt — tells Claude it is working
# with anonymised aliases, not real identities.
# Prevents security warnings and stops the model
# from treating PII label words ("SSN", "DOB")
# as meaningful context signals.
# ─────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful assistant. The names, email addresses, phone numbers, "
    "and other personal details in this conversation are anonymised aliases — "
    "placeholders for the user's real information. Treat every alias as you "
    "would treat real information: answer helpfully and naturally. Do not "
    "comment on the aliases, do not flag them as unusual, and do not issue "
    "security warnings about them. The user's privacy is already protected "
    "by the system they are using."
)


class ClaudeChat:
    """
    Manages a single conversation with the Claude API.

    Attributes:
        conversation: The Conversation dataclass holding all state.
        _client:      provider adapter (chatbot/providers.py).
    """

    def __init__(self, conversation: Optional[Conversation] = None) -> None:
        """
        Initialise the chat handler.

        Args:
            conversation: Existing Conversation to continue, or None to
                          create a new one.
        """
        settings = load_settings()
        self._provider: str = settings.get("llm_provider", "claude")
        self._client = self._init_client()
        self.conversation = conversation or Conversation()
        logger.debug(f"[ClaudeChat] Conversation ID: {self.conversation.id} provider={self._provider}")

    def _init_client(self):
        """Build the adapter for the configured provider (chatbot/providers.py)."""
        return providers.build(self._provider)

    def _send_to_api(self, api_payload: list) -> str:
        """Send *api_payload* and return the reply text.

        Raises providers.ProviderError (incl. EmptyResponse and
        ProviderAuthError) — never returns None or an empty string.
        """
        try:
            return providers.complete(self._client, api_payload, SYSTEM_PROMPT,
                                      provider=self._provider)
        except providers.ProviderError as exc:
            logger.error(f"[ClaudeChat] {exc}")
            raise

    def send(self, sanitised_message: str, *, display_message: Optional[str] = None,
             context_prefix: str = "") -> str:
        """
        Send a sanitised message and return the raw reply (still surrogates).

        Two histories (audit I18):
          - api_messages: sanitised text (surrogates) — re-sent every turn
          - messages:     what the user typed and (after the pipeline calls
                          update_last_assistant_message) the restored reply

        *context_prefix* (RAG excerpts) goes into this request only; it is
        never stored, so it is not re-sent on later turns. Nothing is
        appended until the provider has answered: a failed call leaves both
        histories unchanged (no dangling user turn).

        Args:
            sanitised_message: User message with PII already replaced.
            display_message:   The real user text for the display history
                               (defaults to *sanitised_message*).
            context_prefix:    Text prepended to this request only.

        Raises:
            providers.ProviderError: the call failed; nothing was recorded.
        """
        api_payload = self.conversation.to_api_history() + [
            {"role": "user", "content": context_prefix + sanitised_message}]
        assistant_text = self._send_to_api(api_payload)

        conv = self.conversation
        conv.api_messages.append(ConversationMessage(role="user", content=sanitised_message))
        conv.api_messages.append(ConversationMessage(role="assistant", content=assistant_text))
        # placeholder reply: the pipeline restores originals and calls
        # update_last_assistant_message() right after
        conv.messages.append(ConversationMessage(
            role="user", content=sanitised_message if display_message is None else display_message))
        conv.messages.append(ConversationMessage(role="assistant", content=assistant_text))
        return assistant_text

    def update_last_assistant_message(self, restored_text: str) -> None:
        """
        Replace the last assistant message in the DISPLAY history with
        the restored (real-values) text.

        The API history (api_messages) is intentionally NOT touched here —
        it must always retain surrogate values so future turns never send
        real PII to Claude.

        Args:
            restored_text: Response text with originals restored by ResolvePass.
        """
        for msg in reversed(self.conversation.messages):
            if msg.role == "assistant":
                msg.content = restored_text
                return

    def save(self) -> None:
        """
        Persist both display and API histories to <conversations_dir>/<id>.json.

        Two lists are saved:
          messages     — display history (real values, for the user to read)
          api_messages — API history (surrogates only, for Claude context)

        Sealed with the transcript key and AAD (storage.logic, audit I11);
        written atomically, mode 0600. Raises OSError if the write fails.
        """
        conv_id = validate_id(self.conversation.id)

        def _serialise(msgs):
            return [
                {"role": m.role, "content": m.content, "timestamp": m.timestamp}
                for m in msgs
            ]

        data = {
            "id": conv_id,
            "created": self.conversation.created,
            "rag_mode": self.conversation.rag_mode,
            "messages": _serialise(self.conversation.messages),
            "api_messages": _serialise(self.conversation.api_messages),
        }
        plaintext = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        path = conversations_dir() / f"{conv_id}.json"
        write_private(path, seal(_derive_key(conv_id, KIND_TRANSCRIPT), conv_id,
                                 KIND_TRANSCRIPT, plaintext))
        logger.debug(f"[ClaudeChat] Saved conversation → {path}")

    @staticmethod
    def _read_transcript(conversation_id: str, path: Path) -> dict:
        """Decrypt a transcript. Plaintext legacy files start with ``{``;
        anything else must decrypt, else ValueError (audit I19)."""
        raw = path.read_bytes()
        if raw[:1] == b"{":
            logger.warning(
                f"[ClaudeChat] Conversation {conversation_id!r} is a legacy "
                "plaintext file — it is re-encrypted on the next save."
            )
            return json.loads(raw.decode("utf-8"))
        try:
            plaintext = unseal(_derive_key(conversation_id, KIND_TRANSCRIPT), conversation_id,
                               KIND_TRANSCRIPT, raw, legacy=_legacy_key(conversation_id))
        except CorruptStoreError as exc:
            raise ValueError(
                f"conversation {conversation_id} cannot be decrypted with this "
                f"device key ({exc})"
            ) from exc
        return json.loads(plaintext.decode("utf-8"))

    @classmethod
    def load(cls, conversation_id: str) -> "ClaudeChat":
        """
        Load an existing conversation from disk and return a ClaudeChat instance.

        Raises:
            ValueError:        invalid id, or the file cannot be decrypted/parsed.
            FileNotFoundError: the conversation does not exist.
        """
        conversation_id = validate_id(conversation_id)
        path = conversations_dir() / f"{conversation_id}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Conversation '{conversation_id}' not found at {path}"
            )
        try:
            data = cls._read_transcript(conversation_id, path)

            def _deserialise(raw_list):
                return [
                    ConversationMessage(
                        role=m["role"],
                        content=m["content"],
                        timestamp=m.get("timestamp", ""),
                    )
                    for m in raw_list
                ]

            messages     = _deserialise(data.get("messages", []))
            api_messages = _deserialise(data.get("api_messages", []))

            # Back-compat: old files only have "messages" (pre-dual-history format).
            # Do NOT copy display messages into api_messages — those display messages
            # may contain real PII values (from before the history privacy fix).
            # Starting with an empty api_messages is safe: Claude will lose old context
            # but will never receive real PII. The display history remains readable.
            if not api_messages:
                logger.warning(
                    f"[ClaudeChat] Old-format conversation {conversation_id!r} has no "
                    "api_messages. Starting fresh API context to prevent PII leakage. "
                    "Display history is preserved."
                )
                api_messages = []

            conv = Conversation(
                id=data["id"],
                messages=messages,
                api_messages=api_messages,
                created=data.get("created", ""),
                rag_mode=data.get("rag_mode", False),
            )
            logger.info(
                f"[ClaudeChat] Loaded conversation {conversation_id} "
                f"({len(messages)} display msgs, {len(api_messages)} api msgs)"
            )
            return cls(conversation=conv)
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError) as exc:
            raise ValueError(
                f"Conversation file at {path} is corrupt or invalid: {exc}"
            ) from exc

    @staticmethod
    def delete(conversation_id: str) -> bool:
        """
        Delete the conversation transcript (no decryption needed). Returns
        False when there was no transcript.

        The shadow map is deleted separately by storage.logic.erase().
        """
        path = conversations_dir() / f"{validate_id(conversation_id)}.json"
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        logger.info(f"[ClaudeChat] Deleted conversation file: {path}")
        return True

    @staticmethod
    def list_conversations() -> List[dict]:
        """
        Return metadata for all saved conversations.

        Returns:
            List of dicts with 'id', 'created', 'message_count', 'rag_mode'
            and 'readable' (False if the file cannot be decrypted or parsed).
        """
        conv_dir = conversations_dir()
        if not conv_dir.exists():
            return []
        results = []
        for json_file in sorted(conv_dir.glob("*.json")):
            conv_id = json_file.stem
            try:
                validate_id(conv_id)
                data = ClaudeChat._read_transcript(conv_id, json_file)
                results.append({
                    "id": data.get("id", conv_id),
                    "created": data.get("created", "unknown"),
                    "message_count": len(data.get("messages", [])),
                    "rag_mode": data.get("rag_mode", False),
                    "readable": True,
                })
            except (ValueError, OSError, StorageError) as exc:
                logger.warning(f"[ClaudeChat] Unreadable conversation {json_file.name}: {exc}")
                results.append({"id": conv_id, "created": "unreadable",
                                "message_count": 0, "rag_mode": False, "readable": False})
        return results
