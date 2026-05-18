"""
chatbot/chat.py — Claude API Conversation Handler

Manages multi-turn conversations with the Claude API.
Maintains conversation history as a list of {role, content} dicts
and persists conversations to conversations/<conv_id>.json.

This module is intentionally isolated — it has NO imports from
detection, generation, storage, or reconstruction. The pipeline.py
module connects everything. chatbot.chat only talks to the API.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import anthropic

from config import CLAUDE_MODEL, SHADOWMAP_DIR
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
        _client:      Anthropic API client instance.
    """

    def __init__(self, conversation: Optional[Conversation] = None) -> None:
        """
        Initialise the chat handler.

        Args:
            conversation: Existing Conversation to continue, or None to
                          create a new one.
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Export it before running SurrogateShield."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self.conversation = conversation or Conversation()
        logger.debug(f"[ClaudeChat] Conversation ID: {self.conversation.id}")

    def send(self, sanitised_message: str) -> str:
        """
        Send a sanitised message to Claude and return the raw API response.

        Maintains two separate histories:
          - api_messages: sanitised text (surrogates) — sent to Claude every turn
          - messages:     display text (real values) — updated by pipeline after
                          ResolvePass runs, never sent to the API

        This separation is what prevents real PII from leaking into the
        multi-turn context window on subsequent turns.

        Args:
            sanitised_message: User message with PII already replaced by surrogates.

        Returns:
            Raw assistant response text (still containing surrogates).
        """
        # Append sanitised user turn to the API history
        self.conversation.api_messages.append(
            ConversationMessage(role="user", content=sanitised_message)
        )

        # Build API payload using to_api_history() — uses api_messages (surrogates only)
        api_payload = self.conversation.to_api_history()

        try:
            response = self._client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=api_payload,
            )
            assistant_text: str = response.content[0].text
        except anthropic.APIConnectionError as exc:
            logger.error(f"[ClaudeChat] Connection error: {exc}")
            raise
        except anthropic.AuthenticationError:
            logger.error("[ClaudeChat] Authentication failed — check ANTHROPIC_API_KEY")
            raise
        except Exception as exc:
            logger.error(f"[ClaudeChat] API call failed: {exc}")
            raise

        # Append raw assistant response (surrogates) to API history
        self.conversation.api_messages.append(
            ConversationMessage(role="assistant", content=assistant_text)
        )

        # Append a placeholder to display history — pipeline will restore
        # real values and call update_last_assistant_message() immediately after
        self.conversation.messages.append(
            ConversationMessage(role="user", content=sanitised_message)
        )
        self.conversation.messages.append(
            ConversationMessage(role="assistant", content=assistant_text)
        )

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
        Persist both display and API histories to conversations/<conv_id>.json.

        Two lists are saved:
          messages     — display history (real values, for the user to read)
          api_messages — API history (surrogates only, for Claude context)
        """
        try:
            Path(SHADOWMAP_DIR).mkdir(parents=True, exist_ok=True)
            path = Path(SHADOWMAP_DIR) / f"{self.conversation.id}.json"

            def _serialise(msgs):
                return [
                    {"role": m.role, "content": m.content, "timestamp": m.timestamp}
                    for m in msgs
                ]

            data = {
                "id": self.conversation.id,
                "created": self.conversation.created,
                "rag_mode": self.conversation.rag_mode,
                "messages": _serialise(self.conversation.messages),
                "api_messages": _serialise(self.conversation.api_messages),
            }
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.debug(f"[ClaudeChat] Saved conversation → {path}")
        except OSError as exc:
            logger.error(f"[ClaudeChat] Failed to save conversation: {exc}")

    @classmethod
    def load(cls, conversation_id: str) -> "ClaudeChat":
        """
        Load an existing conversation from disk and return a ClaudeChat instance.

        Args:
            conversation_id: The UUID of the conversation to load.

        Returns:
            ClaudeChat instance with history populated from disk.

        Raises:
            FileNotFoundError: If the conversation file does not exist.
        """
        path = Path(SHADOWMAP_DIR) / f"{conversation_id}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Conversation '{conversation_id}' not found at {path}"
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))

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
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(
                f"Conversation file at {path} is corrupt or invalid: {exc}"
            ) from exc

    @staticmethod
    def delete(conversation_id: str) -> None:
        """
        Delete conversation JSON file from disk.

        The ShadowMap file is deleted separately by ShadowMap.delete().

        Args:
            conversation_id: UUID of the conversation to delete.
        """
        path = Path(SHADOWMAP_DIR) / f"{conversation_id}.json"
        try:
            if path.exists():
                path.unlink()
                logger.info(f"[ClaudeChat] Deleted conversation file: {path}")
            else:
                logger.warning(
                    f"[ClaudeChat] Conversation file not found: {path}"
                )
        except OSError as exc:
            logger.error(f"[ClaudeChat] Failed to delete {path}: {exc}")

    @staticmethod
    def list_conversations() -> List[dict]:
        """
        Return metadata for all saved conversations.

        Returns:
            List of dicts with 'id', 'created', 'message_count', 'rag_mode'.
        """
        conv_dir = Path(SHADOWMAP_DIR)
        if not conv_dir.exists():
            return []
        results = []
        for json_file in sorted(conv_dir.glob("*.json")):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                results.append({
                    "id": data.get("id", json_file.stem),
                    "created": data.get("created", "unknown"),
                    "message_count": len(data.get("messages", [])),
                    "rag_mode": data.get("rag_mode", False),
                })
            except Exception:
                results.append({"id": json_file.stem, "created": "?", "message_count": 0})
        return results