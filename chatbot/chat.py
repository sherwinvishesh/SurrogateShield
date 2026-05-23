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
import time
from pathlib import Path
from typing import List, Optional

import anthropic

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from config import (
    CLAUDE_MODEL, SHADOWMAP_DIR,
    GEMINI_MODEL, OPENAI_MODEL, LOCAL_LLM_MODEL, LOCAL_LLM_HOST,
    AES_NONCE_SIZE,
)
from settings_manager import load_settings
from storage.logic import _derive_key
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
        settings = load_settings()
        self._provider: str = settings.get("llm_provider", "claude")
        self._client = self._init_client()
        self.conversation = conversation or Conversation()
        logger.debug(f"[ClaudeChat] Conversation ID: {self.conversation.id} provider={self._provider}")

    def _init_client(self):
        """Initialise and return the API client for the configured provider."""
        if self._provider == "claude":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "ANTHROPIC_API_KEY is not set. Add it to your .env file."
                )
            return anthropic.Anthropic(api_key=api_key)

        if self._provider == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "GEMINI_API_KEY is not set. Add it to your .env file."
                )
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                return genai
            except ImportError:
                raise EnvironmentError(
                    "google-generativeai package not installed. Run: pip install google-generativeai"
                )

        if self._provider == "chatgpt":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "OPENAI_API_KEY is not set. Add it to your .env file."
                )
            try:
                import openai
                return openai.OpenAI(api_key=api_key)
            except ImportError:
                raise EnvironmentError(
                    "openai package not installed. Run: pip install openai"
                )

        if self._provider == "local":
            host = os.environ.get("LOCAL_LLM_HOST", LOCAL_LLM_HOST)
            try:
                import ollama
                return ollama.Client(host=host)
            except ImportError:
                raise EnvironmentError(
                    "ollama package not installed. Run: pip install ollama"
                )

        raise EnvironmentError(f"Unknown LLM provider: {self._provider!r}")

    def _send_to_api(self, api_payload: list) -> str:
        """Route the API call to the configured provider and return the response text."""
        _retryable = (anthropic.RateLimitError, anthropic.APIConnectionError)
        _base_delay = 5  # seconds

        for attempt in range(3):
            try:
                if self._provider == "claude":
                    response = self._client.messages.create(
                        model=CLAUDE_MODEL,
                        max_tokens=4096,
                        system=SYSTEM_PROMPT,
                        messages=api_payload,
                    )
                    return response.content[0].text

                if self._provider == "gemini":
                    model = self._client.GenerativeModel(
                        model_name=GEMINI_MODEL,
                        system_instruction=SYSTEM_PROMPT,
                    )
                    history = [
                        {
                            "role": "user" if m["role"] == "user" else "model",
                            "parts": [m["content"]],
                        }
                        for m in api_payload[:-1]
                    ]
                    chat = model.start_chat(history=history)
                    response = chat.send_message(api_payload[-1]["content"])
                    return response.text

                if self._provider == "chatgpt":
                    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + api_payload
                    response = self._client.chat.completions.create(
                        model=OPENAI_MODEL,
                        max_tokens=4096,
                        messages=messages,
                    )
                    return response.choices[0].message.content

                if self._provider == "local":
                    model = os.environ.get("LOCAL_LLM_MODEL", LOCAL_LLM_MODEL)
                    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + api_payload
                    response = self._client.chat(model=model, messages=messages)
                    return response.message.content

                raise EnvironmentError(f"Unknown provider: {self._provider!r}")

            except anthropic.AuthenticationError:
                logger.error("[ClaudeChat] Authentication failed — check ANTHROPIC_API_KEY")
                raise

            except _retryable as exc:
                if attempt < 2:
                    delay = _base_delay * (2 ** attempt)
                    if isinstance(exc, anthropic.RateLimitError):
                        try:
                            ra = exc.response.headers.get("retry-after")
                            if ra:
                                delay = float(ra)
                        except Exception:
                            pass
                        label = "Rate limited"
                    else:
                        label = "Connection error"
                    logger.warning(
                        f"[ClaudeChat] {label} — retrying in {delay:.0f}s"
                        f" (attempt {attempt + 1}/3)"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"[ClaudeChat] API call failed after 3 attempts: {exc}")
                    raise

            except Exception as exc:
                logger.error(f"[ClaudeChat] API call failed ({self._provider}): {exc}")
                raise

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

        assistant_text = self._send_to_api(api_payload)

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

        The file is AES-256-GCM encrypted (same scheme as ShadowMap):
          nonce (12 bytes) || ciphertext
        Key is derived via HKDF-SHA256 from device secret + conversation_id.
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
            plaintext = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
            key = _derive_key(self.conversation.id)
            nonce = os.urandom(AES_NONCE_SIZE)
            ciphertext = AESGCM(key).encrypt(nonce, plaintext, None)
            path.write_bytes(nonce + ciphertext)
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
            raw = path.read_bytes()
            data = None
            # Try encrypted format first (nonce || ciphertext)
            if len(raw) > AES_NONCE_SIZE:
                try:
                    key = _derive_key(conversation_id)
                    nonce, ct = raw[:AES_NONCE_SIZE], raw[AES_NONCE_SIZE:]
                    plaintext = AESGCM(key).decrypt(nonce, ct, None)
                    data = json.loads(plaintext.decode("utf-8"))
                except Exception:
                    pass
            # Backward compat: old plaintext JSON files
            if data is None:
                logger.warning(
                    f"[ClaudeChat] Conversation {conversation_id!r} is unencrypted "
                    "(legacy format) — loading as plaintext."
                )
                data = json.loads(raw.decode("utf-8"))

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
                raw = json_file.read_bytes()
                data = None
                if len(raw) > AES_NONCE_SIZE:
                    try:
                        key = _derive_key(json_file.stem)
                        nonce, ct = raw[:AES_NONCE_SIZE], raw[AES_NONCE_SIZE:]
                        plaintext = AESGCM(key).decrypt(nonce, ct, None)
                        data = json.loads(plaintext.decode("utf-8"))
                    except Exception:
                        pass
                if data is None:
                    data = json.loads(raw.decode("utf-8"))
                results.append({
                    "id": data.get("id", json_file.stem),
                    "created": data.get("created", "unknown"),
                    "message_count": len(data.get("messages", [])),
                    "rag_mode": data.get("rag_mode", False),
                })
            except Exception:
                results.append({"id": json_file.stem, "created": "?", "message_count": 0})
        return results