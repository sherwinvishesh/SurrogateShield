"""
storage/logic.py — ShadowMap

AES-256-GCM encrypted, conversation-tied mapping cache.

Maps surrogate → original for every PII substitution in a conversation.
Persisted to disk as <conv_id>.shadowmap (binary, encrypted).
Key is derived per-conversation via HKDF from conversation_id + device_secret.
Device secret is generated once and stored in ~/.surrogateshield/device.key.

Design:
    - In-memory dict: {surrogate: original}
    - Encrypted to disk: nonce (12 bytes) || ciphertext
    - Key derivation: HKDF-SHA256(conversation_id.encode() + device_secret)
    - If .shadowmap missing on load → start with empty mapping (graceful)
    - On conversation delete → delete both .json and .shadowmap files
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

from config import (
    AES_NONCE_SIZE,
    DEVICE_KEY_PATH,
    HKDF_INFO,
    SHADOWMAP_DIR,
)
from util import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Device secret management
# ─────────────────────────────────────────────

def _get_device_secret() -> bytes:
    """
    Load the device secret from disk, creating it if it does not exist.

    The secret is stored in ~/.surrogateshield/device.key as raw bytes.
    The directory is created on first run.

    Returns:
        32-byte device secret.
    """
    key_path = Path(DEVICE_KEY_PATH).expanduser()
    try:
        key_path.parent.mkdir(parents=True, exist_ok=True)
        if key_path.exists():
            return key_path.read_bytes()
        # First run: generate and persist
        secret = os.urandom(32)
        key_path.write_bytes(secret)
        logger.info(f"[ShadowMap] Generated new device secret at {key_path}")
        return secret
    except OSError as exc:
        logger.error(f"[ShadowMap] Could not read/write device secret: {exc}")
        # Fallback: generate an ephemeral secret (data won't survive restart)
        logger.warning("[ShadowMap] Using ephemeral device secret — mappings lost on restart")
        return os.urandom(32)


# ─────────────────────────────────────────────
# Key derivation
# ─────────────────────────────────────────────

def _derive_key(conversation_id: str) -> bytes:
    """
    Derive a 32-byte AES key for a specific conversation.

    Uses HKDF-SHA256 with the device secret as input key material
    and the conversation_id as salt material.

    Args:
        conversation_id: Unique conversation identifier.

    Returns:
        32-byte derived key.
    """
    device_secret = _get_device_secret()
    ikm = conversation_id.encode("utf-8") + device_secret
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=HKDF_INFO,
    )
    return hkdf.derive(ikm)


# ─────────────────────────────────────────────
# ShadowMap class
# ─────────────────────────────────────────────

class ShadowMap:
    """
    Encrypted conversation-tied surrogate mapping store.

    Attributes:
        conversation_id: The conversation this map belongs to.
        _mappings:       In-memory dict: {surrogate: original}.
        _key:            Derived AES-256 key for this conversation.
        _path:           Path to the .shadowmap file on disk.
    """

    def __init__(self, conversation_id: str) -> None:
        """
        Initialise ShadowMap for a conversation.

        Derives the encryption key and attempts to load existing
        mappings from disk. If the file is missing or corrupt,
        starts with an empty mapping.

        Args:
            conversation_id: Unique identifier for the conversation.
        """
        self.conversation_id = conversation_id
        self._key: bytes = _derive_key(conversation_id)
        self._path: Path = (
            Path(SHADOWMAP_DIR) / f"{conversation_id}.shadowmap"
        )
        self._mappings: Dict[str, str] = {}
        self._load()

    # ── CRUD ────────────────────────────────────────────────────

    def add(self, surrogate: str, original: str) -> None:
        """
        Add a surrogate → original mapping.

        Args:
            surrogate: The fake value sent to the API.
            original:  The real PII value to restore later.
        """
        self._mappings[surrogate] = original

    def get(self, surrogate: str) -> Optional[str]:
        """
        Look up the original value for a surrogate.

        Args:
            surrogate: The fake value.

        Returns:
            The original value, or None if not found.
        """
        return self._mappings.get(surrogate)

    def update(self, new_mappings: Dict[str, str]) -> None:
        """
        Merge a dict of {surrogate: original} into the map.

        Args:
            new_mappings: New entries to add/update.
        """
        self._mappings.update(new_mappings)

    def all_mappings(self) -> Dict[str, str]:
        """Return a copy of all current surrogate→original mappings."""
        return dict(self._mappings)

    # ── Persistence ─────────────────────────────────────────────

    def save(self) -> None:
        """
        Encrypt the current mappings and write to disk.

        Encryption: AESGCM with a fresh 12-byte nonce per save.
        File format: nonce (12 bytes) || ciphertext.
        """
        try:
            Path(SHADOWMAP_DIR).mkdir(parents=True, exist_ok=True)
            plaintext = json.dumps(self._mappings).encode("utf-8")
            nonce = os.urandom(AES_NONCE_SIZE)
            aesgcm = AESGCM(self._key)
            ciphertext = aesgcm.encrypt(nonce, plaintext, None)
            self._path.write_bytes(nonce + ciphertext)
            logger.debug(
                f"[ShadowMap] Saved {len(self._mappings)} mappings → {self._path}"
            )
        except OSError as exc:
            logger.error(f"[ShadowMap] Failed to save to {self._path}: {exc}")

    def _load(self) -> None:
        """
        Attempt to load and decrypt mappings from the .shadowmap file.

        If the file is absent, empty, or corrupt — starts with empty
        mappings and logs a warning. Never raises.
        """
        if not self._path.exists():
            logger.debug(
                f"[ShadowMap] No existing shadowmap at {self._path} — "
                "starting with empty mappings"
            )
            return
        try:
            data = self._path.read_bytes()
            if len(data) <= AES_NONCE_SIZE:
                logger.warning(
                    f"[ShadowMap] File too short to be valid: {self._path}"
                )
                return
            nonce = data[:AES_NONCE_SIZE]
            ciphertext = data[AES_NONCE_SIZE:]
            aesgcm = AESGCM(self._key)
            plaintext = aesgcm.decrypt(nonce, ciphertext, None)
            self._mappings = json.loads(plaintext.decode("utf-8"))
            logger.info(
                f"[ShadowMap] Loaded {len(self._mappings)} mappings from {self._path}"
            )
        except Exception as exc:
            logger.warning(
                f"[ShadowMap] Could not decrypt {self._path}: {exc}. "
                "Starting with empty mappings."
            )
            self._mappings = {}

    def delete(self) -> None:
        """
        Permanently delete the .shadowmap file from disk.

        Called when a conversation is deleted. Does nothing if the
        file does not exist.
        """
        try:
            if self._path.exists():
                self._path.unlink()
                logger.info(f"[ShadowMap] Deleted {self._path}")
        except OSError as exc:
            logger.error(f"[ShadowMap] Failed to delete {self._path}: {exc}")

    def __len__(self) -> int:
        """Return the number of surrogate→original mappings."""
        return len(self._mappings)

    def __repr__(self) -> str:
        return f"ShadowMap(conversation_id={self.conversation_id!r}, entries={len(self)})"
