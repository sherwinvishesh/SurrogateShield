"""
storage/shadow_map.py — ShadowMap

Dual-mode surrogate mapping store: memory-only or AES-256-GCM encrypted disk.

MEMORY MODE (storage_dir=None):
  All mappings held in a Python dict. Nothing written to disk. flush() clears
  the dict.

PERSISTENT MODE (storage_dir is a path):
  After every update the mapping is encrypted and written to
  storage_dir/session_id.shadowmap using AES-256-GCM.  The per-session key
  is derived with HKDF-SHA256 from a randomly generated 32-byte session key
  stored at storage_dir/session_id.key (owner-only, 0o600).  flush() deletes
  both files.

File format (persistent): nonce (12 bytes) || ciphertext.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_AES_NONCE_SIZE = 12
_HKDF_INFO = b"shadowmap"


def _derive_key(session_key: bytes, session_id: str) -> bytes:
    """Derive a 32-byte AES key via HKDF-SHA256."""
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=session_id.encode("utf-8"),
        info=_HKDF_INFO,
    )
    return hkdf.derive(session_key)


class ShadowMap:
    """
    Surrogate→original mapping store with optional encrypted persistence.

    Args:
        session_id:  Unique identifier for this session.
        storage_dir: Directory path for persistent mode, or None for memory-only.
    """

    def __init__(self, session_id: str, storage_dir: Optional[str] = None) -> None:
        self._session_id = session_id
        self._storage_dir = storage_dir
        self._mappings: Dict[str, str] = {}

        if storage_dir is not None:
            self._dir = Path(storage_dir)
            self._map_path = self._dir / f"{session_id}.shadowmap"
            self._key_path = self._dir / f"{session_id}.key"
            self._key: Optional[bytes] = self._load_or_create_key()
            self._load()
        else:
            self._dir = None
            self._map_path = None
            self._key_path = None
            self._key = None

    # ── Key management ─────────────────────────────────────────────────────────

    def _load_or_create_key(self) -> bytes:
        """Load session key from disk or generate a new one (owner-only perms)."""
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            if self._key_path.exists():
                return self._key_path.read_bytes()
            raw = os.urandom(32)
            fd = os.open(str(self._key_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            os.write(fd, raw)
            os.close(fd)
            logger.debug(f"[ShadowMap] Generated session key at {self._key_path}")
            return raw
        except OSError as exc:
            logger.error(f"[ShadowMap] Could not read/write session key: {exc}")
            return os.urandom(32)  # ephemeral fallback

    # ── Public interface ───────────────────────────────────────────────────────

    def update(self, new_mappings: Dict[str, str]) -> None:
        """Merge new surrogate→original mappings and persist if configured."""
        self._mappings.update(new_mappings)
        if self._storage_dir is not None:
            self._save()

    def get_all(self) -> Dict[str, str]:
        """Return a copy of all current surrogate→original mappings."""
        return dict(self._mappings)

    def flush(self) -> None:
        """Clear all mappings and delete disk files if in persistent mode."""
        self._mappings.clear()
        if self._storage_dir is not None:
            for path in (self._map_path, self._key_path):
                if path is not None and path.exists():
                    try:
                        path.unlink()
                        logger.debug(f"[ShadowMap] Deleted {path}")
                    except OSError as exc:
                        logger.warning(f"[ShadowMap] Could not delete {path}: {exc}")

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save(self) -> None:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            derived = _derive_key(self._key, self._session_id)
            plaintext = json.dumps(self._mappings).encode("utf-8")
            nonce = os.urandom(_AES_NONCE_SIZE)
            aesgcm = AESGCM(derived)
            ciphertext = aesgcm.encrypt(nonce, plaintext, None)
            self._map_path.write_bytes(nonce + ciphertext)
            logger.debug(f"[ShadowMap] Saved {len(self._mappings)} mappings → {self._map_path}")
        except OSError as exc:
            logger.error(f"[ShadowMap] Failed to save: {exc}")

    def _load(self) -> None:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        if not self._map_path.exists():
            return
        try:
            data = self._map_path.read_bytes()
            if len(data) <= _AES_NONCE_SIZE:
                logger.warning(f"[ShadowMap] File too short: {self._map_path}")
                return
            nonce = data[:_AES_NONCE_SIZE]
            ciphertext = data[_AES_NONCE_SIZE:]
            derived = _derive_key(self._key, self._session_id)
            aesgcm = AESGCM(derived)
            plaintext = aesgcm.decrypt(nonce, ciphertext, None)
            self._mappings = json.loads(plaintext.decode("utf-8"))
            logger.info(f"[ShadowMap] Loaded {len(self._mappings)} mappings from {self._map_path}")
        except Exception as exc:
            logger.warning(
                f"[ShadowMap] Could not decrypt {self._map_path}: {exc}. "
                "Starting with empty mappings."
            )
            self._mappings = {}

    def __len__(self) -> int:
        return len(self._mappings)

    def __repr__(self) -> str:
        return f"ShadowMap(session_id={self._session_id!r}, entries={len(self)}, persistent={self._storage_dir is not None})"
