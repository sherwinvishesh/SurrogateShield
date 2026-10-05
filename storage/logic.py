# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
storage/logic.py — the app's ShadowMap.

``surrogateshield.core.storage.shadow_map`` (one implementation, audit F4)
bound to config.py: the store lives in ``conversations_dir()`` and the app
saves explicitly (``autosave=False``). See that module for the file format,
permissions and failure behaviour (audit I11, F1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import config as _config
from surrogateshield.core.storage.shadow_map import (  # noqa: F401
    KIND_SHADOWMAP,
    KIND_TRANSCRIPT,
    CorruptStoreError,
    ShadowMapStorageError,
    StorageError,
    derive_key,
    device_secret,
    home,
    legacy_key,
    seal,
    unseal,
    validate_id,
    write_private,
)
from surrogateshield.core.storage.shadow_map import ShadowMap as _ShadowMap


def conversations_dir() -> Path:
    """Directory for shadow maps and transcripts (absolute)."""
    if _config.SHADOWMAP_DIR:
        return Path(_config.SHADOWMAP_DIR).expanduser().resolve()
    return home() / "conversations"


def _get_device_secret() -> bytes:
    return device_secret(_config.DEVICE_KEY_PATH,
                         allow_ephemeral=_config.ALLOW_EPHEMERAL_KEY)


def _derive_key(conversation_id: str, kind: bytes = KIND_SHADOWMAP) -> bytes:
    """Per-conversation AES key for *kind* (shadow map or transcript)."""
    return derive_key(_get_device_secret(), validate_id(conversation_id), kind)


def _legacy_key(conversation_id: str) -> bytes:
    return legacy_key(_get_device_secret(), validate_id(conversation_id))


class ShadowMap(_ShadowMap):
    """App shadow map for one conversation, stored in ``conversations_dir()``."""

    def __init__(self, conversation_id: str, storage_dir: Optional[str] = None) -> None:
        super().__init__(
            conversation_id,
            storage_dir if storage_dir is not None else conversations_dir(),
            secret_path=_config.DEVICE_KEY_PATH,
            allow_ephemeral=_config.ALLOW_EPHEMERAL_KEY,
            autosave=False,
        )


def erase(conversation_id: str) -> bool:
    """Delete a conversation's shadow map without decrypting it. Returns False
    when there was none."""
    return _ShadowMap.erase(conversation_id, conversations_dir())


RAG_STORE_ID = "rag_global"


def surrogates_in_use(exclude: str = RAG_STORE_ID) -> set:
    """Every surrogate issued by any conversation's shadow map, read without
    side effects (an unreadable map is skipped with a warning, not moved
    aside). add-doc uses it so a document surrogate never equals a chat
    surrogate that means someone else (audit I16)."""
    used: set = set()
    directory = conversations_dir()
    if not directory.is_dir():
        return used
    import json
    from util import get_logger
    for path in sorted(directory.glob("*.shadowmap")):
        store_id = path.stem
        if store_id == exclude:
            continue
        try:
            blob = path.read_bytes()
            plaintext = unseal(_derive_key(store_id), store_id, KIND_SHADOWMAP, blob,
                               legacy=_legacy_key(store_id))
            used.update(json.loads(plaintext))
        except (OSError, ValueError, CorruptStoreError) as exc:
            get_logger(__name__).warning(f"[storage] skipped unreadable map {path.name}: {exc}")
    return used
