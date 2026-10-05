"""
storage/shadow_map.py — ShadowMap and the private-file primitives behind it.

One implementation for the library and the app (audit F4, I11, F1).

Layout (``home()`` is ``$SURROGATESHIELD_HOME`` or ``~/.surrogateshield``):

    <home>/                    0700
    <home>/device.key          0600   32 random bytes, created once
    <storage_dir>/<id>.shadowmap  0600 sealed {surrogate: original}

Sealed file format (v1)::

    b"SSv1" || nonce (12) || AES-256-GCM(key, plaintext, aad)

    key = HKDF-SHA256(ikm=device secret, salt=id, info=kind)
    aad = kind || b"|" || id           kind ∈ {b"shadowmap-v1", b"transcript-v1"}

so a file only opens as the kind and id it was written for. Ids must match
``^[A-Za-z0-9_-]{1,64}$`` (no path traversal). Every write is atomic (temp
file + fsync + rename) with mode 0600. A file that cannot be decrypted is
moved aside to ``<name>.corrupt-<ts>`` and :class:`CorruptStoreError` is
raised: it never silently becomes an empty map. If the device secret cannot
be persisted :class:`ShadowMapStorageError` is raised unless the caller
opted into an ephemeral secret.

Pre-v1 files (no magic: ``nonce || ct``, HKDF info ``b"shadowmap"``, no AAD)
are still readable and are rewritten as v1 on the next save.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)

ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAGIC = b"SSv1"
NONCE_SIZE = 12
SECRET_SIZE = 32
KIND_SHADOWMAP = b"shadowmap-v1"
KIND_TRANSCRIPT = b"transcript-v1"
_LEGACY_INFO = b"shadowmap"

PathLike = Union[str, os.PathLike]


class StorageError(RuntimeError):
    """Base class for storage failures."""


class ShadowMapStorageError(StorageError):
    """The device secret could not be read or persisted."""


class CorruptStoreError(StorageError):
    """A sealed file could not be decrypted (wrong key, tampered, truncated)."""


# ── Paths and ids ─────────────────────────────────────────────────────────────

def home() -> Path:
    """Base directory for keys and default stores."""
    return Path(os.environ.get("SURROGATESHIELD_HOME") or "~/.surrogateshield").expanduser()


def validate_id(store_id: str) -> str:
    """Return *store_id* if it is safe to use as a file name, else raise ValueError."""
    if not isinstance(store_id, str) or not ID_PATTERN.match(store_id):
        raise ValueError(f"invalid id {store_id!r}: must match {ID_PATTERN.pattern}")
    return store_id


def ensure_private_dir(path: PathLike) -> Path:
    """Create *path* (and parents) with mode 0700; tighten it if it is looser."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.stat().st_mode & 0o077:
        os.chmod(path, 0o700)
    return path


def write_private(path: PathLike, data: bytes) -> None:
    """Atomically replace *path* with *data*, mode 0600, in a 0700 directory."""
    path = Path(path)
    ensure_private_dir(path.parent)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def quarantine(path: PathLike) -> Path:
    """Move an unreadable file aside (never delete it) and return the new path."""
    path = Path(path)
    bad = path.with_name(f"{path.name}.corrupt-{int(time.time())}")
    os.replace(path, bad)
    return bad


# ── Device secret and keys ────────────────────────────────────────────────────

_secret_lock = threading.Lock()
_secret_cache: Dict[Path, bytes] = {}


def device_secret(path: Optional[PathLike] = None, *, allow_ephemeral: bool = False) -> bytes:
    """Return the 32-byte device secret at *path* (default ``home()/device.key``).

    Created once with mode 0600 (create-if-absent is atomic, so two processes
    starting together agree on one secret). Loose permissions are tightened.
    A file of the wrong length raises :class:`ShadowMapStorageError` — keys
    are never derived from a truncated secret. If the secret cannot be read
    or written, :class:`ShadowMapStorageError` is raised unless
    *allow_ephemeral*, in which case one random secret is used for the rest
    of the process (and nothing written with it survives a restart).
    """
    key_path = Path(path).expanduser() if path is not None else home() / "device.key"
    with _secret_lock:
        cached = _secret_cache.get(key_path)
        if cached is not None:
            return cached
        try:
            ensure_private_dir(key_path.parent)
            if not key_path.exists():
                _create_secret(key_path)
            if key_path.stat().st_mode & 0o077:
                os.chmod(key_path, 0o600)
                logger.warning(f"[ShadowMap] {key_path} had loose permissions; fixed to 0600")
            secret = key_path.read_bytes()
        except OSError as exc:
            if not allow_ephemeral:
                raise ShadowMapStorageError(
                    f"cannot read or persist the device secret at {key_path}: {exc}"
                ) from exc
            logger.error(
                f"[ShadowMap] EPHEMERAL device secret in use ({exc}); "
                "nothing stored by this process will be readable after it exits"
            )
            secret = os.urandom(SECRET_SIZE)
        else:
            if len(secret) != SECRET_SIZE:
                raise ShadowMapStorageError(
                    f"{key_path} is corrupt ({len(secret)} bytes, expected {SECRET_SIZE}); "
                    "refusing to derive keys from it"
                )
        _secret_cache[key_path] = secret
        return secret


def _create_secret(key_path: Path) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(key_path.parent), prefix=".tmp-key-")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(os.urandom(SECRET_SIZE))
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, key_path)          # fails if another process won the race
            logger.info(f"[ShadowMap] Generated new device secret at {key_path}")
        except FileExistsError:
            pass
    finally:
        os.unlink(tmp)


def derive_key(secret: bytes, store_id: str, kind: bytes) -> bytes:
    """32-byte AES key: HKDF-SHA256(ikm=secret, salt=id, info=kind)."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    return HKDF(
        algorithm=hashes.SHA256(), length=32, salt=store_id.encode("utf-8"), info=kind,
    ).derive(secret)


def legacy_key(secret: bytes, store_id: str) -> bytes:
    """Key used by pre-v1 files (shadow maps and transcripts alike)."""
    return derive_key(secret, store_id, _LEGACY_INFO)


def _aad(store_id: str, kind: bytes) -> bytes:
    return kind + b"|" + store_id.encode("utf-8")


def seal(key: bytes, store_id: str, kind: bytes, plaintext: bytes) -> bytes:
    """Encrypt *plaintext* bound to (*kind*, *store_id*)."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = os.urandom(NONCE_SIZE)
    return MAGIC + nonce + AESGCM(key).encrypt(nonce, plaintext, _aad(store_id, kind))


def is_sealed(blob: bytes) -> bool:
    return blob.startswith(MAGIC)


def unseal(key: bytes, store_id: str, kind: bytes, blob: bytes,
           legacy: Optional[bytes] = None) -> bytes:
    """Decrypt a v1 blob, or a pre-v1 blob with the *legacy* key.

    Raises :class:`CorruptStoreError` if the blob does not open.
    """
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    try:
        if is_sealed(blob):
            body = blob[len(MAGIC):]
            if len(body) <= NONCE_SIZE:
                raise CorruptStoreError("truncated")
            return AESGCM(key).decrypt(body[:NONCE_SIZE], body[NONCE_SIZE:], _aad(store_id, kind))
        if legacy is None or len(blob) <= NONCE_SIZE:
            raise CorruptStoreError("not a sealed file")
        return AESGCM(legacy).decrypt(blob[:NONCE_SIZE], blob[NONCE_SIZE:], None)
    except InvalidTag as exc:
        raise CorruptStoreError("authentication failed (wrong key, wrong id, or tampered)") from exc


# ── ShadowMap ─────────────────────────────────────────────────────────────────

class ShadowMap:
    """
    Surrogate→original mapping store with optional encrypted persistence.

    Args:
        session_id:      Store id; must match ``^[A-Za-z0-9_-]{1,64}$``.
        storage_dir:     Directory for ``<id>.shadowmap``, or None for memory only.
        secret_path:     Device-secret file (default ``home()/device.key``).
        allow_ephemeral: Use a process-lifetime secret if the device secret
                         cannot be persisted, instead of raising.
        autosave:        Persist after every ``update``/``add`` (library);
                         the app passes False and calls ``save()`` itself.

    Raises:
        ValueError:            invalid id.
        ShadowMapStorageError: device secret unavailable.
        CorruptStoreError:     the existing file could not be decrypted; it
                               has been moved to ``<name>.corrupt-<ts>``.
    """

    def __init__(
        self,
        session_id: str,
        storage_dir: Optional[PathLike] = None,
        *,
        secret_path: Optional[PathLike] = None,
        allow_ephemeral: bool = False,
        autosave: bool = True,
    ) -> None:
        self._session_id = validate_id(session_id)
        self._lock = threading.RLock()
        self._autosave = autosave
        self._mappings: Dict[str, str] = {}
        # Forward index (original → surrogate), maintained alongside the main
        # map so callers never have to copy + invert the whole map per call.
        self._reverse: Dict[str, str] = {}
        self._storage_dir = storage_dir
        if storage_dir is None:
            self._dir = self._map_path = None
            self._key = self._legacy_key = None
            return
        self._dir = Path(storage_dir).expanduser()
        self._map_path = self._dir / f"{self._session_id}.shadowmap"
        # pre-v1 library layout kept a per-session key beside the ciphertext
        self._old_key_path = self._dir / f"{self._session_id}.key"
        secret = device_secret(secret_path, allow_ephemeral=allow_ephemeral)
        self._key = derive_key(secret, self._session_id, KIND_SHADOWMAP)
        self._legacy_key = legacy_key(secret, self._session_id)
        self._load()

    @property
    def session_id(self) -> str:
        return self._session_id

    conversation_id = session_id

    @property
    def persistent(self) -> bool:
        return self._map_path is not None

    @property
    def path(self) -> Optional[Path]:
        return self._map_path

    # ── Mapping interface ──────────────────────────────────────────────────

    def add(self, surrogate: str, original: str) -> None:
        self.update({surrogate: original})

    def update(self, new_mappings: Dict[str, str]) -> None:
        """Merge new surrogate→original mappings (persisted if autosave)."""
        with self._lock:
            self._mappings.update(new_mappings)
            for surrogate, original in new_mappings.items():
                self._reverse[original] = surrogate
            if self._autosave:
                self._save_locked()

    def get(self, surrogate: str) -> Optional[str]:
        return self._mappings.get(surrogate)

    def get_all(self) -> Dict[str, str]:
        """Return a copy of all current surrogate→original mappings."""
        with self._lock:
            return dict(self._mappings)

    all_mappings = get_all

    def lookup_original(self, original: str) -> Optional[str]:
        """Return the surrogate already issued for *original*, or None.
        O(1) via the maintained forward index."""
        return self._reverse.get(original)

    def originals(self):
        """View of all original values currently mapped (forward-index keys)."""
        return self._reverse.keys()

    def forget(self, original: str) -> int:
        """Erase every mapping to *original* (and persist). Returns the count."""
        with self._lock:
            gone = [s for s, o in self._mappings.items() if o == original]
            for s in gone:
                del self._mappings[s]
            self._reverse.pop(original, None)
            if gone and self.persistent:
                self._save_locked()
            return len(gone)

    def flush(self) -> None:
        """Clear all mappings and delete the file if persistent."""
        with self._lock:
            self._mappings.clear()
            self._reverse.clear()
            self.delete()

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self) -> None:
        """Seal the mappings and atomically write them (no-op in memory mode).

        Raises OSError if the write fails: a mapping that cannot be stored
        means a later turn cannot be restored, so the caller must know.
        """
        with self._lock:
            self._save_locked()

    def _save_locked(self) -> None:
        if not self.persistent:
            return
        plaintext = json.dumps(self._mappings).encode("utf-8")
        write_private(self._map_path, seal(self._key, self._session_id, KIND_SHADOWMAP, plaintext))
        if self._old_key_path.exists():
            self._old_key_path.unlink()
        logger.debug(f"[ShadowMap] Saved {len(self._mappings)} mappings → {self._map_path}")

    def _load(self) -> None:
        if not self._map_path.exists():
            return
        blob = self._map_path.read_bytes()
        legacy = self._legacy_key
        if not is_sealed(blob) and self._old_key_path.exists():
            legacy = legacy_key(self._old_key_path.read_bytes(), self._session_id)
        try:
            mappings = json.loads(unseal(self._key, self._session_id, KIND_SHADOWMAP, blob, legacy))
            if not isinstance(mappings, dict):
                raise CorruptStoreError("not a mapping")
        except (CorruptStoreError, ValueError) as exc:
            bad = quarantine(self._map_path)
            raise CorruptStoreError(
                f"shadow map {self._map_path.name} could not be decrypted ({exc}); "
                f"moved to {bad.name}"
            ) from exc
        self._mappings = mappings
        self._reverse = {v: k for k, v in mappings.items()}
        logger.info(f"[ShadowMap] Loaded {len(mappings)} mappings from {self._map_path}")

    def delete(self) -> None:
        """Delete the file (and any pre-v1 key file); keep the in-memory map."""
        if not self.persistent:
            return
        for path in (self._map_path, self._old_key_path):
            try:
                path.unlink()
                logger.debug(f"[ShadowMap] Deleted {path}")
            except FileNotFoundError:
                pass

    @staticmethod
    def erase(session_id: str, storage_dir: PathLike) -> None:
        """Delete a stored map without opening it (works on unreadable files)."""
        path = Path(storage_dir).expanduser() / f"{validate_id(session_id)}.shadowmap"
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def __len__(self) -> int:
        return len(self._mappings)

    def __repr__(self) -> str:
        return (f"ShadowMap(session_id={self._session_id!r}, entries={len(self)}, "
                f"persistent={self.persistent})")
