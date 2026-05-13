"""
pipeline.py — SurrogateShield Full Message Pipeline

Orchestrates the end-to-end message flow for every turn:

    User message
        │
        ▼
    SentinelLayer (PatternScan → EntityTrace → ContextGuard)
        │  confirmed_entities, needs_confirmation
        ▼
    User confirmation for borderline entities (if any)
        │  final_entities
        ▼
    MimicGen → generate surrogates
        │  surrogate_map: {original: surrogate}
        ▼
    Apply substitutions → sanitised_message
        │
        ▼
    ShadowMap.update({surrogate: original})
    ShadowMap.save()
        │
        ▼
    [Optional] RAG query → build context prompt
        │
        ▼
    Claude API → raw_response (surrogates)
        │
        ▼
    ResolvePass → restored_response (originals back)
        │
        ▼
    Display to user

The chatbot/ sub-package has NO knowledge of detection, generation,
storage, or reconstruction — this file is the only connector.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from util import (
    DetectedEntity,
    get_logger,
    print_detection_table,
    print_needs_confirmation,
)
from detection import logic as sentinel_layer
from generation.logic import MimicGen
from storage.logic import ShadowMap
from reconstruction.logic import ResolvePass
from chatbot.chat import ClaudeChat
from chatbot.rag import RAGStore

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────

class Pipeline:
    """
    End-to-end SurrogateShield pipeline for a single conversation.

    Holds all stateful components for one conversation session:
    MimicGen (session-level surrogate collision avoidance),
    ShadowMap (persistent encrypted mapping), ResolvePass,
    ClaudeChat API handler, and optional RAGStore.

    Attributes:
        chat:     Claude API handler.
        shadow:   Encrypted surrogate↔original mapping store.
        mimic:    Surrogate generator.
        resolve:  Response reconstructor.
        rag:      Optional RAG vector store.
    """

    def __init__(
        self,
        chat: ClaudeChat,
        rag: Optional[RAGStore] = None,
    ) -> None:
        """
        Initialise the pipeline with a chat handler.

        Args:
            chat: ClaudeChat instance (new or loaded conversation).
            rag:  Optional RAGStore for RAG-mode conversations.
        """
        self.chat = chat
        self.rag = rag
        self.shadow = ShadowMap(chat.conversation.id)
        self.mimic = MimicGen()
        self.resolve = ResolvePass()

    # ── Internal helpers ────────────────────────────────────────

    def _apply_surrogates(
        self,
        text: str,
        surrogate_map: Dict[str, str],
    ) -> str:
        """
        Replace all original PII values in *text* with their surrogates.

        Args:
            text:          Original text with real PII.
            surrogate_map: Dict mapping original_text → surrogate_text.

        Returns:
            Sanitised text with surrogates substituted in.
        """
        result = text
        # Sort by length descending to avoid substring replacement conflicts
        for original in sorted(surrogate_map, key=len, reverse=True):
            surrogate = surrogate_map[original]
            result = result.replace(original, surrogate)
        return result

    def _invert_map(self, surrogate_map: Dict[str, str]) -> Dict[str, str]:
        """
        Invert {original: surrogate} → {surrogate: original} for ShadowMap.

        Args:
            surrogate_map: Dict mapping original → surrogate.

        Returns:
            Inverted dict mapping surrogate → original.
        """
        return {v: k for k, v in surrogate_map.items()}

    # ── Main turn handler ────────────────────────────────────────

    def process_turn(
        self,
        user_message: str,
        interactive: bool = True,
    ) -> Tuple[str, List[DetectedEntity], Dict[str, str]]:
        """
        Process a single conversation turn end-to-end.

        Steps:
          1. Detect PII via SentinelLayer cascade
          2. [optional] Prompt user to confirm borderline entities
          3. Generate surrogates for all confirmed entities
          4. Substitute surrogates into the message
          5. Update and save ShadowMap
          6. [optional] RAG retrieval
          7. Send to Claude API
          8. Reconstruct originals in the response

        Args:
            user_message: Raw user message (may contain PII).
            interactive:  If True, ask user to confirm borderline entities.
                          Set to False for batch/programmatic use.

        Returns:
            Tuple of:
                restored_response  — final response with real values
                confirmed_entities — entities that were replaced
                surrogate_map      — {original: surrogate} mapping used
        """
        # ── Step 1: Detection ──────────────────────────────────
        logger.info("[Pipeline] Running SentinelLayer cascade")
        confirmed, needs_confirmation = sentinel_layer.run_cascade(user_message)

        # ── Step 2: User confirmation for borderlines ──────────
        if interactive and needs_confirmation:
            approved = print_needs_confirmation(needs_confirmation)
            confirmed.extend(approved)

        confirmed = sentinel_layer.deduplicate(confirmed)

        # ── Step 3: Generate surrogates ─────────────────────────
        surrogate_map: Dict[str, str] = {}
        if confirmed:
            surrogate_map = self.mimic.generate_all(confirmed)
            # Display detection table in terminal
            print_detection_table(confirmed, surrogate_map)
        else:
            logger.info("[Pipeline] No PII detected — message sent as-is")

        # ── Step 4: Sanitise message ────────────────────────────
        sanitised = self._apply_surrogates(user_message, surrogate_map)

        # ── Step 5: Update ShadowMap ────────────────────────────
        if surrogate_map:
            inverted = self._invert_map(surrogate_map)
            self.shadow.update(inverted)
            self.shadow.save()

        # ── Step 6: RAG context retrieval ──────────────────────
        if self.rag is not None:
            chunks = self.rag.query(sanitised)
            if chunks:
                context_prefix = self.rag.build_context_prompt(chunks)
                sanitised = context_prefix + sanitised
                logger.info(f"[Pipeline] RAG: prepended {len(chunks)} chunks")

        # ── Step 7: Send to Claude API ──────────────────────────
        logger.info("[Pipeline] Sending sanitised message to Claude API")
        raw_response = self.chat.send(sanitised)

        # ── Step 8: Reconstruct originals ──────────────────────
        all_mappings = self.shadow.all_mappings()
        restored_response = self.resolve.resolve(raw_response, all_mappings)

        # Update the stored assistant message to the restored version
        self.chat.update_last_assistant_message(restored_response)

        # Persist conversation
        self.chat.save()

        return restored_response, confirmed, surrogate_map

    def add_rag_document(
        self,
        raw_text: str,
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Anonymise and index a document into the RAG store.

        Args:
            raw_text:  Document text that may contain PII.
            metadata:  Optional metadata for the document.

        Returns:
            Number of chunks indexed.

        Raises:
            RuntimeError: If no RAG store is initialised.
        """
        if self.rag is None:
            raise RuntimeError("RAG is not enabled for this pipeline. Use --rag flag.")

        # Anonymise before indexing
        confirmed, _ = sentinel_layer.run_cascade(raw_text)
        surrogate_map = self.mimic.generate_all(confirmed) if confirmed else {}
        sanitised = self._apply_surrogates(raw_text, surrogate_map)

        # Update ShadowMap for any PII found in documents
        if surrogate_map:
            self.shadow.update(self._invert_map(surrogate_map))
            self.shadow.save()

        return self.rag.add_document(sanitised, metadata=metadata)
