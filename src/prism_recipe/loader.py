"""Egalitarian HF FineWeb-Edu loader (VAL-RECIPE-002).

Workers stream the pinned FineWeb-Edu revision themselves (or from a local HF
cache). A live master FineWeb mount is **not** required.

Contract
--------
- Fixed dataset id + immutable revision (commit SHA).
- Fixed global token start offset (EQUAL_OFFSET) identical for every architecture.
- Default prod token_budget = TOKEN_BUDGET_PROD (2_500_000_000).
- Single-pass / one epoch only (no multi-epoch rescan of the window).
- Smoke may shrink budget via ``PRISM_RECIPE_TOKEN_BUDGET`` without changing
  offset pin identity.

Token accounting uses a simple whitespace word split when no encode_fn is
supplied, so unit tests and offline smoke need no tiktoken/network. Production
train code may inject a real tokenizer encode function.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from prism_recipe.config import (
    EQUAL_OFFSET,
    PROD_DATASET_ID,
    PROD_DATASET_REVISION,
    TOKEN_BUDGET_PROD,
    DataWindowPin,
    prod_data_window,
    resolve_token_budget,
)

EncodeFn = Callable[[str], Sequence[int]]
TextDocument = dict[str, Any]


class LoaderError(RuntimeError):
    """Raised when the egalitarian loader cannot satisfy its contract."""


@dataclass(frozen=True, slots=True)
class LoaderPlan:
    """Resolved load plan (pin identity + effective budget)."""

    pin: DataWindowPin
    token_budget: int
    single_pass: bool = True
    epochs: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.pin.dataset_id,
            "dataset_revision": self.pin.dataset_revision,
            "token_start_offset": self.pin.token_start_offset,
            "token_budget": self.token_budget,
            "epochs": self.epochs,
            "single_pass": self.single_pass,
            "equal_offset": EQUAL_OFFSET,
            "token_budget_prod": TOKEN_BUDGET_PROD,
            "master_fineweb_mount_required": False,
        }


@dataclass(frozen=True, slots=True)
class DocumentSlice:
    """A document contribution inside the egalitarian window."""

    document: TextDocument
    token_start: int  # inclusive index into encoded tokens
    token_count: int  # number of tokens taken from this document

    @property
    def text(self) -> str:
        return _document_text(self.document)


def _fnv1a_32(data: bytes) -> int:
    """FNV-1a 32-bit (same constants as smoke_train.fixture_encode)."""
    h = 2166136261
    for byte in data:
        h ^= byte
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def default_encode(text: str) -> list[int]:
    """Offline-safe default: one synthetic token id per whitespace word.

    Real train loops should pass a tokenizer encode_fn. Word split keeps unit
    tests deterministic without optional train extras (tiktoken/torch).

    Token ids are derived via FNV-1a (not builtin hash()) so results are stable
    across PYTHONHASHSEED values — matching the smoke fixture encode path.
    """
    if not text or not text.strip():
        return []
    # Stable synthetic ids from FNV-1a word hashes (positive 31-bit).
    out: list[int] = []
    for word in text.split():
        h = _fnv1a_32(word.encode("utf-8"))
        out.append((h % (2**31 - 1)) or 1)
    return out


def count_tokens(text: str, encode_fn: EncodeFn | None = None) -> int:
    """Count tokens for a document using encode_fn or the default encoder."""
    encoder = encode_fn or default_encode
    return len(encoder(text))


def _document_text(doc: TextDocument) -> str:
    text = doc.get("text")
    if text is None:
        text = doc.get("content")
    if not isinstance(text, str):
        return ""
    return text


def apply_offset_and_budget(
    documents: Iterable[TextDocument],
    *,
    token_start_offset: int,
    token_budget: int,
    encode_fn: EncodeFn | None = None,
    epochs: int = 1,
) -> Iterator[DocumentSlice]:
    """Skip ``token_start_offset`` tokens, then yield slices until ``token_budget``.

    Enforces single-pass: ``epochs`` must be 1; the input iterable is scanned
    at most once (no wrap / multi-epoch rescan).
    """
    if epochs != 1:
        raise LoaderError(
            f"single-pass enforced: epochs must be 1, got {epochs} "
            "(multi-epoch rescan is forbidden for the egalitarian window)"
        )
    if token_start_offset < 0:
        raise LoaderError("token_start_offset must be >= 0")
    if token_budget < 1:
        raise LoaderError("token_budget must be >= 1")

    encoder = encode_fn or default_encode
    skipped = 0
    emitted = 0

    for doc in documents:
        if emitted >= token_budget:
            return
        text = _document_text(doc)
        tokens = list(encoder(text))
        n = len(tokens)
        if n == 0:
            continue

        start_in_doc = 0
        if skipped < token_start_offset:
            remaining_skip = token_start_offset - skipped
            if n <= remaining_skip:
                skipped += n
                continue
            start_in_doc = remaining_skip
            skipped = token_start_offset

        available = n - start_in_doc
        room = token_budget - emitted
        take = available if available <= room else room
        if take <= 0:
            return
        yield DocumentSlice(document=doc, token_start=start_in_doc, token_count=take)
        emitted += take
        if emitted >= token_budget:
            return

    # Stream exhausted before budget filled: stop cleanly (single-pass, no rescan).
    return


def iter_budgeted_tokens(
    documents: Iterable[TextDocument],
    *,
    token_start_offset: int,
    token_budget: int,
    encode_fn: EncodeFn | None = None,
    epochs: int = 1,
) -> Iterator[int]:
    """Yield individual token ids after offset, stopping at budget (single-pass)."""
    encoder = encode_fn or default_encode
    for slc in apply_offset_and_budget(
        documents,
        token_start_offset=token_start_offset,
        token_budget=token_budget,
        encode_fn=encoder,
        epochs=epochs,
    ):
        tokens = list(encoder(_document_text(slc.document)))
        end = slc.token_start + slc.token_count
        yield from tokens[slc.token_start : end]


class EgalitarianFineWebLoader:
    """Deterministic FineWeb-Edu window loader for the recipe image.

    Network/HF access is optional: pass ``document_source`` for offline/mocked
    streams (unit tests), or call ``iter_text_shards`` / ``open_hf_stream`` when
    HuggingFace ``datasets`` is available on the worker.
    """

    def __init__(
        self,
        pin: DataWindowPin | None = None,
        *,
        document_source: Iterable[TextDocument] | None = None,
        encode_fn: EncodeFn | None = None,
        token_budget: int | None = None,
    ) -> None:
        self.pin = pin or prod_data_window()
        self.pin.validate()
        self._document_source = document_source
        self._encode_fn = encode_fn
        self._token_budget_override = token_budget

    def plan(self) -> LoaderPlan:
        if self._token_budget_override is not None:
            budget = self._token_budget_override
        else:
            budget = resolve_token_budget(default=self.pin.token_budget)
        if self.pin.epochs != 1:
            raise LoaderError(
                f"single-pass enforced: pin.epochs must be 1, got {self.pin.epochs}"
            )
        return LoaderPlan(
            pin=self.pin,
            token_budget=budget,
            single_pass=True,
            epochs=1,
        )

    def open_hf_stream(self) -> Iterator[TextDocument]:
        """Stream documents from the pinned FineWeb-Edu revision via HF datasets.

        Uses streaming mode so workers do not need a master FineWeb mount. Local
        HF caches are honored automatically by the datasets library when present.
        """
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - optional runtime dep
            raise LoaderError(
                "the 'datasets' package is required to stream FineWeb-Edu; "
                "install train extras or pass document_source= for offline use"
            ) from exc

        plan = self.plan()
        stream = load_dataset(
            plan.pin.dataset_id,
            name=plan.pin.dataset_config,
            split=plan.pin.dataset_split,
            revision=plan.pin.dataset_revision,
            streaming=True,
        )
        for record in stream:
            if isinstance(record, dict):
                yield dict(record)
            else:  # pragma: no cover - defensive
                yield {"text": str(record)}

    def _documents(self) -> Iterable[TextDocument]:
        if self._document_source is not None:
            return self._document_source
        return self.open_hf_stream()

    def iter_documents(self) -> Iterator[DocumentSlice]:
        """Yield document slices inside the egalitarian window."""
        plan = self.plan()
        yield from apply_offset_and_budget(
            self._documents(),
            token_start_offset=plan.pin.token_start_offset,
            token_budget=plan.token_budget,
            encode_fn=self._encode_fn,
            epochs=plan.epochs,
        )

    def iter_tokens(self) -> Iterator[int]:
        """Yield token ids for the egalitarian window (stops at budget)."""
        plan = self.plan()
        yield from iter_budgeted_tokens(
            self._documents(),
            token_start_offset=plan.pin.token_start_offset,
            token_budget=plan.token_budget,
            encode_fn=self._encode_fn,
            epochs=plan.epochs,
        )

    def iter_text_shards(self) -> Iterator[str]:
        """Yield raw text for documents that contribute to the window."""
        for slc in self.iter_documents():
            text = slc.text
            if text:
                yield text

    def collect_token_count(self) -> int:
        """Materialize the window token count (for tests / preflight stats)."""
        return sum(1 for _ in self.iter_tokens())


def build_loader(
    pin: DataWindowPin | None = None,
    *,
    document_source: Iterable[TextDocument] | None = None,
    encode_fn: EncodeFn | None = None,
    token_budget: int | None = None,
) -> EgalitarianFineWebLoader:
    """Factory used by harness entrypoints."""
    return EgalitarianFineWebLoader(
        pin=pin,
        document_source=document_source,
        encode_fn=encode_fn,
        token_budget=token_budget,
    )


# Public pin constants re-exported for discoverability.
__all__ = [
    "EQUAL_OFFSET",
    "DocumentSlice",
    "EncodeFn",
    "EgalitarianFineWebLoader",
    "LoaderError",
    "LoaderPlan",
    "PROD_DATASET_ID",
    "PROD_DATASET_REVISION",
    "TOKEN_BUDGET_PROD",
    "TextDocument",
    "apply_offset_and_budget",
    "build_loader",
    "count_tokens",
    "default_encode",
    "iter_budgeted_tokens",
]
