"""FineWeb-Edu streaming helpers for chronologically bounded SN38 pretraining.

Design goals:
- never require a full FineWeb-Edu download;
- mix years via weights while keeping **only one** crawl iterator open (avoids
  the multi-dump OOM / resolve thrash of the old interleaver);
- stay on a dump for `docs_per_turn` docs before switching;
- optional FineWeb-Edu `int_score` / `score` quality floor;
- pack documents with EOS separators;
- save/restore stream + packing state for cheap resume.

This module intentionally does *not* infer temporal safety from page text.  The
caller must provide only crawl configs allowed by the target cutoff year.
"""

from __future__ import annotations

import hashlib
import itertools
import random
import re
from collections import Counter
from typing import Iterator

import torch
from datasets import load_dataset


_DUMP_YEAR_RE = re.compile(r"CC-MAIN-(\d{4})-\d+")


def dump_year(dump: str) -> int:
    """Extract the Common Crawl year from a config such as CC-MAIN-2018-51."""
    match = _DUMP_YEAR_RE.fullmatch(dump)
    if match is None:
        raise ValueError(f"Unsupported FineWeb-Edu dump name: {dump!r}")
    return int(match.group(1))


def validate_cutoff_dumps(dumps: list[str], cutoff_year: int) -> None:
    """Fail fast if any configured crawl is later than the model cutoff."""
    if not dumps:
        raise ValueError("At least one training dump is required")
    too_new = [d for d in dumps if dump_year(d) > cutoff_year]
    if too_new:
        raise ValueError(
            f"Cutoff model is {cutoff_year}, but later crawl configs were supplied: {too_new}"
        )


def _stable_seed(base_seed: int, dump: str, cycle: int) -> int:
    """Stable per-source seed; unlike Python hash(), this is process-independent."""
    raw = f"{base_seed}|{dump}|{cycle}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & 0x7FFFFFFF


def _normalized_source_weights(
    dumps: list[str], year_weights: dict[int, float] | None
) -> list[float]:
    """Split each year's mass equally across that year's crawl configs."""
    counts = Counter(dump_year(d) for d in dumps)
    if year_weights is None:
        return [1.0 / len(dumps)] * len(dumps)

    missing = sorted(set(counts) - set(year_weights))
    if missing:
        raise ValueError(f"Missing data.year_weights entries for years: {missing}")

    positive = {y: float(year_weights[y]) for y in counts if float(year_weights[y]) > 0}
    if not positive:
        raise ValueError("data.year_weights must contain at least one positive weight")

    total_year_mass = sum(positive.values())
    weights = []
    for dump in dumps:
        y = dump_year(dump)
        year_mass = positive.get(y, 0.0) / total_year_mass
        weights.append(year_mass / counts[y])

    total = sum(weights)
    if total <= 0:
        raise ValueError("Configured source weights sum to zero")
    return [w / total for w in weights]


def _row_passes_quality(row: dict, min_int_score: int | None, min_score: float | None) -> bool:
    if min_int_score is not None:
        value = row.get("int_score")
        if value is None:
            return True  # older rows / missing field — keep
        if int(value) < int(min_int_score):
            return False
    if min_score is not None:
        value = row.get("score")
        if value is None:
            return True
        if float(value) < float(min_score):
            return False
    return True


class CycleFineWebStream:
    """Weighted FineWeb-Edu stream with a single open crawl (OOM-safe mix).

    - Pick the next dump by `year_weights` (or uniform / sequential).
    - Emit `docs_per_turn` kept documents from that dump, then switch.
    - Close the previous iterator before opening the next (max 1 resident).
    - Optional `min_int_score` / `min_score` filters for stronger edu pages.
    """

    def __init__(
        self,
        dumps: list[str],
        dataset: str = "HuggingFaceFW/fineweb-edu",
        *,
        cutoff_year: int,
        year_weights: dict[int, float] | None = None,
        docs_per_turn: int = 4096,
        shuffle_buffer_size: int = 64,
        seed: int = 42,
        min_int_score: int | None = None,
        min_score: float | None = None,
        sequential: bool = False,
    ):
        validate_cutoff_dumps(dumps, cutoff_year)
        if docs_per_turn <= 0:
            raise ValueError("docs_per_turn must be > 0")
        if shuffle_buffer_size < 0:
            raise ValueError("shuffle_buffer_size must be >= 0")

        self.dumps = list(dumps)
        self.dataset = dataset
        self.cutoff_year = int(cutoff_year)
        self.docs_per_turn = int(docs_per_turn)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.seed = int(seed)
        self.min_int_score = None if min_int_score is None else int(min_int_score)
        self.min_score = None if min_score is None else float(min_score)
        self.sequential = bool(sequential)

        self.source_weights = _normalized_source_weights(self.dumps, year_weights)
        self._weight_by_dump = dict(zip(self.dumps, self.source_weights))

        self._rng = random.Random(self.seed)
        self._offsets = {d: 0 for d in self.dumps}
        self._cycles = {d: 0 for d in self.dumps}
        self._active_dump: str | None = None
        self._remaining_in_turn = 0
        self._seq_idx = 0
        self._iter: Iterator | None = None

        mode = "sequential" if self.sequential else "weighted-single-open"
        print(
            f"[data] stream={mode} dumps={len(self.dumps)} "
            f"docs_per_turn={self.docs_per_turn} "
            f"min_int_score={self.min_int_score} min_score={self.min_score}"
        )
        if not self.sequential and year_weights:
            pretty = ", ".join(f"{y}:{w:g}" for y, w in sorted(year_weights.items()))
            print(f"[data] year_weights={{ {pretty} }}")

    def __iter__(self):
        return self

    def _make_source_iter(self, dump: str) -> Iterator:
        ds = load_dataset(self.dataset, name=dump, split="train", streaming=True)
        cycle = self._cycles[dump]
        if self.shuffle_buffer_size > 1:
            ds = ds.shuffle(
                seed=_stable_seed(self.seed, dump, cycle),
                buffer_size=self.shuffle_buffer_size,
            )
        offset = self._offsets[dump]
        if offset:
            ds = ds.skip(offset)
        return iter(ds)

    def _open_dump(self, dump: str) -> None:
        print(
            f"[data] open {dump} "
            f"(offset={self._offsets[dump]:,} cycle={self._cycles[dump]})"
        )
        self._active_dump = dump
        self._iter = self._make_source_iter(dump)

    def _close_dump(self) -> None:
        self._iter = None
        self._active_dump = None

    def _pick_next_dump(self) -> str:
        if self.sequential:
            dump = self.dumps[self._seq_idx % len(self.dumps)]
            self._seq_idx += 1
            if self._seq_idx % len(self.dumps) == 0:
                # mark a full list pass for logging/resume clarity
                pass
            return dump
        return self._rng.choices(self.dumps, weights=self.source_weights, k=1)[0]

    def _start_turn(self) -> None:
        dump = self._pick_next_dump()
        if dump != self._active_dump:
            self._close_dump()
            self._open_dump(dump)
        self._remaining_in_turn = self.docs_per_turn

    def _exhaust_and_recycle(self, dump: str) -> None:
        self._cycles[dump] += 1
        self._offsets[dump] = 0
        self._close_dump()
        self._open_dump(dump)

    def __next__(self) -> str:
        while True:
            if self._active_dump is None or self._remaining_in_turn <= 0:
                self._start_turn()

            dump = self._active_dump
            assert dump is not None and self._iter is not None

            try:
                row = next(self._iter)
                self._offsets[dump] += 1
            except StopIteration:
                self._exhaust_and_recycle(dump)
                continue

            text = row.get("text")
            if not text:
                continue
            if not _row_passes_quality(row, self.min_int_score, self.min_score):
                continue

            self._remaining_in_turn -= 1
            return text

    def state_dict(self) -> dict:
        return {
            "version": "cycle_v2",
            "rng_state": self._rng.getstate(),
            "offsets": dict(self._offsets),
            "cycles": dict(self._cycles),
            "active_dump": self._active_dump,
            "remaining_in_turn": int(self._remaining_in_turn),
            "seq_idx": int(self._seq_idx),
        }

    def load_state_dict(self, state: dict) -> None:
        version = state.get("version")
        if version == "cycle_v1":
            # Best-effort migrate from plain sequential cycle checkpoints.
            dump_idx = int(state.get("dump_idx", 0))
            if not 0 <= dump_idx < len(self.dumps):
                raise ValueError(f"Resume dump_idx={dump_idx} out of range")
            dump = self.dumps[dump_idx]
            self._offsets = {d: 0 for d in self.dumps}
            self._offsets[dump] = int(state.get("offset", 0))
            self._cycles = {d: 0 for d in self.dumps}
            self._cycles[dump] = int(state.get("list_cycle", 0))
            self._active_dump = dump
            self._remaining_in_turn = self.docs_per_turn
            self._seq_idx = dump_idx
            self._iter = None
            return

        if version not in {None, "cycle_v2"} and "offsets" not in state:
            raise ValueError(
                "Checkpoint data_state is incompatible with the current stream; "
                "use legacy sequence replay or restart from --init-from / scratch."
            )

        if "rng_state" in state:
            self._rng.setstate(state["rng_state"])
        self._offsets = {d: int(state.get("offsets", {}).get(d, 0)) for d in self.dumps}
        self._cycles = {d: int(state.get("cycles", {}).get(d, 0)) for d in self.dumps}
        active = state.get("active_dump")
        if active is not None and active not in self.dumps:
            raise ValueError(f"Resume state references unknown dump: {active}")
        self._active_dump = active
        self._remaining_in_turn = int(state.get("remaining_in_turn", 0))
        self._seq_idx = int(state.get("seq_idx", 0))
        self._iter = None


def cycle_dumps(dumps: list[str], dataset: str) -> Iterator[str]:
    """Infinite sequential cycle over dumps (no resume state)."""
    while True:
        yield from iter_fineweb_edu(dumps, dataset=dataset)


class PackedCausalStream:
    """Stateful EOS-separated packing into fixed-length causal LM blocks."""

    def __init__(self, texts: Iterator[str], tokenizer, seq_len: int):
        if seq_len <= 1:
            raise ValueError("seq_len must be > 1")
        self.texts = texts
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.buffer: list[int] = []

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        eos = self.tokenizer.eos_token_id
        while len(self.buffer) < self.seq_len:
            text = next(self.texts)
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            self.buffer.extend(ids)
            if eos is not None:
                self.buffer.append(eos)

        chunk = self.buffer[: self.seq_len]
        del self.buffer[: self.seq_len]
        input_ids = torch.tensor(chunk, dtype=torch.long)
        return {"input_ids": input_ids}

    def state_dict(self) -> dict:
        state = {"buffer": list(self.buffer)}
        if hasattr(self.texts, "state_dict"):
            state["text_stream"] = self.texts.state_dict()
        return state

    def load_state_dict(self, state: dict) -> None:
        self.buffer = [int(x) for x in state.get("buffer", [])]
        text_state = state.get("text_stream")
        if text_state is not None:
            if not hasattr(self.texts, "load_state_dict"):
                raise ValueError("Underlying text stream cannot restore saved state")
            self.texts.load_state_dict(text_state)


def _stream_kwargs(
    *,
    cutoff_year: int,
    year_weights: dict[int, float] | None,
    docs_per_turn: int,
    shuffle_buffer_size: int,
    seed: int,
    min_int_score: int | None,
    min_score: float | None,
    sequential: bool,
) -> dict:
    return dict(
        cutoff_year=cutoff_year,
        year_weights=year_weights,
        docs_per_turn=docs_per_turn,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
        min_int_score=min_int_score,
        min_score=min_score,
        sequential=sequential,
    )


def build_packed_stream(
    dumps: list[str],
    dataset: str,
    tokenizer,
    *,
    cutoff_year: int,
    seq_len: int,
    year_weights: dict[int, float] | None = None,
    docs_per_turn: int = 4096,
    shuffle_buffer_size: int = 64,
    seed: int = 42,
    min_int_score: int | None = None,
    min_score: float | None = None,
    sequential: bool = False,
    **_ignored,
) -> PackedCausalStream:
    text_stream = CycleFineWebStream(
        dumps,
        dataset,
        **_stream_kwargs(
            cutoff_year=cutoff_year,
            year_weights=year_weights,
            docs_per_turn=docs_per_turn,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
            min_int_score=min_int_score,
            min_score=min_score,
            sequential=sequential,
        ),
    )
    return PackedCausalStream(text_stream, tokenizer, seq_len)


def iter_fineweb_edu(
    dumps: list[str], dataset: str = "HuggingFaceFW/fineweb-edu"
) -> Iterator[str]:
    """Simple sequential stream retained for inspection/debug utilities."""
    for dump in dumps:
        ds = load_dataset(dataset, name=dump, split="train", streaming=True)
        for row in ds:
            text = row.get("text")
            if text:
                yield text


def take_texts(dumps: list[str], dataset: str, n: int) -> list[str]:
    """Return the first `n` non-empty documents for quick dataset inspection."""
    return list(itertools.islice(iter_fineweb_edu(dumps, dataset=dataset), n))


def build_text_stream(
    dumps: list[str],
    dataset: str,
    *,
    cutoff_year: int,
    year_weights: dict[int, float] | None = None,
    docs_per_turn: int = 4096,
    shuffle_buffer_size: int = 64,
    seed: int = 42,
    min_int_score: int | None = None,
    min_score: float | None = None,
    sequential: bool = False,
    **_ignored,
) -> CycleFineWebStream:
    """Same mix policy as pretraining (also used for cutoff tokenizer training)."""
    return CycleFineWebStream(
        dumps,
        dataset,
        **_stream_kwargs(
            cutoff_year=cutoff_year,
            year_weights=year_weights,
            docs_per_turn=docs_per_turn,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
            min_int_score=min_int_score,
            min_score=min_score,
            sequential=sequential,
        ),
    )
