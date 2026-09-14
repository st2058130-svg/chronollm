"""FineWeb-Edu streaming helpers for chronologically bounded SN38 pretraining.

Design goals:
- never require a full FineWeb-Edu download;
- cycle cutoff-safe crawl configs one at a time (sb38-style) to avoid OOM from
  keeping many Hub streams / resolve buffers open;
- optional small per-dump shuffle buffer;
- pack documents efficiently with EOS separators;
- save/restore stream + packing state for cheap resume.

This module intentionally does *not* infer temporal safety from page text.  The
caller must provide only crawl configs allowed by the target cutoff year.
"""

from __future__ import annotations

import hashlib
import itertools
import re
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


class CycleFineWebStream:
    """Stateful sequential FineWeb-Edu stream (one crawl open at a time).

    Walks `dumps` in order, then repeats forever — same idea as sb38 `cycle_dumps`,
    but with cutoff checks, optional shuffle, and exact resume via offsets.
    Only one Hub streaming iterator is resident, which avoids the RAM spikes /
    OOM kills from the old weighted multi-dump interleaver.
    """

    def __init__(
        self,
        dumps: list[str],
        dataset: str = "HuggingFaceFW/fineweb-edu",
        *,
        cutoff_year: int,
        shuffle_buffer_size: int = 0,
        seed: int = 42,
    ):
        validate_cutoff_dumps(dumps, cutoff_year)
        if shuffle_buffer_size < 0:
            raise ValueError("shuffle_buffer_size must be >= 0")

        self.dumps = list(dumps)
        self.dataset = dataset
        self.cutoff_year = int(cutoff_year)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.seed = int(seed)

        self._dump_idx = 0
        self._offset = 0  # rows consumed in current dump (incl. empty); for ds.skip
        self._list_cycle = 0  # how many full passes over the dump list
        self._iter: Iterator | None = None

    def __iter__(self):
        return self

    def _current_dump(self) -> str:
        return self.dumps[self._dump_idx]

    def _open_current(self) -> None:
        dump = self._current_dump()
        print(
            f"[data] open {dump} "
            f"(idx={self._dump_idx}/{len(self.dumps)} offset={self._offset:,} "
            f"list_cycle={self._list_cycle})"
        )
        ds = load_dataset(self.dataset, name=dump, split="train", streaming=True)
        if self.shuffle_buffer_size > 1:
            ds = ds.shuffle(
                seed=_stable_seed(self.seed, dump, self._list_cycle),
                buffer_size=self.shuffle_buffer_size,
            )
        if self._offset:
            ds = ds.skip(self._offset)
        self._iter = iter(ds)

    def _advance_dump(self) -> None:
        self._dump_idx = (self._dump_idx + 1) % len(self.dumps)
        if self._dump_idx == 0:
            self._list_cycle += 1
        self._offset = 0
        self._iter = None

    def __next__(self) -> str:
        if self._iter is None:
            self._open_current()

        while True:
            assert self._iter is not None
            try:
                row = next(self._iter)
                self._offset += 1
            except StopIteration:
                self._advance_dump()
                self._open_current()
                continue

            text = row.get("text")
            if text:
                return text

    def state_dict(self) -> dict:
        return {
            "version": "cycle_v1",
            "dump_idx": int(self._dump_idx),
            "offset": int(self._offset),
            "list_cycle": int(self._list_cycle),
        }

    def load_state_dict(self, state: dict) -> None:
        version = state.get("version")
        if version != "cycle_v1" and "dump_idx" not in state:
            raise ValueError(
                "Checkpoint data_state is from the old interleaved stream; "
                "cannot restore cycle stream exactly. Delete data_state resume "
                "or restart from --init-from / scratch."
            )
        dump_idx = int(state.get("dump_idx", 0))
        if not 0 <= dump_idx < len(self.dumps):
            raise ValueError(f"Resume dump_idx={dump_idx} out of range for {len(self.dumps)} dumps")
        self._dump_idx = dump_idx
        self._offset = int(state.get("offset", 0))
        self._list_cycle = int(state.get("list_cycle", 0))
        self._iter = None  # rebuilt with ds.skip(offset) on next __next__


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


def build_packed_stream(
    dumps: list[str],
    dataset: str,
    tokenizer,
    *,
    cutoff_year: int,
    seq_len: int,
    shuffle_buffer_size: int = 0,
    seed: int = 42,
    **_ignored,
) -> PackedCausalStream:
    """Pack a cutoff-safe cycling FineWeb stream into fixed-length blocks.

    Extra kwargs (year_weights, max_open_sources, …) are ignored for backward
    compatibility with older train.py / config keys.
    """
    text_stream = CycleFineWebStream(
        dumps,
        dataset,
        cutoff_year=cutoff_year,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
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
    shuffle_buffer_size: int = 0,
    seed: int = 42,
    **_ignored,
) -> CycleFineWebStream:
    """Build the same cutoff-safe cycling text stream used for pretraining.

    Also used to train a custom tokenizer so tokenizer provenance obeys the
    same temporal cutoff as model training.
    """
    return CycleFineWebStream(
        dumps,
        dataset,
        cutoff_year=cutoff_year,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
    )
