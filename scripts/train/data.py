"""FineWeb-Edu streaming helpers for chronologically bounded SN38 pretraining.

Design goals:
- never require a full FineWeb-Edu download;
- interleave cutoff-safe crawl configs from the beginning of training;
- make the historical mixture explicit through year weights;
- use deterministic bounded shuffle inside each crawl;
- pack documents efficiently with EOS separators;
- save/restore exact stream + packing state for cheap resume.

This module intentionally does *not* infer temporal safety from page text.  The
caller must provide only crawl configs allowed by the target cutoff year.
"""

from __future__ import annotations

import hashlib
import itertools
import random
import re
from collections import Counter, OrderedDict
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
    """Split each year's requested mass equally across that year's crawl configs."""
    counts = Counter(dump_year(d) for d in dumps)
    if year_weights is None:
        # Equal probability per configured source if no policy is supplied.
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


class InterleavedFineWebStream:
    """Stateful, deterministic weighted stream across FineWeb-Edu crawl configs.

    A source is selected according to explicit year weights, then `docs_per_turn`
    non-empty documents are yielded from that source before another source is
    selected.  Each source is independently shuffled with a bounded streaming
    buffer.  The state stores source offsets plus scheduler RNG state, so resume
    reconstructs the current location instead of replaying the entire mixed
    training stream.

    `max_open_sources` caps how many crawl iterators stay resident. When the cap
    is reached, source picks stay inside the open set most of the time so we do
    not thrash Hub "Resolving data files" + expensive ds.skip(offset) rebuilds.
    Occasional explores (`open_explore_prob`) rotate in a new crawl.
    """

    def __init__(
        self,
        dumps: list[str],
        dataset: str = "HuggingFaceFW/fineweb-edu",
        *,
        cutoff_year: int,
        year_weights: dict[int, float] | None = None,
        docs_per_turn: int = 32,
        shuffle_buffer_size: int = 1024,
        seed: int = 42,
        max_open_sources: int | None = None,
        open_explore_prob: float = 0.02,
    ):
        validate_cutoff_dumps(dumps, cutoff_year)
        if docs_per_turn <= 0:
            raise ValueError("docs_per_turn must be > 0")
        if shuffle_buffer_size < 0:
            raise ValueError("shuffle_buffer_size must be >= 0")
        if max_open_sources is not None and max_open_sources < 1:
            raise ValueError("max_open_sources must be >= 1 when set")
        if not 0.0 <= open_explore_prob <= 1.0:
            raise ValueError("open_explore_prob must be in [0, 1]")

        self.dumps = list(dumps)
        self.dataset = dataset
        self.cutoff_year = int(cutoff_year)
        self.docs_per_turn = int(docs_per_turn)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.seed = int(seed)
        self.max_open_sources = (
            None if max_open_sources is None else int(max_open_sources)
        )
        self.open_explore_prob = float(open_explore_prob)
        self.source_weights = _normalized_source_weights(self.dumps, year_weights)
        self._weight_by_dump = dict(zip(self.dumps, self.source_weights))

        self._rng = random.Random(self.seed)
        self._offsets = {d: 0 for d in self.dumps}  # shuffled rows consumed per source
        self._cycles = {d: 0 for d in self.dumps}
        self._iters: OrderedDict[str, Iterator] = OrderedDict()
        self._active_dump: str | None = None
        self._remaining_in_turn = 0

    def __iter__(self):
        return self

    def _make_source_iter(self, dump: str):
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

    def _evict_one(self) -> None:
        """Drop the cheapest-to-rebuild open source (smallest skip offset)."""
        if not self._iters:
            return
        victim = min(self._iters.keys(), key=lambda d: self._offsets[d])
        del self._iters[victim]

    def _get_source_iter(self, dump: str) -> Iterator:
        if dump in self._iters:
            self._iters.move_to_end(dump)
            return self._iters[dump]

        if self.max_open_sources is not None:
            while len(self._iters) >= self.max_open_sources:
                self._evict_one()

        source_iter = self._make_source_iter(dump)
        self._iters[dump] = source_iter
        return source_iter

    def _pick_from(self, candidates: list[str]) -> str:
        weights = [self._weight_by_dump[d] for d in candidates]
        return self._rng.choices(candidates, weights=weights, k=1)[0]

    def _pick_source(self) -> str:
        # Fill the open set first, then stay sticky to avoid resolve/skip thrash.
        if self.max_open_sources is None or len(self._iters) < self.max_open_sources:
            return self._pick_from(self.dumps)

        open_dumps = list(self._iters.keys())
        if open_dumps and self._rng.random() >= self.open_explore_prob:
            return self._pick_from(open_dumps)
        return self._pick_from(self.dumps)

    def __next__(self) -> str:
        while True:
            if self._active_dump is None or self._remaining_in_turn <= 0:
                self._active_dump = self._pick_source()
                self._remaining_in_turn = self.docs_per_turn

            dump = self._active_dump
            source_iter = self._get_source_iter(dump)

            try:
                row = next(source_iter)
                # Offset counts rows emitted by the deterministic shuffled stream,
                # including empty rows.  This is what ds.skip(offset) must restore.
                self._offsets[dump] += 1
            except StopIteration:
                # Extremely long real runs may exhaust a source. Start a new,
                # deterministically re-shuffled cycle of that source.
                self._cycles[dump] += 1
                self._offsets[dump] = 0
                self._iters.pop(dump, None)
                self._iters[dump] = self._make_source_iter(dump)
                continue

            text = row.get("text")
            if not text:
                continue

            self._remaining_in_turn -= 1
            return text

    def state_dict(self) -> dict:
        return {
            "rng_state": self._rng.getstate(),
            "offsets": dict(self._offsets),
            "cycles": dict(self._cycles),
            "active_dump": self._active_dump,
            "remaining_in_turn": self._remaining_in_turn,
        }

    def load_state_dict(self, state: dict) -> None:
        self._rng.setstate(state["rng_state"])
        self._offsets = {d: int(state.get("offsets", {}).get(d, 0)) for d in self.dumps}
        self._cycles = {d: int(state.get("cycles", {}).get(d, 0)) for d in self.dumps}
        active = state.get("active_dump")
        if active is not None and active not in self.dumps:
            raise ValueError(f"Resume state references unknown dump: {active}")
        self._active_dump = active
        self._remaining_in_turn = int(state.get("remaining_in_turn", 0))
        self._iters.clear()  # lazily rebuilt with ds.skip(offset)


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
    year_weights: dict[int, float] | None,
    docs_per_turn: int,
    shuffle_buffer_size: int,
    seed: int,
    max_open_sources: int | None = None,
    open_explore_prob: float = 0.02,
) -> PackedCausalStream:
    text_stream = InterleavedFineWebStream(
        dumps,
        dataset,
        cutoff_year=cutoff_year,
        year_weights=year_weights,
        docs_per_turn=docs_per_turn,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
        max_open_sources=max_open_sources,
        open_explore_prob=open_explore_prob,
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
    year_weights: dict[int, float] | None,
    docs_per_turn: int,
    shuffle_buffer_size: int,
    seed: int,
    max_open_sources: int | None = None,
    open_explore_prob: float = 0.02,
) -> InterleavedFineWebStream:
    """Build the same cutoff-safe weighted text stream used for pretraining.

    This is also used to train a custom tokenizer, ensuring tokenizer provenance
    obeys the same temporal cutoff and source-mixture policy as model training.
    """
    return InterleavedFineWebStream(
        dumps,
        dataset,
        cutoff_year=cutoff_year,
        year_weights=year_weights,
        docs_per_turn=docs_per_turn,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
        max_open_sources=max_open_sources,
        open_explore_prob=open_explore_prob,
    )
