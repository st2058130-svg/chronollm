"""FineWeb-Edu streaming helpers for chronologically bounded SN38 pretraining.

Design goals:
- never require a full FineWeb-Edu download;
- mix years via weights while keeping **only one** crawl iterator open (avoids
  the multi-dump OOM / resolve thrash of the old interleaver);
- stay on a dump for `docs_per_turn` docs before switching;
- optional FineWeb-Edu `int_score` / `score` quality floor;
- optional SkillMixStream upsampling of Stage-2 skill families (heuristic tags);
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


def _row_date_year(row: dict, date_field: str | None) -> int | None:
    """Parse calendar year from FineWeb `date` (or similar) field."""
    if not date_field:
        return None
    raw = row.get(date_field)
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        ts = float(raw)
        if ts > 1e12:
            ts /= 1000.0
        if ts > 1e9:
            from datetime import datetime

            return datetime.utcfromtimestamp(ts).year
        if 1900 <= int(ts) <= 2100:
            return int(ts)
        return None
    text = str(raw).strip()
    if re.fullmatch(r"\d{4}", text):
        return int(text)
    if re.match(r"\d{4}-\d{2}-\d{2}", text):
        return int(text[:4])
    if re.match(r"\d{8}", text):
        return int(text[:4])
    try:
        from datetime import datetime

        return datetime.fromisoformat(text.replace("Z", "+00:00")).year
    except ValueError:
        return None


def _row_passes_date(
    row: dict,
    *,
    date_field: str | None,
    date_year_min: int | None,
    date_year_max: int | None,
    require_date: bool,
) -> bool:
    if date_field is None and date_year_min is None and date_year_max is None:
        return True
    field = date_field or "date"
    year = _row_date_year(row, field)
    if year is None:
        return not require_date
    if date_year_min is not None and year < int(date_year_min):
        return False
    if date_year_max is not None and year > int(date_year_max):
        return False
    return True


# Aligned with sn38.template.quality_prompts.CATEGORIES (skills, not fixed prompts).
SKILL_CATEGORIES = [
    "reading_comprehension",
    "language_understanding",
    "world_knowledge",
    "commonsense_reasoning",
    "language_modeling",
    "causal_reasoning",
    "logical_inference",
    "temporal_reasoning",
    "math_reasoning",
    "truthfulness",
    "pronoun_resolution",
    "paraphrase_detection",
    "word_sense",
]

# Lightweight keyword / phrase heuristics over FineWeb text (+ optional URL).
_CATEGORY_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "reading_comprehension": [
        re.compile(r"\b(according to the|in the passage|the author|paragraph|comprehension)\b", re.I),
        re.compile(r"\b(read the|based on the text|main idea|summar)\b", re.I),
    ],
    "language_understanding": [
        re.compile(r"\b(tone|intent|implied|figurative|metaphor|nuance|means that)\b", re.I),
        re.compile(r"\b(interpretation|suggests that|in other words)\b", re.I),
    ],
    "world_knowledge": [
        re.compile(r"\b(geography|history|biology|chemistry|physics|capital of|photosynthesis)\b", re.I),
        re.compile(r"\b(planet|continent|century|scientist|discovered|encyclopedia)\b", re.I),
        re.compile(r"wikipedia|britannica|edu/", re.I),
    ],
    "commonsense_reasoning": [
        re.compile(r"\b(everyday|common sense|makes sense|you would|in real life)\b", re.I),
        re.compile(r"\b(if you leave|what happens if|because it was too)\b", re.I),
    ],
    "language_modeling": [
        re.compile(r"\b(once upon|the story|chapter|narrator|she walked|he said)\b", re.I),
        re.compile(r"\b(novel|fiction|short story|scene)\b", re.I),
    ],
    "causal_reasoning": [
        re.compile(r"\b(because|therefore|as a result|leads to|caused by|consequently)\b", re.I),
        re.compile(r"\b(feedback|chain of|downstream|upstream|which in turn)\b", re.I),
    ],
    "logical_inference": [
        re.compile(r"\b(therefore|premise|conclusion|if and only if|all .* are|syllogism)\b", re.I),
        re.compile(r"\b(implies|deduce|logically|cannot conclude)\b", re.I),
    ],
    "temporal_reasoning": [
        re.compile(r"\b(first|then|after that|before|finally|sequence|timeline|next step)\b", re.I),
        re.compile(r"\b(previously|afterwards|in order|step \d)\b", re.I),
    ],
    "math_reasoning": [
        re.compile(r"\b(equation|fraction|percent|algebra|geometry|calculate|solve for)\b", re.I),
        re.compile(r"\b(\d+\s*[+\-*/×÷=]\s*\d+|square root|proportion|ratio)\b", re.I),
        re.compile(r"\b(math|mathematics|word problem)\b", re.I),
    ],
    "truthfulness": [
        re.compile(r"\b(myth|misconception|false claim|is it true|debunk|actually not)\b", re.I),
        re.compile(r"\b(contrary to popular|fact check|urban legend)\b", re.I),
    ],
    "pronoun_resolution": [
        re.compile(r"\b(he|she|they|it|them|his|her|their)\b.*\b(he|she|they|it|them)\b", re.I),
        re.compile(r"\b(who does|refers to|the pronoun|ambiguous)\b", re.I),
    ],
    "paraphrase_detection": [
        re.compile(r"\b(in other words|that is to say|equivalently|same meaning|paraphrase)\b", re.I),
        re.compile(r"\b(restated|rewritten|means the same)\b", re.I),
    ],
    "word_sense": [
        re.compile(r"\b(means|sense of|ambiguous|homonym|definition|polysem)\b", re.I),
        re.compile(r"\b(depending on context|could mean|word sense)\b", re.I),
    ],
}


def classify_skill(text: str, url: str = "") -> str | None:
    """Heuristic skill label for a FineWeb doc, or None if no category matches."""
    if not text:
        return None
    blob = f"{text[:5000]}\n{url or ''}"
    scores = {
        cat: sum(1 for pat in patterns if pat.search(blob))
        for cat, patterns in _CATEGORY_PATTERNS.items()
    }
    best_cat, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score <= 0:
        return None
    return best_cat


def _normalize_category_weights(
    category_weights: dict[str, float] | None,
) -> tuple[list[str], list[float]]:
    cats = list(SKILL_CATEGORIES)
    if not category_weights:
        return cats, [1.0] * len(cats)
    weights = []
    for c in cats:
        weights.append(float(category_weights.get(c, 0.0)))
    if sum(weights) <= 0:
        return cats, [1.0] * len(cats)
    return cats, weights


class SkillMixStream:
    """Mix general FineWeb docs with skill-upsampled docs (rejection sampling).

    Still uses a single underlying crawl stream (OOM-safe). With probability
    `skill_fraction`, hunt for a document matching a sampled quality category
    (aligned with validator Stage-2 skill families). Otherwise emit the next
    general document unchanged.
    """

    def __init__(
        self,
        base: CycleFineWebStream,
        *,
        skill_fraction: float = 0.2,
        category_weights: dict[str, float] | None = None,
        max_skips: int = 64,
        accept_any_skill_fallback: bool = True,
        min_chars: int = 200,
        seed: int = 42,
    ):
        if not 0.0 <= skill_fraction <= 1.0:
            raise ValueError("skill_fraction must be in [0, 1]")
        if max_skips < 1:
            raise ValueError("max_skips must be >= 1")

        self.base = base
        self.skill_fraction = float(skill_fraction)
        self.max_skips = int(max_skips)
        self.accept_any_skill_fallback = bool(accept_any_skill_fallback)
        self.min_chars = int(min_chars)
        self._rng = random.Random(int(seed) + 90_017)
        self._categories, self._cat_weights = _normalize_category_weights(category_weights)
        self._skill_hits = 0
        self._skill_attempts = 0
        self._general_hits = 0

        print(
            f"[data] skill_mix enabled fraction={self.skill_fraction:.2f} "
            f"max_skips={self.max_skips} categories={len(self._categories)}"
        )

    def __iter__(self):
        return self

    def _pick_category(self) -> str:
        return self._rng.choices(self._categories, weights=self._cat_weights, k=1)[0]

    def __next__(self) -> str:
        if self._rng.random() >= self.skill_fraction:
            self._general_hits += 1
            return next(self.base)

        self._skill_attempts += 1
        target = self._pick_category()
        fallback_any: str | None = None
        last = ""
        for _ in range(self.max_skips):
            text = next(self.base)
            last = text
            if len(text) < self.min_chars:
                continue
            cat = classify_skill(text)
            if cat == target:
                self._skill_hits += 1
                return text
            if (
                self.accept_any_skill_fallback
                and fallback_any is None
                and cat is not None
            ):
                fallback_any = text

        self._skill_hits += 1
        return fallback_any if fallback_any is not None else last

    def state_dict(self) -> dict:
        return {
            "version": "skill_mix_v1",
            "rng_state": self._rng.getstate(),
            "base": self.base.state_dict(),
            "skill_hits": self._skill_hits,
            "skill_attempts": self._skill_attempts,
            "general_hits": self._general_hits,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("version") not in {None, "skill_mix_v1"} and "base" not in state:
            # Plain cycle checkpoint under a skill wrapper: restore base only.
            self.base.load_state_dict(state)
            return
        if "rng_state" in state:
            self._rng.setstate(state["rng_state"])
        base_state = state.get("base", state)
        self.base.load_state_dict(base_state)
        self._skill_hits = int(state.get("skill_hits", 0))
        self._skill_attempts = int(state.get("skill_attempts", 0))
        self._general_hits = int(state.get("general_hits", 0))


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
        date_field: str | None = None,
        date_year_min: int | None = None,
        date_year_max: int | None = None,
        require_date: bool = False,
    ):
        validate_cutoff_dumps(dumps, cutoff_year)
        if docs_per_turn <= 0:
            raise ValueError("docs_per_turn must be > 0")
        if shuffle_buffer_size < 0:
            raise ValueError("shuffle_buffer_size must be >= 0")
        if date_year_max is not None and int(date_year_max) > int(cutoff_year):
            raise ValueError(
                f"date_year_max={date_year_max} > cutoff_year={cutoff_year} (leak risk)"
            )
        if (
            date_year_min is not None
            and date_year_max is not None
            and int(date_year_min) > int(date_year_max)
        ):
            raise ValueError("date_year_min must be <= date_year_max")

        self.dumps = list(dumps)
        self.dataset = dataset
        self.cutoff_year = int(cutoff_year)
        self.docs_per_turn = int(docs_per_turn)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.seed = int(seed)
        self.min_int_score = None if min_int_score is None else int(min_int_score)
        self.min_score = None if min_score is None else float(min_score)
        self.sequential = bool(sequential)
        self.date_field = date_field
        self.date_year_min = None if date_year_min is None else int(date_year_min)
        self.date_year_max = None if date_year_max is None else int(date_year_max)
        self.require_date = bool(require_date)
        if (self.date_year_min is not None or self.date_year_max is not None) and not self.date_field:
            self.date_field = "date"

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
        if self.date_field and (self.date_year_min is not None or self.date_year_max is not None):
            print(
                f"[data] date_filter field={self.date_field} "
                f"year=[{self.date_year_min},{self.date_year_max}] "
                f"require_date={self.require_date}"
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
            if not _row_passes_date(
                row,
                date_field=self.date_field,
                date_year_min=self.date_year_min,
                date_year_max=self.date_year_max,
                require_date=self.require_date,
            ):
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
    date_field: str | None = None,
    date_year_min: int | None = None,
    date_year_max: int | None = None,
    require_date: bool = False,
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
        date_field=date_field,
        date_year_min=date_year_min,
        date_year_max=date_year_max,
        require_date=require_date,
    )


def _maybe_wrap_skill_mix(
    base,
    *,
    seed: int,
    skill_mix: dict | None,
):
    """Wrap any text stream that implements __next__ / optional state_dict."""
    if not skill_mix or not skill_mix.get("enabled"):
        return base
    cat_w = skill_mix.get("category_weights") or None
    if cat_w is not None:
        cat_w = {str(k): float(v) for k, v in cat_w.items()}
    return SkillMixStream(
        base,
        skill_fraction=float(skill_mix.get("skill_fraction", 0.2)),
        category_weights=cat_w,
        max_skips=int(skill_mix.get("max_skips", 64)),
        accept_any_skill_fallback=bool(skill_mix.get("accept_any_skill_fallback", True)),
        min_chars=int(skill_mix.get("min_chars", 200)),
        seed=seed,
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
    skill_mix: dict | None = None,
    date_field: str | None = None,
    date_year_min: int | None = None,
    date_year_max: int | None = None,
    require_date: bool = False,
    **_ignored,
) -> PackedCausalStream:
    text_stream: CycleFineWebStream | SkillMixStream = CycleFineWebStream(
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
            date_field=date_field,
            date_year_min=date_year_min,
            date_year_max=date_year_max,
            require_date=require_date,
        ),
    )
    text_stream = _maybe_wrap_skill_mix(text_stream, seed=seed, skill_mix=skill_mix)
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
    skill_mix: dict | None = None,
    date_field: str | None = None,
    date_year_min: int | None = None,
    date_year_max: int | None = None,
    require_date: bool = False,
    **_ignored,
) -> CycleFineWebStream | SkillMixStream:
    """Same mix policy as pretraining (also used for cutoff tokenizer training)."""
    text_stream: CycleFineWebStream | SkillMixStream = CycleFineWebStream(
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
            date_field=date_field,
            date_year_min=date_year_min,
            date_year_max=date_year_max,
            require_date=require_date,
        ),
    )
    return _maybe_wrap_skill_mix(text_stream, seed=seed, skill_mix=skill_mix)
