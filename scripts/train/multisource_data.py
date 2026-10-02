"""Multi-source cutoff-safe streams for ChronoLLM (FineWeb + Wikipedia + HF / local text).

Uses official HuggingFace / Wikimedia dumps, plus optional local .txt / .jsonl
fact packs (file or directory). Each source must be dated for the model cutoff
year (crawl year, wiki snapshot YYYYMMDD, per-row date field, or explicit
snapshot_year for local files/dirs).
"""

from __future__ import annotations

import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from datasets import load_dataset

from scripts.train.data import (
    CycleFineWebStream,
    PackedCausalStream,
    SkillMixStream,
    _maybe_wrap_skill_mix,
    validate_cutoff_dumps,
)

_WIKI_SNAP_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})\.")


def wiki_snapshot_year(config_name: str) -> int:
    """Parse year from configs like 20220301.en."""
    match = _WIKI_SNAP_RE.match(config_name)
    if match is None:
        raise ValueError(
            f"Wikipedia config must look like YYYYMMDD.lang (got {config_name!r})"
        )
    return int(match.group(1))


def validate_wiki_snapshot(config_name: str, cutoff_year: int) -> None:
    year = wiki_snapshot_year(config_name)
    if year > cutoff_year:
        raise ValueError(
            f"Wikipedia snapshot {config_name!r} is year {year} > cutoff {cutoff_year}"
        )


def _row_date_year(row: dict, date_field: str | None) -> int | None:
    if not date_field:
        return None
    raw = row.get(date_field)
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        # Unix seconds or milliseconds
        ts = float(raw)
        if ts > 1e12:
            ts /= 1000.0
        if ts > 1e9:
            return datetime.utcfromtimestamp(ts).year
        # Plain year
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
        return datetime.fromisoformat(text.replace("Z", "+00:00")).year
    except ValueError:
        return None


class WikipediaStream:
    """Stream English Wikipedia articles from a dated HF snapshot (cutoff-safe)."""

    def __init__(
        self,
        *,
        dataset: str = "wikimedia/wikipedia",
        config_name: str = "20220301.en",
        cutoff_year: int,
        text_field: str = "text",
        min_chars: int = 200,
        seed: int = 42,
        start_offset: int = 0,
    ):
        validate_wiki_snapshot(config_name, cutoff_year)
        self.dataset = dataset
        self.config_name = config_name
        self.cutoff_year = int(cutoff_year)
        self.text_field = text_field
        self.min_chars = int(min_chars)
        self.seed = int(seed)
        self._cycle = 0
        self._offset = max(0, int(start_offset))
        self._iter: Iterator | None = None
        print(
            f"[data] wikipedia dataset={dataset} config={config_name} "
            f"snapshot_year={wiki_snapshot_year(config_name)} cutoff={cutoff_year} "
            f"start_offset={self._offset}"
        )

    def __iter__(self):
        return self

    def _open(self) -> None:
        print(f"[data] open wikipedia {self.config_name} (offset={self._offset} cycle={self._cycle})")
        try:
            ds = load_dataset(
                self.dataset,
                self.config_name,
                split="train",
                streaming=True,
            )
        except ValueError as exc:
            raise ValueError(
                f"Wikipedia config {self.config_name!r} unavailable on {self.dataset!r}. "
                f"For cutoff year {self.cutoff_year}, do NOT use 20231101.* "
                f"(post-cutoff leak). Prefer type=hf_text with a dated 2022 parquet "
                f"mirror (e.g. tinhpx2911/wikipedia_20220620_cleaned, snapshot_year=2022). "
                f"Original error: {exc}"
            ) from exc
        if self._offset:
            ds = ds.skip(self._offset)
        self._iter = iter(ds)

    def __next__(self) -> str:
        while True:
            if self._iter is None:
                self._open()
            assert self._iter is not None
            try:
                row = next(self._iter)
                self._offset += 1
            except StopIteration:
                self._cycle += 1
                self._offset = 0
                self._iter = None
                continue
            text = row.get(self.text_field) or ""
            if isinstance(text, str) and len(text) >= self.min_chars:
                return text

    def state_dict(self) -> dict:
        return {
            "version": "wiki_v1",
            "cycle": self._cycle,
            "offset": self._offset,
            "config_name": self.config_name,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("config_name") not in {None, self.config_name}:
            raise ValueError("Wikipedia stream config_name mismatch")
        self._cycle = int(state.get("cycle", 0))
        self._offset = int(state.get("offset", 0))
        self._iter = None


class HFTextStream:
    """Generic HF text stream with optional per-row date filter for cutoff safety."""

    def __init__(
        self,
        *,
        dataset: str,
        cutoff_year: int,
        name: str | None = None,
        split: str = "train",
        text_field: str = "text",
        date_field: str | None = None,
        require_date: bool = False,
        snapshot_year: int | None = None,
        min_year: int | None = None,
        min_chars: int = 200,
        seed: int = 42,
        start_offset: int = 0,
    ):
        if snapshot_year is not None and int(snapshot_year) > int(cutoff_year):
            raise ValueError(
                f"hf_text snapshot_year={snapshot_year} > cutoff {cutoff_year}"
            )
        if min_year is not None and int(min_year) > int(cutoff_year):
            raise ValueError(f"hf_text min_year={min_year} > cutoff {cutoff_year}")
        if date_field is None and snapshot_year is None and require_date:
            raise ValueError("hf_text require_date=true needs date_field or snapshot_year")
        if min_year is not None and date_field is None and snapshot_year is None:
            raise ValueError("hf_text min_year requires date_field (or a dated snapshot_year-only dump)")
        self.dataset = dataset
        self.name = name
        self.split = split
        self.cutoff_year = int(cutoff_year)
        self.text_field = text_field
        self.date_field = date_field
        self.require_date = bool(require_date)
        self.snapshot_year = None if snapshot_year is None else int(snapshot_year)
        self.min_year = None if min_year is None else int(min_year)
        self.min_chars = int(min_chars)
        self.seed = int(seed)
        self._cycle = 0
        self._offset = max(0, int(start_offset))
        self._iter: Iterator | None = None
        print(
            f"[data] hf_text dataset={dataset} name={name} "
            f"date_field={date_field} min_year={min_year} "
            f"snapshot_year={snapshot_year} cutoff={cutoff_year} "
            f"start_offset={self._offset}"
        )

    def __iter__(self):
        return self

    def _open(self) -> None:
        label = self.name or "default"
        print(f"[data] open hf_text {self.dataset}:{label} (offset={self._offset} cycle={self._cycle})")
        kwargs = {"split": self.split, "streaming": True}
        if self.name:
            ds = load_dataset(self.dataset, self.name, **kwargs)
        else:
            ds = load_dataset(self.dataset, **kwargs)
        if self._offset:
            ds = ds.skip(self._offset)
        self._iter = iter(ds)

    def _keep_row(self, row: dict) -> bool:
        # Dated whole-dump (e.g. GDELT split=2023): keep all rows.
        if self.snapshot_year is not None and self.date_field is None:
            return True
        year = _row_date_year(row, self.date_field)
        if year is None:
            if self.require_date or self.min_year is not None:
                return False
            return self.date_field is None
        if year > self.cutoff_year:
            return False
        if self.min_year is not None and year < self.min_year:
            return False
        return True

    def __next__(self) -> str:
        while True:
            if self._iter is None:
                self._open()
            assert self._iter is not None
            try:
                row = next(self._iter)
                self._offset += 1
            except StopIteration:
                self._cycle += 1
                self._offset = 0
                self._iter = None
                continue
            if not self._keep_row(row):
                continue
            text = row.get(self.text_field) or ""
            if isinstance(text, str) and len(text) >= self.min_chars:
                return text

    def state_dict(self) -> dict:
        return {
            "version": "hf_text_v2",
            "cycle": self._cycle,
            "offset": self._offset,
            "dataset": self.dataset,
            "name": self.name,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("dataset") not in {None, self.dataset}:
            raise ValueError("hf_text dataset mismatch")
        if state.get("name") not in {None, self.name}:
            raise ValueError("hf_text name mismatch")
        self._cycle = int(state.get("cycle", 0))
        self._offset = int(state.get("offset", 0))
        self._iter = None


class LocalTextStream:
    """Cycle paragraphs from a local .txt / .jsonl file, or a directory of them.

    Directory mode recursively loads ``*.txt``, ``*.md``, ``*.jsonl``, ``*.json``.
    JSONL/JSON rows use text-like fields when present (text, content, abstract, …).
    """

    _TEXT_SUFFIXES = {".txt", ".md", ".jsonl", ".json"}
    _JSON_TEXT_KEYS = (
        "text",
        "content",
        "body",
        "abstract",
        "title",
        "summary",
        "passage",
        "document",
    )

    def __init__(
        self,
        *,
        path: str | Path,
        cutoff_year: int,
        snapshot_year: int | None = None,
        min_chars: int = 80,
        seed: int = 42,
        start_offset: int = 0,
    ):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"local_text path not found: {self.path.resolve()}")
        snap = int(cutoff_year if snapshot_year is None else snapshot_year)
        if snap > int(cutoff_year):
            raise ValueError(
                f"local_text {self.path} snapshot_year={snap} > cutoff {cutoff_year}"
            )
        self.cutoff_year = int(cutoff_year)
        self.snapshot_year = snap
        self.min_chars = int(min_chars)
        self.seed = int(seed)
        docs = self._load_docs(self.path)
        if not docs:
            raise ValueError(
                f"local_text {self.path} has no usable chunks (min_chars={self.min_chars})"
            )
        self.docs = docs
        self._cycle = 0
        # Local packs are tiny and cycle; offset is modulo length.
        self._offset = max(0, int(start_offset)) % len(self.docs)
        kind = "dir" if self.path.is_dir() else "file"
        print(
            f"[data] local_text {kind}={self.path.resolve()} docs={len(self.docs)} "
            f"snapshot_year={self.snapshot_year} cutoff={self.cutoff_year} "
            f"start_offset={self._offset}"
        )

    def _load_docs(self, path: Path) -> list[str]:
        if path.is_file():
            return self._load_file(path)
        files = sorted(
            p
            for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in self._TEXT_SUFFIXES
        )
        if not files:
            raise FileNotFoundError(
                f"local_text directory has no .txt/.md/.jsonl/.json files: {path.resolve()}"
            )
        docs: list[str] = []
        for fp in files:
            docs.extend(self._load_file(fp))
        return docs

    def _load_file(self, path: Path) -> list[str]:
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".json"}:
            return self._load_jsonish(path)
        raw = path.read_text(encoding="utf-8", errors="ignore")
        chunks = re.split(r"\n\s*\n+", raw)
        docs = [c.strip() for c in chunks if len(c.strip()) >= self.min_chars]
        if not docs:
            docs = [ln.strip() for ln in raw.splitlines() if len(ln.strip()) >= self.min_chars]
        return docs

    def _extract_json_text(self, obj: object) -> str | None:
        if isinstance(obj, str):
            return obj.strip() or None
        if not isinstance(obj, dict):
            return None
        for key in self._JSON_TEXT_KEYS:
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        # Common nested shapes: {"title": "...", "abstract": "..."}
        parts = [
            str(obj[k]).strip()
            for k in ("title", "abstract", "text", "content")
            if isinstance(obj.get(k), str) and str(obj.get(k)).strip()
        ]
        if parts:
            return "\n\n".join(parts)
        return None

    def _load_jsonish(self, path: Path) -> list[str]:
        import json

        docs: list[str] = []
        raw = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not raw:
            return docs
        # Prefer JSONL (one object per line); fall back to a JSON array/object.
        if "\n" in raw or path.suffix.lower() == ".jsonl":
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    if len(line) >= self.min_chars:
                        docs.append(line)
                    continue
                text = self._extract_json_text(obj)
                if text and len(text) >= self.min_chars:
                    docs.append(text)
            if docs:
                return docs
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return docs
        if isinstance(payload, list):
            for obj in payload:
                text = self._extract_json_text(obj)
                if text and len(text) >= self.min_chars:
                    docs.append(text)
        else:
            text = self._extract_json_text(payload)
            if text and len(text) >= self.min_chars:
                docs.append(text)
        return docs

    def __iter__(self):
        return self

    def __next__(self) -> str:
        if self._offset >= len(self.docs):
            self._cycle += 1
            self._offset = 0
        text = self.docs[self._offset]
        self._offset += 1
        return text

    def state_dict(self) -> dict:
        return {
            "version": "local_text_v2",
            "cycle": self._cycle,
            "offset": self._offset,
            "path": str(self.path),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("path") not in {None, str(self.path)}:
            raise ValueError("local_text path mismatch")
        self._cycle = int(state.get("cycle", 0))
        self._offset = int(state.get("offset", 0))


class WeightedMultiSourceStream:
    """Mix sources by weight, or cycle them in config list order when sequential."""

    def __init__(
        self,
        sources: list[tuple[str, Any, float]],
        *,
        docs_per_turn: int = 4096,
        seed: int = 42,
        sequential: bool = False,
    ):
        if not sources:
            raise ValueError("At least one source is required")
        if docs_per_turn <= 0:
            raise ValueError("docs_per_turn must be > 0")
        self.names = [n for n, _, _ in sources]
        self.streams = [s for _, s, _ in sources]
        weights = [float(w) for _, _, w in sources]
        total = sum(weights)
        if total <= 0:
            raise ValueError("source weights must sum to > 0")
        self.weights = [w / total for w in weights]
        self.docs_per_turn = int(docs_per_turn)
        self.sequential = bool(sequential)
        self._rng = random.Random(int(seed) + 4242)
        self._active_idx: int | None = None
        self._remaining = 0
        self._seq_idx = 0
        pretty = ", ".join(f"{n}:{w:.2f}" for n, w in zip(self.names, self.weights))
        mode = "sequential" if self.sequential else "weighted"
        print(
            f"[data] multisource={mode} {{ {pretty} }} "
            f"docs_per_turn={self.docs_per_turn}"
        )

    def __iter__(self):
        return self

    def _start_turn(self) -> None:
        if self.sequential:
            self._active_idx = self._seq_idx % len(self.streams)
            self._seq_idx += 1
        else:
            self._active_idx = self._rng.choices(
                range(len(self.streams)), weights=self.weights, k=1
            )[0]
        self._remaining = self.docs_per_turn
        print(f"[data] multisource turn -> {self.names[self._active_idx]}")

    def __next__(self) -> str:
        if self._active_idx is None or self._remaining <= 0:
            self._start_turn()
        assert self._active_idx is not None
        text = next(self.streams[self._active_idx])
        self._remaining -= 1
        return text

    def state_dict(self) -> dict:
        return {
            "version": "multisource_v1",
            "rng_state": self._rng.getstate(),
            "active_idx": self._active_idx,
            "remaining": self._remaining,
            "seq_idx": self._seq_idx,
            "sequential": self.sequential,
            "names": list(self.names),
            "streams": [
                s.state_dict() if hasattr(s, "state_dict") else None for s in self.streams
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("version") not in {None, "multisource_v1"}:
            raise ValueError(f"Unsupported multisource state version: {state.get('version')}")
        if state.get("names") not in {None, self.names} and state.get("names") != self.names:
            raise ValueError("multisource source name list mismatch")
        self._rng.setstate(state["rng_state"])
        self._active_idx = state.get("active_idx")
        self._remaining = int(state.get("remaining", 0))
        self._seq_idx = int(state.get("seq_idx", 0))
        saved = state.get("streams") or []
        for stream, sub in zip(self.streams, saved):
            if sub is not None and hasattr(stream, "load_state_dict"):
                stream.load_state_dict(sub)


def _build_one_source(spec: dict, *, cutoff_year: int, seed: int) -> tuple[str, Any, float]:
    stype = str(spec.get("type", "")).strip().lower()
    name = str(spec.get("name", stype))
    weight = float(spec.get("weight", 1.0))
    if weight <= 0:
        raise ValueError(f"source {name!r} weight must be > 0")

    if stype in {"fineweb_edu", "fineweb"}:
        dumps = list(spec.get("dumps") or [])
        validate_cutoff_dumps(dumps, cutoff_year)
        yw = spec.get("year_weights") or {}
        year_weights = {int(k): float(v) for k, v in yw.items()} or None
        min_int = spec.get("min_int_score", None)
        min_score = spec.get("min_score", None)
        stream = CycleFineWebStream(
            dumps,
            str(spec.get("dataset", "HuggingFaceFW/fineweb-edu")),
            cutoff_year=cutoff_year,
            year_weights=year_weights,
            docs_per_turn=int(spec.get("docs_per_turn", 65536)),
            shuffle_buffer_size=int(spec.get("shuffle_buffer_size", 64)),
            seed=seed + int(spec.get("seed_offset", 0)),
            min_int_score=int(min_int) if min_int is not None else None,
            min_score=float(min_score) if min_score is not None else None,
            sequential=bool(spec.get("sequential", False)),
        )
        return name, stream, weight

    if stype == "wikipedia":
        stream = WikipediaStream(
            dataset=str(spec.get("dataset", "wikimedia/wikipedia")),
            config_name=str(spec.get("config_name", "20220301.en")),
            cutoff_year=cutoff_year,
            text_field=str(spec.get("text_field", "text")),
            min_chars=int(spec.get("min_chars", 200)),
            seed=seed + int(spec.get("seed_offset", 11)),
            start_offset=int(spec.get("start_offset", 0)),
        )
        return name, stream, weight

    if stype == "hf_text":
        stream = HFTextStream(
            dataset=str(spec["dataset"]),
            cutoff_year=cutoff_year,
            name=spec.get("config_name") or spec.get("subset"),
            split=str(spec.get("split", "train")),
            text_field=str(spec.get("text_field", "text")),
            date_field=spec.get("date_field"),
            require_date=bool(spec.get("require_date", False)),
            snapshot_year=spec.get("snapshot_year"),
            min_year=spec.get("min_year"),
            min_chars=int(spec.get("min_chars", 200)),
            seed=seed + int(spec.get("seed_offset", 22)),
            start_offset=int(spec.get("start_offset", 0)),
        )
        return name, stream, weight

    if stype == "local_text":
        stream = LocalTextStream(
            path=str(spec["path"]),
            cutoff_year=cutoff_year,
            snapshot_year=spec.get("snapshot_year"),
            min_chars=int(spec.get("min_chars", 80)),
            seed=seed + int(spec.get("seed_offset", 33)),
            start_offset=int(spec.get("start_offset", 0)),
        )
        return name, stream, weight

    raise ValueError(
        f"Unknown source type {stype!r}; use fineweb_edu, wikipedia, hf_text, or local_text"
    )


def build_multisource_text_stream(
    cfg: dict,
    *,
    cutoff_year: int,
    seed: int,
    skill_mix: dict | None = None,
) -> WeightedMultiSourceStream | SkillMixStream:
    specs = list(cfg.get("sources") or [])
    if not specs:
        raise ValueError("config.sources must be a non-empty list")
    built = [
        _build_one_source(spec, cutoff_year=cutoff_year, seed=seed) for spec in specs
    ]
    docs_per_turn = int(cfg.get("data", {}).get("multisource_docs_per_turn", 8192))
    sequential = bool(cfg.get("data", {}).get("multisource_sequential", False))
    mixer = WeightedMultiSourceStream(
        built,
        docs_per_turn=docs_per_turn,
        seed=seed,
        sequential=sequential,
    )
    if skill_mix and skill_mix.get("enabled"):
        # SkillMixStream duck-types any __next__ text stream.
        return _maybe_wrap_skill_mix(mixer, seed=seed, skill_mix=skill_mix)  # type: ignore[arg-type]
    return mixer


def build_multisource_packed_stream(
    cfg: dict,
    tokenizer,
    *,
    seq_len: int,
    seed: int,
    skill_mix: dict | None = None,
) -> PackedCausalStream:
    text = build_multisource_text_stream(
        cfg,
        cutoff_year=int(cfg["year"]),
        seed=seed,
        skill_mix=skill_mix,
    )
    return PackedCausalStream(text, tokenizer, seq_len)
