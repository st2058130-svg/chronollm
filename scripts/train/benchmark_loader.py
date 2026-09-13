"""Load SN38 Stage-1 benchmarks without wallet or TEE authentication.

Official validator probes live on the private TEE-protected API. For local
mining workflows this module either:

1. Builds benchmarks from the bundled chronological fact bank + live public
   /config thresholds (default), or
2. Loads pre-exported benchmark JSON from --benchmark-dir.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_FACTS_PATH = Path(__file__).resolve().parent / "benchmark_data" / "facts.json"


def fetch_public_config(backend_url: str) -> dict:
    import requests

    resp = requests.get(f"{backend_url.rstrip('/')}/config", timeout=60)
    resp.raise_for_status()
    return resp.json()


def fetch_public_years(backend_url: str, round_num: int | None = None) -> list[int]:
    import requests

    params = {"round_num": round_num} if round_num is not None else {}
    resp = requests.get(f"{backend_url.rstrip('/')}/years", params=params, timeout=60)
    resp.raise_for_status()
    years = resp.json().get("years", [])
    if not years:
        raise RuntimeError("No evaluation years returned by backend")
    return years


def load_facts(path: Path | None = None) -> list[dict]:
    facts_path = path or DEFAULT_FACTS_PATH
    if not facts_path.is_file():
        raise FileNotFoundError(f"Missing benchmark fact bank: {facts_path}")
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    if not isinstance(facts, list) or not facts:
        raise ValueError(f"Fact bank must be a non-empty list: {facts_path}")
    return facts


def build_benchmark(cutoff_year: int, known: bool, config: dict, facts: list[dict]) -> dict:
    """Build a validator-compatible benchmark dict for one cutoff year."""
    epsilon = config.get("leak_epsilon", -11.51)
    cutoff_weight = config.get("known_cutoff_weight", 5)

    if known:
        items = []
        for fact in facts:
            if fact["year"] <= cutoff_year:
                weight = cutoff_weight if fact["year"] == cutoff_year else 1
                items.append(
                    {
                        "prompt": fact["prompt"],
                        "phrase": fact["phrase"],
                        "weight": weight,
                    }
                )
        return {
            "items": items,
            "threshold": config.get("known_threshold", 0.7),
            "epsilon": epsilon,
        }

    items = [
        {"prompt": fact["prompt"], "phrase": fact["phrase"], "weight": 1}
        for fact in facts
        if fact["year"] > cutoff_year
    ]
    return {
        "items": items,
        "threshold": config.get("leak_threshold", 0.1),
        "epsilon": epsilon,
    }


def benchmark_file_path(benchmark_dir: Path, year: int, known: bool) -> Path:
    name = "known.json" if known else "unknown.json"
    return benchmark_dir / str(year) / name


def load_benchmark_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_year_benchmarks(
    cutoff_year: int,
    config: dict,
    facts: list[dict],
    benchmark_dir: Path | None,
) -> dict[str, dict]:
    if benchmark_dir is not None:
        unknown_path = benchmark_file_path(benchmark_dir, cutoff_year, known=False)
        known_path = benchmark_file_path(benchmark_dir, cutoff_year, known=True)
        if not unknown_path.is_file() or not known_path.is_file():
            raise FileNotFoundError(
                f"Missing benchmark files for year {cutoff_year} under {benchmark_dir}"
            )
        return {
            "unknown": load_benchmark_file(unknown_path),
            "known": load_benchmark_file(known_path),
        }

    return {
        "unknown": build_benchmark(cutoff_year, known=False, config=config, facts=facts),
        "known": build_benchmark(cutoff_year, known=True, config=config, facts=facts),
    }


def preload_benchmarks(
    years: list[int],
    config: dict,
    facts: list[dict] | None = None,
    benchmark_dir: Path | None = None,
) -> dict[int, dict[str, dict]]:
    fact_bank = facts if facts is not None else load_facts()
    benchmarks = {}
    for year in years:
        year_benchmarks = load_year_benchmarks(year, config, fact_bank, benchmark_dir)
        if not year_benchmarks["unknown"]["items"]:
            raise RuntimeError(f"No unknown (post-cutoff) items for year {year}")
        if not year_benchmarks["known"]["items"]:
            raise RuntimeError(f"No known (pre-cutoff) items for year {year}")
        benchmarks[year] = year_benchmarks
    return benchmarks
