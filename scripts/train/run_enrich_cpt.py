"""Enrich chronollm/cpt.jsonl in-place with multi-style fields."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from scripts.train.enrich_styles import enrich_record


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    src = root / "cpt.jsonl"
    dst = root / "cpt.jsonl.enrich.tmp"
    kept = bad = 0
    style_hist: dict[str, int] = {}
    t0 = time.time()

    with src.open("r", encoding="utf-8", errors="replace") as fin, dst.open(
        "w", encoding="utf-8", newline="\n"
    ) as fout:
        for i, line in enumerate(fin, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                bad += 1
                continue
            enriched = enrich_record(obj, target_year=2024)
            fout.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            kept += 1
            for style in enriched.get("styles") or []:
                style_hist[style] = style_hist.get(style, 0) + 1
            if i % 100000 == 0:
                print(f"progress i={i} kept={kept} {time.time() - t0:.1f}s", flush=True)

    os.replace(dst, src)
    print("done")
    print("kept", kept, "bad", bad)
    print("new_bytes", src.stat().st_size)
    print("seconds", round(time.time() - t0, 1))
    top = sorted(style_hist.items(), key=lambda x: -x[1])[:24]
    print("style_counts", dict(top))
    with src.open("r", encoding="utf-8") as f:
        sample = json.loads(f.readline())
    print("sample_keys", sorted(sample.keys()))
    print("sample_styles", sample.get("styles"))
    print("sample_fact", (sample.get("fact") or "")[:220])
    print("sample_qa", (sample.get("qa") or "")[:220])


if __name__ == "__main__":
    main()
