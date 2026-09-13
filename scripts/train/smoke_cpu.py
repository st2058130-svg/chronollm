"""CPU environment checks before GPU training.

Verifies:
  1) nanochrono init + 1 forward/backward on CPU (offline)
  2) FineWeb-Edu streaming (needs network / HF_TOKEN)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.train.env import load_train_env

# Longer HF timeouts for slow links
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")


def test_model():
    print("[1/2] tokenizer + nanochrono init (offline)…")
    from transformers import AutoTokenizer
    import torch
    import sn38.architectures  # noqa: F401
    from sn38.architectures.nanochrono.configuration_nanochrono import NanochronoConfig
    from sn38.architectures.nanochrono.modeling_nanochrono import NanochronoForCausalLM

    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Tiny model for CPU smoke (not the production 28-layer net)
    config = NanochronoConfig(
        vocab_size=len(tok),
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        sliding_window=128,
        bos_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    model = NanochronoForCausalLM(config)
    model.train()
    n = sum(p.numel() for p in model.parameters())
    print(f"      tiny smoke model params={n / 1e6:.2f}M")

    ids = torch.randint(0, len(tok), (1, 64))
    out = model(input_ids=ids, labels=ids, use_cache=False)
    out.loss.backward()
    print(f"      forward/backward OK loss={out.loss.item():.4f}")
    return True


def test_stream():
    print("[2/2] streaming FineWeb-Edu CC-MAIN-2018-51…")
    from scripts.train.data import take_texts

    texts = take_texts(["CC-MAIN-2018-51"], "HuggingFaceFW/fineweb-edu", n=3)
    assert texts, "no texts returned"
    print(f"      got {len(texts)} docs, first {len(texts[0])} chars")
    return True


def main():
    load_train_env()
    print("=== SN38 train env smoke (CPU) ===")
    test_model()
    try:
        test_stream()
    except Exception as e:
        print(f"      SKIP stream (network/HF): {type(e).__name__}: {e}")
        print("      Set HF_TOKEN in scripts/train/.env or the environment, then retry.")
        print("=== PARTIAL OK — code+deps ready; HF stream needs network ===")
        return
    print("=== OK — env ready; plug in GPU and run train.py ===")


if __name__ == "__main__":
    main()
