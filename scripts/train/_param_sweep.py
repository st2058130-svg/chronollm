import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import sn38.architectures  # noqa: F401
from transformers import AutoTokenizer
from sn38.architectures.nanochrono.configuration_nanochrono import NanochronoConfig
from sn38.architectures.nanochrono.modeling_nanochrono import NanochronoForCausalLM

tok = AutoTokenizer.from_pretrained("gpt2")
v = len(tok)


def count(**kw):
    defaults = dict(
        vocab_size=v,
        hidden_size=1792,
        intermediate_size=7168,
        num_hidden_layers=28,
        num_attention_heads=14,
        num_key_value_heads=14,
        max_position_embeddings=2048,
        sliding_window=512,
        bos_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    defaults.update(kw)
    m = NanochronoForCausalLM(NanochronoConfig(**defaults))
    n = sum(p.numel() for p in m.parameters())
    print(
        f"{n / 1e9:.3f}B  layers={defaults['num_hidden_layers']} "
        f"hidden={defaults['hidden_size']} inter={defaults['intermediate_size']}"
    )


for layers in [28, 26, 25, 24, 23]:
    count(num_hidden_layers=layers)
