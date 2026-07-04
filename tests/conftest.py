"""Shared fixtures. Everything is tiny + synthetic: no downloads, seconds-scale."""
import numpy as np
import pytest

TYPES = ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"]


def heavy_tailed(nrow, n, seed):
    """Heavy-tailed test weights (cubic -> heavy tails, per-row scale spread)."""
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((nrow, n)) ** 3).astype(np.float32)
    x *= rng.uniform(0.2, 3.0, size=(nrow, 1)).astype(np.float32)
    return x


@pytest.fixture(scope="session")
def tiny_qwen3():
    """A 2-layer random Qwen3ForCausalLM (all divisible by 256 for real quant paths)."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(
        vocab_size=512, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        max_position_embeddings=128, tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg).eval()
    return model, cfg


@pytest.fixture(scope="session")
def tiny_ids():
    import torch
    torch.manual_seed(1)
    return torch.randint(0, 512, (1, 48))
