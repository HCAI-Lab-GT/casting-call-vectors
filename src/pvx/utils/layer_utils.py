"""Resolve architecture-relative layer choices to absolute decoder-block indices."""

from __future__ import annotations


def num_decoder_layers(model_id: str) -> int:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    n = getattr(cfg, "num_hidden_layers", None)
    if n is None and hasattr(cfg, "text_config"):
        n = getattr(cfg.text_config, "num_hidden_layers", None)
    if n is None:
        raise ValueError(f"Cannot determine layer count for {model_id}")
    return int(n)


def layer_from_fraction(fraction: float, num_layers: int) -> int:
    """Index of the decoder block at `fraction` of depth (0.5 of 32 layers -> 16)."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"layer fraction must be in [0, 1], got {fraction}")
    return min(int(round(fraction * num_layers)), num_layers - 1)


def resolve_layers(
    model_id: str,
    layers: list[int] | None = None,
    layer_fractions: list[float] | None = None,
) -> list[int]:
    """Absolute layers from explicit indices and/or fractions of depth; sorted, de-duplicated."""
    resolved = list(layers or [])
    if layer_fractions:
        n = num_decoder_layers(model_id)
        resolved += [layer_from_fraction(f, n) for f in layer_fractions]
    if not resolved:
        raise ValueError("Provide at least one of layers / layer_fractions")
    return sorted(set(resolved))
