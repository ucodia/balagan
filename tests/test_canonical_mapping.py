"""Tests for balagan.core.canonical_mapping."""

from pathlib import Path

import torch

from balagan.core.canonical_mapping import load_canonical_mapping


class StubMapping(torch.nn.Module):
    """Minimal stand-in for a StyleGAN2 MappingNetwork."""

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)


class StubGenerator(torch.nn.Module):
    """Minimal generator: a mapping network plus a synthesis network."""

    def __init__(self):
        super().__init__()
        self.mapping = StubMapping()
        self.synthesis = torch.nn.Linear(4, 4)


def test_extracts_the_g_ema_mapping_network():
    g_ema = StubGenerator()
    plain = StubGenerator()

    def loader(path):
        return {"G_ema": g_ema, "G": plain}

    mapping = load_canonical_mapping("snap.pkl", "cpu", pkl_loader=loader)
    assert mapping is g_ema.mapping
    assert mapping is not plain.mapping


def test_moves_the_mapping_to_the_inference_device():
    def loader(path):
        return {"G_ema": StubGenerator()}

    mapping = load_canonical_mapping("snap.pkl", "meta", pkl_loader=loader)
    assert next(mapping.parameters()).device.type == "meta"


def test_passes_the_pkl_path_to_the_loader():
    seen: list[Path] = []

    def loader(path):
        seen.append(path)
        return {"G_ema": StubGenerator()}

    load_canonical_mapping("runs/network-snapshot-002544.pkl", "cpu", pkl_loader=loader)
    assert seen == [Path("runs/network-snapshot-002544.pkl")]
