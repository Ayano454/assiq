from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class NeuronCoverageTracker:
    """
    Simple neuron coverage tracker.

    For convolutional outputs, each channel is treated as one neuron by taking the
    mean activation over spatial dimensions. For linear outputs, each feature is a neuron.
    A neuron is marked covered when its activation exceeds the threshold at least once.
    """

    def __init__(self, threshold: float = 0.2) -> None:
        self.threshold = threshold
        self.covered: Dict[str, torch.Tensor] = {}
        self.handles = []

    def _hook(self, name: str):
        def fn(_module: nn.Module, _inputs, output: torch.Tensor) -> None:
            if not isinstance(output, torch.Tensor):
                return

            feats = output.detach()

            if feats.dim() > 2:
                feats = feats.mean(dim=tuple(range(2, feats.dim())))

            if feats.dim() != 2:
                return

            hits = (feats > self.threshold).any(dim=0).cpu()

            # 같은 ReLU 모듈이 서로 다른 채널 수로 여러 번 호출될 수 있으므로
            # 채널 수까지 포함해서 별도 키를 만든다.
            key = f"{name}:{hits.numel()}"

            if key not in self.covered:
                self.covered[key] = torch.zeros_like(hits, dtype=torch.bool)

            self.covered[key] |= hits

        return fn

    def register(self, model: nn.Module) -> None:
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                self.handles.append(module.register_forward_hook(self._hook(name)))

    def coverage(self) -> float:
        total = sum(t.numel() for t in self.covered.values())
        active = sum(t.sum().item() for t in self.covered.values())
        if total == 0:
            return 0.0
        return float(active) / float(total)

    def summary(self) -> Dict[str, float]:
        total = sum(t.numel() for t in self.covered.values())
        active = sum(t.sum().item() for t in self.covered.values())
        return {
            "covered_neurons": int(active),
            "total_neurons": int(total),
            "coverage": self.coverage(),
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()