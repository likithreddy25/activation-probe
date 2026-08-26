"""Latency-optimized single-layer activation extraction for serving.

`ActivationExtractor` (extract.py) uses nnsight to capture the *entire*
residual stream in one pass, which is what you want for offline research
(layer sweeps, re-analysis). But the served probe only ever reads one fixed
layer, so at inference time we don't need the rest of the network: a forward
hook on the target decoder layer captures the pooled activation and then
raises to abort the forward pass, skipping every later layer (and the LM
head). For a mid-network layer like the one this probe uses, that's roughly
half the model's compute skipped on every request -- part of the "cheap
and at scale" argument this project is trying to demonstrate with real
numbers (see scripts/benchmark_latency.py).
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class _StopForward(Exception):
    pass


class FastLayerExtractor:
    def __init__(self, model_name: str, layer: int, device: str | None = None):
        self.model_name = model_name
        self.layer = layer
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()
        self._captured: torch.Tensor | None = None
        self.model.model.layers[layer].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self._captured = hidden.mean(dim=1).detach()
        raise _StopForward()

    @torch.no_grad()
    def extract(self, prompt: str) -> np.ndarray:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        self._captured = None
        try:
            self.model(**inputs)
        except _StopForward:
            pass
        if self._captured is None:
            raise RuntimeError("forward hook never fired -- layer index out of range?")
        return self._captured[0].float().cpu().numpy()
