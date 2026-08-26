"""Extract mean-pooled residual-stream activations from a small open-weight model.

Design choice: prompts are wrapped in the model's chat template (as a single
user turn, with the generation prompt appended) before extraction, rather than
fed as raw text. That mirrors how a real guardrail would see the input -- the
target model receives the same chat-formatted turn -- and keeps the special
tokens/formatting consistent with how the base model actually processes chat
input during deployment.

We use nnsight so the whole residual stream (the output of every decoder
layer) is captured in a single forward pass per prompt, rather than doing N
separate forward passes for N layers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from nnsight import LanguageModel
from tqdm import tqdm

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class ActivationBatch:
    # shape: (n_prompts, n_layers, hidden_dim) -- mean-pooled residual stream
    # after each decoder layer, for every prompt.
    activations: np.ndarray
    model_name: str
    n_layers: int
    hidden_dim: int


class ActivationExtractor:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        self.model_name = model_name
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = LanguageModel(model_name, device_map=self.device, dispatch=True)
        self.n_layers = self.model.config.num_hidden_layers
        self.hidden_dim = self.model.config.hidden_size

    def _format(self, prompt: str) -> str:
        return self.model.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def extract_one(self, prompt: str) -> np.ndarray:
        """Return array of shape (n_layers, hidden_dim): mean-pooled residual
        stream after each decoder layer for a single prompt."""
        text = self._format(prompt)
        saved = []
        with torch.no_grad(), self.model.trace(text):
            for i in range(self.n_layers):
                pooled = self.model.model.layers[i].output.mean(dim=1).save()
                saved.append(pooled)
        return np.stack([s[0].float().detach().cpu().numpy() for s in saved], axis=0)

    def extract_batch(self, prompts: list[str], show_progress: bool = True) -> ActivationBatch:
        rows = []
        it = tqdm(prompts, desc=f"extracting activations ({self.model_name})") if show_progress else prompts
        for p in it:
            rows.append(self.extract_one(p))
        acts = np.stack(rows, axis=0)
        return ActivationBatch(
            activations=acts,
            model_name=self.model_name,
            n_layers=self.n_layers,
            hidden_dim=self.hidden_dim,
        )
