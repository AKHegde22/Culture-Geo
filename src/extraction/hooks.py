"""
Hook registration for extracting internal activations from LLMs.

Uses raw HuggingFace transformer hooks (compatible with Llama-3-8B).
TransformerLens has limited Llama-3 support, so we use direct hooks.

Key hook points:
    - hidden_states[layer_idx]: Residual stream at each layer
    - attn_weights[layer_idx]: Attention patterns at each layer
    - lm_logits: Final logits (for logit lens)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ActivationCache:
    """Container for extracted activations from a single forward pass."""
    hidden_states: Dict[int, torch.Tensor] = field(default_factory=dict)
    attn_weights: Optional[Dict[int, torch.Tensor]] = None
    lm_logits: Optional[torch.Tensor] = None
    input_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None

    @property
    def num_layers(self) -> int:
        return len(self.hidden_states)

    def to_numpy(self) -> "ActivationCache":
        """Convert all tensors to numpy for storage efficiency."""
        import torch
        for k in self.hidden_states:
            if isinstance(self.hidden_states[k], torch.Tensor):
                self.hidden_states[k] = self.hidden_states[k].cpu().numpy()
        if self.attn_weights:
            for k in self.attn_weights:
                if isinstance(self.attn_weights[k], torch.Tensor):
                    self.attn_weights[k] = self.attn_weights[k].cpu().numpy()
        if self.lm_logits is not None:
            if isinstance(self.lm_logits, torch.Tensor):
                self.lm_logits = self.lm_logits.cpu().numpy()
        return self


class ActivationExtractor:
    """
    Registers forward hooks on a HuggingFace model to capture activations.

    Usage:
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
        extractor = ActivationExtractor(model)
        cache = extractor.extract(input_ids)
        # cache.hidden_states[0] through cache.hidden_states[32]
    """

    def __init__(
        self,
        model: nn.Module,
        include_attention: bool = False,
        device: str = "cuda",
    ):
        self.model = model
        self.include_attention = include_attention
        self.device = device
        self.hooks = []
        self._cache = {}

        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on transformer layers."""
        # Clear any existing hooks
        self.remove_hooks()

        # Access the model's transformer layers
        # Works for LlamaForCausalLM, MistralForCausalLM, etc.
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            layers = self.model.transformer.h  # GPT-2 style
        else:
            raise ValueError(
                f"Cannot find transformer layers in model: {type(self.model).__name__}. "
                f"Model attributes: {[a for a in dir(self.model) if not a.startswith('_')]}"
            )

        self.num_layers = len(layers)

        for idx, layer in enumerate(layers):
            # Hook for residual stream output (post-layer)
            hook = layer.register_forward_hook(
                self._make_hidden_state_hook(idx)
            )
            self.hooks.append(hook)

            # Hook for attention weights (optional)
            if self.include_attention and hasattr(layer, "self_attn"):
                attn_hook = layer.self_attn.register_forward_hook(
                    self._make_attention_hook(idx)
                )
                self.hooks.append(attn_hook)

    def _make_hidden_state_hook(self, layer_idx: int):
        """Create a hook function that captures the residual stream."""
        def hook_fn(module, input, output):
            # For most transformer layers, output is a tuple:
            # (hidden_states, attention_weights, present_key_value)
            # The first element is the residual stream
            if isinstance(output, tuple):
                hidden_state = output[0]
            else:
                hidden_state = output

            self._cache[f"hidden_state_{layer_idx}"] = hidden_state.detach()
        return hook_fn

    def _make_attention_hook(self, layer_idx: int):
        """Create a hook function that captures attention weights."""
        def hook_fn(module, input, output):
            # attention_output, attn_weights, present_key_value
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                self._cache[f"attn_{layer_idx}"] = output[1].detach()
        return hook_fn

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def extract(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        capture_logits: bool = True,
    ) -> ActivationCache:
        """
        Run a forward pass and capture all activations.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            capture_logits: Whether to capture the final logits

        Returns:
            ActivationCache with all captured activations
        """
        self._cache.clear()
        self.model.eval()

        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            if capture_logits:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,  # We use hooks instead
                    return_dict=True,
                )
                lm_logits = outputs.logits
            else:
                self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                )
                lm_logits = None

        # Collect activations from cache
        hidden_states = {}
        attn_weights = {}
        for key, value in self._cache.items():
            if key.startswith("hidden_state_"):
                layer_idx = int(key.split("_")[-1])
                hidden_states[layer_idx] = value
            elif key.startswith("attn_"):
                layer_idx = int(key.split("_")[-1])
                attn_weights[layer_idx] = value

        return ActivationCache(
            hidden_states=hidden_states,
            attn_weights=attn_weights if attn_weights else None,
            lm_logits=lm_logits.detach() if lm_logits is not None else None,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def __del__(self):
        self.remove_hooks()
