"""SAE-based steering: inject steering vectors into the residual stream.

Provides ``SAESteerer`` which wraps a ``ModelWrapper`` to evaluate prompts
with and without a steering hook.  Used by C1, C2, C3 for all forward passes.

The steering hook adds a vector to the residual stream at the injection layer,
at the **last token position** only.  This is the standard activation-addition
approach: the model's processing of the final token (the one that produces the
next-token prediction for "Answer:") is shifted by the steering vector.
"""

from __future__ import annotations

from typing import Any

import torch

from src.models.wrapper import ModelWrapper, locate_hidden_tensor
from src.sae.wrapper import SAEWrapper
from src.utils.logging import log


# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------

def _extract_answer_logits(
    logits: torch.Tensor,
    tokenizer: Any,
    letters: tuple[str, ...] = ("A", "B", "C"),
) -> dict[str, float]:
    """Extract logits for each answer letter from the model's output.

    Parameters
    ----------
    logits : Tensor
        Shape ``(1, seq_len, vocab_size)`` — raw model output logits.
    tokenizer : Any
        HuggingFace tokenizer with ``encode`` method.
    letters : tuple[str, ...]
        Answer option letters to extract.

    Returns
    -------
    dict
        ``{letter: logit_value}`` for each letter.
    """
    # Last token position logits -> next-token prediction
    last_logits = logits[0, -1, :]  # (vocab_size,)

    result = {}
    for letter in letters:
        # Tokenize the letter; take first token if multi-token
        token_ids = tokenizer.encode(letter, add_special_tokens=False)
        if not token_ids:
            result[letter] = float("-inf")
            continue
        tid = token_ids[0]
        result[letter] = float(last_logits[tid].item())

    return result


def _top_answer(answer_logits: dict[str, float]) -> str:
    """Return the letter with the highest logit."""
    if not answer_logits:
        return ""
    return max(answer_logits, key=lambda k: answer_logits[k])


# ---------------------------------------------------------------------------
# SAESteerer
# ---------------------------------------------------------------------------

class SAESteerer:
    """Evaluates prompts with optional steering-vector injection.

    Parameters
    ----------
    wrapper : ModelWrapper
        The model wrapper (provides hook registration, tokenizer, model).
    sae : SAEWrapper
        The SAE for the injection layer (kept for potential future use;
        not required for pure steering).
    injection_layer : int
        Decoder layer at which the steering vector is injected.
    """

    def __init__(
        self,
        wrapper: ModelWrapper,
        sae: SAEWrapper,
        injection_layer: int,
    ) -> None:
        self.wrapper = wrapper
        self.sae = sae
        self.injection_layer = injection_layer
        self.device = wrapper.device
        self.dtype = next(wrapper.model.parameters()).dtype

    # ── Baseline (no hooks) ──────────────────────────────────────────

    def evaluate_baseline(
        self,
        prompt: str,
        *,
        letters: tuple[str, ...] = ("A", "B", "C"),
    ) -> dict[str, Any]:
        """Run a forward pass WITHOUT steering hooks.

        Returns
        -------
        dict
            ``model_answer``, ``answer_logits``.
        """
        tokenizer = self.wrapper.tokenizer
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.wrapper.model(**inputs)

        logits = outputs.logits  # (1, seq_len, vocab_size)
        answer_logits = _extract_answer_logits(logits, tokenizer, letters)

        return {
            "model_answer": _top_answer(answer_logits),
            "answer_logits": answer_logits,
        }

    def evaluate_baseline_mcq(
        self,
        prompt: str,
        *,
        letters: tuple[str, ...] = ("A", "B", "C", "D"),
    ) -> dict[str, Any]:
        """Convenience alias for baseline evaluation with configurable letters.

        Used by C3 (MedQA has 5 options, MMLU has 4).
        """
        return self.evaluate_baseline(prompt, letters=letters)

    # ── Steered (with hook) ──────────────────────────────────────────

    def steer_and_evaluate(
        self,
        prompt: str,
        steering_vec: torch.Tensor,
        *,
        letters: tuple[str, ...] = ("A", "B", "C"),
    ) -> dict[str, Any]:
        """Run a forward pass WITH steering vector injection.

        The steering vector is added to the residual stream at
        ``self.injection_layer``, at the **last token position** only.

        Parameters
        ----------
        prompt : str
            The formatted prompt.
        steering_vec : Tensor
            Shape ``(hidden_dim,)``.  Added to the hidden state.
        letters : tuple[str, ...]
            Answer option letters to extract logits for.

        Returns
        -------
        dict
            ``model_answer``, ``answer_logits``, ``degenerated``.
        """
        tokenizer = self.wrapper.tokenizer
        hidden_dim = self.wrapper.hidden_dim
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

        # Ensure steering vec is on device and correct dtype
        vec = steering_vec.to(device=self.device, dtype=self.dtype)

        # Build the steering hook
        def steering_hook(
            module: Any, args: Any, output: Any,
        ) -> Any:
            h = locate_hidden_tensor(output, hidden_dim)
            # h shape: (1, seq_len, hidden_dim)
            # Add steering vec at last token position only
            h[:, -1, :] = h[:, -1, :] + vec
            return output

        # Register hook, run forward, remove hook
        layer_module = self.wrapper.get_layer(self.injection_layer)
        handle = layer_module.register_forward_hook(steering_hook)

        try:
            with torch.no_grad():
                outputs = self.wrapper.model(**inputs)
        finally:
            handle.remove()

        logits = outputs.logits
        answer_logits = _extract_answer_logits(logits, tokenizer, letters)

        # Degeneration check (imported lazily to avoid circular dep at module level)
        from src.metrics.bias_metrics import is_degenerated
        degenerated = is_degenerated(answer_logits, options=letters)

        return {
            "model_answer": _top_answer(answer_logits),
            "answer_logits": answer_logits,
            "degenerated": degenerated,
        }
