"""SAE wrapper: load pre-trained Sparse Autoencoders and expose encode/decode.

Supports Llama Scope / OpenMOSS checkpoints via safetensors loading from
HuggingFace Hub or local directories.

Ported from lgbtqmi/src/sae_localization/sae_wrapper.py.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from src.utils.logging import log


class SAEWrapper:
    """Thin wrapper around a pre-trained Sparse Autoencoder.

    Parameters
    ----------
    checkpoint_path_or_id : str
        Local directory containing the SAE checkpoint **or** a HuggingFace
        repo id (e.g. ``OpenMOSS-Team/Llama3_1-8B-Base-LXR-32x``).
    layer : int
        Transformer layer index the SAE was trained on.
    site : str
        Hook site.  ``"R"`` = residual stream (post-MLP).
    expansion : int
        Width multiplier (8 → 32 768 features, 32 → 131 072 features).
    device : str
        Target device for SAE weights.
    """

    def __init__(
        self,
        checkpoint_path_or_id: str,
        layer: int,
        site: str = "R",
        expansion: int = 32,
        device: str = "cpu",
    ) -> None:
        self._layer = layer
        self._site = site
        self._expansion = expansion
        self._device = device

        # Weights (populated by _load_*)
        self._W_enc: torch.Tensor   # (d_model, d_sae)
        self._b_enc: torch.Tensor   # (d_sae,)
        self._W_dec: torch.Tensor   # (d_sae, d_model)
        self._b_dec: torch.Tensor   # (d_model,)
        self._threshold: torch.Tensor  # (d_sae,) — JumpReLU threshold
        self._norm_factor: float = 1.0
        self._scale_by_decoder_norm: bool = False
        self._n_features: int = 0
        self._hidden_dim: int = 0

        path = Path(checkpoint_path_or_id)
        if path.is_dir():
            self._load_local(path, layer, site, expansion)
        else:
            self._load_hub(checkpoint_path_or_id, layer, site, expansion)

        log(
            f"SAEWrapper: loaded layer={layer} site={site} "
            f"expansion={expansion}x  features={self._n_features}  "
            f"hidden_dim={self._hidden_dim}  device={device}"
        )

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_local(
        self, base_dir: Path, layer: int, site: str, expansion: int,
    ) -> None:
        """Load from a local snapshot (already downloaded)."""
        candidates = list(base_dir.glob(f"*L{layer}{site}-{expansion}x*"))
        if candidates:
            sae_dir = candidates[0]
        elif (base_dir / "hyperparams.json").exists():
            sae_dir = base_dir
        elif (base_dir / "checkpoints" / "final.safetensors").exists():
            sae_dir = base_dir
        else:
            raise FileNotFoundError(
                f"Cannot locate SAE checkpoint for layer={layer} site={site} "
                f"expansion={expansion}x under {base_dir}"
            )
        self._load_safetensors_dir(sae_dir)

    def _load_hub(
        self, repo_id: str, layer: int, site: str, expansion: int,
    ) -> None:
        """Download from HuggingFace Hub, then load locally."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for downloading SAE checkpoints. "
                "Install with: pip install huggingface-hub"
            ) from exc

        log(f"Downloading SAE from {repo_id} (layer {layer}, {expansion}x) ...")
        patterns = [
            f"*L{layer}{site}-{expansion}x*/hyperparams.json",
            f"*L{layer}{site}-{expansion}x*/checkpoints/final.safetensors",
            f"*L{layer}{site}-{expansion}x*/lm_config.json",
        ]
        local = Path(
            snapshot_download(repo_id, allow_patterns=patterns)
        )

        has_match = any(list(local.glob(pat)) for pat in patterns)
        if not has_match:
            raise FileNotFoundError(
                "Downloaded SAE snapshot did not contain expected files for "
                f"repo_id={repo_id} layer={layer} site={site} "
                f"expansion={expansion}x. "
                f"Tried allow_patterns={patterns}."
            )
        self._load_local(local, layer, site, expansion)

    def _load_safetensors_dir(self, sae_dir: Path) -> None:
        """Parse hyperparams.json + load final.safetensors."""
        from safetensors.torch import load_file

        # --- hyperparams ---
        hp_path = sae_dir / "hyperparams.json"
        hp: dict = {}
        if hp_path.exists():
            with open(hp_path) as f:
                hp = json.load(f)
            self._scale_by_decoder_norm = hp.get(
                "sparsity_include_decoder_norm", False,
            )
            norm_act = hp.get("norm_activation", "none")
            if norm_act == "dataset-wise":
                dataset_norm = hp.get("dataset_average_activation_norm", {})
                if isinstance(dataset_norm, dict):
                    norm_in = next(
                        (v for v in dataset_norm.values()
                         if isinstance(v, (int, float))),
                        None,
                    )
                else:
                    norm_in = float(dataset_norm)
                d_model = hp.get("d_model", 4096)
                if norm_in and norm_in > 0:
                    self._norm_factor = math.sqrt(d_model) / norm_in
                log(
                    f"  Dataset-wise norm: factor={self._norm_factor:.4f} "
                    f"(raw norm_in={norm_in})"
                )
        else:
            log("  No hyperparams.json found; using defaults")

        # --- weights ---
        st_path = sae_dir / "checkpoints" / "final.safetensors"
        if not st_path.exists():
            candidates = list(sae_dir.glob("*.safetensors"))
            if not candidates:
                raise FileNotFoundError(
                    f"No .safetensors found under {sae_dir}"
                )
            st_path = candidates[0]

        weights = load_file(str(st_path), device=self._device)

        # Support multiple naming conventions.
        if "encoder.weight" in weights:
            # Llama Scope: encoder is (d_sae, d_model)
            self._W_enc = weights["encoder.weight"].T  # → (d_model, d_sae)
            self._b_enc = weights["encoder.bias"]
            self._W_dec = weights["decoder.weight"].T  # → (d_sae, d_model)
            self._b_dec = weights["decoder.bias"]
        elif "W_E" in weights:
            # lm-saes v1 format
            self._W_enc = weights["W_E"]
            self._b_enc = weights["b_E"]
            self._W_dec = weights["W_D"]
            self._b_dec = weights["b_D"]
        else:
            raise KeyError(
                f"Unrecognised weight keys: {list(weights.keys())}. "
                "Expected 'encoder.weight' (Llama Scope) or 'W_E' (lm-saes)."
            )

        # JumpReLU threshold
        log_thresh_key = "activation_function.log_jumprelu_threshold"
        if log_thresh_key in weights:
            self._threshold = torch.exp(weights[log_thresh_key])
        elif hp_path.exists() and "jump_relu_threshold" in hp:
            t_val = float(hp["jump_relu_threshold"])
            self._threshold = torch.full(
                (self._W_enc.shape[1],), t_val, device=self._device,
            )
            log(f"  JumpReLU threshold (global): {t_val}")
        else:
            log("  WARNING: no JumpReLU threshold found; using ReLU")
            self._threshold = torch.zeros(
                self._W_enc.shape[1], device=self._device,
            )

        self._n_features = int(self._W_enc.shape[1])
        self._hidden_dim = int(self._W_enc.shape[0])

        # Pre-compute decoder column norms for sparsity_include_decoder_norm.
        if self._scale_by_decoder_norm:
            self._dec_norms = torch.norm(self._W_dec, dim=1).clamp(min=1e-8)
        else:
            self._dec_norms = torch.ones(
                self._n_features, device=self._device,
            )

        # Fold dataset-norm into weights so encode() is just a matmul.
        if abs(self._norm_factor - 1.0) > 1e-6:
            self._W_enc = self._W_enc * self._norm_factor
            self._b_enc = self._b_enc * self._norm_factor
            self._norm_factor = 1.0
            log("  Folded dataset-norm into encoder weights")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        """Encode raw activations into sparse SAE feature space.

        Parameters
        ----------
        activations : Tensor
            Shape ``(batch, hidden_dim)`` or ``(hidden_dim,)``.

        Returns
        -------
        Tensor
            Shape ``(batch, n_features)`` or ``(n_features,)`` — sparse.
        """
        squeeze = activations.dim() == 1
        if squeeze:
            activations = activations.unsqueeze(0)

        x = activations.to(self._W_enc.dtype).to(self._device)

        hidden_pre = x @ self._W_enc + self._b_enc

        if self._scale_by_decoder_norm:
            hidden_pre = hidden_pre * self._dec_norms.unsqueeze(0)

        # JumpReLU: x * (x > threshold)
        mask = hidden_pre > self._threshold.unsqueeze(0)
        feature_acts = hidden_pre * mask

        if self._scale_by_decoder_norm:
            feature_acts = feature_acts / self._dec_norms.unsqueeze(0)

        if squeeze:
            feature_acts = feature_acts.squeeze(0)
        return feature_acts

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Reconstruct activation from SAE feature activations.

        Parameters
        ----------
        feature_acts : Tensor
            Shape ``(batch, n_features)`` or ``(n_features,)``.

        Returns
        -------
        Tensor
            Shape ``(batch, hidden_dim)`` or ``(hidden_dim,)``.
        """
        squeeze = feature_acts.dim() == 1
        if squeeze:
            feature_acts = feature_acts.unsqueeze(0)
        x = feature_acts.to(self._W_dec.dtype).to(self._device)
        out = x @ self._W_dec + self._b_dec
        if squeeze:
            out = out.squeeze(0)
        return out

    def get_decoder_matrix(self) -> np.ndarray:
        """Return the decoder weight matrix with L2-normalised rows.

        Shape ``(n_features, hidden_dim)``.
        """
        W = self._W_dec.detach().float().cpu().numpy()
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return W / norms

    def get_feature_direction(self, feature_idx: int) -> np.ndarray:
        """Return the L2-normalised decoder direction for a single feature."""
        d = self._W_dec[feature_idx].detach().float().cpu().numpy()
        norm = np.linalg.norm(d)
        if norm > 1e-8:
            d = d / norm
        return d

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def layer(self) -> int:
        return self._layer

    @property
    def device(self) -> str:
        return self._device
