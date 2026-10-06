"""Export-time replacement for Gemma4 audio ``Tensor.unfold``.

``Gemma4AudioAttention._extract_block_context`` windows K/V with
``hidden_states.unfold(1, context_size, chunk_size)``. Core AI convert
rejects ``aten.unfold.default``. Gather along the padded sequence dim
matches ``unfold(1, window, step)`` + ``movedim(-1, 2)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gather_seq_windows(x: torch.Tensor, window: int, step: int) -> torch.Tensor:
    """``x.unfold(1, window, step)`` then ``movedim(-1, 2)`` via ``index_select``.

    ``x`` is ``[B, S, ...]``. Result is ``[B, n_win, window, ...]``.
    """
    seq_len = int(x.shape[1])
    window = int(window)
    step = int(step)
    if window <= 0 or step <= 0:
        raise ValueError(f"window={window} step={step}")
    n_win = (seq_len - window) // step + 1
    if n_win <= 0:
        raise ValueError(f"n_win={n_win} for S={seq_len} window={window} step={step}")
    starts = torch.arange(n_win, device=x.device, dtype=torch.long) * step
    offsets = torch.arange(window, device=x.device, dtype=torch.long)
    idx = (starts.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1)
    gathered = x.index_select(1, idx)
    return gathered.reshape(x.shape[0], n_win, window, *x.shape[2:])


def _extract_block_context_gather(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = F.pad(
        hidden_states,
        (0, 0, 0, 0, self.max_past_horizon, self.max_future_horizon + self.chunk_size - 1),
    )
    hidden_states = gather_seq_windows(hidden_states, self.context_size, self.chunk_size)
    return hidden_states.contiguous()


def apply_audio_unfold_patch() -> dict[str, int]:
    """Monkey-patch ``Gemma4AudioAttention._extract_block_context`` for export."""
    import transformers.models.gemma4.modeling_gemma4 as g4

    g4.Gemma4AudioAttention._extract_block_context = _extract_block_context_gather
    return {"patched": 1, "window_op": "index_select"}
