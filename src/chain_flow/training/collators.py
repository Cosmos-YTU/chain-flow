from __future__ import annotations

import torch


def collate_teacher_windows(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    out = {
        "context_hidden": torch.stack([item["context_hidden"] for item in batch], dim=0),
        "target_hidden": torch.stack([item["target_hidden"] for item in batch], dim=0),
        "future_tokens": torch.stack([item["future_tokens"] for item in batch], dim=0),
    }
    if "prev_token" in batch[0]:
        out["prev_token"] = torch.stack([item["prev_token"] for item in batch], dim=0)
    if "lag_token" in batch[0]:
        out["lag_token"] = torch.stack([item["lag_token"] for item in batch], dim=0)
    return out
