from __future__ import annotations

import os

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

from datasets import Dataset, load_dataset, load_from_disk
import torch
from torch.utils.data import Dataset as TorchDataset

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is normally present through datasets/transformers.
    tqdm = None


FLOW_CACHE_VERSION = 1
FLOW_CACHE_METADATA = "metadata.json"
FLOW_CACHE_FILES = {
    "hidden": "hidden.pt",
    "input_ids": "input_ids.pt",
    "row_offsets": "row_offsets.pt",
    "row_lengths": "row_lengths.pt",
    "prompt_lengths": "prompt_lengths.pt",
    "original_row_indices": "original_row_indices.pt",
}


def is_flow_window_cache(path: str | Path) -> bool:
    path = Path(path)
    return (
        path.is_dir()
        and (path / FLOW_CACHE_METADATA).exists()
        and all((path / filename).exists() for filename in FLOW_CACHE_FILES.values())
    )


def _load_teacher_source(path: str, *, split: str) -> Dataset:
    if Path(path).exists():
        print(f"loading teacher dataset from disk: {path}", flush=True)
        return load_from_disk(path)
    print(f"loading teacher dataset from hub: {path} split={split}", flush=True)
    return load_dataset(path, split=split)


def _scan_valid_rows(
    dataset: Dataset,
    *,
    draft_length: int,
) -> tuple[list[tuple[int, int, int]], list[int], list[int]]:
    print("formatting flow dataset: scanning teacher-window rows", flush=True)
    lengths = dataset["num_tokens"]
    prompt_lengths = dataset["prompt_length"] if "prompt_length" in dataset.column_names else None
    valid_rows: list[tuple[int, int, int]] = []
    row_lengths: list[int] = []
    row_prompt_lengths: list[int] = []
    iterator = enumerate(lengths)
    if tqdm is not None:
        iterator = tqdm(
            iterator,
            total=len(lengths),
            desc="scanning flow rows",
            unit="row",
        )
    for row_idx, length_value in iterator:
        length = int(length_value)
        prompt_length = int(prompt_lengths[row_idx]) if prompt_lengths is not None else 1
        min_t = max(0, prompt_length - 1)
        max_t = length - draft_length - 1
        if max_t >= min_t:
            valid_rows.append((row_idx, min_t, max_t))
            row_lengths.append(length)
            row_prompt_lengths.append(prompt_length)
    if not valid_rows:
        raise ValueError("dataset contains no response-side rows long enough for the requested draft_length")
    return valid_rows, row_lengths, row_prompt_lengths


def _torch_dtype_from_cache_string(dtype: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    normalized = dtype.lower()
    if normalized not in mapping:
        raise ValueError(f"unsupported hidden_dtype: {dtype}")
    return mapping[normalized]


def build_flow_window_cache(
    dataset_path: str,
    output_dir: str | Path,
    *,
    split: str = "train",
    draft_length: int = 2,
    hidden_dtype: str = "float32",
    overwrite: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"flow cache output dir is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = _load_teacher_source(dataset_path, split=split)
    print(f"teacher dataset loaded: rows={len(dataset)} columns={dataset.column_names}", flush=True)
    valid_rows, row_lengths, row_prompt_lengths = _scan_valid_rows(dataset, draft_length=draft_length)
    input_ids_column = dataset["input_ids"]
    hidden_column = dataset["final_hidden"]

    first_row_idx = valid_rows[0][0]
    first_hidden = torch.as_tensor(hidden_column[first_row_idx])
    if first_hidden.ndim != 2:
        raise ValueError(f"final_hidden row {first_row_idx} must have shape [tokens, hidden_size]")
    hidden_size = int(first_hidden.shape[1])
    total_tokens = int(sum(row_lengths))
    cache_dtype = _torch_dtype_from_cache_string(hidden_dtype)
    print(
        f"allocating flow cache tensors: rows={len(valid_rows)} tokens={total_tokens} "
        f"hidden_size={hidden_size} hidden_dtype={hidden_dtype}",
        flush=True,
    )
    hidden = torch.empty((total_tokens, hidden_size), dtype=cache_dtype)
    input_ids = torch.empty((total_tokens,), dtype=torch.long)
    row_offsets = torch.empty((len(valid_rows),), dtype=torch.long)
    row_lengths_tensor = torch.tensor(row_lengths, dtype=torch.long)
    prompt_lengths_tensor = torch.tensor(row_prompt_lengths, dtype=torch.long)
    original_row_indices = torch.empty((len(valid_rows),), dtype=torch.long)

    offset = 0
    iterator = enumerate(valid_rows)
    if tqdm is not None:
        iterator = tqdm(
            iterator,
            total=len(valid_rows),
            desc="building flow cache",
            unit="row",
        )
    for cache_row_idx, (row_idx, _, _) in iterator:
        length = int(row_lengths[cache_row_idx])
        row_input_ids = torch.as_tensor(input_ids_column[row_idx], dtype=torch.long)
        row_hidden = torch.as_tensor(hidden_column[row_idx], dtype=cache_dtype)
        if row_hidden.ndim != 2:
            raise ValueError(f"final_hidden row {row_idx} must have shape [tokens, hidden_size]")
        if int(row_hidden.shape[0]) != length:
            raise ValueError(f"final_hidden row {row_idx} length does not match num_tokens")
        if int(row_hidden.shape[1]) != hidden_size:
            raise ValueError(f"final_hidden row {row_idx} hidden_size changed")
        if row_input_ids.ndim != 1 or int(row_input_ids.shape[0]) < length:
            raise ValueError(f"input_ids row {row_idx} must have at least num_tokens entries")
        row_offsets[cache_row_idx] = offset
        original_row_indices[cache_row_idx] = int(row_idx)
        hidden[offset : offset + length] = row_hidden.contiguous()
        input_ids[offset : offset + length] = row_input_ids[:length].contiguous()
        offset += length

    torch.save(hidden, output_dir / FLOW_CACHE_FILES["hidden"])
    torch.save(input_ids, output_dir / FLOW_CACHE_FILES["input_ids"])
    torch.save(row_offsets, output_dir / FLOW_CACHE_FILES["row_offsets"])
    torch.save(row_lengths_tensor, output_dir / FLOW_CACHE_FILES["row_lengths"])
    torch.save(prompt_lengths_tensor, output_dir / FLOW_CACHE_FILES["prompt_lengths"])
    torch.save(original_row_indices, output_dir / FLOW_CACHE_FILES["original_row_indices"])

    metadata: dict[str, Any] = {
        "cache_type": "chained_flow.flow_window_cache",
        "cache_version": FLOW_CACHE_VERSION,
        "dataset_path": dataset_path,
        "dataset_split": split,
        "draft_length": draft_length,
        "hidden_dtype": hidden_dtype,
        "hidden_size": hidden_size,
        "num_rows": len(valid_rows),
        "total_tokens": total_tokens,
        "source_rows": len(dataset),
        "columns": list(dataset.column_names),
    }
    import json

    with (output_dir / FLOW_CACHE_METADATA).open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    hidden_gib = hidden.numel() * hidden.element_size() / 1024**3
    print(
        f"flow cache saved: {output_dir} rows={len(valid_rows)} tokens={total_tokens} "
        f"hidden_gib={hidden_gib:.2f}",
        flush=True,
    )
    return metadata


@dataclass(frozen=True)
class TeacherWindow:
    context_hidden: torch.Tensor
    target_hidden: torch.Tensor
    future_tokens: torch.Tensor


class TeacherWindowDataset(TorchDataset):
    lag_context: bool = False        # set by the trainer; see build_context() in tree_vae_flow

    def __init__(
        self,
        dataset: Dataset,
        *,
        context_size: int,
        draft_length: int,
        windows_per_epoch: int | None = None,
        seed: int = 0,
        materialize_rows: bool = True,
    ):
        self.dataset = dataset
        self.context_size = context_size
        self.draft_length = draft_length
        self.seed = seed
        self.materialize_rows = materialize_rows
        self.row_tensors: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None
        self.valid_rows, _, _ = _scan_valid_rows(dataset, draft_length=draft_length)
        self.available_windows = sum(max_t - min_t + 1 for _, min_t, max_t in self.valid_rows)
        self.windows_per_epoch = windows_per_epoch or self.available_windows
        print(
            f"flow dataset formatted: valid_rows={len(self.valid_rows)} "
            f"available_windows={self.available_windows} windows_per_epoch={self.windows_per_epoch}",
            flush=True,
        )
        if materialize_rows:
            self.row_tensors = self._materialize_valid_rows(dataset)

    def _materialize_valid_rows(self, dataset: Dataset) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        print("materializing flow dataset rows: caching input_ids and final_hidden tensors", flush=True)
        input_ids_column = dataset["input_ids"]
        hidden_column = dataset["final_hidden"]
        row_tensors: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        total_tokens = 0
        total_hidden_elements = 0
        iterator = self.valid_rows
        if tqdm is not None:
            iterator = tqdm(
                iterator,
                total=len(self.valid_rows),
                desc="materializing flow rows",
                unit="row",
            )
        for row_idx, _, _ in iterator:
            input_ids = torch.as_tensor(input_ids_column[row_idx], dtype=torch.long).contiguous()
            hidden = torch.as_tensor(hidden_column[row_idx], dtype=torch.float32).contiguous()
            if hidden.ndim != 2:
                raise ValueError(f"final_hidden row {row_idx} must have shape [tokens, hidden_size]")
            if input_ids.ndim != 1:
                raise ValueError(f"input_ids row {row_idx} must have shape [tokens]")
            if input_ids.shape[0] < hidden.shape[0]:
                raise ValueError(f"input_ids row {row_idx} is shorter than final_hidden")
            row_tensors[int(row_idx)] = (input_ids, hidden)
            total_tokens += int(hidden.shape[0])
            total_hidden_elements += int(hidden.numel())
        approx_hidden_gib = total_hidden_elements * 4 / 1024**3
        print(
            f"flow rows materialized: rows={len(row_tensors)} tokens={total_tokens} "
            f"hidden_gib_float32={approx_hidden_gib:.2f}",
            flush=True,
        )
        return row_tensors

    @classmethod
    def from_path(
        cls,
        path: str,
        *,
        split: str = "train",
        context_size: int,
        draft_length: int,
        windows_per_epoch: int | None = None,
        seed: int = 0,
        materialize_rows: bool = True,
    ) -> "TeacherWindowDataset":
        if is_flow_window_cache(path):
            return FlowWindowCacheDataset.from_cache_dir(
                path,
                context_size=context_size,
                draft_length=draft_length,
                windows_per_epoch=windows_per_epoch,
                seed=seed,
            )
        if Path(path).exists():
            print(f"loading flow dataset from disk: {path}", flush=True)
            dataset = load_from_disk(path)
        else:
            print(f"loading flow dataset from hub: {path} split={split}", flush=True)
            dataset = load_dataset(path, split=split)
        print(f"flow dataset source loaded: rows={len(dataset)} columns={dataset.column_names}", flush=True)
        return cls(
            dataset,
            context_size=context_size,
            draft_length=draft_length,
            windows_per_epoch=windows_per_epoch,
            seed=seed,
            materialize_rows=materialize_rows,
        )

    @classmethod
    def from_disk(
        cls,
        path: str,
        *,
        context_size: int,
        draft_length: int,
        windows_per_epoch: int | None = None,
        seed: int = 0,
        materialize_rows: bool = True,
    ) -> "TeacherWindowDataset":
        return cls.from_path(
            path,
            context_size=context_size,
            draft_length=draft_length,
            windows_per_epoch=windows_per_epoch,
            seed=seed,
            materialize_rows=materialize_rows,
        )

    def __len__(self) -> int:
        return self.windows_per_epoch

    def _sample_row_and_t(self, index: int) -> tuple[int, int]:
        rng = random.Random(self.seed + index)
        row_idx, min_t, max_t = rng.choice(self.valid_rows)
        return int(row_idx), rng.randint(min_t, max_t)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row_idx, t = self._sample_row_and_t(index)
        if self.row_tensors is not None:
            input_ids, hidden = self.row_tensors[row_idx]
        else:
            row = self.dataset[row_idx]
            input_ids = torch.as_tensor(row["input_ids"], dtype=torch.long)
            hidden = torch.as_tensor(row["final_hidden"], dtype=torch.float32)

        context_start = t - self.context_size + 1
        if context_start >= 0:
            context_hidden = hidden[context_start : t + 1]
        else:
            pad = hidden[:1].expand(-context_start, -1)
            context_hidden = torch.cat([pad, hidden[: t + 1]], dim=0)

        target_hidden = hidden[t : t + self.draft_length]
        future_tokens = input_ids[t + 1 : t + self.draft_length + 1]

        out = {
            "context_hidden": context_hidden,
            "target_hidden": target_hidden,
            "future_tokens": future_tokens,
            "prev_token": input_ids[t],          # see FlowWindowCacheDataset: position 0's real parent
        }
        if self.lag_context:
            cs = t - self.context_size
            if cs >= 0:
                ctx = hidden[cs:t]
            else:
                pad = hidden[:1].expand(-cs, -1)
                ctx = torch.cat([pad, hidden[:t]], dim=0)
            out["context_hidden"] = ctx
            out["lag_token"] = input_ids[t]
        return out

class FlowWindowCacheDataset(TorchDataset):
    lag_context: bool = False        # set by the trainer; see build_context() in tree_vae_flow

    def __init__(
        self,
        *,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        row_offsets: torch.Tensor,
        row_lengths: torch.Tensor,
        prompt_lengths: torch.Tensor,
        metadata: dict[str, Any],
        context_size: int,
        draft_length: int,
        windows_per_epoch: int | None = None,
        seed: int = 0,
    ):
        if hidden.ndim != 2:
            raise ValueError("hidden cache must have shape [tokens, hidden_size]")
        if input_ids.ndim != 1:
            raise ValueError("input_ids cache must have shape [tokens]")
        self.hidden = hidden.contiguous()
        self.input_ids = input_ids.contiguous()
        self.row_offsets = row_offsets.long().contiguous()
        self.row_lengths = row_lengths.long().contiguous()
        self.prompt_lengths = prompt_lengths.long().contiguous()
        self.metadata = metadata
        self.context_size = context_size
        self.draft_length = draft_length
        self.seed = seed
        self.materialize_rows = False
        self.row_tensors = None
        self.dataset = None
        self.num_rows = int(self.row_offsets.numel())
        self.valid_rows: list[tuple[int, int, int]] = []
        for row_idx in range(self.num_rows):
            length = int(self.row_lengths[row_idx].item())
            prompt_length = int(self.prompt_lengths[row_idx].item())
            min_t = max(0, prompt_length - 1)
            max_t = length - draft_length - 1
            if max_t >= min_t:
                self.valid_rows.append((row_idx, min_t, max_t))
        if not self.valid_rows:
            raise ValueError("flow cache contains no rows long enough for the requested draft_length")
        self.available_windows = sum(max_t - min_t + 1 for _, min_t, max_t in self.valid_rows)
        self.windows_per_epoch = windows_per_epoch or self.available_windows
        print(
            f"flow cache dataset initialized: rows={self.num_rows} tokens={int(self.hidden.shape[0])} "
            f"hidden_size={int(self.hidden.shape[1])} valid_rows={len(self.valid_rows)} "
            f"available_windows={self.available_windows} windows_per_epoch={self.windows_per_epoch}",
            flush=True,
        )

    @classmethod
    def from_cache_dir(
        cls,
        cache_dir: str | Path,
        *,
        context_size: int,
        draft_length: int,
        windows_per_epoch: int | None = None,
        seed: int = 0,
    ) -> "FlowWindowCacheDataset":
        import json

        cache_dir = Path(cache_dir)
        print(f"loading flow window cache: {cache_dir}", flush=True)
        with (cache_dir / FLOW_CACHE_METADATA).open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        # mmap the big one. `hidden.pt` is the whole cache -- 186 GB for the 27B Turkish v2 mix --
        # and a plain torch.load gives EVERY DDP rank its own private copy: 4 ranks x 192 GB = 768 GB
        # of host RAM, which is what takes a machine down before the first step. Memory-mapping
        # leaves a single copy in the page cache that all ranks share, and pages fault in on
        # access, so a random-window reader touches only what it reads.
        #
        # CF_CACHE_MMAP=0 restores the eager load. Only worth doing if the cache is on storage too
        # slow to fault against -- on local NVMe it is strictly better.
        _mmap = os.environ.get("CF_CACHE_MMAP", "1") not in ("0", "false", "False", "no")
        hidden = torch.load(cache_dir / FLOW_CACHE_FILES["hidden"], map_location="cpu", mmap=_mmap)
        input_ids = torch.load(cache_dir / FLOW_CACHE_FILES["input_ids"], map_location="cpu")
        row_offsets = torch.load(cache_dir / FLOW_CACHE_FILES["row_offsets"], map_location="cpu")
        row_lengths = torch.load(cache_dir / FLOW_CACHE_FILES["row_lengths"], map_location="cpu")
        prompt_lengths = torch.load(cache_dir / FLOW_CACHE_FILES["prompt_lengths"], map_location="cpu")
        hidden_gib = hidden.numel() * hidden.element_size() / 1024**3
        print(
            f"flow window cache loaded: rows={int(row_offsets.numel())} tokens={int(hidden.shape[0])} "
            f"hidden_dtype={hidden.dtype} hidden_gib={hidden_gib:.2f}",
            flush=True,
        )
        return cls(
            hidden=hidden,
            input_ids=input_ids,
            row_offsets=row_offsets,
            row_lengths=row_lengths,
            prompt_lengths=prompt_lengths,
            metadata=metadata,
            context_size=context_size,
            draft_length=draft_length,
            windows_per_epoch=windows_per_epoch,
            seed=seed,
        )

    def __len__(self) -> int:
        return self.windows_per_epoch

    def _sample_row_and_t(self, index: int) -> tuple[int, int]:
        rng = random.Random(self.seed + index)
        row_idx, min_t, max_t = rng.choice(self.valid_rows)
        return int(row_idx), rng.randint(min_t, max_t)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row_idx, t = self._sample_row_and_t(index)
        offset = int(self.row_offsets[row_idx].item())
        length = int(self.row_lengths[row_idx].item())
        row_hidden = self.hidden[offset : offset + length]
        row_input_ids = self.input_ids[offset : offset + length]

        context_start = t - self.context_size + 1
        if context_start >= 0:
            context_hidden = row_hidden[context_start : t + 1]
        else:
            pad = row_hidden[:1].expand(-context_start, -1)
            context_hidden = torch.cat([pad, row_hidden[: t + 1]], dim=0)

        target_hidden = row_hidden[t : t + self.draft_length].float()
        future_tokens = row_input_ids[t + 1 : t + self.draft_length + 1]
        out = {
            "context_hidden": context_hidden.float(),
            "target_hidden": target_hidden,
            "future_tokens": future_tokens,
            # the LAST COMMITTED token: position 0's real parent. _parent_stack masks position 0 out
            # entirely because it only sees future_tokens, so training zeroed the path residual and
            # markov bias there while inference feeds them from `known0` -- a train/inference
            # mismatch at the single highest-leverage slot (accept is a prefix).
            "prev_token": row_input_ids[t],
        }
        if self.lag_context:
            # Shift the context back one so it ends at h(t-1) -- what vLLM can actually supply --
            # and hand over token t separately. Targets are UNCHANGED, so accept stays comparable.
            cs = t - self.context_size
            if cs >= 0:
                ctx = row_hidden[cs:t]
            else:
                pad = row_hidden[:1].expand(-cs, -1)
                ctx = torch.cat([pad, row_hidden[:t]], dim=0)
            out["context_hidden"] = ctx.float()
            out["lag_token"] = row_input_ids[t]
        return out

