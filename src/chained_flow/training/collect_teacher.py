from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence as TypingSequence

from datasets import Dataset, Features, Sequence, Value, load_dataset, load_from_disk
import torch
from tqdm.auto import tqdm

from chained_flow.context import ChainedFlowContext
from chained_flow.frozen_lm import DEFAULT_MODEL_ID
from chained_flow.timing import TimingStats, timed_section


@dataclass(frozen=True)
class TeacherCollectionConfig:
    model_id: str = DEFAULT_MODEL_ID
    dataset_name: str = "gsm8k"
    dataset_config: str = "main"
    split: str = "train"
    source: str = "gsm8k"
    format_name: str = "qwen_chat_qa"
    limit: int | None = None
    dataset_start: int = 0
    dataset_end: int | None = None
    streaming: bool = False
    stream_take: int | None = None
    data_files: str | None = None
    target_layer_ids: list[int] | None = None  # if set, store concat of these backbone layers as final_hidden
    max_tokens: int | None = None
    generation_max_new_tokens: int = 256
    batch_size: int = 1
    generation_batch_size: int | None = None
    hidden_batch_size: int | None = None
    storage_dtype: str = "float32"
    local_files_only: bool = False
    device: str | None = None
    dtype: str | None = None
    seed: int = 0
    tmp_output_dir: str | None = None
    tmp_push_to_hub: str | None = None
    push_to_hub: str | None = None
    answer_dataset_path: str | None = None
    answer_dataset_split: str | None = None
    private: bool = False


def _hidden_feature_dtype(storage_dtype: str) -> str:
    if storage_dtype == "float16":
        return "float16"
    if storage_dtype == "float32":
        return "float32"
    raise ValueError("HF dataset hidden storage supports float32 or float16")


def teacher_dataset_features(storage_dtype: str = "float32") -> Features:
    hidden_dtype = _hidden_feature_dtype(storage_dtype)
    return Features(
        {
            "text": Value("string"),
            "prompt_text": Value("string"),
            "generated_text": Value("string"),
            "input_ids": Sequence(Value("int32")),
            "final_hidden": Sequence(Sequence(Value(hidden_dtype))),
            "example_id": Value("string"),
            "source": Value("string"),
            "split": Value("string"),
            "format_name": Value("string"),
            "model_id": Value("string"),
            "hidden_dtype": Value("string"),
            "num_tokens": Value("int32"),
            "prompt_length": Value("int32"),
        }
    )


def teacher_answer_dataset_features() -> Features:
    return Features(
        {
            "text": Value("string"),
            "prompt_text": Value("string"),
            "generated_text": Value("string"),
            "input_ids": Sequence(Value("int32")),
            "example_id": Value("string"),
            "source": Value("string"),
            "split": Value("string"),
            "format_name": Value("string"),
            "model_id": Value("string"),
            "hidden_dtype": Value("string"),
            "num_tokens": Value("int32"),
            "prompt_length": Value("int32"),
        }
    )


def format_gsm8k_prompt(example: dict[str, Any], tokenizer: Any) -> str:
    question = example["question"].strip()
    messages = [{"role": "user", "content": question}]
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"Question:\n{question}\n\nAnswer:\n"


def format_nemotron_prompt(example: dict[str, Any], tokenizer: Any) -> str | None:
    """Build a prompt from a Nemotron `messages` row.

    Uses only the leading system/user turns before the first assistant turn (the dataset's
    assistant answers are from other/larger models, so we discard them and let the teacher
    generate its own). Returns None for degenerate prompts so the caller can skip them.
    """
    messages = example.get("messages") or []
    prompt_messages: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            break
        if role in ("system", "user"):
            prompt_messages.append({"role": role, "content": (message.get("content") or "").strip()})
    user_text = " ".join(m["content"] for m in prompt_messages if m["role"] == "user").strip()
    if len(user_text) < 8:
        return None
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    return user_text + "\n"


def _apply_user_chat(tokenizer: Any, content: str | None) -> str | None:
    content = (content or "").strip()
    if len(content) < 8:
        return None
    messages = [{"role": "user", "content": content}]
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{content}\n"


def format_pretemplated(example: dict[str, Any], tokenizer: Any) -> str | None:
    """Pass a prompt that is ALREADY chat-templated through verbatim.

    For corpora whose rows carry a full `<|im_start|>...<|im_start|>assistant\\n` prefix
    (multi-turn, or a system turn carrying tool definitions), re-wrapping the raw text as a
    single user message -- what `prompt_chat` does -- would flatten the roles and change the
    prompt distribution. This handler keeps the prefix exactly as stored. Only valid when the
    prefix was produced by the SAME chat template this tokenizer applies.
    """
    # No stripping: the generation prefix legitimately ends in whitespace
    # (`<think>\n\n</think>\n\n` in Qwen3.5 non-thinking mode), and trimming it would
    # move the first generated token off-distribution.
    prompt = example.get("prompt") or ""
    return prompt if len(prompt.strip()) >= 8 else None


def format_mbpp_prompt(example: dict[str, Any], tokenizer: Any) -> str | None:
    return _apply_user_chat(tokenizer, example.get("prompt"))


def format_dolly_prompt(example: dict[str, Any], tokenizer: Any) -> str | None:
    instruction = (example.get("instruction") or "").strip()
    context = (example.get("context") or "").strip()
    content = f"{instruction}\n\n{context}".strip() if context else instruction
    return _apply_user_chat(tokenizer, content)


def format_alpaca_prompt(example: dict[str, Any], tokenizer: Any) -> str | None:
    instruction = (example.get("instruction") or "").strip()
    extra = (example.get("input") or "").strip()
    content = f"{instruction}\n\n{extra}".strip() if extra else instruction
    return _apply_user_chat(tokenizer, content)


def format_prompt_chat(example: dict[str, Any], tokenizer: Any) -> str | None:
    """Generic: chat-wrap a single `prompt` field (e.g. eval benchmarks)."""
    return _apply_user_chat(tokenizer, example.get("prompt"))


def format_summarize_article(example: dict[str, Any], tokenizer: Any) -> str | None:
    """Summarization: wrap an `article` (or `document`) field in a summarize instruction."""
    article = (example.get("article") or example.get("document") or "").strip()
    if len(article) < 64:
        return None
    article = article[:6000]  # cap very long articles to keep prompts in-budget
    return _apply_user_chat(tokenizer, f"Summarize the following article in a few sentences:\n\n{article}")


def format_writingprompt(example: dict[str, Any], tokenizer: Any) -> str | None:
    """Creative prose: strip the leading [WP]/[EU]/[TT] tag and ask for a short story."""
    import re

    p = (example.get("prompt") or "").strip()
    p = re.sub(r"^\[\s*[A-Z]{2,3}\s*\]\s*", "", p).strip()  # drop [ WP ] / [EU] / [TT] tags
    if len(p) < 12:
        return None
    return _apply_user_chat(tokenizer, f"Write a short story based on this prompt:\n\n{p}")


def format_translate_en_fr(example: dict[str, Any], tokenizer: Any) -> str | None:
    """Translation: build an English->French instruction from a `translation` {en, fr} row."""
    tr = example.get("translation") or {}
    en = (tr.get("en") or "").strip()
    if len(en) < 12:
        return None
    en = en[:2000]
    return _apply_user_chat(tokenizer, f"Translate the following English text to French:\n\n{en}")


FORMAT_HANDLERS = {
    "qwen_chat_qa": format_gsm8k_prompt,
    # alias: the Turkish 9B collect configs were written against this name before
    # `pretemplated` landed. Same behaviour -- one implementation, two names.
    "raw_prompt": format_pretemplated,
    "nemotron_messages": format_nemotron_prompt,
    "mbpp": format_mbpp_prompt,
    "dolly": format_dolly_prompt,
    "alpaca": format_alpaca_prompt,
    "prompt_chat": format_prompt_chat,
    "pretemplated": format_pretemplated,
    "summarize_article": format_summarize_article,
    "writingprompt": format_writingprompt,
    "translate_en_fr": format_translate_en_fr,
}


def select_formatter(format_name: str):
    handler = FORMAT_HANDLERS.get(format_name)
    if handler is None:
        raise ValueError(f"unknown format_name={format_name!r}; known: {sorted(FORMAT_HANDLERS)}")
    return handler


def _storage_torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    raise ValueError("storage_dtype must be one of: float32, float16")


def _model_torch_dtype(name: str | None) -> torch.dtype | None:
    if name is None:
        return None
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    raise ValueError("dtype must be one of: float32, float16")


def _backbone(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "base_model"):
        return model.base_model
    raise AttributeError("could not find a backbone module on the causal LM")


def _resolve_range(total_rows: int, *, start: int, end: int | None, limit: int | None) -> tuple[int, int]:
    if start < 0:
        raise ValueError("dataset_start must be non-negative")
    if end is not None and end < start:
        raise ValueError("dataset_end must be greater than or equal to dataset_start")
    resolved_end = end
    if resolved_end is None and limit is not None:
        resolved_end = start + limit
    if resolved_end is None:
        resolved_end = total_rows
    return min(start, total_rows), min(resolved_end, total_rows)


def _iter_range(dataset: TypingSequence[dict[str, Any]], start: int, end: int) -> Iterable[tuple[int, dict[str, Any]]]:
    for index in range(start, end):
        yield index, dataset[index]


def _batched(items: Iterable[tuple[int, dict[str, Any]]], batch_size: int) -> Iterable[list[tuple[int, dict[str, Any]]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batch: list[tuple[int, dict[str, Any]]] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _effective_generation_batch_size(config: TeacherCollectionConfig) -> int:
    return config.generation_batch_size or config.batch_size


def _effective_hidden_batch_size(config: TeacherCollectionConfig) -> int:
    return config.hidden_batch_size or config.batch_size


def _ensure_padding(tokenizer: Any, eos_token_id: int | None) -> int:
    tokenizer.padding_side = "left"
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return int(pad_token_id)
    if eos_token_id is None:
        raise ValueError("tokenizer has no pad_token_id and model has no eos_token_id")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = eos_token_id
    return int(eos_token_id)


def _eos_token_ids(tokenizer: Any, eos_token_id: int | list[int] | None) -> list[int]:
    ids: list[int] = []
    if isinstance(eos_token_id, int):
        ids.append(eos_token_id)
    elif isinstance(eos_token_id, list):
        ids.extend(int(item) for item in eos_token_id)

    for token in ("<|im_end|>", "<|endoftext|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            ids.append(token_id)

    return sorted(set(ids))


def _sequence_spans(
    input_ids: torch.Tensor,
    pad_token_id: int,
    eos_token_id: int | list[int] | None,
    prompt_lengths: TypingSequence[int],
) -> list[tuple[int, int]]:
    eos_ids = set(eos_token_id if isinstance(eos_token_id, list) else ([] if eos_token_id is None else [eos_token_id]))
    spans: list[tuple[int, int]] = []
    for row, prompt_length in zip(input_ids, prompt_lengths, strict=True):
        nonpad = (row != pad_token_id).nonzero(as_tuple=False)
        if not len(nonpad):
            spans.append((0, 0))
            continue
        start = int(nonpad[0].item())
        end = int(nonpad[-1].item() + 1)
        prompt_end = min(start + int(prompt_length), end)
        if eos_ids:
            eos_positions = torch.tensor(
                [offset for offset, token_id in enumerate(row[prompt_end:end].tolist()) if token_id in eos_ids],
                device=row.device,
            )
            if len(eos_positions):
                end = prompt_end + int(eos_positions[0].item() + 1)
        spans.append((start, end))
    return spans


def _left_pad_sequences(
    sequences: TypingSequence[torch.Tensor],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(seq.shape[0] for seq in sequences)
    input_ids = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long, device=device)
    for row_idx, seq in enumerate(sequences):
        seq = seq.to(device)
        input_ids[row_idx, -seq.shape[0] :] = seq
        attention_mask[row_idx, -seq.shape[0] :] = 1
    return input_ids, attention_mask


def _load_answer_dataset(path_or_repo: str, split: str | None) -> Dataset:
    path = Path(path_or_repo)
    if path.exists():
        return load_from_disk(str(path))
    return load_dataset(path_or_repo, split=split or "train")


def _answer_row_from_dataset_row(row: dict[str, Any], *, storage_dtype: str) -> dict[str, Any]:
    input_ids = [int(token_id) for token_id in row["input_ids"]]
    return {
        "text": row["text"],
        "prompt_text": row["prompt_text"],
        "generated_text": row["generated_text"],
        "input_ids": input_ids,
        "example_id": str(row["example_id"]),
        "source": row["source"],
        "split": row["split"],
        "format_name": row["format_name"],
        "model_id": row["model_id"],
        "hidden_dtype": storage_dtype,
        "num_tokens": int(row.get("num_tokens", len(input_ids))),
        "prompt_length": int(row["prompt_length"]),
    }


def _ranged_answer_rows(
    answer_dataset: Dataset,
    *,
    start: int,
    end: int,
    storage_dtype: str,
) -> list[dict[str, Any]]:
    return [
        _answer_row_from_dataset_row(answer_dataset[index], storage_dtype=storage_dtype)
        for index in range(start, end)
    ]


def _push_teacher_dataset_if_requested(
    dataset: Dataset,
    config: TeacherCollectionConfig,
    timings: TimingStats,
    device: torch.device,
) -> None:
    if config.push_to_hub:
        print(f"pushing teacher dataset: {config.push_to_hub}", flush=True)
        with timed_section(timings, "dataset_push_to_hub", device):
            dataset.push_to_hub(config.push_to_hub, private=config.private)


@torch.inference_mode()
def collect_teacher_dataset(config: TeacherCollectionConfig) -> tuple[Dataset, TimingStats]:
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    timings = TimingStats()
    print(f"loading model: {config.model_id}", flush=True)
    context = ChainedFlowContext.from_pretrained(
        config.model_id,
        device=config.device,
        dtype=_model_torch_dtype(config.dtype),
        local_files_only=config.local_files_only,
    )
    timings.merge(context.timings)
    wrapper = context.frozen_lm
    storage_dtype = _storage_torch_dtype(config.storage_dtype)
    pad_token_id = _ensure_padding(wrapper.tokenizer, wrapper.eos_token_id)
    eos_token_ids = _eos_token_ids(wrapper.tokenizer, wrapper.eos_token_id)
    print(f"model loaded: {config.model_id}", flush=True)
    print(f"model device: {wrapper.device}", flush=True)

    answer_rows: list[dict[str, Any]] = []
    with timed_section(timings, "teacher_collection", wrapper.device):
        if config.answer_dataset_path:
            print(
                f"loading answer dataset: {config.answer_dataset_path} split={config.answer_dataset_split or 'train'}",
                flush=True,
            )
            with timed_section(timings, "answer_dataset_load", wrapper.device):
                answer_dataset = _load_answer_dataset(config.answer_dataset_path, config.answer_dataset_split)
            print(f"answer dataset loaded: rows={len(answer_dataset)}", flush=True)
            range_start, range_end = _resolve_range(
                len(answer_dataset),
                start=config.dataset_start,
                end=config.dataset_end,
                limit=config.limit,
            )
            print(f"answer dataset range: [{range_start}:{range_end}]", flush=True)
            with timed_section(timings, "answer_dataset_rows", wrapper.device):
                answer_rows = _ranged_answer_rows(
                    answer_dataset,
                    start=range_start,
                    end=range_end,
                    storage_dtype=config.storage_dtype,
                )
        else:
            print(
                f"loading dataset: {config.dataset_name}/{config.dataset_config} split={config.split}",
                flush=True,
            )
            with timed_section(timings, "dataset_load", wrapper.device):
                if config.data_files:
                    raw_dataset = load_dataset(
                        config.dataset_name,
                        data_files=config.data_files,
                        split=config.split,
                    )
                elif config.streaming:
                    import itertools

                    take = config.stream_take or config.dataset_end or config.limit
                    if take is None:
                        raise ValueError("streaming collection requires stream_take, dataset_end, or limit")
                    stream = load_dataset(
                        config.dataset_name,
                        config.dataset_config,
                        split=config.split,
                        streaming=True,
                    )
                    raw_dataset = list(itertools.islice(stream, int(take)))
                else:
                    raw_dataset = load_dataset(
                        config.dataset_name,
                        config.dataset_config,
                        split=config.split,
                    )
            print(f"dataset loaded: rows={len(raw_dataset)}", flush=True)
            range_start, range_end = _resolve_range(
                len(raw_dataset),
                start=config.dataset_start,
                end=config.dataset_end,
                limit=config.limit,
            )
            print(f"dataset range: [{range_start}:{range_end}]", flush=True)

            total = range_end - range_start
            generation_batch_size = _effective_generation_batch_size(config)
            batches = list(_batched(_iter_range(raw_dataset, range_start, range_end), generation_batch_size))
            formatter = select_formatter(config.format_name)
            generation_bar = tqdm(total=total, desc="phase 1/2 generating answers")
            with timed_section(timings, "teacher_generation", wrapper.device):
                for batch in batches:
                    batch_indices: list[int] = []
                    batch_examples: list[dict[str, Any]] = []
                    prompt_texts: list[str] = []
                    for index, example in batch:
                        prompt = formatter(example, wrapper.tokenizer)
                        if prompt is None:
                            continue
                        batch_indices.append(index)
                        batch_examples.append(example)
                        prompt_texts.append(prompt)
                    if not prompt_texts:
                        generation_bar.update(len(batch))
                        continue
                    encoded = wrapper.tokenizer(
                        prompt_texts,
                        return_tensors="pt",
                        padding=True,
                    )
                    prompt_ids = encoded.input_ids.to(wrapper.device)
                    prompt_attention_mask = encoded.attention_mask.to(wrapper.device)
                    prompt_lengths = prompt_attention_mask.sum(dim=1).tolist()
                    generated_ids = wrapper.model.generate(
                        input_ids=prompt_ids,
                        attention_mask=prompt_attention_mask,
                        max_new_tokens=config.generation_max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                        pad_token_id=pad_token_id,
                        eos_token_id=eos_token_ids,
                    )
                    generation_bar.update(len(batch))
                    input_ids = generated_ids.to(wrapper.device)
                    if config.max_tokens is not None:
                        input_ids = input_ids[:, : config.max_tokens]
                    spans = _sequence_spans(input_ids, pad_token_id, eos_token_ids, prompt_lengths)
                    for row_idx, (start, end) in enumerate(spans):
                        input_row = input_ids[row_idx, start:end]
                        if input_row.shape[0] < 2:
                            continue
                        prompt_length = min(int(prompt_lengths[row_idx]), input_row.shape[0])
                        text = wrapper.decode(input_row, skip_special_tokens=False)
                        prompt_text = wrapper.decode(input_row[:prompt_length], skip_special_tokens=False)
                        generated_text = wrapper.decode(input_row[prompt_length:], skip_special_tokens=False)
                        answer_rows.append(
                            {
                                "text": text,
                                "prompt_text": prompt_text,
                                "generated_text": generated_text,
                                "input_ids": input_row.detach().cpu().to(torch.int32).tolist(),
                                "example_id": str(batch_examples[row_idx].get("id", batch_indices[row_idx])),
                                "source": config.source,
                                "split": config.split,
                                "format_name": config.format_name,
                                "model_id": config.model_id,
                                "hidden_dtype": config.storage_dtype,
                                "num_tokens": int(input_row.shape[0]),
                                "prompt_length": int(prompt_length),
                            }
                        )
            generation_bar.close()

            with timed_section(timings, "tmp_answer_dataset_build", wrapper.device):
                answer_dataset = Dataset.from_list(answer_rows, features=teacher_answer_dataset_features())
            if config.tmp_output_dir:
                print(f"saving temporary answer dataset: {config.tmp_output_dir}", flush=True)
                answer_dataset.save_to_disk(config.tmp_output_dir)
            if config.tmp_push_to_hub:
                print(f"pushing temporary answer dataset: {config.tmp_push_to_hub}", flush=True)
                answer_dataset.push_to_hub(config.tmp_push_to_hub, private=config.private)

        rows: list[dict[str, Any]] = []
        hidden_bar = tqdm(total=len(answer_rows), desc="phase 2/2 extracting hidden states")
        with timed_section(timings, "teacher_hidden_extraction", wrapper.device):
            hidden_batch_size = _effective_hidden_batch_size(config)
            for batch in _batched(list(enumerate(answer_rows)), hidden_batch_size):
                batch_rows = [row for _, row in batch]
                sequences = [
                    torch.tensor(row["input_ids"], dtype=torch.long, device=wrapper.device)
                    for row in batch_rows
                ]
                padded_ids, attention_mask = _left_pad_sequences(
                    sequences,
                    pad_token_id=pad_token_id,
                    device=wrapper.device,
                )
                with timed_section(timings, "hidden_extraction", wrapper.device):
                    outputs = _backbone(wrapper.model)(
                        input_ids=padded_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                if config.target_layer_ids:
                    # DFlash-style multi-layer conditioning: concat selected layers along feature dim.
                    # hidden_states has len = num_layers + 1 (index 0 = embeddings); +1 offset to skip it.
                    final_hidden = torch.cat(
                        [outputs.hidden_states[lid + 1] for lid in config.target_layer_ids], dim=-1
                    )
                else:
                    final_hidden = outputs.hidden_states[-1]
                for row_idx, row in enumerate(batch_rows):
                    length = int(row["num_tokens"])
                    left_pad = padded_ids.shape[1] - length
                    hidden = final_hidden[row_idx, left_pad : left_pad + length]
                    rows.append(
                        {
                            **row,
                            "final_hidden": hidden.detach().cpu().to(storage_dtype).tolist(),
                        }
                    )
                hidden_bar.update(len(batch_rows))
        hidden_bar.close()

    with timed_section(timings, "dataset_build"):
        dataset = Dataset.from_list(rows, features=teacher_dataset_features(config.storage_dtype))
    _push_teacher_dataset_if_requested(dataset, config, timings, wrapper.device)
    return dataset, timings
