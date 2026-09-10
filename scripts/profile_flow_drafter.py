from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chain_flow.context import ChainedFlowContext
from chain_flow.frozen_lm import DEFAULT_MODEL_ID
from chain_flow.generation import generate_with_drafter
from chain_flow.timing import synchronize_if_needed
from chain_flow.training.collators import collate_teacher_windows
from chain_flow.training.eval_chunked_flow import (
    collect_speedup_prompts,
    find_flow_config_dir,
    generate_greedy_baseline,
    load_flow_training_module,
    torch_dtype_from_string,
)
from chain_flow.training.train_chunked_flow import ChunkedFlowModelArguments
from chain_flow.training.window_dataset import TeacherWindowDataset


SECTION_NAMES = [
    "vae.encode_context",
    "noise_init",
    "flow.integrate",
    "vae.decode",
    "lm_head",
    "argmax",
]

FLOW_SECTION_NAMES = [
    "flow.dtype_setup",
    "flow.tau",
    "flow.query",
    "flow.kv",
    "flow.self_attn",
    "flow.cross_attn",
    "flow.ffn",
    "flow.out_proj",
    "flow.concat_velocity",
    "flow.euler_update",
]

BACKBONE_SECTION_NAMES = [
    "backbone.forward_k_tokens",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the flow drafter pipeline on sampled teacher windows.")
    parser.add_argument("--flow_dir", required=True, help="Flow run or checkpoint directory to profile.")
    parser.add_argument("--dataset_path", default="data/flow_cache/gsm8k_1k_test")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--num_examples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--warmup_batches", type=int, default=2)
    parser.add_argument(
        "--profile_flow_ops",
        action="store_true",
        help="Break flow.integrate into expert-level operations. This only changes profiler instrumentation.",
    )
    parser.add_argument(
        "--profile_backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also time a regular Qwen backbone forward over the same K-token span.",
    )
    parser.add_argument(
        "--profile_generation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also time end-to-end greedy backbone generation versus drafter+verifier generation.",
    )
    parser.add_argument(
        "--generation_num_prompts",
        type=int,
        default=16,
        help="Number of prompts for end-to-end generation timing.",
    )
    parser.add_argument(
        "--generation_warmup_prompts",
        type=int,
        default=2,
        help="Warmup prompts for end-to-end generation timing.",
    )
    parser.add_argument(
        "--generation_max_new_tokens",
        type=int,
        default=32,
        help="New tokens per prompt for end-to-end generation timing.",
    )
    parser.add_argument("--window_seed", type=int, default=0)
    parser.add_argument("--noise_seed", type=int, default=0)
    parser.add_argument("--materialize_rows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_path", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def load_flow_model_args(flow_dir: str | Path) -> ChunkedFlowModelArguments:
    config_dir = find_flow_config_dir(flow_dir)
    with (config_dir / "chain_flow_chunked_flow_config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    return ChunkedFlowModelArguments(**config["model_args"])


def time_section(device: torch.device, fn) -> tuple[Any, float]:
    synchronize_if_needed(device)
    start = time.perf_counter()
    value = fn()
    synchronize_if_needed(device)
    return value, time.perf_counter() - start


def add_time(stats: dict[str, float], name: str, seconds: float) -> None:
    stats[name] = stats.get(name, 0.0) + seconds


def slice_batch(batch: dict[str, torch.Tensor], count: int) -> dict[str, torch.Tensor]:
    return {key: value[:count] for key, value in batch.items()}


def profile_integrate_latents(*, drafter, context_latents: torch.Tensor, z0: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    timings: dict[str, float] = {}

    def dtype_setup() -> tuple[torch.Tensor, torch.Tensor]:
        context = context_latents.to(device=drafter.frozen_lm.device, dtype=drafter._expert_dtype())
        z_init = z0.to(device=context.device, dtype=context.dtype)
        return context, z_init

    (context_latents, z), seconds = time_section(device, dtype_setup)
    add_time(timings, "flow.dtype_setup", seconds)

    if drafter.config.architecture == "hidden_kv_flow":
        return profile_hidden_kv_integrate_latents(
            drafter=drafter,
            context_latents=context_latents,
            z=z,
            device=device,
            timings=timings,
        )

    if drafter.config.architecture == "block_causal_experts":
        return profile_block_causal_integrate_latents(
            drafter=drafter,
            context_latents=context_latents,
            z=z,
            device=device,
            timings=timings,
        )

    return profile_independent_integrate_latents(
        drafter=drafter,
        context_latents=context_latents,
        z=z,
        device=device,
        timings=timings,
    )


def profile_hidden_kv_integrate_latents(
    *,
    drafter,
    context_latents: torch.Tensor,
    z: torch.Tensor,
    device: torch.device,
    timings: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    steps = drafter.config.num_flow_steps
    dt = 1.0 / steps
    batch = context_latents.shape[0]

    for step in range(steps):
        tau, seconds = time_section(
            device,
            lambda: torch.full((batch,), step * dt, device=context_latents.device, dtype=context_latents.dtype),
        )
        add_time(timings, "flow.tau", seconds)

        velocities = []
        for chunk_idx, expert in enumerate(drafter._chunk_experts()):
            start = chunk_idx * drafter.config.chunk_size
            end = start + drafter.config.chunk_size
            previous_hidden = z[:, :start, :]
            if drafter.config.detach_previous_chunks:
                previous_hidden = previous_hidden.detach()
            current_h_tau = z[:, start:end, :]

            velocity, seconds = time_section(
                device,
                lambda expert=expert, previous_hidden=previous_hidden, current_h_tau=current_h_tau, start=start: expert(
                    context_hidden=context_latents,
                    previous_hidden=previous_hidden,
                    current_h_tau=current_h_tau,
                    tau=tau,
                    chunk_start=start,
                ),
            )
            add_time(timings, f"flow.chunk{chunk_idx}.hidden_kv_expert", seconds)
            velocities.append(velocity)

        full_velocity, seconds = time_section(device, lambda: torch.cat(velocities, dim=1))
        add_time(timings, "flow.concat_velocity", seconds)

        z, seconds = time_section(device, lambda: z + dt * full_velocity)
        add_time(timings, "flow.euler_update", seconds)

    return z, timings


def profile_block_causal_integrate_latents(
    *,
    drafter,
    context_latents: torch.Tensor,
    z: torch.Tensor,
    device: torch.device,
    timings: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    steps = drafter.config.num_flow_steps
    dt = 1.0 / steps
    batch = context_latents.shape[0]

    for step in range(steps):
        tau, seconds = time_section(
            device,
            lambda: torch.full((batch,), step * dt, device=context_latents.device, dtype=context_latents.dtype),
        )
        add_time(timings, "flow.tau", seconds)

        velocities = []
        for chunk_idx, expert in enumerate(drafter._chunk_experts()):
            start = chunk_idx * drafter.config.chunk_size
            end = start + drafter.config.chunk_size
            previous_latents = z[:, :start, :]
            if drafter.config.detach_previous_chunks:
                previous_latents = previous_latents.detach()
            current_z_tau = z[:, start:end, :]

            velocity, seconds = time_section(
                device,
                lambda expert=expert, previous_latents=previous_latents, current_z_tau=current_z_tau, start=start: expert(
                    context_latents=context_latents,
                    previous_latents=previous_latents,
                    current_z_tau=current_z_tau,
                    tau=tau,
                    chunk_start=start,
                ),
            )
            add_time(timings, f"flow.chunk{chunk_idx}.fused_expert", seconds)
            velocities.append(velocity)

        full_velocity, seconds = time_section(device, lambda: torch.cat(velocities, dim=1))
        add_time(timings, "flow.concat_velocity", seconds)

        z, seconds = time_section(device, lambda: z + dt * full_velocity)
        add_time(timings, "flow.euler_update", seconds)

    return z, timings


def profile_independent_integrate_latents(
    *,
    drafter,
    context_latents: torch.Tensor,
    z: torch.Tensor,
    device: torch.device,
    timings: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    expert = drafter.expert
    steps = drafter.config.num_flow_steps
    dt = 1.0 / steps
    batch = context_latents.shape[0]
    slot_ids = torch.arange(expert.chunk_size, device=context_latents.device)

    for step in range(steps):
        tau, seconds = time_section(
            device,
            lambda: torch.full((batch,), step * dt, device=context_latents.device, dtype=context_latents.dtype),
        )
        add_time(timings, "flow.tau", seconds)

        def query_block() -> torch.Tensor:
            tau_2d = tau[:, None]
            q = expert.query_proj(z)
            q = q + expert.time_mlp(tau_2d).unsqueeze(1)
            q = q + expert.slot_embedding(slot_ids).unsqueeze(0)
            return expert.query_norm(q)

        q, seconds = time_section(device, query_block)
        add_time(timings, "flow.query", seconds)

        def kv_block() -> tuple[torch.Tensor, torch.Tensor]:
            k = expert.context_norm(expert.key_proj(context_latents))
            v = expert.context_norm(expert.value_proj(context_latents))
            return k, v

        (k, v), seconds = time_section(device, kv_block)
        add_time(timings, "flow.kv", seconds)

        self_out, seconds = time_section(device, lambda: expert.self_attn(q, q, q, need_weights=False)[0])
        add_time(timings, "flow.self_attn", seconds)
        q = q + self_out

        cross_out, seconds = time_section(device, lambda: expert.cross_attn(q, k, v, need_weights=False)[0])
        add_time(timings, "flow.cross_attn", seconds)
        q = q + cross_out

        ffn_out, seconds = time_section(device, lambda: expert.ffn(q))
        add_time(timings, "flow.ffn", seconds)
        q = q + ffn_out

        velocity, seconds = time_section(device, lambda: expert.out_proj(expert.out_norm(q)))
        add_time(timings, "flow.out_proj", seconds)

        z, seconds = time_section(device, lambda: z + dt * velocity)
        add_time(timings, "flow.euler_update", seconds)

    return z, timings

def profile_batch(
    *,
    module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    generator: torch.Generator,
    profile_flow_ops: bool = False,
) -> tuple[dict[str, float], dict[str, float]]:
    timings: dict[str, float] = {}
    flow_timings: dict[str, float] = {}
    context_hidden = batch["context_hidden"].to(device)

    z_ctx, seconds = time_section(device, lambda: module.drafter.encode_hidden(context_hidden))
    add_time(timings, "vae.encode_context", seconds)

    z0, seconds = time_section(device, lambda: module.drafter.init_latents(z_ctx, generator=generator))
    add_time(timings, "noise_init", seconds)

    if profile_flow_ops:
        pred_latent, flow_timings = profile_integrate_latents(
            drafter=module.drafter,
            context_latents=z_ctx,
            z0=z0,
            device=device,
        )
        add_time(timings, "flow.integrate", sum(flow_timings.values()))
    else:
        pred_latent, seconds = time_section(device, lambda: module.drafter.integrate_latents(z_ctx, z0=z0))
        add_time(timings, "flow.integrate", seconds)

    pred_hidden, seconds = time_section(device, lambda: module.drafter.decode_latent(pred_latent))
    add_time(timings, "vae.decode", seconds)

    logits, seconds = time_section(device, lambda: module.lm_head(pred_hidden))
    add_time(timings, "lm_head", seconds)

    _, seconds = time_section(device, lambda: logits.argmax(dim=-1))
    add_time(timings, "argmax", seconds)
    timings["total_drafter"] = sum(timings[name] for name in SECTION_NAMES)
    return timings, flow_timings


def profile_backbone_batch(*, frozen_lm, batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, float]:
    future_tokens = batch["future_tokens"].to(device)
    _, seconds = time_section(device, lambda: frozen_lm._forward(future_tokens, use_cache=True))
    return {"backbone.forward_k_tokens": seconds}



def summarize_generation_profile(*, frozen_lm, module, dataset, num_prompts: int, warmup_prompts: int, max_new_tokens: int) -> dict[str, Any]:
    if num_prompts < 1:
        raise ValueError("generation_num_prompts must be >= 1")
    if warmup_prompts < 0:
        raise ValueError("generation_warmup_prompts must be >= 0")
    if max_new_tokens < 1:
        raise ValueError("generation_max_new_tokens must be >= 1")

    warmups, prompts = collect_speedup_prompts(dataset, num_prompts=num_prompts, warmup_prompts=warmup_prompts)
    draft_len = module.drafter.config.draft_length
    context = ChainedFlowContext(frozen_lm)

    for prompt in warmups:
        generate_greedy_baseline(frozen_lm, prompt, max_new_tokens=max_new_tokens)
        generate_with_drafter(context, module.drafter, prompt, max_new_tokens=max_new_tokens, draft_len=draft_len)

    backbone_seconds = 0.0
    backbone_decode_seconds = 0.0
    backbone_tokens = 0.0
    flow_seconds = 0.0
    flow_decode_seconds = 0.0
    flow_tokens = 0.0
    accepted = 0.0
    drafted = 0.0
    steps = 0.0
    flow_timing_sections: dict[str, float] = {}

    for prompt in prompts:
        baseline = generate_greedy_baseline(frozen_lm, prompt, max_new_tokens=max_new_tokens)
        backbone_seconds += float(baseline["seconds"])
        backbone_decode_seconds += float(baseline["decode_seconds"])
        backbone_tokens += float(baseline["generated_tokens"])

        flow = generate_with_drafter(context, module.drafter, prompt, max_new_tokens=max_new_tokens, draft_len=draft_len)
        total_generation = float(flow.timings.get("total_generation"))
        prefill = float(flow.timings.get("prefill"))
        flow_seconds += total_generation
        flow_decode_seconds += max(0.0, total_generation - prefill)
        flow_tokens += float(flow.generated_token_count)
        for name, seconds in flow.timings.sections.items():
            if name in {"total_generation", "tokens_per_second"}:
                continue
            flow_timing_sections[name] = flow_timing_sections.get(name, 0.0) + float(seconds)
        for step in flow.step_stats:
            accepted += float(step.accepted_len)
            drafted += float(step.draft_len)
            steps += 1.0

    backbone_tps = backbone_tokens / backbone_seconds if backbone_seconds > 0 else 0.0
    flow_tps = flow_tokens / flow_seconds if flow_seconds > 0 else 0.0
    backbone_decode_tps = backbone_tokens / backbone_decode_seconds if backbone_decode_seconds > 0 else 0.0
    flow_decode_tps = flow_tokens / flow_decode_seconds if flow_decode_seconds > 0 else 0.0
    mean_accept = accepted / steps if steps > 0 else 0.0
    mean_draft = drafted / steps if steps > 0 else 0.0

    section_total = sum(flow_timing_sections.values())
    section_details = {
        name: {
            "seconds": seconds,
            "percent_generation": 100.0 * seconds / flow_seconds if flow_seconds > 0 else 0.0,
            "percent_profiled_sections": 100.0 * seconds / section_total if section_total > 0 else 0.0,
        }
        for name, seconds in sorted(flow_timing_sections.items(), key=lambda item: item[1], reverse=True)
    }

    grouped_sections: dict[str, float] = {
        "prefill": flow_timing_sections.get("prefill", 0.0),
        "anchor_next_token": flow_timing_sections.get("next_token", 0.0),
        "anchor_forward_with_cache": flow_timing_sections.get("forward_with_cache", 0.0),
        "drafter": sum(seconds for name, seconds in flow_timing_sections.items() if name.startswith("drafter.")),
        "verifier_forward": flow_timing_sections.get("verifier.verifier_forward", 0.0),
        "verifier_acceptance": flow_timing_sections.get("verifier.acceptance", 0.0),
        "verifier_cache_repair": flow_timing_sections.get("verifier.cache_repair", 0.0),
        "verifier_nested_forward_with_cache": flow_timing_sections.get("verifier.forward_with_cache", 0.0),
    }
    grouped_total = sum(grouped_sections.values())
    grouped_details = {
        name: {
            "seconds": seconds,
            "percent_generation": 100.0 * seconds / flow_seconds if flow_seconds > 0 else 0.0,
            "percent_grouped_sections": 100.0 * seconds / grouped_total if grouped_total > 0 else 0.0,
        }
        for name, seconds in sorted(grouped_sections.items(), key=lambda item: item[1], reverse=True)
    }

    return {
        "num_prompts": len(prompts),
        "warmup_prompts": len(warmups),
        "max_new_tokens": max_new_tokens,
        "draft_len": draft_len,
        "backbone_seconds": backbone_seconds,
        "backbone_decode_seconds": backbone_decode_seconds,
        "backbone_generated_tokens": int(backbone_tokens),
        "backbone_tokens_per_second": backbone_tps,
        "backbone_decode_tokens_per_second": backbone_decode_tps,
        "drafter_verifier_seconds": flow_seconds,
        "drafter_verifier_decode_seconds": flow_decode_seconds,
        "drafter_verifier_generated_tokens": int(flow_tokens),
        "drafter_verifier_tokens_per_second": flow_tps,
        "drafter_verifier_decode_tokens_per_second": flow_decode_tps,
        "real_speedup": flow_tps / backbone_tps if backbone_tps > 0 else 0.0,
        "decode_only_speedup": flow_decode_tps / backbone_decode_tps if backbone_decode_tps > 0 else 0.0,
        "mean_accept_len": mean_accept,
        "mean_draft_len": mean_draft,
        "draft_steps": int(steps),
        "timing_sections": section_details,
        "timing_sections_profiled_total_seconds": section_total,
        "timing_groups": grouped_details,
        "timing_groups_profiled_total_seconds": grouped_total,
    }


def summarize_timing_group(timings: dict[str, float], names: list[str], *, num_examples: int, draft_length: int, total: float) -> dict[str, dict[str, float]]:
    sections: dict[str, dict[str, float]] = {}
    for name in names:
        seconds = timings.get(name, 0.0)
        sections[name] = {
            "seconds": seconds,
            "ms_per_example": 1000.0 * seconds / num_examples if num_examples else 0.0,
            "ms_per_draft_token": 1000.0 * seconds / (num_examples * draft_length) if num_examples and draft_length else 0.0,
            "percent_total": 100.0 * seconds / total if total > 0 else 0.0,
        }
    return sections


def summarize_timings(
    timings: dict[str, float],
    *,
    num_examples: int,
    draft_length: int,
    flow_timings: dict[str, float] | None = None,
    backbone_timings: dict[str, float] | None = None,
) -> dict[str, Any]:
    total = timings.get("total_drafter", 0.0)
    sections = summarize_timing_group(
        timings,
        SECTION_NAMES,
        num_examples=num_examples,
        draft_length=draft_length,
        total=total,
    )
    sections["total_drafter"] = {
        "seconds": total,
        "ms_per_example": 1000.0 * total / num_examples if num_examples else 0.0,
        "ms_per_draft_token": 1000.0 * total / (num_examples * draft_length) if num_examples and draft_length else 0.0,
        "percent_total": 100.0,
    }
    summary: dict[str, Any] = {
        "sections": sections,
        "examples_per_second": num_examples / total if total > 0 else 0.0,
        "draft_tokens_per_second": (num_examples * draft_length) / total if total > 0 else 0.0,
    }
    if flow_timings:
        flow_total = sum(flow_timings.values())
        flow_section_names = [name for name in FLOW_SECTION_NAMES if name in flow_timings]
        flow_section_names.extend(name for name in sorted(flow_timings) if name not in flow_section_names)
        summary["flow_sections"] = summarize_timing_group(
            flow_timings,
            flow_section_names,
            num_examples=num_examples,
            draft_length=draft_length,
            total=flow_total,
        )
        summary["flow_total_seconds"] = flow_total
    if backbone_timings:
        backbone_total = sum(backbone_timings.values())
        summary["backbone_sections"] = summarize_timing_group(
            backbone_timings,
            BACKBONE_SECTION_NAMES,
            num_examples=num_examples,
            draft_length=draft_length,
            total=backbone_total,
        )
        summary["backbone_total_seconds"] = backbone_total
        summary["backbone_tokens_per_second"] = (num_examples * draft_length) / backbone_total if backbone_total > 0 else 0.0
        summary["drafter_vs_backbone_forward_ratio"] = total / backbone_total if backbone_total > 0 else 0.0
        summary["backbone_forward_vs_drafter_ratio"] = backbone_total / total if total > 0 else 0.0
    return summary


def print_summary(result: dict[str, Any]) -> None:
    metadata = result["metadata"]
    summary = result["summary"]
    sections = summary["sections"]
    total = sections["total_drafter"]
    measured_sections = {name: value for name, value in sections.items() if name != "total_drafter"}
    slowest = sorted(measured_sections.items(), key=lambda item: item[1]["seconds"], reverse=True)

    print("", flush=True)
    print("Flow drafter profile", flush=True)
    print("====================", flush=True)
    print(f"Flow checkpoint : {metadata['flow_dir']}", flush=True)
    print(f"Dataset         : {metadata['dataset_path']} ({metadata['dataset_split']})", flush=True)
    print(f"Device / dtype  : {metadata['device']} / {metadata['dtype']}", flush=True)
    print(
        "Model shape     : "
        f"context={metadata['context_size']}, K={metadata['draft_length']}, "
        f"hidden={metadata['hidden_size']}, latent={metadata['latent_size']}, "
        f"expert_dim={metadata['expert_dim']}, heads={metadata['num_heads']}, "
        f"ffn_multiplier={metadata['ffn_multiplier']}, flow_steps={metadata['num_flow_steps']}, "
        f"init_mode={metadata['init_mode']}, architecture={metadata['architecture']}",
        flush=True,
    )
    print(
        "Profile size    : "
        f"{metadata['num_examples']} examples, batch_size={metadata['batch_size']}, "
        f"warmup_batches={metadata['warmup_batches']}",
        flush=True,
    )

    print("", flush=True)
    print("Throughput", flush=True)
    print("----------", flush=True)
    print(f"Total drafter time      : {total['seconds']:.4f} s", flush=True)
    print(f"Milliseconds / example  : {total['ms_per_example']:.3f} ms", flush=True)
    print(f"Milliseconds / token    : {total['ms_per_draft_token']:.3f} ms", flush=True)
    print(f"Examples / second       : {summary['examples_per_second']:.2f}", flush=True)
    print(f"Draft tokens / second   : {summary['draft_tokens_per_second']:.2f}", flush=True)

    print("", flush=True)
    print("Timing breakdown", flush=True)
    print("----------------", flush=True)
    for name, section in measured_sections.items():
        print(
            f"{name:<20} "
            f"{section['seconds']:>8.4f} s  "
            f"{section['ms_per_example']:>8.3f} ms/example  "
            f"{section['ms_per_draft_token']:>8.3f} ms/token  "
            f"{section['percent_total']:>5.1f}%",
            flush=True,
        )

    flow_sections = summary.get("flow_sections")
    if flow_sections:
        flow_total = summary.get("flow_total_seconds", 0.0)
        print("", flush=True)
        print("Flow integrate detail", flush=True)
        print("---------------------", flush=True)
        print(f"Total profiled flow time: {flow_total:.4f} s", flush=True)
        for name, section in flow_sections.items():
            print(
                f"{name:<20} "
                f"{section['seconds']:>8.4f} s  "
                f"{section['ms_per_example']:>8.3f} ms/example  "
                f"{section['ms_per_draft_token']:>8.3f} ms/token  "
                f"{section['percent_total']:>5.1f}% of flow",
                flush=True,
            )

    backbone_sections = summary.get("backbone_sections")
    if backbone_sections:
        backbone_total = summary.get("backbone_total_seconds", 0.0)
        print("", flush=True)
        print("Backbone reference", flush=True)
        print("------------------", flush=True)
        print(f"Qwen forward over same K-token span : {backbone_total:.4f} s", flush=True)
        print(f"Backbone draft-token throughput     : {summary.get('backbone_tokens_per_second', 0.0):.2f} tokens/s", flush=True)
        print(f"Drafter / backbone forward time     : {summary.get('drafter_vs_backbone_forward_ratio', 0.0):.3f}x", flush=True)
        print(f"Backbone forward / drafter time     : {summary.get('backbone_forward_vs_drafter_ratio', 0.0):.3f}x", flush=True)
        for name, section in backbone_sections.items():
            print(
                f"{name:<28} "
                f"{section['seconds']:>8.4f} s  "
                f"{section['ms_per_example']:>8.3f} ms/example  "
                f"{section['ms_per_draft_token']:>8.3f} ms/token",
                flush=True,
            )

    generation = summary.get("generation_e2e")
    if generation:
        print("", flush=True)
        print("End-to-end generation", flush=True)
        print("------------------------", flush=True)
        print(f"Prompts / max new tokens       : {generation['num_prompts']} / {generation['max_new_tokens']}", flush=True)
        print(f"Backbone alone seconds         : {generation['backbone_seconds']:.4f} s", flush=True)
        print(f"Drafter + verifier seconds     : {generation['drafter_verifier_seconds']:.4f} s", flush=True)
        print(f"Backbone tokens / second       : {generation['backbone_tokens_per_second']:.2f}", flush=True)
        print(f"Drafter + verifier tokens/sec  : {generation['drafter_verifier_tokens_per_second']:.2f}", flush=True)
        print(f"Real speedup                   : {generation['real_speedup']:.3f}x", flush=True)
        print(f"Decode-only speedup            : {generation['decode_only_speedup']:.3f}x", flush=True)
        print(f"Mean accepted / drafted        : {generation['mean_accept_len']:.3f} / {generation['mean_draft_len']:.3f}", flush=True)
        timing_groups = generation.get("timing_groups", {})
        if timing_groups:
            print("", flush=True)
            print("Drafter + verifier timing groups", flush=True)
            print("--------------------------------", flush=True)
            for name, section in timing_groups.items():
                print(
                    f"{name:<36} "
                    f"{section['seconds']:>8.4f} s  "
                    f"{section['percent_generation']:>5.1f}% of generation",
                    flush=True,
                )

    print("", flush=True)
    print("Largest costs", flush=True)
    print("-------------", flush=True)
    for rank, (name, section) in enumerate(slowest[:3], start=1):
        print(f"{rank}. {name}: {section['percent_total']:.1f}% of drafter time", flush=True)
    if flow_sections:
        slowest_flow = sorted(flow_sections.items(), key=lambda item: item[1]["seconds"], reverse=True)
        for rank, (name, section) in enumerate(slowest_flow[:3], start=1):
            print(f"flow {rank}. {name}: {section['percent_total']:.1f}% of flow.integrate time", flush=True)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.num_examples < 1:
        raise ValueError("num_examples must be >= 1")
    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if args.warmup_batches < 0:
        raise ValueError("warmup_batches must be >= 0")
    if args.profile_generation and args.generation_num_prompts < 1:
        raise ValueError("generation_num_prompts must be >= 1")
    if args.profile_generation and args.generation_warmup_prompts < 0:
        raise ValueError("generation_warmup_prompts must be >= 0")
    if args.profile_generation and args.generation_max_new_tokens < 1:
        raise ValueError("generation_max_new_tokens must be >= 1")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch_dtype_from_string(args.dtype)
    print(f"loading backbone: model_id={args.model_id} dtype={args.dtype} device={device}", flush=True)
    context = ChainedFlowContext.from_pretrained(
        args.model_id,
        device=device,
        dtype=dtype,
        local_files_only=args.local_files_only,
    )
    frozen_lm = context.frozen_lm

    print(f"loading flow checkpoint: {args.flow_dir}", flush=True)
    module, config = load_flow_training_module(args.flow_dir, frozen_lm=frozen_lm, device=device)
    model_args = load_flow_model_args(args.flow_dir)

    total_windows = args.num_examples + args.warmup_batches * args.batch_size
    dataset = TeacherWindowDataset.from_path(
        args.dataset_path,
        split=args.dataset_split,
        context_size=model_args.context_size,
        draft_length=model_args.draft_length,
        windows_per_epoch=total_windows,
        seed=args.window_seed,
        materialize_rows=args.materialize_rows,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_teacher_windows)
    generator = torch.Generator(device=device).manual_seed(args.noise_seed)

    iterator = iter(dataloader)
    for _ in range(args.warmup_batches):
        try:
            warmup_batch = next(iterator)
        except StopIteration:
            break
        profile_batch(
            module=module,
            batch=warmup_batch,
            device=device,
            generator=generator,
            profile_flow_ops=args.profile_flow_ops,
        )
        if args.profile_backbone:
            profile_backbone_batch(frozen_lm=frozen_lm, batch=warmup_batch, device=device)

    timings = {name: 0.0 for name in SECTION_NAMES}
    flow_timings: dict[str, float] = {}
    backbone_timings = {name: 0.0 for name in BACKBONE_SECTION_NAMES}
    measured = 0
    while measured < args.num_examples:
        try:
            batch = next(iterator)
        except StopIteration:
            break
        remaining = args.num_examples - measured
        batch_size = min(remaining, int(batch["context_hidden"].shape[0]))
        batch = slice_batch(batch, batch_size)
        batch_timings, batch_flow_timings = profile_batch(
            module=module,
            batch=batch,
            device=device,
            generator=generator,
            profile_flow_ops=args.profile_flow_ops,
        )
        for name, seconds in batch_timings.items():
            add_time(timings, name, seconds)
        for name, seconds in batch_flow_timings.items():
            add_time(flow_timings, name, seconds)
        if args.profile_backbone:
            batch_backbone_timings = profile_backbone_batch(frozen_lm=frozen_lm, batch=batch, device=device)
            for name, seconds in batch_backbone_timings.items():
                add_time(backbone_timings, name, seconds)
        measured += batch_size

    if measured != args.num_examples:
        raise RuntimeError(f"profiled {measured} examples, expected {args.num_examples}")

    summary = summarize_timings(
        timings,
        num_examples=measured,
        draft_length=model_args.draft_length,
        flow_timings=flow_timings if args.profile_flow_ops else None,
        backbone_timings=backbone_timings if args.profile_backbone else None,
    )
    if args.profile_generation:
        print(
            "profiling end-to-end generation: "
            f"prompts={args.generation_num_prompts} warmup={args.generation_warmup_prompts} "
            f"max_new_tokens={args.generation_max_new_tokens}",
            flush=True,
        )
        summary["generation_e2e"] = summarize_generation_profile(
            frozen_lm=frozen_lm,
            module=module,
            dataset=dataset,
            num_prompts=args.generation_num_prompts,
            warmup_prompts=args.generation_warmup_prompts,
            max_new_tokens=args.generation_max_new_tokens,
        )
    metadata = {
        "flow_dir": args.flow_dir,
        "dataset_path": args.dataset_path,
        "dataset_split": args.dataset_split,
        "model_id": args.model_id,
        "device": str(device),
        "dtype": args.dtype,
        "num_examples": measured,
        "batch_size": args.batch_size,
        "warmup_batches": args.warmup_batches,
        "profile_flow_ops": args.profile_flow_ops,
        "profile_backbone": args.profile_backbone,
        "profile_generation": args.profile_generation,
        "generation_num_prompts": args.generation_num_prompts,
        "generation_warmup_prompts": args.generation_warmup_prompts,
        "generation_max_new_tokens": args.generation_max_new_tokens,
        "context_size": model_args.context_size,
        "draft_length": model_args.draft_length,
        "chunk_size": model_args.chunk_size,
        "hidden_size": module.drafter.hidden_size,
        "latent_size": module.drafter.latent_size,
        "expert_dim": model_args.expert_dim,
        "num_heads": model_args.num_heads,
        "ffn_multiplier": model_args.ffn_multiplier,
        "num_flow_steps": model_args.num_flow_steps,
        "init_mode": model_args.init_mode,
        "architecture": model_args.architecture,
        "noise_seed": args.noise_seed,
        "window_seed": args.window_seed,
        "config": config,
    }
    result = {"metadata": metadata, "summary": summary}
    print_summary(result)

    if args.output_path is not None:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"profile saved: {output_path}", flush=True)


if __name__ == "__main__":
    main()
