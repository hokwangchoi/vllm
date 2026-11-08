# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU cost of scheduling and KV cache management per engine step.

Drives the real ``Scheduler`` with a fake model runner on a hybrid
attention + Mamba ("align" mode) KV cache layout, with multi-turn
conversations whose turns re-send the history, so later turns hit the
prefix cache. Reports the time per step spent in ``schedule()``,
``update_from_output()`` and in each ``KVCacheManager`` method.

Example (Qwen3-Next-like layout, TP 2, ~1.85M-token KV cache):

    python benchmarks/benchmark_kv_cache_manager.py --concurrency 4 32 128
"""

import cProfile
import functools
import gc
import os
import pstats
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch
from tabulate import tabulate

from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

# The scheduler only needs the model config for limits, not the weights.
os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

EOS_TOKEN_ID = 50256
VOCAB = 150_000


def make_scheduler(args) -> Scheduler:
    model_config = ModelConfig(
        model="facebook/opt-125m",
        dtype="float16",
        seed=0,
        max_model_len=args.max_model_len,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=args.max_num_seqs or max(args.concurrency_value, 1),
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        enable_chunked_prefill=True,
        is_encoder_decoder=False,
    )
    cache_config = CacheConfig(
        block_size=args.block_size,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=True,
        mamba_cache_mode="align" if args.mamba_groups else "none",
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
    )
    groups = [
        KVCacheGroupSpec(
            [f"attn.{g}"],
            FullAttentionSpec(
                block_size=args.block_size,
                num_kv_heads=1,
                head_size=256,
                dtype=torch.bfloat16,
            ),
        )
        for g in range(args.attn_groups)
    ] + [
        KVCacheGroupSpec(
            [f"mamba.{g}"],
            MambaSpec(
                block_size=args.block_size,
                shapes=((1, 1),),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
            ),
        )
        for g in range(args.mamba_groups)
    ]
    kv_cache_config = KVCacheConfig(
        num_blocks=args.num_blocks, kv_cache_tensors=[], kv_cache_groups=groups
    )
    cache_config.num_gpu_blocks = args.num_blocks
    register_all_kvcache_specs(vllm_config)
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=args.block_size,
        hash_block_size=args.block_size,
        log_stats=True,
    )


@dataclass
class Conversation:
    """A chat whose turns re-send the history: prompt = history + new turn."""

    idx: int
    rng: random.Random
    history: list[int]
    turns_left: int


@dataclass
class Timers:
    totals: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def wrap(self, obj, name: str, label: str) -> None:
        fn = getattr(obj, name)
        totals, calls = self.totals, self.calls

        @functools.wraps(fn)
        def timed(*a, **kw):
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                totals[label] += time.perf_counter() - t0
                calls[label] += 1

        setattr(obj, name, timed)


KV_MANAGER_METHODS = (
    "get_computed_blocks",
    "allocate_slots",
    "free",
    "pop_blocks_for_free",
    "cache_blocks",
    "get_block_ids",
    "get_blocks",
    "remove_skipped_blocks",
    "new_step_starts",
    "take_events",
    "take_new_block_ids",
    "take_kv_cache_block_copies",
    "take_boundary_state_offloads",
)


def run(args, concurrency: int) -> dict:
    args.concurrency_value = concurrency
    scheduler = make_scheduler(args)
    hasher = get_request_block_hasher(args.block_size, sha256)
    timers = Timers()
    for name in KV_MANAGER_METHODS:
        if hasattr(scheduler.kv_cache_manager, name):
            timers.wrap(scheduler.kv_cache_manager, name, f"kv.{name}")

    master = random.Random(args.seed)
    shared = [master.randrange(VOCAB) for _ in range(args.shared_prefix_tokens)]
    convs = []
    for i in range(args.num_conversations or concurrency):
        rng = random.Random(master.random())
        own = args.prompt_tokens - args.shared_prefix_tokens
        convs.append(
            Conversation(
                i, rng, shared + [rng.randrange(VOCAB) for _ in range(own)], args.turns
            )
        )
    pending = list(reversed(convs))
    active: dict[str, tuple[Conversation, Request]] = {}
    sampling_params = SamplingParams(
        max_tokens=args.output_tokens, ignore_eos=True, temperature=0.0
    )
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)

    next_id = 0
    hash_time = 0.0
    num_new_requests = 0
    prompt_tokens = cached_tokens = 0

    def admit(conv: Conversation) -> None:
        nonlocal next_id, hash_time, num_new_requests
        prompt = conv.history
        t0 = time.perf_counter()
        req = Request(
            request_id=f"c{conv.idx}-{next_id}",
            prompt_token_ids=list(prompt),
            sampling_params=sampling_params,
            pooling_params=None,
            block_hasher=hasher,
        )
        hash_time += time.perf_counter() - t0
        num_new_requests += 1
        next_id += 1
        scheduler.add_request(req)
        active[req.request_id] = (conv, req)

    for _ in range(min(concurrency, len(pending))):
        admit(pending.pop())

    step_times: list[tuple[float, float, int, int]] = []
    profiler = cProfile.Profile() if args.profile else None
    gc.collect()
    wall0 = time.perf_counter()
    while active:
        if profiler:
            profiler.enable()
        t0 = time.perf_counter()
        so = scheduler.schedule()
        t1 = time.perf_counter()
        for new_req in so.scheduled_new_reqs:
            stats = scheduler.requests[new_req.req_id].prefill_stats
            if stats is not None:
                prompt_tokens += stats.num_prompt_tokens
                cached_tokens += stats.num_local_cached_tokens
        req_ids = list(so.num_scheduled_tokens)
        sampled = []
        for rid in req_ids:
            req = scheduler.requests[rid]
            if req.num_computed_tokens >= req.num_tokens:
                conv = active[rid][0]
                sampled.append([conv.rng.randrange(VOCAB)])
            else:
                sampled.append([])
        mro = ModelRunnerOutput(req_ids=req_ids, sampled_token_ids=sampled)
        t2 = time.perf_counter()
        scheduler.update_from_output(so, mro)
        t3 = time.perf_counter()
        if profiler:
            profiler.disable()
        num_prefill_tokens = sum(
            n for rid, n in so.num_scheduled_tokens.items() if n > 1
        )
        step_times.append((t1 - t0, t3 - t2, len(req_ids), num_prefill_tokens))

        for rid, (conv, req) in list(active.items()):
            if RequestStatus.is_finished(req.status):
                del active[rid]
                conv.turns_left -= 1
                if conv.turns_left > 0:
                    conv.history = list(req.all_token_ids) + [
                        conv.rng.randrange(VOCAB) for _ in range(args.turn_tokens)
                    ]
                    pending.insert(0, conv)
        while pending and len(active) < concurrency:
            admit(pending.pop())
        if len(step_times) >= args.max_steps:
            break
    wall = time.perf_counter() - wall0

    sched = np.array([s[0] for s in step_times]) * 1e6
    upd = np.array([s[1] for s in step_times]) * 1e6
    nreq = np.array([s[2] for s in step_times])
    prefill = np.array([s[3] for s in step_times]) > 0
    nsteps = len(step_times)
    kv_total = sum(v for k, v in timers.totals.items() if k.startswith("kv."))
    result = {
        "concurrency": concurrency,
        "steps": nsteps,
        "prefill_steps": int(prefill.sum()),
        "mean_batch": float(nreq.mean()),
        "schedule_us": float(sched.mean()),
        "schedule_p99_us": float(np.percentile(sched, 99)),
        "update_us": float(upd.mean()),
        "decode_step_us": float((sched + upd)[~prefill].mean())
        if (~prefill).any()
        else float("nan"),
        "prefill_step_us": float((sched + upd)[prefill].mean())
        if prefill.any()
        else float("nan"),
        "kv_us": kv_total / nsteps * 1e6,
        "kv_share": kv_total / max(1e-9, (sched.sum() + upd.sum()) / 1e6),
        "hash_ms_per_req": hash_time / max(1, num_new_requests) * 1e3,
        "wall_s": wall,
        "preemptions": scheduler.num_preemptions
        if hasattr(scheduler, "num_preemptions")
        else None,
        "hit_rate": cached_tokens / prompt_tokens if prompt_tokens else None,
        "timers": {
            k: (v / nsteps * 1e6, timers.calls[k]) for k, v in timers.totals.items()
        },
    }
    if profiler:
        ps = pstats.Stats(profiler).sort_stats("tottime")
        print(f"\n=== cProfile, concurrency {concurrency} (top by tottime) ===")
        ps.print_stats(args.profile_top)
    return result


def main(args) -> None:
    init_none_hash(sha256)
    results = [run(args, c) for c in args.concurrency]
    rows = [
        [
            r["concurrency"],
            r["steps"],
            r["prefill_steps"],
            f"{r['mean_batch']:.1f}",
            f"{r['schedule_us']:.0f}",
            f"{r['schedule_p99_us']:.0f}",
            f"{r['update_us']:.0f}",
            f"{r['decode_step_us']:.0f}",
            f"{r['prefill_step_us']:.0f}",
            f"{r['kv_us']:.0f}",
            f"{100 * r['kv_share']:.0f}%",
            f"{r['hash_ms_per_req']:.2f}",
            "-" if r["hit_rate"] is None else f"{100 * r['hit_rate']:.0f}%",
        ]
        for r in results
    ]
    print(
        tabulate(
            rows,
            headers=[
                "conc",
                "steps",
                "prefill\nsteps",
                "batch",
                "schedule\nus",
                "schedule\np99 us",
                "update\nus",
                "decode\nstep us",
                "prefill\nstep us",
                "KV mgr\nus/step",
                "KV mgr\nshare",
                "hash ms\n/request",
                "hit\nrate",
            ],
        )
    )
    for r in results:
        print(f"\nKVCacheManager per step, concurrency {r['concurrency']}:")
        print(
            tabulate(
                sorted(
                    ([k, f"{us:.1f}", n] for k, (us, n) in r["timers"].items()),
                    key=lambda row: -float(row[1]),
                ),
                headers=["method", "us/step", "calls"],
            )
        )


def invoke_main() -> None:
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 32, 128])
    parser.add_argument("--num-conversations", type=int, default=None)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--prompt-tokens", type=int, default=25_000)
    parser.add_argument("--shared-prefix-tokens", type=int, default=2_000)
    parser.add_argument("--turn-tokens", type=int, default=500)
    parser.add_argument("--output-tokens", type=int, default=650)
    parser.add_argument("--block-size", type=int, default=544)
    parser.add_argument("--num-blocks", type=int, default=13_600)
    parser.add_argument("--attn-groups", type=int, default=1)
    parser.add_argument("--mamba-groups", type=int, default=3)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=262_144)
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-top", type=int, default=30)
    main(parser.parse_args())


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
