# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Without the extensible KV cache, CUDA graph memory is profiled against a
throwaway KV cache allocated before the real one is sized. That cache must not
grow with `max_num_seqs`, or startup fails when free memory is tight."""

import torch

from tests.conftest import VllmRunner
from tests.utils import create_new_process_for_each_test


@create_new_process_for_each_test()
def test_cudagraph_profiling_kv_cache_fits_tight_memory(
    vllm_runner: type[VllmRunner],
) -> None:
    free, total = torch.accelerator.get_memory_info()
    keep_free = 4 << 30
    # Hold the rest so the engine starts with ~4 GiB free. One 64-token block
    # per request (512) of Qwen3-0.6B's KV cache would take ~3.7 GiB.
    hold = torch.empty(
        free - keep_free,
        dtype=torch.uint8,
        device=torch.accelerator.current_accelerator(),
    )
    with vllm_runner(
        "Qwen/Qwen3-0.6B",
        enable_extensible_kv_cache=False,
        gpu_memory_utilization=(keep_free - (512 << 20)) / total,
        max_model_len=1024,
        max_num_seqs=512,
        block_size=64,
        compilation_config={"decode_cudagraph_capture_request_sizes": [1, 512]},
    ) as vllm_model:
        (output,) = vllm_model.generate_greedy(["The capital of France is"], 8)
    assert output[1]
    del hold
