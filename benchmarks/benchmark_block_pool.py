# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BlockPool operations: allocate, free, touch and prefix-cache hit search."""

import gc

import torch
from benchmark_utils import TimeCollector
from tabulate import tabulate

from vllm.sampling_params import SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    MambaManager,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.request import Request

BLOCK_SIZE = 16


def make_cached_pool(num_gpu_blocks: int, num_cached: int) -> tuple[BlockPool, list]:
    """A pool whose first ``num_cached`` blocks hold one request's prefix."""
    pool = BlockPool(num_gpu_blocks, enable_caching=True, hash_block_size=BLOCK_SIZE)
    request = Request(
        request_id="0",
        prompt_token_ids=list(range(num_cached * BLOCK_SIZE)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )
    blocks = pool.get_new_blocks(num_cached)
    pool.cache_full_blocks(request, blocks, 0, num_cached, BLOCK_SIZE, 0)
    pool.free_blocks(reversed(blocks))
    return pool, request.block_hashes


def bench_alloc_free(args, n: int) -> list:
    gc.collect()
    pool = BlockPool(args.num_gpu_blocks, enable_caching=True, hash_block_size=16)
    get_t, free_t = TimeCollector(TimeCollector.US), TimeCollector(TimeCollector.US)
    for _ in range(args.num_iteration):
        with get_t:
            blocks = pool.get_new_blocks(n)
        with free_t:
            pool.free_blocks(blocks)
    return [get_t.avg(), free_t.avg()]


def bench_cached(args, n: int) -> list:
    """Touch, cache-hit search (full attention and Mamba) over n cached blocks."""
    gc.collect()
    pool, hashes = make_cached_pool(args.num_gpu_blocks, n)
    blocks = [
        pool.cached_block_hash_to_block.get_one_block(
            make_block_hash_with_group_id(h, 0)
        )
        for h in hashes
    ]
    touch_t, full_t, mamba_t = (TimeCollector(TimeCollector.US) for _ in range(3))
    full_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float16
    )
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((1,),),
        dtypes=(torch.float16,),
        mamba_cache_mode="align",
    )
    max_length = n * BLOCK_SIZE
    for _ in range(args.num_iteration):
        with touch_t:
            pool.touch(blocks)
        pool.free_blocks(reversed(blocks))
        with full_t:
            hit, _ = FullAttentionManager.find_longest_cache_hit(
                hashes, max_length, [0], pool, full_spec, False, BLOCK_SIZE
            )
        # Worst case for the right-to-left search: only block 0 hits.
        with mamba_t:
            MambaManager.find_longest_cache_hit(
                hashes[:1] + [b"miss"] * (n - 1),
                max_length,
                [0],
                pool,
                mamba_spec,
                False,
                BLOCK_SIZE,
            )
    assert len(hit[0]) == n
    return [touch_t.avg(), full_t.avg(), mamba_t.avg()]


def main(args) -> None:
    init_none_hash(sha256)
    rows = [
        [n] + bench_alloc_free(args, n) + bench_cached(args, n) for n in args.num_blocks
    ]
    print(
        tabulate(
            rows,
            headers=[
                "Blocks",
                "get_new_blocks\nus",
                "free_blocks\nus",
                "touch\nus",
                "full attn hit\nsearch us",
                "Mamba hit\nsearch us",
            ],
            floatfmt=".2f",
        )
    )


def invoke_main() -> None:
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument("--num-gpu-blocks", type=int, default=100000)
    parser.add_argument(
        "--num-iteration",
        type=int,
        default=1000,
        help="Number of iterations to run to stabilize final data readings",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        nargs="*",
        default=[1, 10, 50, 100, 500, 1000],
        help="Number of blocks per operation",
    )
    main(parser.parse_args())


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
