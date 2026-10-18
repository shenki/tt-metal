# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host (CPU) cost of the plugin host sampler (vLLM Sampler via vllm_tt_plugin.host_sampler) at 128000 vocab.

    python models/autoports/liquidai_lfm2_5_24b_a2b/tests/bench_host_sampler.py
"""

import time

import torch
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm_tt_plugin.host_sampler import create_host_sampler

s = create_host_sampler()
V = 128000
print("threads", torch.get_num_threads())
for B, k, rep in [(1, 50, 1.05), (1, 50, 1.0), (1, 20, 1.05), (32, 50, 1.05)]:
    logits = torch.randn(B, V)
    md = SamplingMetadata(
        temperature=torch.full((B,), 0.1),
        all_greedy=False,
        all_random=True,
        top_p=None,
        top_k=torch.full((B,), k),
        generators={},
        max_num_logprobs=None,
        no_penalties=rep == 1.0,
        prompt_token_ids=torch.randint(0, V, (B, 128)) if rep != 1.0 else None,
        frequency_penalties=torch.zeros(B),
        presence_penalties=torch.zeros(B),
        repetition_penalties=torch.full((B,), rep),
        output_token_ids=[list(range(100))] * B,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )
    for _ in range(3):
        s(logits=logits, sampling_metadata=md)
    t = time.perf_counter()
    n = 20
    for _ in range(n):
        s(logits=logits, sampling_metadata=md)
    print(B, k, rep, f"{(time.perf_counter() - t) / n * 1000:.2f} ms")
x = torch.randn(32, 129024).bfloat16()
t = time.perf_counter()
for _ in range(10):
    y = x.float()[:1, :V]
print("bf16->f32 32x129024", (time.perf_counter() - t) / 10 * 1000, "ms")
