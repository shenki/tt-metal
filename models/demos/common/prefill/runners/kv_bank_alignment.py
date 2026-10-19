# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32


def bank_aligned_seq_len_quantum(stripes: int, num_dram_banks: int) -> int:
    return stripes * NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK * num_dram_banks


def check_kv_bank_aligned(max_seq_len: int, stripes: int, num_dram_banks: int) -> None:
    quantum = bank_aligned_seq_len_quantum(stripes, num_dram_banks)
    if max_seq_len % quantum:
        lower = max_seq_len // quantum * quantum
        raise ValueError(
            f"max_seq_len {max_seq_len} over {stripes} stripes does not start every KV slot on DRAM bank 0 "
            f"({num_dram_banks} banks x {NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK} tokens): use a multiple of "
            f"{quantum}, e.g. {lower or quantum} or {lower + quantum}"
        )
