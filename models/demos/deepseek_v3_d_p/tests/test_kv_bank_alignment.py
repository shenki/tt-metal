# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
    bank_aligned_seq_len_quantum,
    kv_page_rows,
)


def _bank_of_row(row, num_dram_banks):
    return (row // NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK) % num_dram_banks


@pytest.mark.parametrize("num_dram_banks", [7, 8])
@pytest.mark.parametrize("stripes", [8, 32])
@pytest.mark.parametrize("layers", [5, 6, 18, 20])
def test_slot_layer_rows_start_on_bank_0(num_dram_banks, stripes, layers):
    quantum = bank_aligned_seq_len_quantum(stripes, num_dram_banks)
    seq_len_local = 3 * quantum // stripes
    for slot in range(4):
        for layer in range(layers):
            start = (slot * layers + layer) * seq_len_local
            assert _bank_of_row(start, num_dram_banks) == 0


@pytest.mark.parametrize("num_dram_banks", [7, 8])
def test_pool_pages_start_on_bank_0(num_dram_banks):
    page_rows = kv_page_rows(num_dram_banks)
    for page in range(64):
        rows = range(page * page_rows, (page + 1) * page_rows, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK)
        assert [_bank_of_row(r, num_dram_banks) for r in rows] == list(range(num_dram_banks))


@pytest.mark.parametrize("num_dram_banks", [7, 8])
def test_slot_divides_into_pool_pages(num_dram_banks):
    quantum = bank_aligned_seq_len_quantum(32, num_dram_banks)
    assert (quantum // 32) % kv_page_rows(num_dram_banks) == 0
