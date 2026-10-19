# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from models.demos.common.prefill.runners.kv_page_pool import (
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
    bank_aligned_seq_len_quantum,
    check_bank_aligned,
    kv_page_rows,
)

CHUNK_SIZE = 5120


def _bank_of_row(row, num_dram_banks):
    return (row // NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK) % num_dram_banks


@pytest.mark.parametrize("num_dram_banks", [7, 8])
@pytest.mark.parametrize("stripes", [8, 32])
@pytest.mark.parametrize("layers", [5, 6, 18, 20])
def test_slot_layer_rows_start_on_bank_0(num_dram_banks, stripes, layers):
    quantum = bank_aligned_seq_len_quantum(stripes, num_dram_banks, CHUNK_SIZE)
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
    quantum = bank_aligned_seq_len_quantum(32, num_dram_banks, CHUNK_SIZE)
    assert (quantum // 32) % kv_page_rows(num_dram_banks) == 0


@pytest.mark.parametrize("max_seq_len, stripes", [(1064960, 32), (256000, 8), (1054720, 8), (61440, 8)])
def test_manifest_lengths_pass_check(max_seq_len, stripes):
    cache = SimpleNamespace(shape=[4, 1, max_seq_len // stripes, 576])
    check_bank_aligned({"kvpe": cache}, 8, max_seq_len, CHUNK_SIZE)


@pytest.mark.parametrize("max_seq_len, stripes", [(1049600, 8), (56320, 8), (256000, 32)])
def test_unaligned_lengths_fail_check(max_seq_len, stripes):
    cache = SimpleNamespace(shape=[4, 1, max_seq_len // stripes, 576])
    quantum = bank_aligned_seq_len_quantum(stripes, 8, CHUNK_SIZE)
    with pytest.raises(ValueError, match=f"multiple of {quantum}"):
        check_bank_aligned({"kvpe": cache}, 8, max_seq_len, CHUNK_SIZE)
