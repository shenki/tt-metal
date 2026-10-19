# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import math
import os
from dataclasses import dataclass

from loguru import logger

import ttnn
from models.demos.common.prefill.runners.migration import _host_tag_int, get_num_dram_banks, migration_table_path

NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32
_POOL_TAG = 0x6B76


@dataclass(frozen=True)
class KvPagePool:
    token_caches: dict
    state_caches: dict
    rows_per_user: dict
    num_users: int
    page_rows: int


def kv_page_rows(num_dram_banks: int) -> int:
    return NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK * num_dram_banks


def bank_aligned_seq_len_quantum(stripes: int, num_dram_banks: int, chunk_size: int) -> int:
    return math.lcm(stripes * kv_page_rows(num_dram_banks), chunk_size)


def check_bank_aligned(token_caches: dict, num_dram_banks: int, max_seq_len: int, chunk_size: int) -> None:
    page_rows = kv_page_rows(num_dram_banks)
    for kind, cache in token_caches.items():
        rows = cache.shape[2]
        if rows % page_rows:
            quantum = bank_aligned_seq_len_quantum(max(1, max_seq_len // rows), num_dram_banks, chunk_size)
            raise ValueError(
                f"{kind} KV cache holds {rows} rows per device, not a multiple of {page_rows} "
                f"({num_dram_banks} DRAM banks x {NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK}), so its slots do not "
                f"start on DRAM bank 0: make PREFILL_MAX_SEQ_LEN={max_seq_len} a multiple of {quantum}"
            )


def _rows_per_slot(kind: str, cache, num_slots: int) -> int:
    if cache.shape[0] % num_slots:
        raise ValueError(f"{kind} KV cache has {cache.shape[0]} rows on dim 0, not a multiple of {num_slots} slots")
    return cache.shape[0] // num_slots


def _allocate_like(cache, rows: int, mesh_device):
    tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape([rows, *list(cache.shape)[1:]]), cache.dtype, cache.layout, mesh_device, cache.memory_config()
    )
    tensor.update_tensor_topology(cache.tensor_topology())
    return tensor


def allocate_kv_page_pool(mesh_device, token_caches: dict, state_caches: dict, *, num_slots: int, num_users: int):
    rows_per_user = {
        kind: _rows_per_slot(kind, cache, num_slots) for kind, cache in {**token_caches, **state_caches}.items()
    }
    if num_users == 0:
        return KvPagePool({}, {}, rows_per_user, 0, kv_page_rows(get_num_dram_banks(mesh_device)))
    return KvPagePool(
        token_caches={
            kind: _allocate_like(cache, num_users * rows_per_user[kind], mesh_device)
            for kind, cache in token_caches.items()
        },
        state_caches={
            kind: _allocate_like(cache, num_users * rows_per_user[kind], mesh_device)
            for kind, cache in state_caches.items()
        },
        rows_per_user=rows_per_user,
        num_users=num_users,
        page_rows=kv_page_rows(get_num_dram_banks(mesh_device)),
    )


def kv_page_pool_path() -> str:
    return os.path.splitext(migration_table_path())[0] + "_page_pool.json"


def _local_pool_doc(pool: KvPagePool, mesh_device, rank: int) -> dict:
    rows, cols = mesh_device.shape[0], mesh_device.shape[1]
    fnids = []
    for r in range(rows):
        row = []
        for c in range(cols):
            fid = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(r, c))
            row.append([int(fid.mesh_id), int(fid.chip_id)])
        fnids.append(row)
    caches = []
    for role, tensors in (("token", pool.token_caches), ("state", pool.state_caches)):
        for kind, tensor in tensors.items():
            caches.append(
                {
                    "kind": kind,
                    "role": role,
                    "base_addr": int(tensor.buffer_address()),
                    "shape": list(tensor.shape),
                    "rows_per_user": pool.rows_per_user[kind],
                    "dtype": str(tensor.dtype),
                    "layout": str(tensor.layout),
                }
            )
    return {
        "rank": rank,
        "num_users": pool.num_users,
        "num_banks": get_num_dram_banks(mesh_device),
        "host_tag": _host_tag_int(),
        "fnids": fnids,
        "caches": caches,
    }


def publish_kv_page_pool(pool: KvPagePool, mesh_device, *, rank: int, num_slots: int):
    path = kv_page_pool_path()
    if rank == 0 and os.path.exists(path):
        os.remove(path)
    local = json.dumps(_local_pool_doc(pool, mesh_device, rank)).encode()
    sizes = ttnn.distributed_context_allgather_int(len(local))
    if rank != 0:
        ttnn.distributed_context_send_bytes(local, 0, _POOL_TAG)
        return None
    ranks = [json.loads(local)]
    for src in range(1, len(sizes)):
        ranks.append(json.loads(ttnn.distributed_context_recv_bytes(sizes[src], src, _POOL_TAG)))
    num_users = min(r["num_users"] for r in ranks)
    doc = {"num_slots": num_slots, "page_rows": pool.page_rows, "num_users": num_users, "ranks": ranks}
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    logger.info(f"[kv pool] {num_users} pool users + {num_slots} slots -> {path}")
    return path
