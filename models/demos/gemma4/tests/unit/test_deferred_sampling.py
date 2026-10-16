# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from models.demos.gemma4.tt.generator import ChunkedPrefillPageTableGuardMixin
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.generator import DeferredDecodeSampling


def test_deferred_decode_uses_the_shared_one_shot_payload(expect_error):
    """Gemma4 must not return raw logits when the runner defers sampling."""

    forwarded = []
    harness = SimpleNamespace(
        _deferred_decode_sampling_failed=False,
        _pending_deferred_decode_sampling=None,
        mode=Mode.DECODE,
        model=[SimpleNamespace(switch_mode=lambda _mode: None)],
        data_parallel=1,
        _clear_sequential_batch_page_tables=lambda: None,
        _decode_forward_no_trace_text=lambda **kwargs: forwarded.append(kwargs) or ["tt-logits"],
    )
    tokens = torch.tensor([[17]], dtype=torch.int32)
    start_pos = torch.tensor([9], dtype=torch.int32)

    payload = ChunkedPrefillPageTableGuardMixin.decode_forward(
        harness,
        tokens,
        start_pos,
        enable_trace=False,
        read_from_device=False,
        defer_device_sampling=True,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=True,
        reset_sampling_state=False,
    )

    assert isinstance(payload, DeferredDecodeSampling)
    assert payload.tt_logits == ["tt-logits"]
    assert len(payload.start_pos) == 1
    torch.testing.assert_close(payload.start_pos[0], start_pos)
    assert payload.reload_inputs is True
    assert payload.reload_sampling_params is True
    assert harness._pending_deferred_decode_sampling is payload
    assert forwarded[0]["on_device_sampling"] is True

    with expect_error(RuntimeError, "previous deferred device-sampling payload"):
        ChunkedPrefillPageTableGuardMixin.decode_forward(
            harness,
            tokens,
            start_pos,
            enable_trace=False,
            read_from_device=False,
            defer_device_sampling=True,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=True,
            reset_sampling_state=False,
        )
