# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-11 full-phase roofline collector (``roofline_command`` of ``benchmark_stage run``).

    python benchmark_roofline.py --run-dir <run> --concurrency {32|1} --action {check|collect}

Timing source: the adapter's event log (``LFM2_VLLM_EVENTS``, enabled by ``benchmark_server.py``).  The adapter appends
one record per ``prefill_forward`` / ``decode_forward`` call when the call returns, with ``mono`` (CLOCK_MONOTONIC),
``dur_s`` (host time inside the call), the prefill rows ``[start, end, slot, computed_from]`` and, for decode, the active
rows' positions.  No device synchronization is added: prefill calls return the sampled token (their own completion
boundary); decode calls enqueue a traced step whose completion the plugin waits for in its readback before the next
call, so that wait falls into the decode interval below.

Measured window: from the earliest client request start (``perf-bN.json`` ``start_times``; vllm bench uses
``time.perf_counter`` = CLOCK_MONOTONIC on Linux, the same clock as ``mono``) to the last ``release`` of a measured
request (the plugin releases a slot after its final token was read back).  The warmup run and the client's initial
probe end before the window opens; every admission (prefill row with start 0) inside the window must belong to the
measured requests (count = ``completed``, each a 4096-token prompt), else collection fails.

Phase time: the calls inside the window are ordered on the common timeline and every interval from one call's start
to the next call's start (the last: to the window end) is charged to that call's phase.  The prefill and decode
seconds are therefore non-overlapping, include dispatch, all ops, CCL, sampling, readback, scheduler work and host
gaps, and sum to the window.

Work: modeled from the executed shapes, the HF config and the selected precision policy (C1,
``doc/datatype_sweep/selected_precision_config.json``):
* prefill FLOPs = useful FLOPs of every computed prompt token: 2 x the active matmul parameters (attention q/k/v/o,
  ShortConv in/out + depthwise taps, router, 4 of 64 experts, dense MLP of the 2 dense layers) + causal attention
  2 x 2 x heads x head_dim x (position + 1) per attention layer + the LM head once per request (its last token);
* decode DRAM bytes per step, summed over the 4 chips (tensor parallel 4, each chip reads its quarter of every
  sharded weight; router and norms replicated): attention / ShortConv / dense weights and the LM head at their stored
  block-float sizes, experts: the compact path (bucket <= 4) reads bucket x 4 quarter-experts, the larger buckets'
  sparse matmul reads the union of the experts the active rows selected -- not logged, modeled as the expected union
  64 x (1 - (1 - 4/64)^n) under uniform routing; KV reads of every active row's cached positions + its new K/V row
  (BFP8), ShortConv state read+write (bf16).  Activations / L1 traffic are not counted.
Peaks (tt-perf-report 1.x ``ArchitectureSpec`` "blackhole", the source tt-metal's own roofline tooling uses): matmul
4096 FLOP/cycle/core x 1.35 GHz / 2 (HiFi2, the policy's prefill fidelity) x 110 worker cores x 4 chips; DRAM
512 GB/s x 4 chips."""

import argparse
import hashlib
import json
import lzma
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(os.environ.get("TT_MODEL_BRINGUP_ROOT", "/nonexistent")) / "runtime"))
from benchmark_stage.subsets import digest  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parents[1]
SELECTED = MODEL_DIR / "doc" / "datatype_sweep" / "selected_precision_config.json"
SNAPSHOTS = Path.home() / ".cache/huggingface/hub/models--LiquidAI--LFM2.5-24B-A2B/snapshots"
CHIPS = 4
BYTES = {"bf16": 2.0, "bfp8": 1088 / 1024, "bfp4": 576 / 1024}
DECODE_BUCKETS = (1, 2, 4, 8, 16, 32)  # tt/optimized_decoder.py DECODE_BUCKETS
COMPACT_MAX = 4  # Policy.moe_compact_max_users
DENSE_I_DECODE = 3072  # per-chip dense intermediate as stored (2944 padded; ds_dense_w13 / ds_dense_w2_decode)
HEAD_COLS = 32256  # per-chip vocab columns as stored (32000 padded)
CORES, CLOCK_HZ, FLOP_PER_CYCLE, DRAM_BPS = 110, 1.35e9, 4096, 512e9
FIDELITY_DIV = {"lofi": 1, "hifi2": 2, "hifi3": 3, "hifi4": 4}


class Model:
    def __init__(self, server):
        cfg = json.loads((SNAPSHOTS / server["model_revision"] / "config.json").read_text())
        sel = json.loads(SELECTED.read_text())
        if not server["precision"].startswith(sel["config_id"] + " "):
            raise ValueError(f"server precision {server['precision']} is not the selected {sel['config_id']}")
        if server["layer_count"] != cfg["num_hidden_layers"]:
            raise ValueError("server does not run all layers")
        self.pol, self.head_dtype = sel["policy"], sel["head"]["dtype"]
        self.H, self.V = cfg["hidden_size"], cfg["vocab_size"]
        self.nq, self.nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.hd = self.H // self.nq
        self.E, self.K, self.I = cfg["num_experts"], cfg["num_experts_per_tok"], cfg["moe_intermediate_size"]
        self.dense_I, self.Lc = cfg["intermediate_size"], cfg["conv_L_cache"]
        types = cfg["layer_types"]
        self.n_attn = sum(t == "full_attention" for t in types)
        self.n_dense = cfg["num_dense_layers"]  # the first layers (conv) use the dense MLP
        self.n_conv = len(types) - self.n_attn
        self.n_moe = len(types) - self.n_dense
        fids = {self.pol[k] for k in ("expert_fidelity", "attn_fidelity", "conv_fidelity", "dense_fidelity")}
        fids |= {self.pol["sdpa_prefill_fidelity"], self.pol["dispatch_fidelity"]}
        if len(fids) != 1:
            raise ValueError(f"mixed prefill matmul fidelities {fids}: the single-peak accounting does not apply")
        self.fidelity = fids.pop()
        self.peak_flops = FLOP_PER_CYCLE * CLOCK_HZ / FIDELITY_DIV[self.fidelity] * CORES * CHIPS
        self.peak_dram = DRAM_BPS * CHIPS

    # ---------------- prefill FLOPs ----------------
    def linear_params_per_token(self):
        H, hd = self.H, self.hd
        attn = H * (self.nq + 2 * self.nkv) * hd + self.nq * hd * H
        conv = H * 3 * H + H * H + H * self.Lc
        moe = H * self.E + self.K * 3 * H * self.I
        dense = 3 * H * self.dense_I
        return self.n_attn * attn + self.n_conv * conv + self.n_moe * moe + self.n_dense * dense

    def prefill_flops(self, start, end, final):
        """Useful FLOPs of computing prompt positions [start, end) of one request."""
        n = end - start
        lin = 2 * self.linear_params_per_token() * n
        ctx = (start + 1 + end) * n / 2  # sum over positions p of (p + 1)
        attn = self.n_attn * 2 * 2 * self.nq * self.hd * ctx
        head = 2 * self.H * self.V if final else 0
        return lin + attn + head

    # ---------------- decode DRAM bytes ----------------
    def b(self, k):
        return BYTES[self.pol[k]]

    def chip_weight_bytes(self):
        H, hd, tp = self.H, self.hd, CHIPS
        nq, nkv, I = self.nq // tp, self.nkv // tp, self.I // tp
        attn = H * (nq + 2 * nkv) * hd * self.b("attn_qkv") + nq * hd * H * self.b("attn_out")
        conv = H * 3 * (H // tp) * self.b("conv_in") + (H // tp) * H * self.b("conv_out")
        router = H * 256 * 2.0  # bf16, N padded
        dense = H * 2 * DENSE_I_DECODE * self.b("dense_w13") + DENSE_I_DECODE * H * self.b("dense_w2")
        head = H * HEAD_COLS * BYTES[self.head_dtype]
        expert = H * 2 * I * self.b("expert_gate_up") + I * H * self.b("expert_down")
        fixed = self.n_attn * attn + self.n_conv * conv + self.n_moe * router + self.n_dense * dense + head
        return fixed, expert

    def experts_read(self, n):
        bucket = next(x for x in DECODE_BUCKETS if x >= n)
        if bucket <= COMPACT_MAX:
            return bucket * self.K
        return self.E * (1 - (1 - self.K / self.E) ** n)

    def decode_step_bytes(self, positions):
        """All-chip DRAM bytes of one decode step whose active rows hold ``positions`` (next position per row)."""
        fixed, expert = self.chip_weight_bytes()
        n = len(positions)
        kv_row = 2 * (self.nkv // CHIPS) * self.hd * BYTES[self.pol["kv_cache"]]  # one position, K + V, per chip
        kv = self.n_attn * kv_row * (sum(positions) + n)  # read the cached positions, write the new one
        conv_state = self.n_conv * n * 2 * (self.H // CHIPS) * (self.Lc - 1) * 2.0
        chip = fixed + self.n_moe * self.experts_read(n) * expert + kv + conv_state
        return CHIPS * chip

    def describe(self):
        fixed, expert = self.chip_weight_bytes()
        return dict(
            linear_params_per_token=self.linear_params_per_token(),
            chip_fixed_weight_bytes_per_step=fixed,
            chip_quarter_expert_bytes=expert,
            prefill_fidelity=self.fidelity,
            peak_flops_per_second=self.peak_flops,
            peak_dram_bytes_per_second=self.peak_dram,
            layers=dict(attention=self.n_attn, conv=self.n_conv, moe=self.n_moe, dense=self.n_dense),
        )


def events(path):
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def check(run, n):
    server = json.loads((run / f"perf-b{n}-server.json").read_text())
    model = Model(server)
    ev_path = Path(server["events_log"])
    if not ev_path.is_file():
        raise SystemExit(f"timing instrumentation is off: no event log {ev_path}")
    evs = events(ev_path)
    kinds = {e.get("kind") for e in evs}
    if "traces_ready" not in kinds:
        raise SystemExit(f"{ev_path} lacks the server's warmup records: not this server's event log")
    if not any("mono" in e for e in evs):
        raise SystemExit(f"{ev_path} records lack the monotonic clock (adapter predates the stage-11 timing fields)")
    print(
        json.dumps(
            dict(
                action="check",
                concurrency=n,
                events_log=str(ev_path),
                records=len(evs),
                accounting=model.describe(),
                sample_step_bytes_n1_pos4096=model.decode_step_bytes([4096]),
                sample_step_bytes_n32_pos4096=model.decode_step_bytes([4096] * 32),
            ),
            indent=1,
        )
    )


def collect(run, n):
    perf_path = run / f"perf-b{n}.json"
    server_path = run / f"perf-b{n}-server.json"
    perf, server = json.loads(perf_path.read_text()), json.loads(server_path.read_text())
    model = Model(server)
    ev_path = Path(server["events_log"])
    time.sleep(3)  # the last measured request's release record follows its final response
    evs = [e for e in events(ev_path) if "mono" in e]
    starts = perf.get("start_times") or []
    if len(starts) != perf["completed"]:
        raise SystemExit("perf result lacks per-request start_times (--save-detailed)")
    w0 = min(starts)
    calls = []
    for e in evs:
        if e["kind"] in ("prefill", "decode"):
            calls.append(dict(kind=e["kind"], start=e["mono"] - e["dur_s"], end=e["mono"], ev=e))
    calls.sort(key=lambda c: c["start"])
    win = [c for c in calls if c["start"] >= w0]
    admissions = [(c, r) for c in win if c["kind"] == "prefill" for r in c["ev"]["rows"] if r[0] == 0]
    if len(admissions) != perf["completed"]:
        raise SystemExit(f"{len(admissions)} admissions in the window, expected {perf['completed']} measured requests")
    pre_window = [c for c in calls if c["start"] < w0]
    if pre_window and pre_window[-1]["end"] > w0:
        raise SystemExit("a pre-window call overlaps the measured window (warmup/probe still running)")
    releases = [e["mono"] for e in evs if e["kind"] == "release" and e["mono"] >= w0]
    if len(releases) < perf["completed"]:
        raise SystemExit(f"only {len(releases)} releases after the window opened, expected {perf['completed']}")
    w1 = max(max(releases), win[-1]["end"])
    if w1 - w0 > perf["duration"] + 1.0:
        raise SystemExit(f"server window {w1 - w0:.3f} s exceeds the client duration {perf['duration']:.3f} s")
    seconds = {"prefill": 0.0, "decode": 0.0}
    flops, dram, steps, tokens, rows_hist = 0.0, 0.0, 0, 0, {}
    per_call = []
    for i, c in enumerate(win):
        nxt = win[i + 1]["start"] if i + 1 < len(win) else w1
        seg = nxt - c["start"]
        seconds[c["kind"]] += seg
        if c["kind"] == "prefill":
            f = 0.0
            for P, E, _slot, Q in c["ev"]["rows"]:
                if E > 4096:
                    raise SystemExit(f"prefill row ends at {E} > 4096: not a measured request")
                f += model.prefill_flops(Q, E, final=(E == 4096))
                tokens += E - Q
            flops += f
            per_call.append(
                dict(
                    kind="prefill",
                    start=c["start"],
                    dur_s=c["ev"]["dur_s"],
                    segment_s=seg,
                    rows=c["ev"]["rows"],
                    flops=f,
                )
            )
        else:
            pos = c["ev"].get("active_pos")
            if pos is None:
                raise SystemExit("decode records lack active_pos")
            if not pos:
                per_call.append(
                    dict(kind="decode", start=c["start"], dur_s=c["ev"]["dur_s"], segment_s=seg, active=0, bytes=0)
                )
                continue
            by = model.decode_step_bytes(pos)
            dram += by
            steps += 1
            rows_hist[len(pos)] = rows_hist.get(len(pos), 0) + 1
            per_call.append(
                dict(
                    kind="decode",
                    start=c["start"],
                    dur_s=c["ev"]["dur_s"],
                    segment_s=seg,
                    active=len(pos),
                    pos_min=min(pos),
                    pos_max=max(pos),
                    bytes=by,
                )
            )
    expected_tokens = perf["total_input_tokens"]
    if tokens != expected_tokens:
        raise SystemExit(f"prefilled {tokens} tokens in the window, the measured prompts hold {expected_tokens}")
    # each request's first output token comes from its prefill; the rest from decode steps
    decode_tokens = sum(len(c["ev"]["active_pos"]) for c in win if c["kind"] == "decode")
    tag = f"b{n}"
    raw_copy = run / f"roofline-{tag}-events.jsonl.xz"
    with lzma.open(raw_copy, "wt") as f:
        for c in calls:
            if c["start"] >= w0 - 1.0:
                f.write(json.dumps(c["ev"]) + "\n")
    evidence = dict(
        concurrency=n,
        events_log=str(ev_path),
        raw_window_events=raw_copy.name,
        raw_window_events_sha256=hashlib.sha256(raw_copy.read_bytes()).hexdigest(),
        window_mono=[w0, w1],
        window_seconds=w1 - w0,
        client_duration_s=perf["duration"],
        calls=len(win),
        prefill_calls=sum(c["kind"] == "prefill" for c in win),
        decode_steps=steps,
        decode_row_tokens=decode_tokens,
        decode_active_rows_histogram=rows_hist,
        prefill_tokens=tokens,
        seconds=seconds,
        flops=flops,
        dram_bytes=dram,
        accounting=model.describe(),
        method=__doc__,
        per_call=per_call,
    )
    ev_file = run / f"roofline-{tag}-evidence.json"
    ev_file.write_text(json.dumps(evidence, indent=0) + "\n")
    ev_sha = hashlib.sha256(ev_file.read_bytes()).hexdigest()
    timing = (
        "adapter call records (CLOCK_MONOTONIC) in the measured window; each interval from a call's start to the "
        "next call's start charged to that call's phase (non-overlapping, includes host gaps, CCL, sampling and "
        "readback waits)"
    )
    peak_src = (
        "tt-perf-report ArchitectureSpec 'blackhole' (110 worker cores, 4096 FLOP/cycle/core at 1.35 GHz, "
        "HiFi2 = /2; DRAM 512 GB/s per chip) x 4 chips"
    )
    row = dict(
        performance_sha256=hashlib.sha256(perf_path.read_bytes()).hexdigest(),
        server_identity_sha256=digest(server),
        requests=perf["completed"],
        input_tokens=perf["total_input_tokens"],
        output_tokens=perf["total_output_tokens"],
        prefill=dict(
            flops=flops,
            seconds=seconds["prefill"],
            peak_flops_per_second=model.peak_flops,
            timing_scope="full_phase_wall_time",
            timing_method=timing,
            work_method="useful prompt FLOPs from executed prefill rows x HF config (2 x active matmul params "
            "+ causal attention + LM head once per request)",
            peak_source=peak_src,
            evidence=ev_file.name,
            evidence_sha256=ev_sha,
        ),
        decode=dict(
            dram_bytes=dram,
            seconds=seconds["decode"],
            peak_dram_bytes_per_second=model.peak_dram,
            timing_scope="full_phase_wall_time",
            timing_method=timing,
            work_method="per-step DRAM bytes at C1 stored dtypes over 4 chips: weights (experts: compact "
            "bucket x 4, else expected union of active rows' top-4), KV reads/writes at the "
            "logged positions, ShortConv state",
            peak_source=peak_src,
            evidence=ev_file.name,
            evidence_sha256=ev_sha,
        ),
    )
    out = run / "roofline.json"
    data = json.loads(out.read_text()) if out.exists() else {}
    data[str(n)] = row
    out.write_text(json.dumps(data, indent=1) + "\n")
    pf = 100 * flops / seconds["prefill"] / model.peak_flops
    dd = 100 * dram / seconds["decode"] / model.peak_dram
    print(
        json.dumps(
            dict(
                action="collect",
                concurrency=n,
                window_s=w1 - w0,
                seconds=seconds,
                prefill_tokens=tokens,
                decode_steps=steps,
                flops=flops,
                dram_bytes=dram,
                prefill_pct=pf,
                decode_pct=dd,
            ),
            indent=1,
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--concurrency", required=True, type=int, choices=(1, 32))
    ap.add_argument("--action", required=True, choices=("check", "collect"))
    a = ap.parse_args()
    (check if a.action == "check" else collect)(a.run_dir.resolve(), a.concurrency)


if __name__ == "__main__":
    main()
