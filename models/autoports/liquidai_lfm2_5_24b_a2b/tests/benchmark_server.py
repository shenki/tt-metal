# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-11 server-control hook (``performance_server_command`` of ``benchmark_stage run``).

    python benchmark_server.py --max-num-seqs {32|1} --base-url http://127.0.0.1:8000 --output <run>/perf-bN-server.json

Reuses the running LFM2 vLLM server when it is the one this hook launched with the requested slot count, otherwise stops
only that owned server (its own process group) and launches the requested profile: the stage-10 serving configuration
(``doc/optimized_vllm``: device sampling, trace region 768 MiB, prefill_chunk_alignment 2048, decode interleave 1) with
``--no-enable-prefix-caching``, the ``deepseek_r1`` reasoning parser (the chat template opens ``<think>``; the parser
returns the text after ``</think>`` as the answer) and the adapter's event log (``LFM2_VLLM_EVENTS``) for the roofline
collector.  The written identity is read back from the running server: its ``/proc`` command line, startup log and
``/v1/models``; the startup log lines it relies on are retained as ``configuration-bN.log`` next to the output.
Ownership state lives in ``generated/lfm2_benchmark/`` (``server.json``: pid, command, log, events)."""

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MODEL = "LiquidAI/LFM2.5-24B-A2B"
MODEL_DIR = Path(__file__).resolve().parents[1]
TT_METAL = MODEL_DIR.parents[2]
IMPLEMENTATION = "models/autoports/liquidai_lfm2_5_24b_a2b"
GENERATOR_MODULE = "models.autoports.liquidai_lfm2_5_24b_a2b.tt.generator_vllm"
PLUGIN_ROOT = Path(os.environ.get("VLLM_TT_PLUGIN_ROOT", TT_METAL.parent / "vllm-tt-plugin"))
STATE_DIR = TT_METAL / "generated" / "lfm2_benchmark"
MAX_MODEL_LEN = 131072
TT_CONFIG = {
    "sample_on_device_mode": "all",
    "trace_region_size": 805306368,
    "l1_small_size": 24576,
    "prefill_chunk_alignment": 2048,
    "decode_interleave_prefill_steps": 1,
}
READY_TIMEOUT_S = 1200


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _cmdline(pid):
    try:
        return [a.decode() for a in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if a]
    except OSError:
        return []


def _device_pids():
    pids = set()
    for f in Path("/proc/driver/tenstorrent").glob("*/pids"):
        pids |= {int(x) for x in f.read_text().split() if x.strip().isdigit()}
    return pids


def _get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def _models(base_url):
    try:
        return _get(base_url.rstrip("/") + "/v1/models")
    except Exception:
        return None


def _git(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True).stdout.strip()


def _commit(path):
    sha = _git(path, "rev-parse", "HEAD")
    dirty = _git(path, "status", "--porcelain", "--untracked-files=no")
    return sha + ("+dirty" if dirty else "")


def _stop(state):
    pid = state["pid"]
    if not _alive(pid):
        return
    print(f"stopping owned server pid {pid} (process group)", flush=True)
    for sig, wait in ((signal.SIGINT, 90), (signal.SIGTERM, 60), (signal.SIGKILL, 30)):
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline and _alive(pid):
            time.sleep(0.5)
        if not _alive(pid):
            break
    deadline = time.monotonic() + 120  # EngineCore children release the chips after the API server
    while time.monotonic() < deadline and _device_pids():
        time.sleep(1)
    if _device_pids():
        raise RuntimeError(f"chips still held after stopping the owned server: pids {sorted(_device_pids())}")


def _launch(n, port):
    if _device_pids():
        raise RuntimeError(f"chips are held by processes this hook does not own: {sorted(_device_pids())}")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    log, events = STATE_DIR / f"server-b{n}-{stamp}.log", STATE_DIR / f"events-b{n}-{stamp}.jsonl"
    env = dict(os.environ)
    env.update(
        TT_METAL_HOME=str(TT_METAL),
        # the adapter implements readiness_check.contract.Generator (tt-model-bringup runtime)
        PYTHONPATH=f"{Path(os.environ['TT_MODEL_BRINGUP_ROOT']) / 'runtime'}:{TT_METAL}",
        TT_MODEL_CLASS_OVERRIDES=f"TTLfm2MoeForCausalLM={GENERATOR_MODULE}:TTLfm2MoeForCausalLM",
        VLLM_TT_PLUGIN_ROOT=str(PLUGIN_ROOT),
        HF_HUB_OFFLINE="1",
        MESH_DEVICE="P300x2",
        HF_MODEL=MODEL,
        VLLM_RPC_TIMEOUT="300000",
        LFM2_VLLM_EVENTS=str(events),
    )
    env.pop("LFM2_VLLM_LAYERS", None)
    argv = [
        str(TT_METAL / "python_env" / "bin" / "python"),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL,
        "--block-size",
        "64",
        "--max-num-seqs",
        str(n),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--no-enable-prefix-caching",
        "--reasoning-parser",
        "deepseek_r1",
        "--port",
        str(port),
        "--additional-config",
        json.dumps({"tt": TT_CONFIG}),
    ]
    with open(log, "w") as out:
        proc = subprocess.Popen(
            argv, cwd=TT_METAL, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True
        )
    state = dict(pid=proc.pid, max_num_seqs=n, argv=argv, log=str(log), events=str(events), launched=time.time())
    (STATE_DIR / "server.json").write_text(json.dumps(state, indent=1))
    print(f"launched pid {proc.pid}: log {log}", flush=True)
    return state


def _wait_ready(state, base_url):
    t0 = time.monotonic()
    while time.monotonic() - t0 < READY_TIMEOUT_S:
        if not _alive(state["pid"]):
            raise RuntimeError(f"server exited during startup; see {state['log']}")
        if _models(base_url):
            print(f"ready after {time.monotonic() - t0:.1f} s", flush=True)
            return time.monotonic() - t0
        time.sleep(2)
    _stop(state)
    raise RuntimeError(f"server not ready after {READY_TIMEOUT_S} s; stopped it; see {state['log']}")


def _matches(state, n):
    argv = _cmdline(state["pid"]) if _alive(state["pid"]) else []
    return bool(argv) and argv == state["argv"] and state["max_num_seqs"] == n


def _find(pattern, text, what):
    m = re.search(pattern, text, re.M)
    if not m:
        raise RuntimeError(f"server log lacks {what} ({pattern!r})")
    return m


def identity(state, n, base_url, output):
    log = Path(state["log"]).read_text(errors="replace")
    argv = _cmdline(state["pid"])
    models = _models(base_url)
    card = next(m for m in models["data"] if m["id"] == MODEL)
    nondefault = _find(r"non-default args: (\{.*\})$", log, "vLLM non-default args").group(0)
    engine = _find(r"Initializing a V1 LLM engine \(v([\d.]+)\) with config: (.*)$", log, "engine config")
    snapshot = _find(r"model='[^']*/snapshots/([0-9a-f]{40})'", engine.group(2), "model snapshot").group(1)
    tok_snapshot = _find(r"tokenizer='[^']*/snapshots/([0-9a-f]{40})'", engine.group(2), "tokenizer snapshot").group(1)
    prefix = _find(r"enable_prefix_caching=(\w+)", engine.group(2), "prefix caching setting").group(1)
    init = _find(
        r"TTLfm2MoeForCausalLM: precision (\S+) \((\S+)\), layers (\d+), max_model_len (\d+), .* module (\S+?),",
        log,
        "adapter init line",
    )
    arch = _find(r"Creating TopologyDiscovery for architecture: (\w+)", log, "chip architecture").group(1)
    chips = _find(r"Opening local chip ids/PCIe ids: \{([\d, ]+)\}", log, "local chips").group(1)
    hf_config = json.loads(
        (
            Path.home() / ".cache/huggingface/hub/models--LiquidAI--LFM2.5-24B-A2B/snapshots" / snapshot / "config.json"
        ).read_text()
    )
    n_chips = len([c for c in chips.split(",") if c.strip()])
    ev_lines = [
        l
        for l in log.splitlines()
        if re.search(
            r"non-default args|Initializing a V1 LLM engine|"
            r"TTLfm2MoeForCausalLM: |TopologyDiscovery for architecture|Opening local chip ids|"
            r"Chunked prefill enabled|reasoning|Application startup complete|init engine",
            l,
        )
    ]
    out_dir = Path(output).parent
    evidence = out_dir / f"configuration-b{n}.log"
    evidence.write_text(
        "\n".join(
            [
                f"# server pid {state['pid']} /proc cmdline:",
                json.dumps(argv),
                f"# owned-server state: {json.dumps(state)}",
                f"# GET {base_url}/v1/models:",
                json.dumps(models),
                f"# startup log {state['log']} (identity lines):",
                *ev_lines,
            ]
        )
        + "\n"
    )
    if init.group(5) != str(MODEL_DIR / "tt" / "generator_vllm.py"):
        raise RuntimeError(f"server imported {init.group(5)}, not this checkout's adapter")
    return {
        "model": MODEL,
        "implementation": IMPLEMENTATION,
        "generator_module": GENERATOR_MODULE,
        "generator_file": init.group(5),
        "model_revision": snapshot,
        "tokenizer_revision": tok_snapshot,
        "precision": f"{init.group(1)} ({Path(init.group(2)).relative_to(TT_METAL)})",
        "layer_count": int(init.group(3)),
        "configured_layer_count": int(hf_config["num_hidden_layers"]),
        "source_commits": {
            "tt-metal": _commit(TT_METAL),
            "vllm-tt-plugin": _commit(PLUGIN_ROOT),
            "vllm": engine.group(1),
        },
        "hardware": f"{n_chips}x {arch} (QuietBox p150, chips {{{chips}}}), 1x4 mesh (the adapter rejects any other mesh), tensor parallel 4, FABRIC_1D_RING",
        "server_command": argv,
        "prefix_caching": {"False": False, "True": True}.get(prefix, prefix),
        "max_num_seqs": int(re.search(r"'max_num_seqs': (\d+)", nondefault).group(1)),
        "max_model_len": int(card["max_model_len"]),
        "base_url": base_url,
        "configuration_evidence": evidence.name,
        "configuration_evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        "events_log": state["events"],
        "server_log": state["log"],
        "server_pid": state["pid"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-num-seqs", type=int, required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    port = int(a.base_url.rsplit(":", 1)[1].split("/")[0])
    path = STATE_DIR / "server.json"
    state = json.loads(path.read_text()) if path.exists() else None
    t0 = time.monotonic()
    action = "reused"
    if state and _matches(state, a.max_num_seqs) and _models(a.base_url):
        print(f"reusing owned server pid {state['pid']} ({a.max_num_seqs} slots)", flush=True)
    else:
        if state and _alive(state["pid"]):
            _stop(state)
        elif _models(a.base_url):
            raise RuntimeError(f"{a.base_url} is served by a process this hook does not own")
        state = _launch(a.max_num_seqs, port)
        _wait_ready(state, a.base_url)
        action = "launched"
    ident = identity(state, a.max_num_seqs, a.base_url, a.output)
    if ident["max_num_seqs"] != a.max_num_seqs or ident["prefix_caching"] is not False:
        raise RuntimeError(f"running server does not match the request: {ident}")
    ident["hook_action"] = action
    ident["hook_seconds"] = round(time.monotonic() - t0, 1)
    Path(a.output).write_text(json.dumps(ident, indent=1) + "\n")
    print(json.dumps(ident, indent=1))


if __name__ == "__main__":
    sys.exit(main())
