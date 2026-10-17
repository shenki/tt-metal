# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pareto charts of the datatype sweep: accuracy (main AIME24 reference top-1 / top-5) vs traced teacher-forcing decode
t/s/u, one point per evaluated config (``doc/datatype_sweep/sweep_results.json``), the non-dominated frontier as a step
line, the selected config in red, a dotted vertical line at the minimum allowed accuracy (``thresholds.json``).

Usage: python -m models.autoports.liquidai_lfm2_5_24b_a2b.tests.plot_datatype_sweep [--selected ID]
Writes doc/datatype_sweep/{top1,top5}_perf_pareto.png.

Colors are slots of the dataviz reference palette (light surface): passing points / frontier slot 1 blue, the selected
point red (required by the stage goal) with a larger marker and a text label (identity never by color alone); failing
configs hollow gray; text in text tokens.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SWEEP = Path(__file__).resolve().parents[1] / "doc" / "datatype_sweep"
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, RED, GRAY = "#2a78d6", "#e34948", "#9a9993"
ORANGE = "#eb8a28"


def frontier(points):
    """Non-dominated points (maximize accuracy and speed), sorted by accuracy."""
    out = []
    for p in points:
        if not any(q != p and q[0] >= p[0] and q[1] >= p[1] and (q[0] > p[0] or q[1] > p[1]) for q in points):
            out.append(p)
    return sorted(out)


def base_id(cid):
    """Repeats (``<id>_r<n>``, ``H2H<n>_<id>``) belong to their base config."""
    import re

    cid = re.sub(r"^H2H\d+_", "", cid)
    return re.sub(r"_r\d+$", "", cid)


def grouped(rows):
    """One point per distinct config: median TF decode over its runs (repeats / head-to-head), its accuracy row."""
    import statistics

    out = {}
    for r in rows:
        if not r.get("tf_decode_t_s_u") or r["status"] not in ("pass", "fail"):
            continue
        out.setdefault(base_id(r["config_id"]), []).append(r)
    pts = []
    for cid, rs in out.items():
        base = next((r for r in rs if r["config_id"] == cid), rs[0])
        pt = dict(base)
        pt["config_id"] = cid
        pt["tf_decode_t_s_u"] = statistics.median(r["tf_decode_t_s_u"] for r in rs)
        pt["n_runs"] = len(rs)
        pt["status"] = "pass" if all(r["status"] == "pass" for r in rs) else "fail"
        pts.append(pt)
    return pts


def plot(rows, metric, threshold, selected, path, rejected=None):
    rejected = rejected or {}
    ok = grouped(rows)
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    xs = [100 * r[metric] for r in ok]
    ys = [r["tf_decode_t_s_u"] for r in ok]
    xlo = min(xs + [100 * threshold]) - 2
    ax.axvspan(xlo, 100 * threshold, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.axvline(100 * threshold, color=TEXT2, ls=":", lw=1.5, zorder=1)
    ax.text(
        100 * threshold,
        0.98,
        f"  minimum {metric} {100 * threshold:.0f} %",
        color=TEXT2,
        fontsize=8,
        va="top",
        transform=ax.get_xaxis_transform(),
    )
    front = frontier([(x, y, r["config_id"]) for x, y, r in zip(xs, ys, ok)])
    if front:
        fx, fy = [p[0] for p in front], [p[1] for p in front]
        ax.step(fx, fy, where="post", color=BLUE, lw=2, zorder=2, label="Pareto frontier")
    for x, y, r in zip(xs, ys, ok):
        if r["config_id"] == selected:
            continue
        if r["status"] == "pass" and r["config_id"] in rejected:
            ax.scatter([x], [y], s=64, facecolor=SURFACE, edgecolor=ORANGE, linewidth=2, zorder=3)
        elif r["status"] == "pass":
            ax.scatter([x], [y], s=64, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
        else:
            ax.scatter([x], [y], s=64, facecolor="none", edgecolor=GRAY, linewidth=1.8, zorder=3)
    sel = next((r for r in ok if r["config_id"] == selected), None)
    if sel is not None:
        x, y = 100 * sel[metric], sel["tf_decode_t_s_u"]
        ax.scatter([x], [y], s=150, color=RED, edgecolor=SURFACE, linewidth=2, zorder=4, label=f"selected: {selected}")
    # labels: one column per accuracy value, de-overlapped vertically, leader line to the point
    span = (max(ys) - min(ys)) or 1.0
    gap = span * 0.045
    cols = {}
    for x, y, r in zip(xs, ys, ok):
        cols.setdefault(round(x, 3), []).append((y, r))
    for x, items in cols.items():
        items.sort(key=lambda t: t[0])
        pos, last = [], None
        for y, r in items:
            ly = y if last is None else max(y, last + gap)
            pos.append(ly)
            last = ly
        shift = max(0.0, pos[-1] - (max(ys) + gap)) if pos else 0.0
        pos = [p - shift for p in pos]
        for (y, r), ly in zip(items, pos):
            lab = r["config_id"] + (f" (n={r['n_runs']})" if r.get("n_runs", 1) > 1 else "")
            is_sel = r["config_id"] == selected
            if is_sel:
                lab += " - selected"
            ax.annotate(
                lab,
                (x, y),
                xytext=(x + 0.25 * (100.5 - xlo) / 12.5, ly),
                textcoords="data",
                fontsize=8,
                color=TEXT if is_sel else TEXT2,
                fontweight="bold" if is_sel else "normal",
                va="center",
                arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8, shrinkA=0, shrinkB=4),
            )
    ax.scatter([], [], s=64, color=BLUE, label="passes the accuracy gate")
    if any(r["config_id"] in rejected for r in ok):
        ax.scatter(
            [],
            [],
            s=64,
            facecolor=SURFACE,
            edgecolor=ORANGE,
            linewidth=2,
            label="passes the accuracy gate, rejected by selection rule v2 (probe bars / first token)",
        )
    if any(r["status"] != "pass" for r in ok):
        ax.scatter([], [], s=64, facecolor="none", edgecolor=GRAY, label="fails a gate")
    ax.set_xlabel(f"main-reference {metric} accuracy (%)  [AIME24 chat, 100 tokens]", color=TEXT)
    ax.set_ylabel("traced teacher-forcing decode, t/s/u (batch 1; median over repeats)", color=TEXT)
    ax.set_title(
        f"LFM2.5-24B-A2B datatype sweep: {metric} vs decode speed (1x4 Blackhole p150b)",
        color=TEXT,
        fontsize=11,
        loc="left",
    )
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=TEXT2)
    ax.set_xlim(xlo, 100.5 + (100.5 - xlo) * 0.22)
    import math

    step = 2 if (100 - xlo) > 6 else 0.5
    ax.set_xticks([t for t in (math.ceil(xlo / step) * step + i * step for i in range(60)) if t <= 100.0001])
    if len({round(v, 3) for v in xs}) == 1:
        ax.text(
            0.02,
            0.02,
            f"all evaluated configs reach {xs[0]:.0f} % {metric}: the frontier is the fastest point",
            transform=ax.transAxes,
            fontsize=8,
            color=TEXT2,
        )
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT2, loc="lower left", bbox_to_anchor=(0.0, 0.06))
    if sel is not None and front and selected not in [p[2] for p in front]:
        best = max(front, key=lambda p: p[1])
        pct = 100 * (best[1] / sel["tf_decode_t_s_u"] - 1)
        faster = [p[2] for p in front if p[1] > sel["tf_decode_t_s_u"]]
        if faster and all(c in rejected for c in faster):
            why = f"faster frontier points ({', '.join(faster)}) fail selection rule v2 (thresholds.json)"
        else:
            why = "inside the 0.5 % tie band (build-to-build noise ~1 %), so the simpler config was chosen"
        fig.text(
            0.01,
            0.035,
            f"selected {selected} is {pct:.2f} % below frontier point {best[2]}: {why}",
            fontsize=7.5,
            color=TEXT2,
        )
    dropped = sorted({base_id(r["config_id"]) for r in rows if not r.get("tf_decode_t_s_u")})
    if dropped:
        fig.text(0.01, 0.01, f"not plotted (error / infeasible): {', '.join(dropped)}", fontsize=7, color=TEXT2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", default=None)
    a = ap.parse_args()
    rows = json.loads((SWEEP / "sweep_results.json").read_text())
    th = json.loads((SWEEP / "thresholds.json").read_text())
    selected = a.selected
    if selected is None and (SWEEP / "selected_precision_config.json").exists():
        selected = json.loads((SWEEP / "selected_precision_config.json").read_text())["config_id"]
    rejected = th.get("selection_rule_v2", {}).get("rejected", {})
    plot(rows, "top1", th["top1_min"], selected, SWEEP / "top1_perf_pareto.png", rejected)
    plot(rows, "top5", th["top5_min"], selected, SWEEP / "top5_perf_pareto.png", rejected)


if __name__ == "__main__":
    main()
