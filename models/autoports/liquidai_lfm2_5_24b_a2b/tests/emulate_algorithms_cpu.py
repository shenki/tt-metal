import math

import torch
from transformers import AutoConfig
from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeShortConv, Lfm2MoeSparseMoeBlock

from models.autoports.liquidai_lfm2_5_24b_a2b.tt.functional_decoder import ConvPrefillInputs

torch.manual_seed(0)
cfg = AutoConfig.from_pretrained(
    "/home/ttuser/.cache/huggingface/hub/models--LiquidAI--LFM2.5-24B-A2B/snapshots/1769516ac588c4c558089c94f27d96b0dc67b183"
)
H = cfg.hidden_size
T = 32
conv = Lfm2MoeShortConv(cfg, 0).float().eval()
for p in conv.parameters():
    torch.nn.init.normal_(p, std=0.05)
W_in = conv.in_proj.weight.T
W_out = conv.out_proj.weight.T
w = conv.conv.weight[:, 0, :]
state = [torch.zeros(T, H), torch.zeros(T, H)]


def gated(x):
    bcx = x @ W_in
    return bcx[:, :H] * bcx[:, 2 * H :], bcx[:, H : 2 * H]


def prefill(x, slot, valid, continuation):
    S = x.shape[0]
    bx, c = gated(x)
    c0, c1, t0, t1, keep = [
        t[0, 0]
        for t in ConvPrefillInputs.host_tensors(slot=slot, valid_len=valid, padded_len=S, continuation=continuation)
    ]
    carry = c0 @ state[0] + c1 @ state[1]
    ext = torch.cat([carry, bx])
    out = sum(ext[30 + k : 30 + k + S] * w[:, k] for k in range(3))
    for j, sel in enumerate((t0, t1)):
        state[j] = state[j] * keep + sel @ ext
    return (c * out) @ W_out


def decode(xrows):
    bx, c = gated(xrows)
    out = state[0] * w[:, 0] + state[1] * w[:, 1] + bx * w[:, 2]
    state[0] = state[1].clone()
    state[1] = bx.clone()
    return (c * out) @ W_out


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a -= a.mean()
    b -= b.mean()
    return float(a @ b / (a.norm() * b.norm()))


with torch.no_grad():
    for L, chunk in [(1, 64), (2, 64), (3, 64), (31, 64), (65, 32), (300, 128)]:
        for slot in (0, 5, 31):
            D = 4
            x = torch.randn(L + D, H)
            ref = conv(x.unsqueeze(0))[0]
            # garbage in other slots
            state = [torch.randn(T, H), torch.randn(T, H)]
            other = [s.clone() for s in state]
            outs = []
            pos = 0
            while pos < L:
                n = min(chunk, L - pos)
                S = math.ceil(n / 32) * 32
                xx = torch.zeros(S, H)
                xx[:n] = x[pos : pos + n]
                outs.append(prefill(xx, slot, n, pos > 0)[:n])
                pos += n
            pre = torch.cat(outs)
            dec = []
            for t in range(D):
                rows = torch.randn(T, H)
                rows[slot] = x[L + t]
                dec.append(decode(rows)[slot])
            assert (pre - ref[:L]).abs().max() < 1e-3, (L, slot, (pre - ref[:L]).abs().max())
            assert (torch.stack(dec) - ref[L:]).abs().max() < 1e-3, (L, slot)
    print("shortconv emulation exact")
    # router emulation
    moe = Lfm2MoeSparseMoeBlock(cfg).float().eval()
    for p in moe.parameters():
        torch.nn.init.normal_(p, std=0.02)
    moe.expert_bias.normal_(std=0.01)
    x = torch.randn(64, H)
    s = torch.sigmoid(x @ moe.gate.weight.T)
    idx = torch.topk(s + moe.expert_bias, 4).indices
    sel = s.gather(-1, idx)
    wts = sel / (sel.sum(-1, keepdim=True) + 1e-6)
    dense = torch.zeros_like(s).scatter(-1, idx, wts)
    gu = moe.experts.gate_up_proj
    dn = moe.experts.down_proj
    out = torch.zeros_like(x)
    for e in range(64):
        g, u = (x @ gu[e].T).chunk(2, -1)
        out += dense[:, e : e + 1] * ((torch.nn.functional.silu(g) * u) @ dn[e].T)
    ref = moe(x.unsqueeze(0))[0]
    print("moe emulation max err", (out - ref).abs().max().item())
