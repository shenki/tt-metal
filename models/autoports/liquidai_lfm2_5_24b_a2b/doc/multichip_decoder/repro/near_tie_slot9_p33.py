# Diagnostic for test_multichip_increasing_lengths_same_instance[attn_moe] slot 9 prompt 33 (decode update PCC
# 0.9819 vs HF, deterministic): HF's k-th / (k+1)-th biased router score gap at the first decode position (33) is
# 3.44e-3, outside the harness near-tie window (1e-3).  Re-runs the same schedule with the window at 4e-3: if the
# update PCC vs HF-with-that-swap is >= 0.99, the deviation is exactly one top-k flip at a near tie.
#   python near_tie_slot9_p33.py
import ttnn
from models.autoports.liquidai_lfm2_5_24b_a2b.reference import hf_layer
from models.autoports.liquidai_lfm2_5_24b_a2b.tests import test_functional_decoder as tf
from models.autoports.liquidai_lfm2_5_24b_a2b.tests.harness import User
from models.autoports.liquidai_lfm2_5_24b_a2b.tests.test_multichip_decoder import MC
from models.autoports.liquidai_lfm2_5_24b_a2b.tt.multichip_decoder import mesh_device_params

hf_layer.near_tie_alternatives.__defaults__ = (4e-3,)
p = mesh_device_params()
ttnn.set_fabric_config(p["fabric_config"], router_config=p["fabric_router_config"])
md = ttnn.open_mesh_device(
    ttnn.MeshShape(1, 4), trace_region_size=p["trace_region_size"], l1_small_size=p["l1_small_size"]
)
try:
    lens = [2047, 2048, 2049, 4100, 32, 33]
    fg = [User(slot=9, prompt_len=n, decode_steps=2, seed=500 + n, join_delay=i % 3) for i, n in enumerate(lens)]
    res = tf.run_interleaved(md, tf.LAYERS["attn_moe"], fg, chunk_size=2048, decoder_factory=MC)
    import torch

    from models.autoports.liquidai_lfm2_5_24b_a2b.tests.harness import hf_config, layer_state_dict, pcc

    u = next(u for u, _, _ in res if u.slot == 9 and u.prompt_len == 33)
    n, m = u.prompt_len, len(u.tt_decode)
    upd = torch.stack(u.tt_decode) - u.x[n : n + m]
    for pos in (n, n + 1):
        hl, rope = hf_layer.build_hf_layer(
            hf_config(), tf.LAYERS["attn_moe"], layer_state_dict(tf.LAYERS["attn_moe"], True)
        )
        hf_layer.force_expert_swap(hl, pos, rank=3, alt_rank=4)
        with torch.no_grad():
            alt = hf_layer.hf_layer_forward(hl, rope, u.x.unsqueeze(0))[0]
        print(
            f"NEARTIE slot 9 prompt 33: decode update PCC vs plain HF {u.delta[1]:.6f}; vs HF with experts 4/5 swapped "
            f"at position {pos}: {pcc(upd, alt[n : n + m] - u.x[n : n + m]):.6f}",
            flush=True,
        )
finally:
    ttnn.close_mesh_device(md)
