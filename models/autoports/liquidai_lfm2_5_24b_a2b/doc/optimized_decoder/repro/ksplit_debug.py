import torch

import ttnn

md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
T, H, E = 32, 2048, 64
g = torch.Generator().manual_seed(0)
x = torch.randn(1, 1, T, H, generator=g).bfloat16()
w = (torch.randn(H, E, generator=g) * 0.02).bfloat16()
ref = x.float()[0, 0] @ w.float()
tt = lambda a, dt=ttnn.bfloat16: ttnn.from_torch(a, dtype=dt, layout=ttnn.TILE_LAYOUT, device=md)
ckc = ttnn.init_device_compute_kernel_config(
    md.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)
tx = tt(x)
pcc = lambda a, b: torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
for ks in (8, 16, 32):
    tw = tt(w.reshape(ks, H // ks, E).unsqueeze(0))
    xs = ttnn.permute(ttnn.reshape(tx, (T, ks, H // ks)), (1, 0, 2))
    xs_t = ttnn.to_torch(xs).float()
    exp_xs = x.float()[0, 0].reshape(T, ks, H // ks).permute(1, 0, 2)
    part = ttnn.matmul(ttnn.reshape(xs, (1, ks, T, H // ks)), tw, compute_kernel_config=ckc, dtype=ttnn.float32)
    part_t = ttnn.to_torch(part).float().reshape(ks, T, E)
    exp_part = torch.einsum("ktj,kje->kte", exp_xs, w.float().reshape(ks, H // ks, E))
    rr = ttnn.experimental.fast_reduce_nc(part, dims=[1])
    print("  reduce out shape", rr.shape)
    red = ttnn.to_torch(rr).float().reshape(T, E)
    red2 = ttnn.to_torch(ttnn.sum(part, dim=1)).float().reshape(T, E)
    print(
        f"KS={ks:2d}: permute exact={torch.equal(xs_t, exp_xs)}  partial pcc {pcc(part_t, exp_part):.6f}  "
        f"fast_reduce_nc pcc {pcc(red, ref):.6f} max|err| {(red-ref).abs().max():.4f}  ttnn.sum pcc {pcc(red2, ref):.6f}",
        flush=True,
    )
ttnn.close_mesh_device(md)
