import torch

import ttnn

md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
r = torch.rand(1, 1, 32, 64)
r[0, 0, 5] = float("nan")
m = torch.zeros(1, 1, 32, 1)
m[0, 0, :4] = 1
tr = ttnn.from_torch(r, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md)
tm = ttnn.from_torch(m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md)
for name, f in [("mul", lambda: ttnn.multiply(tr, tm)), ("where", lambda: ttnn.where(tm, tr, 0.0))]:
    try:
        o = ttnn.to_torch(f()).float()
        ref = torch.where(m.bool(), r, torch.zeros(()))
        print(
            name,
            "nan rows:",
            torch.isnan(o).any(-1).nonzero()[:, -1].tolist(),
            "max err active:",
            (o[0, 0, :4] - ref[0, 0, :4]).abs().max().item(),
        )
    except Exception as e:
        print(name, "ERR", str(e).splitlines()[0][:150])
ttnn.close_mesh_device(md)
