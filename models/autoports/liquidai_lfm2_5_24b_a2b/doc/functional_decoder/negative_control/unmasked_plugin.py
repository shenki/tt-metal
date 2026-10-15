"""Negative control: every slot 'active' in conv decode (the pre-fix behaviour)."""
import sys

import torch
from loguru import logger

import ttnn


def pytest_collection_finish(session):
    for name, mod in list(sys.modules.items()):
        if name.endswith("liquidai_lfm2_5_24b_a2b.tests.harness"):
            orig = mod.DecodeInputs.update

            def update(self, x, cur_pos, page_table, _orig=orig):
                _orig(self, x, cur_pos, page_table)
                host = ttnn.from_torch(
                    torch.ones(1, 1, cur_pos.shape[0], 1),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
                ttnn.copy_host_to_device_tensor(host, self.slot_active)

            mod.DecodeInputs.update = update
            logger.warning(f"NEGATIVE CONTROL: patched {name}.DecodeInputs.update (all slots active)")
