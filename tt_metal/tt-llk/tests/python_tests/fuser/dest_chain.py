# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

from helpers.chip_architecture import ChipArchitecture

from .fpu_node import FpuNode
from .pack_node import PackNode
from .sfpu_node import SfpuNode

if TYPE_CHECKING:
    from .fuser_config import GlobalConfig
    from .l1_operation import L1Operation

UNPACK = "dest_dvalid_client::UNPACK"
FPU = "dest_dvalid_client::FPU"
SFPU = "dest_dvalid_client::SFPU"
PACK = "dest_dvalid_client::PACK"
NONE = "dest_dvalid::NONE"
CHAIN_ORDER = (UNPACK, FPU, SFPU, PACK)


def active(config: "GlobalConfig") -> bool:
    return (
        config.architecture == ChipArchitecture.QUASAR
        and config.quasar_use_dvalid
        and not config.skip_sync
    )


def unit_of(node) -> str:
    if isinstance(node, FpuNode):
        return UNPACK if node.unpack_to_dest.value else FPU
    if isinstance(node, SfpuNode):
        return SFPU
    if isinstance(node, PackNode):
        return PACK
    raise TypeError(f"{node} does not use dest")


def section_order(operation: "L1Operation") -> List[str]:
    order = []
    for node in operation.math_nodes + operation.pack_nodes:
        unit = unit_of(node)
        if not order or order[-1] != unit:
            order.append(unit)
    return order


def units(operation: "L1Operation") -> List[str]:
    for node in operation.pack_nodes:
        if isinstance(node, SfpuNode):
            raise NotImplementedError(
                "dest chain: SFPU on the pack thread is not supported"
            )
    used = set(section_order(operation))
    return [unit for unit in CHAIN_ORDER if unit in used]


def pipeline_units(operation: "L1Operation") -> List[str]:
    first = operation
    while first.previous_operation is not None:
        first = first.previous_operation
    used = set()
    op = first
    while op is not None:
        used.update(units(op))
        op = op.next_operation
    return [unit for unit in CHAIN_ORDER if unit in used]


@dataclass(frozen=True)
class Turn:

    unit: str
    next: str
    head: bool
    word_ready: bool

    def cpp(self) -> str:
        head = str(self.head).lower()
        ready = str(self.word_ready).lower()
        return f"DestTurn{{{self.unit}, {self.next}, {head}, {ready}}}"


def turns(operation: "L1Operation") -> List[Turn]:
    order = section_order(operation)
    result = []
    for step, unit in enumerate(order):
        next_unit = order[step + 1] if step + 1 < len(order) else NONE
        result.append(Turn(unit, next_unit, step == 0, order.count(unit) == 1))
    return result


def dst(operation: "L1Operation") -> str:
    return operation.dest_sync.cpp_enum_value


def joins(config: "GlobalConfig", operation: "L1Operation") -> str:
    if not active(config):
        return ""
    code = ""
    if operation.previous_operation is None:
        code += "".join(
            f"_llk_dest_chain_join_<{unit}>();\n" for unit in pipeline_units(operation)
        )
    head = section_order(operation)[0]
    previous = operation.previous_operation
    if previous is not None and section_order(previous)[0] != head:
        code += f"_llk_dest_chain_op_begin_<{head}>();\n"
    for turn in turns(operation):
        if turn.word_ready:
            code += f"_llk_dest_chain_prepare_<{turn.cpp()}, {dst(operation)}>();\n"
    return code


def leaves(config: "GlobalConfig", operation: "L1Operation") -> str:
    if not active(config):
        return ""
    order = section_order(operation)
    if operation.next_operation is None:
        return "".join(
            f"_llk_dest_chain_leave_<{unit}>();\n" for unit in pipeline_units(operation)
        )
    next_head = section_order(operation.next_operation)[0]
    if next_head == order[0]:
        return ""
    return f"_llk_dest_chain_op_end_<{order[-1]}, {next_head}>();\n"


class Section:

    def __init__(self, config: "GlobalConfig", operation: "L1Operation"):
        self.on = active(config)
        self.dst = dst(operation)
        self.args = f"{self.dst}, {config.dest_acc.cpp_enum_value}"
        self.turns = turns(operation)
        self.step = 0

    def take(self, node) -> str:
        if not self.on:
            return ""
        unit = unit_of(node)
        if self.step > 0 and unit == self.turns[self.step - 1].unit:
            return ""
        turn = self.turns[self.step]
        assert unit == turn.unit, "section order drifted from the node order"
        self.step += 1
        if turn.head:
            return f"_llk_dest_chain_start_<{turn.cpp()}, {self.dst}>();\n"
        holder = self.turns[self.step - 2]
        return f"_llk_dest_chain_pass_<{holder.cpp()}, {turn.cpp()}, {self.args}>();\n"

    def takes(self, nodes) -> str:
        return "".join(self.take(node) for node in nodes)

    def end(self) -> str:
        if not self.on:
            return ""
        assert self.step == len(
            self.turns
        ), "a section ended before every unit took dest"
        last = self.turns[-1]
        head = self.turns[0].unit
        return f"_llk_dest_chain_end_<{last.cpp()}, {head}, {self.args}>();\n"
