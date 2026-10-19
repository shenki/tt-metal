// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/tilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/activations.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

namespace ckl = compute_kernel_lib;

template <uint32_t block_ct, uint32_t Mt>
TT_KERNEL void compute(uint32_t wi_start, uint32_t wi_count) {
    // Kimi-K3 uses a fixed four-tap causal convolution, with three preceding rows supplied by history.
    constexpr uint32_t tap_count = 4;
    // Group up to four channel tiles per destination acquire; the host sizes partial for two blocks, so a
    // dst_tiles reservation never waits on the previous tap's block that this loop is still reading.
    constexpr uint32_t dst_tiles = block_ct % 4 == 0 ? 4 : (block_ct % 2 == 0 ? 2 : 1);
    static_assert(dst_tiles <= compute_kernel_lib::DEST_AUTO_LIMIT, "dst_tiles must fit the destination registers");
    compute_kernel_hw_startup(dfb::act_rm, dfb::act_tile, dfb::output);
    DataflowBuffer activation(dfb::act_tile);
    DataflowBuffer weights(dfb::weights);
    DataflowBuffer partial(dfb::partial);
    DataflowBuffer output(dfb::output);

    // Every tap multiplies the activation block by its row-broadcast tap weights. Weights are held for the run of
    // items that share a channel block, so their window stays caller-owned and is addressed at the tap's offset.
    using ActivationTimesWeights = ckl::BinaryFpu<
        ckl::BinaryFpuOp::Mul,
        ckl::input(dfb::act_tile, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
        ckl::input(
            dfb::weights,
            ckl::BroadcastDim::Row,
            ckl::WaitPolicy::None,
            ckl::PopPolicy::None,
            ckl::InputTileMapping::Block,
            ckl::DataFormatReconfig::Enabled,
            ckl::TileAddressing::Offset)>;
    // Later taps add the previous taps' BF16 partial sum in DST.
    using AddPartial = ckl::DestReuseBinary<
        ckl::BinaryFpuOp::Add,
        ckl::input(dfb::partial, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
        ckl::DestReuseType::DEST_TO_SRCB>;
    // The packer keeps the configuration left by tilize; reserve and publish dst_tiles per destination acquire.
    using PackPartial = ckl::PackTile<ckl::output(
        dfb::partial,
        ckl::ReservePolicy::PerBlockSize,
        ckl::PushPolicy::PerBlockSize,
        ckl::DataFormatReconfig::Disabled)>;
    using PackOutput = ckl::PackTile<ckl::output(
        dfb::output,
        ckl::ReservePolicy::PerBlockSize,
        ckl::PushPolicy::PerBlockSize,
        ckl::DataFormatReconfig::Disabled)>;
    // Process dst_tiles channel tiles per destination acquire; every tile still sees the same multiply, BF16 partial
    // add and final SiLU as a one-tile loop.
    constexpr auto shape = ckl::IterationShape::tiles(block_ct).block_size(dst_tiles);

    // Work items are channel-block-major; the reader queues tap weights once per run of items
    // that share a channel block.
    for (uint32_t item = 0; item < wi_count; ++item) {
        const uint32_t work = wi_start + item;
        const uint32_t block = work / Mt;
        if (item == 0 || (work - 1) / Mt != block) {
            weights.wait_front(tap_count * block_ct);
        }
        for (uint32_t tap = 0; tap < tap_count; ++tap) {
            compute_kernel_lib::tilize<block_ct, dfb::act_rm, dfb::act_tile>(1);
            const uint32_t weights_base = tap * block_ct;
            if (tap == 0) {
                ckl::eltwise_chain(
                    shape, ActivationTimesWeights{activation, weights, 0, weights_base}, PackPartial{partial});
            } else if (tap + 1 == tap_count) {
                ckl::eltwise_chain(
                    shape,
                    ActivationTimesWeights{activation, weights, 0, weights_base},
                    AddPartial{partial},
                    ckl::Silu<>{},
                    PackOutput{output});
            } else {
                ckl::eltwise_chain(
                    shape,
                    ActivationTimesWeights{activation, weights, 0, weights_base},
                    AddPartial{partial},
                    PackPartial{partial});
            }
        }
        if (item + 1 == wi_count || (work + 1) / Mt != block) {
            weights.pop_front(tap_count * block_ct);
        }
    }
}
