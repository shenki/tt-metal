// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "cfg_defines.h"
#include "ckernel.h"
#include "llk_defs.h"

namespace ckernel
{

namespace dest_dvalid
{

constexpr dest_dvalid_client NONE = static_cast<dest_dvalid_client>(4);

#if defined(LLK_TRISC_UNPACK) || defined(TRISC_UNPACK)
constexpr dest_dvalid_client THIS_UNIT = dest_dvalid_client::UNPACK;
#elif defined(LLK_TRISC_MATH) || defined(TRISC_MATH)
constexpr dest_dvalid_client THIS_UNIT = dest_dvalid_client::FPU;
#elif defined(LLK_TRISC_ISOLATE_SFPU) || defined(TRISC_ISOLATE_SFPU)
constexpr dest_dvalid_client THIS_UNIT = dest_dvalid_client::SFPU;
#elif defined(LLK_TRISC_PACK) || defined(TRISC_PACK)
constexpr dest_dvalid_client THIS_UNIT = dest_dvalid_client::PACK;
#else
constexpr dest_dvalid_client THIS_UNIT = NONE;
#endif

struct UnitInfo
{
    std::uint32_t bit;
    std::uint32_t ctrl_addr32;
    std::uint32_t wait_res;
    std::uint32_t stall_class;
};

constexpr UnitInfo UNITS[] = {
    {p_cleardvalid::UNPACK_TO_DEST, UNPACK_TO_DEST_DVALID_CTRL_wait_mask_ADDR32, p_stall::UNPACK0, p_stall::STALL_UNPACK},
    {p_cleardvalid::FPU, MATH_DEST_DVALID_CTRL_wait_mask_ADDR32, p_stall::MATH, p_stall::STALL_MATH},
    {p_cleardvalid::SFPU, SFPU_DEST_DVALID_CTRL_wait_mask_ADDR32, p_stall::WAIT_SFPU, p_stall::STALL_SFPU | p_stall::STALL_MATH},
    {p_cleardvalid::PACK, PACK_DEST_DVALID_CTRL_wait_mask_ADDR32, p_stall::PACK, p_stall::STALL_PACK},
};

constexpr UnitInfo info_of(const dest_dvalid_client unit)
{
    return UNITS[static_cast<int>(unit)];
}

constexpr std::uint32_t bit_of(const dest_dvalid_client unit)
{
    return unit == NONE ? 0 : info_of(unit).bit;
}

constexpr std::uint32_t ALL_BITS = 0xF;

constexpr std::uint32_t CTRL_GPR       = 44;
constexpr std::uint32_t CTRL_WORD_MASK = UNPACK_TO_DEST_DVALID_CTRL_wait_mask_MASK | UNPACK_TO_DEST_DVALID_CTRL_wait_polarity_MASK |
                                         UNPACK_TO_DEST_DVALID_CTRL_toggle_mask_MASK | UNPACK_TO_DEST_DVALID_CTRL_disable_auto_bank_id_toggle_MASK;
static_assert(CTRL_WORD_MASK < (1u << 14), "the control word must fit the SETGPR immediate");

static_assert(MATH_DEST_DVALID_CTRL_wait_mask_SHAMT == UNPACK_TO_DEST_DVALID_CTRL_wait_mask_SHAMT);
static_assert(SFPU_DEST_DVALID_CTRL_wait_polarity_SHAMT == UNPACK_TO_DEST_DVALID_CTRL_wait_polarity_SHAMT);
static_assert(PACK_DEST_DVALID_CTRL_toggle_mask_SHAMT == UNPACK_TO_DEST_DVALID_CTRL_toggle_mask_SHAMT);
static_assert(PACK_DEST_DVALID_CTRL_disable_auto_bank_id_toggle_SHAMT == UNPACK_TO_DEST_DVALID_CTRL_disable_auto_bank_id_toggle_SHAMT);

constexpr std::uint32_t ctrl_word(const std::uint32_t wait_mask, const std::uint32_t wait_polarity, const std::uint32_t toggle_mask, const bool hold_bank)
{
    return (wait_mask << UNPACK_TO_DEST_DVALID_CTRL_wait_mask_SHAMT) | (wait_polarity << UNPACK_TO_DEST_DVALID_CTRL_wait_polarity_SHAMT) |
           (toggle_mask << UNPACK_TO_DEST_DVALID_CTRL_toggle_mask_SHAMT) |
           (static_cast<std::uint32_t>(hold_bank) << UNPACK_TO_DEST_DVALID_CTRL_disable_auto_bank_id_toggle_SHAMT);
}

/**
 * @brief Control word of UNIT for one turn with dest.
 *
 * The head waits for every bit clear and never sets its own bit. Every other unit waits for its own bit and
 * clears it when it pulses. The toggle also sets the bit of NEXT, so the toggle names the successor.
 *
 * @param is_head: UNIT went first in the section and nobody set its bit since
 * @param next: unit dest goes to afterwards, NONE when UNIT ends the section
 * @param hold_bank: keep the bank id after the pulse (SyncFull)
 */
template <dest_dvalid_client UNIT>
constexpr std::uint32_t word_of(const bool is_head, const dest_dvalid_client next, const bool hold_bank)
{
    constexpr std::uint32_t OWN = bit_of(UNIT);
    if (is_head)
    {
        return ctrl_word(ALL_BITS, 0, bit_of(next), hold_bank);
    }
    return ctrl_word(OWN, OWN, OWN | bit_of(next), hold_bank);
}

constexpr std::uint32_t PULSE_STALL = p_stall::STALL_THREAD;

} // namespace dest_dvalid

/**
 * @brief Writes UNIT's control word as a Tensix instruction in the thread's own stream.
 *
 * The write lands after every access the thread pushed before it has issued. UNIT's dest accesses then wait
 * until the CFG unit has applied the word, so the next one is judged by the new word.
 */
template <dest_dvalid_client UNIT, std::uint32_t WORD>
inline void _llk_dest_dvalid_write_ctrl_()
{
    constexpr std::uint32_t GPR          = dest_dvalid::CTRL_GPR;
    constexpr dest_dvalid::UnitInfo INFO = dest_dvalid::info_of(UNIT);
    static_assert((WORD & ~dest_dvalid::CTRL_WORD_MASK) == 0, "control word out of range");
    TTI_SETGPR(0, WORD, 0, GPR * 2 + 0);
    TTI_SETGPR(0, 0, 0, GPR * 2 + 1);
    TTI_WRCFG(GPR, p_cfg::WRCFG_32b, INFO.ctrl_addr32);
    TTI_STALLWAIT(INFO.stall_class, p_stall::NOTHING, p_stall::NOTHING, p_stall::CFGEXU);
}

/**
 * @brief Waits for UNIT's engine and pulses its `last`.
 */
template <dest_dvalid_client UNIT>
inline void _llk_dest_dvalid_pulse_()
{
    constexpr dest_dvalid::UnitInfo INFO = dest_dvalid::info_of(UNIT);
    TTI_STALLWAIT(dest_dvalid::PULSE_STALL, p_stall::NOTHING, p_stall::NOTHING, INFO.wait_res);
    TTI_CLEARDVALID(0, 0, 0, 0, INFO.bit, 0);
}

/**
 * @brief Resets UNIT to bank 0. Its word is written when it first takes dest.
 */
template <dest_dvalid_client UNIT>
inline void _llk_dest_dvalid_join_()
{
    TTI_CLEARDVALID(0, 0, 0, dest_dvalid::bit_of(UNIT), 0, 0);
}

/**
 * @brief Takes UNIT out of the handshake: its dest accesses are no longer gated.
 */
template <dest_dvalid_client UNIT>
inline void _llk_dest_dvalid_leave_()
{
    _llk_dest_dvalid_write_ctrl_<UNIT, 0>();
}

/**
 * @brief Programs UNIT's word for a turn, see @ref dest_dvalid::word_of.
 */
template <dest_dvalid_client UNIT, bool IS_HEAD, dest_dvalid_client NEXT, bool HOLD_BANK>
inline void _llk_dest_dvalid_wait_()
{
    _llk_dest_dvalid_write_ctrl_<UNIT, dest_dvalid::word_of<UNIT>(IS_HEAD, NEXT, HOLD_BANK)>();
}

/**
 * @brief PACK zeroes the bank it has just read, once its reads are done. Only PACK clears dest.
 */
template <DstSync DST, bool EN_32BIT_DEST>
inline void _llk_dest_dvalid_pack_clear_(const std::uint32_t bank)
{
    TTI_STALLWAIT(dest_dvalid::PULSE_STALL, p_stall::NOTHING, p_stall::NOTHING, p_stall::PACK);
    if constexpr (DST == DstSync::SyncFull)
    {
        TTI_ZEROACC(p_zeroacc::CLR_ALL, EN_32BIT_DEST, 0, ADDR_MOD_0, 0);
    }
    else if (bank == 0)
    {
        TTI_ZEROACC(p_zeroacc::CLR_HALF, EN_32BIT_DEST, 0, ADDR_MOD_0, 0);
    }
    else
    {
        TTI_ZEROACC(p_zeroacc::CLR_HALF, EN_32BIT_DEST, 0, ADDR_MOD_0, 1);
    }
}

/**
 * @brief Hands dest from UNIT to NEXT, or releases the section when NEXT is NONE.
 *
 * Writes the toggle that names NEXT unless the word already names it, then pulses. A release clears UNIT's own
 * bit, which returns the bank to all zero for the next head. A head releasing alone toggles nothing.
 *
 * @tparam UNIT: unit holding dest, on this thread
 * @tparam NEXT: unit dest goes to, NONE to end the section
 * @tparam IS_HEAD: UNIT went first in the section and nobody set its bit since
 * @tparam DST: destination synchronization mode, values = <SyncHalf/SyncFull>
 * @tparam EN_32BIT_DEST: true when dest is in 32-bit mode
 * @tparam WORD_READY: the word already names NEXT, nothing to write
 * @param bank: bank UNIT is on, the one PACK zeroes in SyncHalf
 */
template <dest_dvalid_client UNIT, dest_dvalid_client NEXT, bool IS_HEAD, DstSync DST, bool EN_32BIT_DEST, bool WORD_READY>
inline void _llk_dest_dvalid_signal_(const std::uint32_t bank)
{
    static_assert(UNIT != dest_dvalid::NONE, "signal needs a unit on this thread");
    if constexpr (!WORD_READY)
    {
        _llk_dest_dvalid_wait_<UNIT, IS_HEAD, NEXT, DST == DstSync::SyncFull>();
    }
    if constexpr (UNIT == dest_dvalid_client::PACK)
    {
        _llk_dest_dvalid_pack_clear_<DST, EN_32BIT_DEST>(bank);
        TTI_CLEARDVALID(0, 0, 0, 0, dest_dvalid::bit_of(UNIT), 0);
    }
    else
    {
        _llk_dest_dvalid_pulse_<UNIT>();
    }
}

/**
 * @brief Moves UNIT to the other bank, for a unit that sat out an odd number of sections.
 *
 * Two pulses on UNIT's own bit, the first with the bank id held and the second without. The bank's bits come
 * back as they were and only the bank id moves. Nobody waits on UNIT's bit in between.
 */
template <dest_dvalid_client UNIT>
inline void _llk_dest_dvalid_bank_flip_()
{
    constexpr std::uint32_t OWN = dest_dvalid::bit_of(UNIT);
    _llk_dest_dvalid_write_ctrl_<UNIT, dest_dvalid::ctrl_word(0, 0, OWN, true)>();
    _llk_dest_dvalid_pulse_<UNIT>();
    _llk_dest_dvalid_write_ctrl_<UNIT, dest_dvalid::ctrl_word(0, 0, OWN, false)>();
    _llk_dest_dvalid_pulse_<UNIT>();
}

/**
 * @brief Clears every dvalid bit and every bank id. For firmware boot, when no unit can be active.
 */
inline void _llk_dest_dvalid_boot_reset_()
{
    TTI_CLEARDVALID(0, 0, dest_dvalid::ALL_BITS, dest_dvalid::ALL_BITS, 0, 0);
}

} // namespace ckernel
