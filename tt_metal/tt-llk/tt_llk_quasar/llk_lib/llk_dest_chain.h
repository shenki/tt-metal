// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "llk_dest_dvalid.h"

namespace ckernel
{

struct DestTurn
{
    dest_dvalid_client unit;
    dest_dvalid_client next = dest_dvalid::NONE;
    bool head               = false;
    bool word_ready         = true;
};

namespace dest_chain
{

using dest_dvalid::NONE;
using dest_dvalid::THIS_UNIT;

constexpr std::uint32_t NUM_BANKS = 2;

struct State
{
    std::uint32_t section                     = 0;
    dest_dvalid_client token_owner[NUM_BANKS] = {NONE, NONE};
    std::uint8_t bank[4]                      = {};
    bool in_chain[4]                          = {};

    dest_dvalid_client holder = NONE;
    dest_dvalid_client head   = NONE;
    std::uint8_t handed       = 0;
};

inline State state;

constexpr int index_of(const dest_dvalid_client unit)
{
    return static_cast<int>(unit);
}

template <dest_dvalid_client UNIT>
constexpr bool mine()
{
    return UNIT == THIS_UNIT;
}

template <DstSync DST>
inline std::uint32_t section_bank()
{
    return DST == DstSync::SyncFull ? 0 : (state.section & 1);
}

/**
 * @brief Calls `fn` with the unit as a compile time constant, for state that is only known when the kernel runs.
 */
template <typename Fn>
inline void for_unit(const dest_dvalid_client unit, Fn fn)
{
    switch (unit)
    {
        case dest_dvalid_client::UNPACK:
            fn(std::integral_constant<dest_dvalid_client, dest_dvalid_client::UNPACK> {});
            break;
        case dest_dvalid_client::FPU:
            fn(std::integral_constant<dest_dvalid_client, dest_dvalid_client::FPU> {});
            break;
        case dest_dvalid_client::SFPU:
            fn(std::integral_constant<dest_dvalid_client, dest_dvalid_client::SFPU> {});
            break;
        case dest_dvalid_client::PACK:
            fn(std::integral_constant<dest_dvalid_client, dest_dvalid_client::PACK> {});
            break;
        default:
            break;
    }
}

struct TokenInfo
{
    std::uint32_t sem[NUM_BANKS];
    std::uint32_t stall_res;
};

constexpr std::uint32_t NO_SEM = 0xFF;

constexpr TokenInfo TOKENS[] = {
    {{12, 13}, p_stall::STALL_UNPACK | p_stall::STALL_SYNC},
    {{14, 15}, p_stall::STALL_MATH | p_stall::STALL_SFPU | p_stall::STALL_SYNC},
    {{0, 3}, p_stall::STALL_SFPU | p_stall::STALL_MATH | p_stall::STALL_SYNC},
    {{NO_SEM, NO_SEM}, 0},
};

constexpr bool can_start(const dest_dvalid_client unit)
{
    return TOKENS[index_of(unit)].sem[0] != NO_SEM;
}

constexpr std::uint32_t sem_bank_sel(const std::uint32_t sem)
{
    return sem >> 3;
}

constexpr std::uint32_t sem_sel(const std::uint32_t sem)
{
    return 1u << (sem & 7);
}

template <dest_dvalid_client UNIT, std::uint32_t BANK>
inline void take_token_on()
{
    constexpr TokenInfo INFO    = TOKENS[index_of(UNIT)];
    constexpr std::uint32_t SEM = INFO.sem[BANK];
    TTI_SEMWAIT(INFO.stall_res, p_stall::STALL_ON_ZERO, sem_bank_sel(SEM), sem_sel(SEM));
    TTI_SEMGET(sem_bank_sel(SEM), sem_sel(SEM));
}

template <dest_dvalid_client UNIT>
inline void take_token(const std::uint32_t bank)
{
    if (bank == 0)
    {
        take_token_on<UNIT, 0>();
    }
    else
    {
        take_token_on<UNIT, 1>();
    }
}

template <dest_dvalid_client TO, std::uint32_t BANK>
inline void give_token_on()
{
    constexpr std::uint32_t SEM = TOKENS[index_of(TO)].sem[BANK];
    TTI_SEMPOST(sem_bank_sel(SEM), sem_sel(SEM));
}

template <dest_dvalid_client TO>
inline void give_token(const std::uint32_t bank)
{
    if (bank == 0)
    {
        give_token_on<TO, 0>();
    }
    else
    {
        give_token_on<TO, 1>();
    }
}

/**
 * @brief OWNER was expected to go first on the bank but NEXT does. OWNER takes its token and passes it on.
 *
 * Only OWNER's semaphore instructions wait, its other work keeps issuing.
 */
template <dest_dvalid_client OWNER, dest_dvalid_client NEXT>
inline void forward_token(const std::uint32_t bank)
{
    const std::uint32_t from = TOKENS[index_of(OWNER)].sem[bank];
    const std::uint32_t to   = TOKENS[index_of(NEXT)].sem[bank];
    TT_SEMWAIT(p_stall::STALL_SYNC, p_stall::STALL_ON_ZERO, sem_bank_sel(from), sem_sel(from));
    TT_SEMGET(sem_bank_sel(from), sem_sel(from));
    TT_SEMPOST(sem_bank_sel(to), sem_sel(to));
}

/**
 * @brief UNIT collects the bank's token from whoever it was addressed to. Runs on every thread, waits on UNIT's.
 */
template <dest_dvalid_client UNIT>
inline void collect_token(const std::uint32_t bank)
{
    const dest_dvalid_client owner = state.token_owner[bank];
    if (owner == NONE)
    {
        return;
    }
    for_unit(
        owner,
        [bank](auto owner_tag)
        {
            constexpr dest_dvalid_client OWNER = decltype(owner_tag)::value;
            if constexpr (OWNER != UNIT && mine<OWNER>() && can_start(OWNER))
            {
                forward_token<OWNER, UNIT>(bank);
            }
        });
    if constexpr (mine<UNIT>())
    {
        take_token<UNIT>(bank);
    }
}

/**
 * @brief Puts UNIT on the bank of the current section, bank 0 by a reset and bank 1 by a flip. Owner thread only.
 *
 * @return true when UNIT had to move. A move pulses with words of its own, so a word written ahead is gone.
 */
template <dest_dvalid_client UNIT, DstSync DST>
inline bool align_bank()
{
    if constexpr (DST == DstSync::SyncFull || !mine<UNIT>())
    {
        return false;
    }
    std::uint8_t& bank         = state.bank[index_of(UNIT)];
    const std::uint32_t target = state.section & 1;
    if (bank == target)
    {
        return false;
    }
    if (target == 0)
    {
        TTI_CLEARDVALID(0, 0, 0, dest_dvalid::bit_of(UNIT), 0, 0);
    }
    else
    {
        _llk_dest_dvalid_bank_flip_<UNIT>();
    }
    bank = target;
    return true;
}

template <dest_dvalid_client UNIT, DstSync DST>
inline void after_pulse()
{
    if constexpr (DST == DstSync::SyncHalf && mine<UNIT>())
    {
        state.bank[index_of(UNIT)] ^= 1;
    }
}

template <DestTurn T, DstSync DST>
inline void write_word()
{
    _llk_dest_dvalid_wait_<T.unit, T.head, T.next, DST == DstSync::SyncFull>();
}

/**
 * @brief The unit of turn T gets ready for its dest accesses: on its bank, with its word in place. Owner thread only.
 */
template <DestTurn T, DstSync DST>
inline void arrive()
{
    if constexpr (!mine<T.unit>())
    {
        return;
    }
    const bool moved = align_bank<T.unit, DST>();
    if (!T.word_ready || moved)
    {
        write_word<T, DST>();
    }
}

/**
 * @brief The head starts a section.
 *
 * @tparam WITH_TOKEN: the head takes the bank's token first, the runtime chain does this every section
 */
template <DestTurn HEAD, DstSync DST, bool WITH_TOKEN>
inline void start()
{
    static_assert(can_start(HEAD.unit), "PACK only reads dest, so it cannot start a section");
    static_assert(HEAD.head, "the first turn of a section is the head's");
    if constexpr (WITH_TOKEN)
    {
        collect_token<HEAD.unit>(section_bank<DST>());
    }
    arrive<HEAD, DST>();
}

/**
 * @brief FROM hands dest to TO: FROM pulses on its thread, TO gets ready on its thread.
 */
template <DestTurn FROM, DestTurn TO, DstSync DST, bool EN_32BIT_DEST>
inline void pass()
{
    static_assert(!FROM.word_ready || FROM.next == TO.unit, "a word written ahead must name the unit it hands to");
    if constexpr (FROM.unit == TO.unit)
    {
        return;
    }
    if constexpr (mine<FROM.unit>())
    {
        _llk_dest_dvalid_signal_<FROM.unit, TO.unit, FROM.head, DST, EN_32BIT_DEST, FROM.word_ready>(state.bank[index_of(FROM.unit)]);
        after_pulse<FROM.unit, DST>();
    }
    arrive<TO, DST>();
}

/**
 * @brief LAST releases dest and the section ends.
 *
 * @tparam HEAD: unit that started the section, the one expected to start the next section on this bank
 * @tparam WITH_TOKEN: LAST posts the bank's token to HEAD, the runtime chain does this every section
 */
template <DestTurn LAST, dest_dvalid_client HEAD, DstSync DST, bool EN_32BIT_DEST, bool WITH_TOKEN>
inline void end()
{
    static_assert(can_start(HEAD), "PACK only reads dest, so it cannot start a section");
    static_assert(!LAST.word_ready || LAST.next == NONE, "a word written ahead must name nobody when the turn ends the section");
    const std::uint32_t bank = section_bank<DST>();
    if constexpr (mine<LAST.unit>())
    {
        _llk_dest_dvalid_signal_<LAST.unit, NONE, LAST.head, DST, EN_32BIT_DEST, LAST.word_ready>(state.bank[index_of(LAST.unit)]);
    }
    if constexpr (mine<LAST.unit>() && WITH_TOKEN)
    {
        give_token<HEAD>(bank);
    }
    if constexpr (!LAST.head)
    {
        after_pulse<LAST.unit, DST>();
    }
    if constexpr (WITH_TOKEN)
    {
        state.token_owner[bank] = HEAD;
    }
    state.section++;
}

inline bool waits_all_clear(const dest_dvalid_client unit)
{
    return (state.handed & dest_dvalid::bit_of(unit)) == 0;
}

template <dest_dvalid_client TO, DstSync DST, bool EN_32BIT_DEST>
inline void pass_from_holder()
{
    for_unit(
        state.holder,
        [](auto holder_tag)
        {
            constexpr dest_dvalid_client FROM = decltype(holder_tag)::value;
            constexpr DestTurn to {TO, NONE, false, false};
            if (waits_all_clear(FROM))
            {
                pass<DestTurn {FROM, TO, true, false}, to, DST, EN_32BIT_DEST>();
            }
            else
            {
                pass<DestTurn {FROM, TO, false, false}, to, DST, EN_32BIT_DEST>();
            }
        });
}

template <dest_dvalid_client LAST, DstSync DST, bool EN_32BIT_DEST>
inline void end_to_head()
{
    for_unit(
        state.head,
        [](auto head_tag)
        {
            constexpr dest_dvalid_client HEAD = decltype(head_tag)::value;
            if constexpr (!can_start(HEAD))
            {
                return;
            }
            else if (waits_all_clear(LAST))
            {
                end<DestTurn {LAST, NONE, true, false}, HEAD, DST, EN_32BIT_DEST, true>();
            }
            else
            {
                end<DestTurn {LAST, NONE, false, false}, HEAD, DST, EN_32BIT_DEST, true>();
            }
        });
}

} // namespace dest_chain

/**
 * @brief Zeroes every token semaphore. For the firmware, before every kernel, from the thread that boots the core.
 */
inline void _llk_dest_chain_boot_init_()
{
    for (const dest_chain::TokenInfo& info : dest_chain::TOKENS)
    {
        for (std::uint32_t bank = 0; bank < dest_chain::NUM_BANKS; bank++)
        {
            if (info.sem[bank] != dest_chain::NO_SEM)
            {
                TT_SEMINIT(1 /*max_value*/, 0 /*init_value*/, dest_chain::sem_bank_sel(info.sem[bank]), dest_chain::sem_sel(info.sem[bank]));
            }
        }
    }
}

/**
 * @brief Adds UNIT to the dest chain.
 *
 * @note Call on every thread from the init of an op that uses dest on UNIT. Pair with @ref _llk_dest_chain_leave_.
 */
template <dest_dvalid_client UNIT>
inline void _llk_dest_chain_join_()
{
    const int u = dest_chain::index_of(UNIT);
    if (dest_chain::state.in_chain[u])
    {
        return;
    }
    if constexpr (dest_chain::mine<UNIT>())
    {
        _llk_dest_dvalid_join_<UNIT>();
    }
    dest_chain::state.in_chain[u] = true;
    dest_chain::state.bank[u]     = 0;
}

/**
 * @brief Removes UNIT from the dest chain.
 *
 * @note Call on every thread from the uninit of the op that joined, outside a section.
 */
template <dest_dvalid_client UNIT>
inline void _llk_dest_chain_leave_()
{
    const int u = dest_chain::index_of(UNIT);
    if (!dest_chain::state.in_chain[u])
    {
        return;
    }
    if constexpr (dest_chain::mine<UNIT>())
    {
        _llk_dest_dvalid_leave_<UNIT>();
    }
    dest_chain::state.in_chain[u] = false;
}

/**
 * @brief Writes the word of turn T once for every section of an operation.
 *
 * A unit that takes one turn per section keeps the same word for the whole operation. Call this before the
 * operation's first section and describe its turns with `word_ready`. A turn that had to move the unit to
 * another bank rewrites the word by itself.
 *
 * @note Call on every thread after the joins of the operation.
 */
template <DestTurn T, DstSync DST>
inline void _llk_dest_chain_prepare_()
{
    if constexpr (dest_chain::mine<T.unit>())
    {
        dest_chain::write_word<T, DST>();
    }
}

/**
 * @brief Opens an operation whose head differs from the previous operation's head.
 *
 * Inside an operation the head never changes, so its own pulses order its sections. Only a change of head
 * between operations needs a token, and both banks are done when an operation ends, so the bank 0 slot
 * covers the boundary. Two operations with the same head need neither this nor @ref _llk_dest_chain_op_end_.
 *
 * @tparam HEAD: unit that starts every section of the operation
 * @note Call on every thread before the operation's first section.
 */
template <dest_dvalid_client HEAD>
inline void _llk_dest_chain_op_begin_()
{
    if constexpr (dest_chain::mine<HEAD>())
    {
        dest_chain::take_token_on<HEAD, 0>();
    }
}

/**
 * @brief Closes an operation whose head differs from the next operation's head.
 *
 * @tparam LAST: unit that ends every section of the operation
 * @tparam NEXT_HEAD: unit that starts every section of the next operation
 * @note Call on every thread after the operation's last section. Pair with @ref _llk_dest_chain_op_begin_.
 */
template <dest_dvalid_client LAST, dest_dvalid_client NEXT_HEAD>
inline void _llk_dest_chain_op_end_()
{
    if constexpr (dest_chain::mine<LAST>())
    {
        dest_chain::give_token_on<NEXT_HEAD, 0>();
    }
}

/**
 * @brief The head starts a section.
 *
 * @tparam HEAD: the head's turn, with `head` set
 * @tparam DST: destination synchronization mode, values = <SyncHalf/SyncFull>
 * @note Call on every thread before the head's first dest access of the section.
 */
template <DestTurn HEAD, DstSync DST>
inline void _llk_dest_chain_start_()
{
    dest_chain::start<HEAD, DST, false /*WITH_TOKEN*/>();
}

/**
 * @brief FROM hands dest to TO.
 *
 * @tparam FROM: turn of the unit holding dest
 * @tparam TO: turn of the unit about to access dest
 * @tparam DST: destination synchronization mode, values = <SyncHalf/SyncFull>
 * @tparam EN_32BIT_DEST: true when dest is in 32-bit mode
 * @note Call on every thread after FROM's last dest access and before TO's first one.
 */
template <DestTurn FROM, DestTurn TO, DstSync DST, bool EN_32BIT_DEST = false>
inline void _llk_dest_chain_pass_()
{
    dest_chain::pass<FROM, TO, DST, EN_32BIT_DEST>();
}

/**
 * @brief LAST releases dest and the section ends.
 *
 * Only PACK zeroes the bank on release. A section that ends on another unit leaves its data in dest.
 *
 * @tparam LAST: turn of the unit holding dest
 * @tparam HEAD: unit that started the section
 * @tparam DST: destination synchronization mode, values = <SyncHalf/SyncFull>
 * @tparam EN_32BIT_DEST: true when dest is in 32-bit mode
 * @note Call on every thread after LAST's last dest access of the section.
 */
template <DestTurn LAST, dest_dvalid_client HEAD, DstSync DST, bool EN_32BIT_DEST = false>
inline void _llk_dest_chain_end_()
{
    dest_chain::end<LAST, HEAD, DST, EN_32BIT_DEST, false /*WITH_TOKEN*/>();
}

/**
 * @brief Starts a section. The bank it uses is the parity of the section count.
 */
inline void _llk_dest_chain_section_begin_()
{
    dest_chain::state.holder = dest_chain::NONE;
    dest_chain::state.head   = dest_chain::NONE;
    dest_chain::state.handed = 0;
}

/**
 * @brief UNIT takes dest: the holder hands it over and UNIT waits for it.
 *
 * The first unit of a section takes the bank's token instead. A unit that already holds dest keeps it, so an
 * op may call this again for a unit still holding dest.
 *
 * @tparam UNIT: unit about to access dest, must have joined
 * @tparam DST: destination synchronization mode, values = <SyncHalf/SyncFull>
 * @tparam EN_32BIT_DEST: true when dest is in 32-bit mode
 * @note Call on every thread before UNIT's dest accesses in a section, each time another unit had dest in between.
 */
template <dest_dvalid_client UNIT, DstSync DST, bool EN_32BIT_DEST = false>
inline void _llk_dest_chain_take_()
{
    dest_chain::State& s = dest_chain::state;
    LLK_ASSERT(s.in_chain[dest_chain::index_of(UNIT)], "a unit must join the dest chain before it takes dest");
    if (s.holder == UNIT)
    {
        return;
    }
    if (s.holder != dest_chain::NONE)
    {
        dest_chain::pass_from_holder<UNIT, DST, EN_32BIT_DEST>();
        s.handed |= dest_dvalid::bit_of(UNIT);
    }
    else if constexpr (dest_chain::can_start(UNIT))
    {
        dest_chain::start<DestTurn {UNIT, dest_chain::NONE, true, false}, DST, true /*WITH_TOKEN*/>();
        s.head = UNIT;
    }
    else
    {
        LLK_ASSERT(false, "PACK only reads dest, so it cannot start a section");
    }
    s.holder = UNIT;
}

/**
 * @brief Ends a section: whichever unit holds dest releases it.
 *
 * @tparam DST: destination synchronization mode, values = <SyncHalf/SyncFull>
 * @tparam EN_32BIT_DEST: true when dest is in 32-bit mode
 */
template <DstSync DST, bool EN_32BIT_DEST = false>
inline void _llk_dest_chain_section_end_()
{
    dest_chain::State& s = dest_chain::state;
    LLK_ASSERT(s.holder != dest_chain::NONE, "a section ends with a unit holding dest");
    dest_chain::for_unit(
        s.holder,
        [](auto holder_tag)
        {
            constexpr dest_dvalid_client LAST = decltype(holder_tag)::value;
            dest_chain::end_to_head<LAST, DST, EN_32BIT_DEST>();
        });
    s.holder = dest_chain::NONE;
}

} // namespace ckernel
