"""CPU reference for MTP + local suffix lookup arbitration.

This module intentionally has no vLLM or torch dependency.  It specifies the
decision that a future GPU implementation must preserve:

* Qwen MTP produces three ordinary draft tokens.
* A suffix match can replace those tokens only on a strong or confirmed match.
* The lookup can extend verification to seven tokens only when it has all seven
  continuation tokens and owns a strong/confirmed head.
* Every replaced sampling row is represented by a point mass.  That is a valid
  rejection-sampling proposal, so it does not alter the target distribution.

The reference is deliberately conservative: a partial lookup continuation is
never used for the tail.  The production experiment can relax this only after
GPU correctness and scheduler tests prove that the fallback path cannot expose
uninitialised draft rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Sequence


MTP_DRAFT_STEPS = 3
MAX_VERIFY_STEPS = 7
DEFAULT_NMIN = 4
DEFAULT_NSTRONG = 6
DEFAULT_AGREE_MIN = 2
DEFAULT_NMIN_TAIL = 4


@dataclass(frozen=True)
class LookupProposal:
    """A continuation found after an earlier occurrence of the current suffix."""

    tokens: tuple[int, ...]
    match_length: int
    valid: int
    match_end: int | None

    @classmethod
    def none(cls, width: int = MAX_VERIFY_STEPS) -> "LookupProposal":
        return cls((0,) * width, 0, 0, None)


@dataclass(frozen=True)
class Arbitration:
    """The safe proposal and the rows whose proposal distribution was replaced."""

    tokens: tuple[int | None, ...]
    lookup_used: tuple[bool, ...]
    verify_tokens: int
    strong_or_confirmed: bool


def find_suffix_match(
    history: Sequence[int],
    *,
    width: int = MAX_VERIFY_STEPS,
    nmin: int = DEFAULT_NMIN,
    nmax: int = 32,
    search_max: int | None = None,
) -> LookupProposal:
    """Find the longest recent continuation of the history's terminal suffix.

    ``history`` ends at the token already accepted by the target.  Candidate
    match ends are strictly before that token, and a candidate may overlap the
    suffix.  Ties prefer the most recent candidate.  ``valid`` is capped by
    both ``width`` and the number of tokens available after the match, so no
    token outside the accepted history is ever proposed.
    """

    if width < 1:
        raise ValueError("width must be positive")
    if nmin < 1 or nmax < nmin:
        raise ValueError("require 1 <= nmin <= nmax")
    if search_max is not None and search_max < 1:
        raise ValueError("search_max must be positive")
    if len(history) < nmin + 2:
        return LookupProposal.none(width)

    suffix_end = len(history) - 1
    first_end = max(nmin - 1, len(history) - (search_max or len(history)))
    best_length = 0
    best_end: int | None = None

    for candidate_end in range(first_end, suffix_end):
        length = 0
        # Compare backwards so overlap is well-defined, matching the Triton
        # design in dflash2-lookup-drafting.patch.
        for offset in range(nmax):
            candidate_pos = candidate_end - offset
            suffix_pos = suffix_end - offset
            if candidate_pos < 0 or history[candidate_pos] != history[suffix_pos]:
                break
            length += 1
        if length >= nmin and (length > best_length or candidate_end > (best_end or -1)):
            best_length = length
            best_end = candidate_end

    if best_end is None:
        return LookupProposal.none(width)

    valid = min(width, suffix_end - best_end)
    return LookupProposal(
        tuple(history[best_end + 1 : best_end + 1 + valid])
        + (0,) * (width - valid),
        best_length,
        valid,
        best_end,
    )


def arbitrate(
    drafter_tokens: Sequence[int],
    lookup: LookupProposal,
    *,
    draft_steps: int = MTP_DRAFT_STEPS,
    verify_steps: int = MAX_VERIFY_STEPS,
    nmin: int = DEFAULT_NMIN,
    nstrong: int = DEFAULT_NSTRONG,
    agree_min: int = DEFAULT_AGREE_MIN,
    nmin_tail: int = DEFAULT_NMIN_TAIL,
) -> Arbitration:
    """Fuse MTP and lookup proposals without exposing an unsafe tail.

    The first ``draft_steps`` positions have an MTP proposal.  A strong match
    wins that head on its own; a merely qualifying match must agree with MTP
    for ``agree_min`` *leading* positions.  The tail is filled only when the
    complete verify block is available and the head was accepted.  Therefore
    the caller can safely schedule either exactly three or exactly seven
    tokens, with no mixed-width request state.
    """

    if not 1 <= draft_steps <= verify_steps:
        raise ValueError("require 1 <= draft_steps <= verify_steps")
    if len(drafter_tokens) < draft_steps:
        raise ValueError("drafter_tokens is shorter than draft_steps")
    if len(lookup.tokens) < verify_steps:
        raise ValueError("lookup proposal is shorter than verify_steps")
    if lookup.valid < 0 or lookup.valid > verify_steps:
        raise ValueError("lookup.valid is outside the verify block")

    # Count only the leading agreement and never treat missing lookup tokens as
    # agreement.  A mismatch after the first equal token is not confirmation.
    agreement = 0
    while agreement < draft_steps and agreement < lookup.valid:
        if drafter_tokens[agreement] != lookup.tokens[agreement]:
            break
        agreement += 1

    strong_or_confirmed = lookup.valid > 0 and (
        lookup.match_length >= nstrong
        or (lookup.match_length >= nmin and agreement >= agree_min)
    )
    full_tail = lookup.valid >= verify_steps
    use_head = strong_or_confirmed
    use_tail = (
        strong_or_confirmed
        and full_tail
        and lookup.match_length >= nmin_tail
    )

    result: list[int | None] = list(drafter_tokens[:draft_steps])
    result.extend([None] * (verify_steps - draft_steps))
    used: list[bool] = [False] * verify_steps
    if use_head:
        for position in range(min(draft_steps, lookup.valid)):
            result[position] = lookup.tokens[position]
            used[position] = True
    if use_tail:
        for position in range(draft_steps, verify_steps):
            result[position] = lookup.tokens[position]
            used[position] = True

    return Arbitration(
        tuple(result),
        tuple(used),
        verify_steps if use_tail else draft_steps,
        strong_or_confirmed,
    )


def point_mass_distribution(token: int, vocab_size: int) -> tuple[float, ...]:
    """Return q=delta(token), the exact proposal for a lookup-owned row."""

    if not 0 <= token < vocab_size:
        raise ValueError("token is outside vocab")
    if vocab_size < 1:
        raise ValueError("vocab_size must be positive")
    row = [0.0] * vocab_size
    row[token] = 1.0
    return tuple(row)


def rewrite_proposal_distributions(
    draft_rows: Sequence[Sequence[float]],
    arbitration: Arbitration,
    *,
    vocab_size: int,
) -> tuple[tuple[float, ...], ...]:
    """Replace only lookup-owned q rows with point masses.

    A real MTP implementation stores processed logits, not probabilities.  It
    should implement the same operation as ``-inf`` everywhere and ``0`` at
    the proposed token.  This CPU helper makes the invariant testable without
    importing torch or vLLM.
    """

    if len(draft_rows) < len(arbitration.lookup_used):
        raise ValueError("draft_rows is shorter than the verify block")
    rewritten = [tuple(row) for row in draft_rows[: len(arbitration.lookup_used)]]
    for position, use_lookup in enumerate(arbitration.lookup_used):
        if not use_lookup:
            continue
        token = arbitration.tokens[position]
        if token is None:
            raise AssertionError("lookup-owned position has no token")
        rewritten[position] = point_mass_distribution(token, vocab_size)
    return tuple(rewritten)


def rejection_output_distribution(
    target: Sequence[float], proposal: Sequence[float]
) -> tuple[float, ...]:
    """Evaluate one exact rejection-sampling step for a finite distribution.

    This is a reference oracle for tests.  It emits accepted proposal mass and
    then the residual ``max(p-q, 0)``.  A point-mass q must reproduce p exactly.
    """

    target = tuple(target)
    proposal = tuple(proposal)
    if len(target) != len(proposal) or not target:
        raise ValueError("target and proposal must have equal non-zero width")
    if any(value < 0 for value in target + tuple(proposal)):
        raise ValueError("distributions must be non-negative")
    if not isclose(sum(target), 1.0, abs_tol=1e-9) or not isclose(
        sum(proposal), 1.0, abs_tol=1e-9
    ):
        raise ValueError("distributions must sum to one")

    accepted = [min(p, q) for p, q in zip(target, proposal)]
    residual = [max(p - q, 0.0) for p, q in zip(target, proposal)]
    residual_mass = sum(residual)
    if residual_mass:
        return tuple(
            a + (1.0 - sum(accepted)) * r / residual_mass
            for a, r in zip(accepted, residual)
        )
    return tuple(accepted)
