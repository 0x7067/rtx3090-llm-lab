"""CPU/static checks for the MTP + suffix lookup proof of concept."""

import unittest

try:
    from .mtp_suffix_lookup import (
        MAX_VERIFY_STEPS,
        MTP_DRAFT_STEPS,
        LookupProposal,
        arbitrate,
        find_suffix_match,
        rejection_output_distribution,
        rewrite_proposal_distributions,
    )
except ImportError:  # Also support ``python experiments/test_mtp_suffix_lookup.py``.
    from mtp_suffix_lookup import (
        MAX_VERIFY_STEPS,
        MTP_DRAFT_STEPS,
        LookupProposal,
        arbitrate,
        find_suffix_match,
        rejection_output_distribution,
        rewrite_proposal_distributions,
    )


class SuffixLookupTests(unittest.TestCase):
    def test_longest_recent_match_and_bounded_continuation(self):
        # The final six tokens repeat an earlier phrase; seven tokens follow
        # that earlier occurrence and are therefore available to verify.
        history = tuple(range(100, 106)) + tuple(range(10, 17)) + tuple(range(100, 106))
        proposal = find_suffix_match(history, nmin=4, nmax=8)
        self.assertEqual(proposal.match_length, 6)
        self.assertEqual(proposal.valid, MAX_VERIFY_STEPS)
        self.assertEqual(proposal.tokens, tuple(range(10, 17)))

    def test_recent_tie_wins(self):
        # Both occurrences have the same suffix match; the later one must win.
        history = (1, 2, 3, 4, 80, 81, 1, 2, 3, 4, 90, 91, 1, 2, 3, 4)
        proposal = find_suffix_match(history, nmin=4, nmax=4)
        self.assertEqual(proposal.match_length, 4)
        self.assertEqual(proposal.tokens[:2], (90, 91))

    def test_multi_token_current_step_is_part_of_suffix(self):
        # A synchronous legacy runner can commit several target tokens in one
        # rejection-sampling result. The lookup must see the complete append,
        # not only its last token.
        history = tuple(range(40, 46)) + tuple(range(10, 17)) + tuple(range(40, 44))
        proposal = find_suffix_match(history, nmin=4, nmax=8)
        self.assertEqual(proposal.match_length, 4)
        self.assertEqual(proposal.valid, MAX_VERIFY_STEPS)
        self.assertEqual(proposal.tokens[:3], (44, 45, 10))

    def test_short_or_missing_match_is_noop(self):
        proposal = find_suffix_match((1, 2, 3, 4, 5), nmin=4)
        self.assertEqual(proposal, LookupProposal.none())

    def test_strong_match_fills_seven_and_marks_every_owned_row(self):
        lookup = LookupProposal((10, 11, 12, 13, 14, 15, 16), 6, 7, 2)
        decision = arbitrate((90, 91, 92), lookup)
        self.assertEqual(decision.verify_tokens, 7)
        self.assertEqual(decision.tokens, (10, 11, 12, 13, 14, 15, 16))
        self.assertEqual(decision.lookup_used, (True,) * 7)

    def test_confirmed_match_requires_leading_agreement(self):
        lookup = LookupProposal((10, 11, 12, 13, 14, 15, 16), 4, 7, 2)
        confirmed = arbitrate((10, 11, 99), lookup)
        self.assertTrue(confirmed.strong_or_confirmed)
        self.assertEqual(confirmed.verify_tokens, 7)
        rejected = arbitrate((10, 99, 12), lookup)
        self.assertFalse(rejected.strong_or_confirmed)
        self.assertEqual(rejected.verify_tokens, MTP_DRAFT_STEPS)
        self.assertEqual(rejected.lookup_used, (False,) * 7)

    def test_partial_continuation_never_exposes_tail(self):
        lookup = LookupProposal((10, 11, 12, 13, 0, 0, 0), 6, 4, 2)
        decision = arbitrate((90, 91, 92), lookup)
        self.assertEqual(decision.verify_tokens, MTP_DRAFT_STEPS)
        self.assertEqual(decision.lookup_used[:3], (True, True, True))
        self.assertEqual(decision.lookup_used[3:], (False,) * 4)
        self.assertEqual(decision.tokens[3:], (None,) * 4)

    def test_point_mass_rewrite_is_lossless_for_rejection_sampling(self):
        lookup = LookupProposal((1, 2, 0, 1, 2, 0, 1), 6, 7, 0)
        decision = arbitrate((90, 91, 92), lookup)
        rows = ((0.1, 0.2, 0.7),) * MAX_VERIFY_STEPS
        rewritten = rewrite_proposal_distributions(rows, decision, vocab_size=3)
        target = (0.2, 0.5, 0.3)
        for position, row in enumerate(rewritten):
            if decision.lookup_used[position]:
                actual = rejection_output_distribution(target, row)
                for actual_value, target_value in zip(actual, target):
                    self.assertAlmostEqual(actual_value, target_value, places=12)

    def test_rewrite_preserves_fallback_rows(self):
        lookup = LookupProposal((1, 2, 3, 4, 5, 6, 7), 0, 0, None)
        decision = arbitrate((90, 91, 92), lookup)
        rows = ((0.1, 0.2, 0.7),) * MAX_VERIFY_STEPS
        self.assertEqual(rewrite_proposal_distributions(rows, decision, vocab_size=3), rows)


if __name__ == "__main__":
    unittest.main()
