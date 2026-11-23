"""Exercise indexed state selection, ragged batches and cache-boundary snapshots."""

import json

import torch
from vllm.third_party.flash_linear_attention.ops import (
    fused_sigmoid_gating_delta_rule_update as reference,
)
from vllm_compact_gdn.kernels import restore_kernel, verify_kernel


@torch.inference_mode()
def main():
    torch.manual_seed(3090)
    rows = []
    for lens in ([8], [8, 8, 8, 8], [2, 8, 4, 6]):
        n = len(lens)
        total = sum(lens)
        slots = 1 + n * 8
        # An interleaved pool reproduces non-dense strides between state slots.
        storage = (
            torch.randn(slots * 2, 48, 128, 128, device="cuda", dtype=torch.float16)
            * 0.01
        )
        state = storage[::2]
        expected = state.clone()
        indices = (
            1
            + torch.arange(n, device="cuda")[:, None] * 8
            + torch.arange(8, device="cuda")
        ).int()
        lengths = torch.tensor(
            [0] + list(torch.tensor(lens).cumsum(0).tolist()),
            device="cuda",
            dtype=torch.int32,
        )
        delta = torch.empty(n, 8, 48, 128, device="cuda")
        keys = torch.empty(n, 8, 16, 128, device="cuda")
        decay = torch.empty(n, 8, 48, device="cuda")
        source = torch.empty(n, device="cuda", dtype=torch.int32)
        previous = torch.ones(n, device="cuda", dtype=torch.int32)
        for step in range(96):
            slow = step >= 48
            q = torch.randn(1, total, 16, 128, device="cuda", dtype=torch.bfloat16)
            k = torch.randn_like(q)
            v = torch.randn(1, total, 48, 128, device="cuda", dtype=torch.bfloat16)
            a = torch.randn(total, 48, device="cuda", dtype=torch.bfloat16)
            b = torch.randn_like(a)
            bias = torch.full((48,), -6.0 if slow else 0.0, device="cuda")
            alog = torch.randn(48, device="cuda") * 0.5
            accepted = torch.tensor(
                [(step + i) % (length + 1) for i, length in enumerate(lens)],
                device="cuda",
                dtype=torch.int32,
            )
            computed = torch.tensor(
                [5180 + (step + i) % 8 for i in range(n)],
                device="cuda",
                dtype=torch.int32,
            )
            mapping = torch.randperm(n + 3, device="cuda", dtype=torch.int32)[:n]
            new_computed = torch.zeros(n + 3, device="cuda", dtype=torch.int32)
            new_computed[mapping.long()] = computed + accepted.clamp_min(1)
            before = state.clone()
            expected.copy_(before)
            wanted = reference(
                A_log=alog,
                a=a,
                b=b,
                dt_bias=bias,
                q=q,
                k=k,
                v=v,
                initial_state=expected,
                cu_seqlens=lengths,
                ssm_state_indices=indices,
                num_accepted_tokens=previous,
                use_qk_l2norm_in_kernel=True,
            )[0]
            actual = torch.empty_like(v)
            verify_kernel[(4, n * 48)](
                q,
                k,
                v,
                a,
                bias,
                alog,
                b,
                state,
                actual,
                delta,
                keys,
                decay,
                1.0,
                20.0,
                128**-0.5,
                lengths,
                indices,
                previous,
                source,
                SS=state.stride(0),
                IS=8,
                T=8,
                H=16,
                HV=48,
                D=128,
                BV=32,
                num_warps=4,
                num_stages=3,
            )
            assert torch.equal(actual, wanted), (lens, step, "output")
            restore_kernel[(4, n * 48)](
                state,
                delta,
                keys,
                decay,
                accepted,
                source,
                indices,
                lengths,
                new_computed,
                mapping,
                BlockSize=5184,
                SS=state.stride(0),
                IS=8,
                T=8,
                H=16,
                HV=48,
                D=128,
                BV=32,
                num_warps=4,
                num_stages=3,
            )
            for i, count in enumerate(accepted.clamp_min(1).tolist()):
                running = int(computed[i]) + 1
                boundary = ((running + count - 1) // 5184) * 5184 - running + 1
                required = {count - 1}
                if 1 <= boundary <= count:
                    required.add(boundary - 1)
                for t in range(8):
                    slot = int(indices[i, t])
                    target = expected[slot] if t in required else before[slot]
                    assert torch.equal(state[slot], target), (lens, step, i, t, "state")
            previous = accepted.clamp_min(1).roll(1)
            indices = indices.roll(1, 0).contiguous()
        # Padded/invalid previous counts must produce zero output and retain cache.
        for invalid in (0, 9):
            before = state.clone()
            previous.fill_(invalid)
            actual.fill_(1)
            verify_kernel[(4, n * 48)](
                q,
                k,
                v,
                a,
                bias,
                alog,
                b,
                state,
                actual,
                delta,
                keys,
                decay,
                1.0,
                20.0,
                128**-0.5,
                lengths,
                indices,
                previous,
                source,
                SS=state.stride(0),
                IS=8,
                T=8,
                H=16,
                HV=48,
                D=128,
                BV=32,
                num_warps=4,
                num_stages=3,
            )
            restore_kernel[(4, n * 48)](
                state,
                delta,
                keys,
                decay,
                accepted,
                source,
                indices,
                lengths,
                new_computed,
                mapping,
                BlockSize=5184,
                SS=state.stride(0),
                IS=8,
                T=8,
                H=16,
                HV=48,
                D=128,
                BV=32,
                num_warps=4,
                num_stages=3,
            )
            assert torch.count_nonzero(actual) == 0 and torch.equal(state, before)
        rows.append({"lengths": lens, "blocks": 96, "bitwise": True})
    print(json.dumps({"passed": True, "cases": rows}), flush=True)


if __name__ == "__main__":
    main()
