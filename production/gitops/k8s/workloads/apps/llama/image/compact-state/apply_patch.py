"""Apply the serving hooks only to the inspected, pinned source files."""

import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
here = Path(__file__).resolve().parent
hashes = json.loads((here / "source-hashes.json").read_text())
for name, digest in hashes.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
for name in hashes:
    p = root / name
    s = p.read_text()
    if name.endswith("qwen_gdn_linear_attn.py"):
        s = (
            "from vllm_compact_gdn.runtime import initialize as compact_initialize, update as compact_update\n"
            + s
        )
        marker = "        self.chunk_gated_delta_rule = ChunkGatedDeltaRule()"
        assert s.count(marker) == 1
        s = s.replace(
            marker, "        compact_initialize(self, vllm_config)\n" + marker
        )
        marker = """            core_attn_out_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update("""
        assert s.count(marker) == 1
        s = s.replace(
            marker,
            """            core_attn_out_spec, last_recurrent_state = (
                compact_update(self, attn_metadata,""",
        )
    elif name == "envs.py":
        marker = "environment_variables: dict[str, Callable[[], Any]] = {"
        assert s.count(marker) == 1
        s = s.replace(
            marker,
            marker
            + '\n    "VLLM_COMPACT_GDN": lambda: os.getenv("VLLM_COMPACT_GDN", "0") == "1",',
        )
    else:
        s = (
            "from vllm_compact_gdn.runtime import begin as compact_begin, finalize as compact_finalize\n"
            + s
        )
        marker = "        self.model_state.postprocess_state(\n            idx_mapping, num_sampled, self.req_states.num_computed_tokens.gpu\n        )"
        assert s.count(marker) == 1
        s = s.replace(
            marker,
            "        compact_finalize(self, idx_mapping, num_sampled)\n" + marker,
        )
        marker = "    ) -> ModelRunnerOutput | IntermediateTensors | None:\n        if not dummy_run:"
        assert s.count(marker) == 1
        s = s.replace(
            marker,
            "    ) -> ModelRunnerOutput | IntermediateTensors | None:\n        compact_begin(dummy_run)\n        if not dummy_run:",
        )
    compile(s, str(p), "exec")
    p.write_text(s)
