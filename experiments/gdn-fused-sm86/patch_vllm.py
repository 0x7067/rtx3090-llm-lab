"""Apply the opt-in SM86 fused-GDN experiment to the v9 vLLM install."""

from pathlib import Path
import sys


path = Path(sys.argv[1])
source = path.read_text()


def replace_once(old: str, new: str) -> None:
    global source
    if source.count(old) != 1:
        raise RuntimeError(f"expected one patch anchor, found {source.count(old)}: {old[:80]!r}")
    source = source.replace(old, new, 1)


def replace_once_after(anchor: str, old: str, new: str) -> None:
    global source
    start = source.index(anchor)
    end = source.index("\n    def forward_xpu(", start)
    prefix, section, suffix = source[:start], source[start:end], source[end:]
    if section.count(old) != 1:
        raise RuntimeError(
            f"expected one patch anchor after {anchor!r}, found {section.count(old)}"
        )
    source = prefix + section.replace(old, new, 1) + suffix


replace_once("from typing import Literal\n", "import os\nfrom typing import Literal\n")

replace_once(
    "logger = init_logger(__name__)\n",
    """logger = init_logger(__name__)

_FUSED_GDN_SM86 = os.environ.get("VLLM_ENABLE_FUSED_GDN_SM86", "").lower() in {
    "1", "true", "on", "sm86", "cuda-sm86"
}
if _FUSED_GDN_SM86:
    torch.ops.load_library("/app/experimental/gdn_fused_sm86_ext.so")
    logger.info("FUSED_GDN_SM86_ACTIVE private dispatcher loaded")
""",
)

replace_once_after(
    "    def forward_cuda(\n",
    """        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)

        if self.gqa_interleaved_layout:
""",
    """        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)

        if _FUSED_GDN_SM86:
            core_attn_out = torch.zeros(
                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed_sm86(
                mixed_qkvz,
                ba,
                core_attn_out,
                layer_name=_encode_layer_name(self.prefix),
            )
            output, _ = self.out_proj(core_attn_out.flatten(-2))
            return output

        if self.gqa_interleaved_layout:
""",
)

replace_once(
    "    def _forward_core(\n",
    """    def _can_use_fused_gdn_sm86(self, attn_metadata: GDNAttentionMetadata) -> bool:
        state_indices = attn_metadata.spec_state_indices_tensor
        return (
            attn_metadata.spec_sequence_masks is not None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes == 0
            and attn_metadata.num_spec_decodes > 0
            and self.kv_cache[1].dtype in (torch.float16, torch.bfloat16)
            and self.num_v_heads % self.num_k_heads == 0
            and state_indices is not None
            and state_indices.size(1) <= 8
            and hasattr(torch.ops.gdn_fused_sm86, "decode_post_conv_mtp")
        )

    def _forward_core_fused_norm_packed_sm86(
        self,
        mixed_qkvz: torch.Tensor,
        ba: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        if attn_metadata_raw is None:
            self._warmup_prefill_kernels(mixed_qkvz[:, :qkv_size], 0)
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        mixed_qkv, output_gate_flat = mixed_qkvz.split(
            [qkv_size, self.value_dim // self.tp_size], dim=-1
        )
        output_gate = output_gate_flat.reshape(
            output_gate_flat.size(0), -1, self.head_v_dim
        )
        b, a = self.split_ba(ba)

        if self._can_use_fused_gdn_sm86(attn_metadata):
            state_indices = attn_metadata.spec_state_indices_tensor
            cu_seqlens = attn_metadata.spec_query_start_loc
            num_accepted_tokens = attn_metadata.num_accepted_tokens
            assert state_indices is not None
            assert cu_seqlens is not None
            assert num_accepted_tokens is not None
            num_requests = attn_metadata.num_spec_decodes
            num_actual_tokens = attn_metadata.num_actual_tokens
            conv_state = (
                self.kv_cache[0]
                if is_conv_state_dim_first()
                else self.kv_cache[0].transpose(-1, -2)
            )
            conv_weights = self.conv1d.weight.view(
                self.conv1d.weight.size(0), self.conv1d.weight.size(2)
            )
            mixed_qkv = causal_conv1d_update(
                mixed_qkv[:num_actual_tokens],
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=state_indices[:num_requests, 0],
                num_accepted_tokens=num_accepted_tokens[:num_requests],
                query_start_loc=cu_seqlens[: num_requests + 1],
                max_query_len=state_indices.size(1),
                validate_data=False,
            )
            torch.ops.gdn_fused_sm86.decode_post_conv_mtp(
                mixed_qkv,
                a[:num_actual_tokens],
                b[:num_actual_tokens],
                self.A_log,
                self.dt_bias,
                state_indices[:num_requests],
                cu_seqlens[: num_requests + 1],
                num_accepted_tokens[:num_requests],
                self.kv_cache[1],
                output_gate[:num_actual_tokens],
                self.norm.weight,
                core_attn_out[:num_actual_tokens],
                self.head_k_dim**-0.5,
                self.layer_norm_epsilon,
            )
            return

        self._forward_core(
            mixed_qkv=mixed_qkv,
            b=b.contiguous(),
            a=a.contiguous(),
            core_attn_out=core_attn_out,
        )
        num_actual_tokens = attn_metadata.num_actual_tokens
        normalized = self.norm(
            core_attn_out[:num_actual_tokens].reshape(-1, self.head_v_dim),
            output_gate[:num_actual_tokens].reshape(-1, self.head_v_dim),
        )
        core_attn_out[:num_actual_tokens].copy_(
            normalized.reshape_as(core_attn_out[:num_actual_tokens])
        )

    def _forward_core(
""",
)

source += """

def qwen_gdn_attention_core_fused_norm_packed_sm86(
    mixed_qkvz: torch.Tensor,
    ba: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._forward_core_fused_norm_packed_sm86(mixed_qkvz, ba, core_attn_out)


def qwen_gdn_attention_core_fused_norm_packed_sm86_fake(
    mixed_qkvz: torch.Tensor,
    ba: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    return


direct_register_custom_op(
    op_name="qwen_gdn_attention_core_fused_norm_packed_sm86",
    op_func=qwen_gdn_attention_core_fused_norm_packed_sm86,
    mutates_args=["core_attn_out"],
    fake_impl=qwen_gdn_attention_core_fused_norm_packed_sm86_fake,
)
"""

path.write_text(source)
