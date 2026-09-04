"""Register the K2 Horizon chat template with SpecForge.

Import this module before `specforge train` / data preprocessing (e.g. add
`import specforge_k2_template` at the top of the training entrypoint, or place
it in specforge/data/template.py). The regenerated data must be produced with
reasoning_effort=high so every assistant turn opens with <ifm|think>\n, which
matches the assistant_header below; keep ignore_token empty so the thinking
tokens are trained (the drafter must draft reasoning too).
"""
from specforge.data.template import TEMPLATE_REGISTRY, ChatTemplate

TEMPLATE_REGISTRY.register(
    name="k2-horizon-thinking",
    template=ChatTemplate(
        assistant_header="<|ifm|im_start|>assistant\n<ifm|think>\n",
        user_header="<|ifm|im_start|>user\n",
        system_prompt=None,
        end_of_turn_token="<|ifm|im_end|>",
        parser_type="thinking",
        enable_thinking=True,
    ),
)

# Thinking disabled (what the template renders for enable_thinking=false or an
# empty reasoning_content): no newline after </ifm|think>, none after im_end.
TEMPLATE_REGISTRY.register(
    name="k2-horizon-nothink",
    template=ChatTemplate(
        assistant_header="<|ifm|im_start|>assistant\n<ifm|think>\n</ifm|think>",
        user_header="<|ifm|im_start|>user\n",
        system_prompt=None,
        end_of_turn_token="<|ifm|im_end|>",
    ),
)
