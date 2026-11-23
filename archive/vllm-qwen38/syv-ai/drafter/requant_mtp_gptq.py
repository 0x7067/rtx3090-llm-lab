"""Re-quantize the MTP module of a model dir with GPTQ (calibrated Hessians from
train_mtp.py --dump-hessians) into a new variant dir. Keeps everything else (lm_head,
draft head, ids) from the source dir.

  python requant_mtp_gptq.py <src_dir> <dst_dir> <hessians.pt> [--bits 4] [--orig <dir with model_extra_tensors.safetensors.bak-mtp>]
"""
import json, os, sys, shutil, copy
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from compressed_tensors.compressors.pack_quantized.base import pack_to_int32
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gptq_utils import gptq_quantize, dequant

S, D, HP = sys.argv[1].rstrip("/") + "/", sys.argv[2].rstrip("/") + "/", sys.argv[3]
BITS = int(sys.argv[sys.argv.index("--bits") + 1]) if "--bits" in sys.argv else 4
ORIG = (sys.argv[sys.argv.index("--orig") + 1] if "--orig" in sys.argv else os.path.join(REPO, "models", "Qwen3.8-27B-W4A16-AutoRound")).rstrip("/") + "/"
GROUP = 128
LIN = ["mtp.fc", "mtp.layers.0.mlp.down_proj", "mtp.layers.0.mlp.gate_proj", "mtp.layers.0.mlp.up_proj",
       "mtp.layers.0.self_attn.q_proj", "mtp.layers.0.self_attn.k_proj", "mtp.layers.0.self_attn.v_proj",
       "mtp.layers.0.self_attn.o_proj"]
os.makedirs(D, exist_ok=True)
for f in os.listdir(S):
    if f.startswith("model-0000") and f.endswith(".safetensors") and not os.path.exists(D + f):
        os.link(S + f, D + f)
for f in ["tokenizer.json"]:
    if not os.path.exists(D + f):
        os.link(S + f, D + f)
for f in ["chat_template.jinja", "generation_config.json", "processor_config.json", "quantization_config.json",
          "tokenizer_config.json", "config.json", "model.safetensors.index.json", "mtp_draft_vocab_ids.pt"]:
    shutil.copy(S + f, D + f)
# bf16 mtp weights
with safe_open(ORIG + "model_extra_tensors.safetensors.bak-mtp", "pt") as f:
    bf = {k: f.get_tensor(k) for k in f.keys()}
# current extra tensors (draft head etc.)
tensors = {}
with safe_open(S + "model_extra_tensors.safetensors", "pt") as f:
    meta = f.metadata()
    for k in f.keys():
        if any(k.startswith(m + ".") for m in LIN):
            continue
        tensors[k] = f.get_tensor(k)
HS = torch.load(HP)
for m in LIN:
    w = bf[m + ".weight"].cuda()
    q, scale = gptq_quantize(w, HS[m].cuda(), bits=BITS, group=GROUP, blocksize=GROUP)
    rel = ((dequant(q, scale) - w.float()).norm() / w.float().norm()).item()
    print(f"  GPTQ {m}: {tuple(w.shape)} int{BITS} rel err {rel:.4f}")
    out_f, in_f = w.shape
    tensors[m + ".weight_packed"] = pack_to_int32(q.cpu(), BITS, packed_dim=1).contiguous()
    tensors[m + ".weight_scale"] = scale.to(torch.float16).cpu().contiguous()
    tensors[m + ".weight_shape"] = torch.tensor([out_f, in_f], dtype=torch.int64)
    del w, q, scale; torch.cuda.empty_cache()
if os.path.exists(D + "model_extra_tensors.safetensors"):
    os.remove(D + "model_extra_tensors.safetensors")
save_file(tensors, D + "model_extra_tensors.safetensors", metadata=meta or {"format": "pt"})
c = json.load(open(D + "config.json"))
qc = c["quantization_config"]
g = qc["config_groups"]["group_3"]
assert g["targets"] == ["re:^mtp\\..*"], g["targets"]
g["weights"]["num_bits"] = BITS
json.dump(c, open(D + "config.json", "w"), indent=2)
print("done", D)
