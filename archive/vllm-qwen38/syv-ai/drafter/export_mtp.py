"""Build a servable model-dir variant from a fine-tuned MTP module (train_mtp.py output).

  python export_mtp.py runs/r1/mtp_bf16.safetensors <src_model_dir> <dst_dir> [--bits 8] [--head-bits 8]
                       [--gptq hessians.pt]     # GPTQ (calibrated) instead of RTN for the mtp linears

dst_dir gets hardlinked weight shards, the pre-MTP-quant config/index, a fresh
model_extra_tensors.safetensors with the trained mtp.* tensors (bf16), then prepare/quant_mtp.py
(int8/int4 RTN) is run on it, and the draft head is written: the trained
mtp.draft_lm_head.weight if present in the checkpoint, else rows sliced from lm_head
(prepare/build_draft_vocab.py --ids).
"""
import json, os, sys, shutil, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from compressed_tensors.compressors.pack_quantized.base import pack_to_int32

ck, S, D = sys.argv[1], sys.argv[2].rstrip("/") + "/", sys.argv[3].rstrip("/") + "/"
BITS = int(sys.argv[sys.argv.index("--bits") + 1]) if "--bits" in sys.argv else 8
HBITS = int(sys.argv[sys.argv.index("--head-bits") + 1]) if "--head-bits" in sys.argv else 8
QS = REPO
os.makedirs(D, exist_ok=True)
for f in os.listdir(S):
    if f.startswith("model-0000") and f.endswith(".safetensors"):
        if not os.path.exists(D + f):
            os.link(S + f, D + f)
if not os.path.exists(D + "tokenizer.json"):
    os.link(S + "tokenizer.json", D + "tokenizer.json")
for f in ["chat_template.jinja", "generation_config.json", "processor_config.json", "quantization_config.json",
          "tokenizer_config.json", "draft_vocab_ids.json"]:
    if os.path.exists(S + f):
        shutil.copy(S + f, D + f)
shutil.copy(S + "config.json.bak-mtp", D + "config.json")
shutil.copy(S + "model.safetensors.index.json.bak-mtp", D + "model.safetensors.index.json")

with safe_open(S + "model_extra_tensors.safetensors.bak-mtp", "pt") as f:
    base = {k: f.get_tensor(k) for k in f.keys()}
with safe_open(ck, "pt") as f:
    trained = {k: f.get_tensor(k) for k in f.keys()}
head = trained.pop("mtp.draft_lm_head.weight", None)
for k in base:
    if k.startswith("mtp.") and k not in trained:
        print("WARNING: trained checkpoint lacks", k, "- keeping original")
out = dict(base)
n_new = 0
for k, v in trained.items():
    assert k in base, ("unexpected key", k)
    assert v.shape == base[k].shape, (k, v.shape, base[k].shape)
    out[k] = v.to(torch.bfloat16).contiguous(); n_new += 1
for f in ["model_extra_tensors.safetensors", "mtp_draft_vocab_ids.pt"]:
    if os.path.exists(D + f):
        os.remove(D + f)   # never truncate a hardlink
save_file(out, D + "model_extra_tensors.safetensors", metadata={"format": "pt"})
print(f"wrote {n_new} trained tensors (+{len(out) - n_new} kept) to {D}model_extra_tensors.safetensors")

GPTQ = sys.argv[sys.argv.index("--gptq") + 1] if "--gptq" in sys.argv else None
if GPTQ is None:
    subprocess.check_call([sys.executable, f"{QS}/prepare/quant_mtp.py", D, "--bits", str(BITS)])
else:
    # same output format as prepare/quant_mtp.py, weights quantized with GPTQ using the dumped Hessians
    import copy
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gptq_utils import gptq_quantize, dequant
    HS = torch.load(GPTQ)
    GROUP = 128
    MTP_LINEARS = ["mtp.fc", "mtp.layers.0.mlp.down_proj", "mtp.layers.0.mlp.gate_proj", "mtp.layers.0.mlp.up_proj",
                   "mtp.layers.0.self_attn.q_proj", "mtp.layers.0.self_attn.k_proj", "mtp.layers.0.self_attn.v_proj",
                   "mtp.layers.0.self_attn.o_proj"]
    idx = json.load(open(D + "model.safetensors.index.json")); wm = idx["weight_map"]
    shard = "model_extra_tensors.safetensors"
    tensors = {}
    with safe_open(D + shard, "pt") as f:
        meta = f.metadata()
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    for m in MTP_LINEARS:
        w = tensors.pop(m + ".weight").cuda()
        H = HS[m].cuda()
        q, scale = gptq_quantize(w, H, bits=BITS, group=GROUP, blocksize=GROUP)
        rel = ((dequant(q, scale) - w.float()).norm() / w.float().norm()).item()
        print(f"  GPTQ {m}: {tuple(w.shape)} int{BITS} rel error {rel:.4f}")
        out_f, in_f = w.shape
        tensors[m + ".weight_packed"] = pack_to_int32(q.cpu(), BITS, packed_dim=1).contiguous()
        tensors[m + ".weight_scale"] = scale.to(torch.float16).cpu().contiguous()
        tensors[m + ".weight_shape"] = torch.tensor([out_f, in_f], dtype=torch.int64)
        del wm[m + ".weight"]
        for s_ in ("weight_packed", "weight_scale", "weight_shape"):
            wm[f"{m}.{s_}"] = shard
    shutil.copy(D + shard, D + shard + ".bak-mtp")
    save_file(tensors, D + shard, metadata=meta or {"format": "pt"})
    shutil.copy(D + "model.safetensors.index.json", D + "model.safetensors.index.json.bak-mtp")
    json.dump(idx, open(D + "model.safetensors.index.json", "w"), indent=2)
    c = json.load(open(D + "config.json"))
    shutil.copy(D + "config.json", D + "config.json.bak-mtp")
    qc = c["quantization_config"]
    qc["ignore"] = [i for i in qc["ignore"] if i not in MTP_LINEARS]
    g = copy.deepcopy(qc["config_groups"]["group_0"])
    g["targets"] = ["re:^mtp\\..*"]
    g["weights"]["num_bits"] = BITS
    qc["config_groups"]["group_3"] = g
    json.dump(c, open(D + "config.json", "w"), indent=2)

if head is None:
    subprocess.check_call([sys.executable, f"{QS}/prepare/build_draft_vocab.py", D, "--ids",
                           os.environ.get("DRAFT_IDS", f"{QS}/prepare/draft_vocab_ids.json")])
else:
    GROUP = 128; QMAX = 2 ** (HBITS - 1) - 1
    w = head.to(torch.float32)
    out_f, in_f = w.shape
    g = w.reshape(out_f, in_f // GROUP, GROUP)
    scale = torch.clamp(g.abs().amax(dim=-1, keepdim=True) / QMAX, min=1e-10)
    q = torch.clamp(torch.round(g / scale), -QMAX - 1, QMAX).to(torch.int8).reshape(out_f, in_f)
    deq = (q.reshape(out_f, -1, GROUP).float() * scale).reshape(out_f, in_f)
    print(f"draft head int{HBITS} round-trip rel error {((deq - w).norm() / w.norm()).item():.4f}")
    extra = D + "model_extra_tensors.safetensors"
    tensors = {}
    with safe_open(extra, "pt") as f:
        meta = f.metadata()
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    tensors["mtp.draft_lm_head.weight_packed"] = pack_to_int32(q, HBITS, packed_dim=1).contiguous()
    tensors["mtp.draft_lm_head.weight_scale"] = scale.squeeze(-1).to(torch.float16).contiguous()
    tensors["mtp.draft_lm_head.weight_shape"] = torch.tensor([out_f, in_f], dtype=torch.int64)
    save_file(tensors, extra, metadata=meta or {"format": "pt"})
    idx = json.load(open(D + "model.safetensors.index.json"))
    for s in ("weight_packed", "weight_scale", "weight_shape"):
        idx["weight_map"][f"mtp.draft_lm_head.{s}"] = "model_extra_tensors.safetensors"
    json.dump(idx, open(D + "model.safetensors.index.json", "w"), indent=2)
    shutil.copy(S + "mtp_draft_vocab_ids.pt", D + "mtp_draft_vocab_ids.pt")
    if HBITS != BITS:
        c = json.load(open(D + "config.json"))
        qc = c["quantization_config"]
        # separate group for the head if its bit-width differs from the rest of mtp.*
        import copy
        g = copy.deepcopy(qc["config_groups"]["group_3"]); g["targets"] = ["re:^mtp\\.draft_lm_head$"]
        g["weights"]["num_bits"] = HBITS
        qc["config_groups"]["group_4"] = g
        json.dump(c, open(D + "config.json", "w"), indent=2)
print("export done:", D)
