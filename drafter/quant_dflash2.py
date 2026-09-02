"""Quantize the DFlash2 drafter (incoai/Qwen3.8-27B-DFlash2, 1.92B params bf16 = 3.85 GB)
to W4A16 compressed-tensors (Marlin) with GPTQ, calibrated on Hessians captured by
capture_dflash2.py from the drafter's own inputs on real traffic.

  python drafter/quant_dflash2.py <src_dir> <dst_dir> <hessians.pt> [--bits 4] [--fc-bits 4|8|16] [--rtn]

What gets quantized (group 128, symmetric, no act-order so Marlin needs no g_idx):
  layers.N.self_attn.{q,k,v,o}_proj, layers.N.mlp.{gate,up,down}_proj   int4 GPTQ   (1.61B params)
  fc (5120 x 25600, the target-hidden-state projection)                   int4 GPTQ by default (--fc-bits)
Left in bf16 (vLLM builds them with quant_config=None anyway): the two grouped-conv
kernel projections per layer, the candidate selector, norms. Result: ~1.0 GB instead of
3.85, i.e. ~3 ms less per decode step on a 3090, and the KV pool survives.

q/k/v share one input (vLLM fuses them into qkv_proj) and so do gate/up (gate_up_proj), so
the Hessians are captured per fused module and reused for each sub-weight. The config.json
gets a compressed-tensors quantization_config with an ignore list for the bf16 modules, in
vLLM's module-prefix namespace (model.layers.N..., model.fc, model.candidate_selector...).
"""
import json, os, sys, shutil, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from compressed_tensors.compressors.pack_quantized.base import pack_to_int32
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gptq_utils import gptq_quantize, rtn_quantize, dequant

S = sys.argv[1].rstrip("/") + "/"; D = sys.argv[2].rstrip("/") + "/"; HP = sys.argv[3]
BITS = int(sys.argv[sys.argv.index("--bits") + 1]) if "--bits" in sys.argv else 4
FC_BITS = int(sys.argv[sys.argv.index("--fc-bits") + 1]) if "--fc-bits" in sys.argv else 4
RTN = "--rtn" in sys.argv
GROUP = 128
dev = "cuda" if torch.cuda.is_available() else "cpu"

cfg = json.load(open(S + "config.json"))
L = cfg["num_hidden_layers"]
# checkpoint name -> (hessian key in capture file)
LIN = {}
for i in range(L):
    for p in ["q_proj", "k_proj", "v_proj"]:
        LIN[f"layers.{i}.self_attn.{p}"] = f"layers.{i}.self_attn.qkv_proj"
    LIN[f"layers.{i}.self_attn.o_proj"] = f"layers.{i}.self_attn.o_proj"
    for p in ["gate_proj", "up_proj"]:
        LIN[f"layers.{i}.mlp.{p}"] = f"layers.{i}.mlp.gate_up_proj"
    LIN[f"layers.{i}.mlp.down_proj"] = f"layers.{i}.mlp.down_proj"
if FC_BITS < 16:
    LIN["fc"] = "fc"

HS = torch.load(HP, map_location="cpu") if not RTN else {}
with safe_open(S + "model.safetensors", "pt") as f:
    meta = f.metadata()
    tensors = {k: f.get_tensor(k) for k in f.keys()}
out = {}
t0 = time.time()
stats = []
for name, w in tensors.items():
    base = name[: -len(".weight")] if name.endswith(".weight") else None
    if base in LIN:
        W = w.to(dev)
        bits = FC_BITS if base == "fc" else BITS
        if RTN or LIN[base] not in HS:
            if not RTN:
                print(f"  !! no Hessian for {LIN[base]}, RTN fallback")
            q, s = rtn_quantize(W, bits, GROUP)
        else:
            H = HS[LIN[base]]["H"].to(dev)
            if base.endswith((".k_proj", ".v_proj")) and "ctx_kv" in HS:
                # k/v rows also project the context rows (fused context-KV precompute):
                # blend the two input distributions by their row counts. GPTQ is
                # row-independent given H, so q_proj keeps the pure query Hessian.
                # NOTE: measured 7% worse greedy acceptance than the query-only Hessian
                # (drafter/README.md); the shipped drafter omits "ctx_kv" from the file.
                nq, Hc, nc = HS[LIN[base]]["n"], HS["ctx_kv"]["H"].to(dev), HS["ctx_kv"]["n"]
                H = (nq * H + nc * Hc) / (nq + nc)
                del Hc
            q, s = gptq_quantize(W, H, bits=bits, group=GROUP, blocksize=GROUP)
            del H
        rel = ((dequant(q, s) - W.float()).norm() / W.float().norm()).item()
        stats.append((base, bits, rel))
        out[base + ".weight_packed"] = pack_to_int32(q.cpu(), bits, packed_dim=1).contiguous()
        out[base + ".weight_scale"] = s.to(torch.float16).cpu().contiguous()
        out[base + ".weight_shape"] = torch.tensor(list(W.shape), dtype=torch.int64)
        del W, q, s
        if dev == "cuda":
            torch.cuda.empty_cache()
    else:
        out[name] = w
for base, bits, rel in stats:
    if base.startswith("layers.0.") or not base.startswith("layers."):
        print(f"  int{bits} {base}: rel err {rel:.4f}")
print(f"quantized {len(stats)} matrices in {time.time()-t0:.0f}s; "
      f"mean rel err {sum(r for _,_,r in stats)/len(stats):.4f}")

os.makedirs(D, exist_ok=True)
save_file(out, D + "model.safetensors", metadata=meta or {"format": "pt"})
for f in os.listdir(S):
    if f not in ("model.safetensors", "config.json") and not f.startswith("."):
        p = os.path.join(S, f)
        if os.path.isfile(p):
            shutil.copy(p, D + f)
ignore = ["re:.*kernel_projection$", "re:.*candidate_selector.*", "re:.*hidden_projection$"]
if FC_BITS >= 16:
    ignore.append("re:.*\\.fc$")
groups = {"group_0": {"format": "pack-quantized", "input_activations": None, "output_activations": None,
                      "targets": ["Linear"],
                      "weights": {"actorder": None, "block_structure": None, "dynamic": False, "group_size": GROUP,
                                  "num_bits": BITS, "observer": "memoryless_minmax", "observer_kwargs": {},
                                  "scale_dtype": None, "strategy": "group", "symmetric": True, "type": "int",
                                  "zp_dtype": None}}}
if FC_BITS < 16 and FC_BITS != BITS:
    groups["group_1"] = json.loads(json.dumps(groups["group_0"]))
    groups["group_1"]["targets"] = ["re:.*\\.fc$"]
    groups["group_1"]["weights"]["num_bits"] = FC_BITS
cfg["quantization_config"] = {"config_groups": groups, "format": "pack-quantized", "global_compression_ratio": None,
                              "ignore": ignore, "kv_cache_scheme": None, "quant_method": "compressed-tensors",
                              "quantization_status": "compressed", "sparsity_config": {}, "transform_config": {},
                              "version": "0.17.0"}
json.dump(cfg, open(D + "config.json", "w"), indent=2)
sz = os.path.getsize(D + "model.safetensors") / 2**30
print(f"wrote {D} ({sz:.2f} GiB)")
