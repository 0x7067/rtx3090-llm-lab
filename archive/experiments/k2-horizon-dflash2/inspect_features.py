import glob, sys, torch
files = sorted(f for f in glob.glob(sys.argv[1] + "/**/*.ckpt", recursive=True) if "vocab_mapping" not in f)
print("files:", len(files))
tot_tokens = 0
tot_bytes = 0
for f in files[:64]:
    d = torch.load(f, map_location="cpu", weights_only=True)
    if isinstance(d, dict):
        n = None
        for k, v in d.items():
            if torch.is_tensor(v):
                if f == files[0]:
                    print(f"  {k}: {tuple(v.shape)} {v.dtype}")
                tot_bytes += v.numel() * v.element_size()
                if k == "input_ids":
                    n = v.numel()
        if n:
            tot_tokens += n
    if f == files[0]:
        print("  keys:", list(d.keys()) if isinstance(d, dict) else type(d))
print(f"tokens: {tot_tokens}, tensor bytes: {tot_bytes/1e9:.2f} GB, bytes/token: {tot_bytes/max(1,tot_tokens):.0f}")
