# Short DFlash image CPU qualification

The authorized build and CPU checks completed on 2026-11-06 in approximately 225 seconds. Image: `127.0.0.1:5000/home-llm-strata:normal120-dflash-short-v1@sha256:83077f60377d9e303d446a2f045381daab28a123e8e314c5bbe1ba53314884a2`.

The exact patched launcher and original vLLM wrapper hashes passed. The custom FP16 GDN plugin is absent and its gate defaults off. Bash syntax and isolated launcher argv checks passed for unchanged default k7 and explicitly gated k3/k5, including retained negative guards. Neither Torch nor vLLM was imported; no NVIDIA device, model request or GPU execution was used. The nonroot read-only CPU pod used deny-all networking and tmpfs. Registry manifest bytes and the actual runtime image identity were verified.

The retained owner unit exited successfully, and the temporary namespace `normal120-dflash-short-cpu` was confirmed absent after cleanup. The build uses the pinned original image, not the rejected FP16 GDN candidate. This receipt establishes packaging and launcher behavior only; serving, GPU correctness, quality and speed remain separate gates. No serving deployment was performed by this workflow.
