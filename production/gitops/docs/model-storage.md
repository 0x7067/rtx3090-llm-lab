# Server model weights

Use `/data/models` as the model library entry point. It points to
`/data/buttercup_6tb/k3s/llama-models` on the Buttercup HDD. The production
deployment mounts read-only serving copies from
[`/srv/llama-nvme/production-models`](../k8s/workloads/apps/llama/README.md#storage).
Inkling uses a serving copy under `/data/development/llama-serving` on the
second NVMe to preserve space for etcd on the root SSD. Its pinned Python and
CUDA runtime lives in `/data/models/inkling-2/vllm-ampere` on Buttercup.
The retired llama.cpp GGUF bundle moved to
`/data/models/inkling-2/archive/07fb508609f0-b12588`. Its previous path under
`/data/development/llama-serving/inkling-2-mini` is a symlink to the archive.

## Layout

```text
/data/models/<family>/<publisher>/<repository>/
/data/models/<family>/local/<experiment-or-export>/
/data/models/<family>/<publisher>/<repository>/hf-cache-user/
/data/models/CATALOG.json
```

Keep model weights, tokenizers, configuration, vision projectors, and adapters
together. Retain the original filenames. Use the `local` publisher when an
artifact was converted or trained locally, or its upstream identity has not
been established. Preserve distinct quantizations and revisions; never replace
an existing filename with different bytes.

Hugging Face cache bundles retain their `blobs`, `refs`, and `snapshots`
structure. The original cache locations point to those bundles so clients keep
using their existing cache settings. Dataset caches stay separate.

Application caches are preserved as complete bundles under
`vision/local/immich-model-cache` and `qwen3.6/local/qwen-dflash-cache`.
Their original Kubernetes local-volume directories point to these HDD locations;
the existing PVC names and workload mount paths are retained.

The catalog records weight paths, sizes, storage identities, and existing
checksum records. Hardlinked paths share one stored copy; they are not backups.
Tokenizer-only GGUF fixtures in source checkouts are not inference weights.

## Compatibility

Old model paths remain as links. Complete benchmark model directories are moved
as bundles so Docker can still mount the old directory paths. The Swift verbatim
benchmark mounts its whole directory, so that directory is preserved as a bundle
under `qwen3.8/local/`; its published weights also appear under `qwen3.8/ukisai/`.
These names use hardlinks on the same disk, without a second copy of the weights.

Do not replace a file within a container-mounted directory with an absolute
host symlink unless the link's target is also mounted in the container. Links
inside the serving volume use relative targets that resolve under `/models`.

## New downloads and verification

Download directly into the appropriate library directory using a temporary
filename. Record the source repository, immutable revision, and expected checksum
when available. Verify the download before publishing its final filename. The
llama fetch configuration already supports pinned revisions and SHA-256 stamps.

Keep this workload off the root SSD, which also hosts k3s/etcd. Store rollback
variants in the library instead of benchmark checkout directories. This library
is one physical copy on Buttercup; consolidation does not provide a second-disk
backup.

The September 15 migration plan, copy-verification receipts, and discovery
inventory are in `benchmarks/model-storage-2026-09-15/`. Cross-filesystem moves
are copied and compared by content before the original bundle is removed.
Same-filesystem moves preserve the original file inodes.

## November 18 legacy serving archive

The cybersecurity GGUF bakeoff archives two legacy on-demand model bundles at
`/mnt/hdd/cyber-campaign-artifacts/legacy-serving-20261118/`:

- `Qwen3.8-27B-Uncensored-W4A16-AutoRound`
- `qwen3.8-27b-cyber-w4a16`

Machine-specific `/etc/fstab` bind mounts preserve the corresponding paths under
`/srv/llama-nvme/hyperqwen-bakeoff/models/`, with
`x-systemd.requires-mounts-for=/mnt/hdd`. The current production staging
script copies only the Pi base into `/srv/llama-nvme/production-models/text`.

The inactive GGUF bundle `qwen3.8-27b-cyber` is preserved under
`/data/development/cyber-gguf-benchmark-20261118/retired-stack/`; its original host
path is a symlink. That link is for host-side access, not a serving alias inside
the models volume. The November 20 storage cleanup archived the benchmark roots under
`/mnt/hdd/archive/cyber-gguf-benchmark-20261118/`: `from-development` holds
source/build artifacts and `from-production-models` holds the former NVMe tree.
The archive retains the OrcaSAQ2 weights, drafters and native runtimes.
The current production staging script omits this retired Orca bundle.
The archive and original hashes remain the provenance sources.

This relocation followed a root-disk-pressure eviction. Kubelet triggers at 5%
free and has a 10% minimum-reclaim target, so recovery requires 15% free space,
followed by its five-minute pressure transition period. Those settings remain
unchanged. Campaign evidence and model hashes are tracked in the private
`0x7067/qwen-cyber-campaign` repository on `codex/gguf-cyber-benchmark`.
