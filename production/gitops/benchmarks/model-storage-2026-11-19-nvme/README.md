# llama-nvme migration, 2026-11-19

The 2026-11-19 disk-pressure incident (root SSD 97%, kubelet evicted all
workloads) emptied `/srv/llama-nvme` off the root SSD, per
`docs/model-storage.md` ("Keep this workload off the root SSD").

Moved into the library at `/data/models/qwen3.8/local/`:

| Bundle | Size | Files | Checksums |
|---|---|---|---|
| `hyperqwen-bakeoff` | 18 GiB real | 92 | 0 diffs |
| `cyber-gguf-benchmark-20261118` | 26 GiB | 179212 | 0 diffs |
| `cyber-cache` | 1.1 GiB | 7256 | 0 diffs |

Method: `rsync -a` copy, then `rsync -rcn --dry-run` checksum verification
with source/destination file-count equality, all checked before the source
directories were removed. The old paths remain as symlinks:
`/srv/llama-nvme/<name>` → the bundle, so existing scripts and docker
mounts keep working.

Not moved (already on HDD, canonical): the two serving models bind-mounted
from `/mnt/hdd/cyber-campaign-artifacts/legacy-serving-20261118/`
(`Qwen3.8-27B-Uncensored-W4A16-AutoRound`, `qwen3.8-27b-cyber-w4a16`, 16 GiB
each). Their fstab bind mounts (lines 31-32) now resolve through the
symlink and remount into the bundle; they were verified empty on the NVMe
side before the swap, so no copy was hidden underneath them.

`models/qwen3.8-27b-cyber` stays a symlink (via `/data/development`) into
`/mnt/hdd`, preserved verbatim.

`CATALOG.json` was generated 2026-09-15, before these bundles existed; it
was not regenerated.
