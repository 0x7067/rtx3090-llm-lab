| Arm | Status | Quality (score / full / n) | Short prose tok/s | Short code tok/s | 64K decode | 64K prefill | 120K decode | 120K prefill | 4-caller aggregate | Peak VRAM MiB | Min host avail GiB | Container peak GiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llamacpp_q4ks | ok | 53.0 / 53 / 54 | 57.83 | 67.16 | 51.54 | 838.6 | 40.71 | 663.8 | 48.32 | 22937 | 29.9 | 16.8 |
| llamacpp_gsq | ok | 54.0 / 54 / 54 | 58.17 | 66.75 | 48.57 | 827.8 | 38.74 | 656.3 | 50.12 | 22191 | 29.3 | 12.2 |
| vllm_hyperqwen | ok | 53.0 / 53 / 54 | 83.07 | 96.85 | 83.59 | 744.5 | 68.13 | 601.9 | 269.14 | 23704 | 33.2 | 13.1 |
| sglang_exl3 | ok | 54.0 / 54 / 54 | 80.45 | 96.72 | 73.36 | 1076.6 | 64.52 | 804.2 | 81.72 | 22564 | 30.0 | 23.7 |
| strata_flashnext_iq3xxs | ok | 50.0 / 50 / 54 | 57.11 | 70.55 | 63.24 | 1750.3 | 66.39 | 1724.4 | 61.2 | 23869 | 8.4 | 31.1 |
