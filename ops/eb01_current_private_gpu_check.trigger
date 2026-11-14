inspect: S05 variant1/2; S06 first; S02 retake variant1
no_gpu_jobs: true
read_only: true

monitor_fix: install required six dependency, verify live Kaggle kernel states

forensic_20261113: inspect S06 failure both attempts; logs only, no GPU

forensic: worker eb01-v7-s06-tail29-3a0a1002da34 ERROR under 2 minutes; inspect actual Kaggle logs

monitor_exact_gpu_job: wolf19loner92/eb01-v7-s06-tail29-17f5716049d7 from run 43443472742

read_only_2026_11_13_ltx_s02: wolf19loner92/eb01-s02-ltx2b-real-5ec733d0369c
reason: check whether live real 125-frame I2V LTX CUDA GPU inference has reached COMPLETE or ERROR, no GPU restart.

2026-11-13T18:05Z: forensic read-only check after LTX S02 worker ERROR; no GPU launch, inspect Kaggle log.

2026-11-14 NEW changed root LTX config: read-only LIVE kernel wolf19loner92/eb01-ltx2b-pilot33-60202f594b77 from GH run 43579646805, inspect stage and hardware. No private files downloaded, no rerun GPU. 

2026-11-14 latest requested status check after failed GitHub monitoring 43579646805: confirm if already submitted private Kaggle kernel eb01-ltx2b-pilot33-60202f594b77 moved beyond QUEUED, read-only no paid GPU/new kernel launch.
