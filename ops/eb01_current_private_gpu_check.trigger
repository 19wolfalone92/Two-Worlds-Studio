inspect: S05 variant1/2; S06 first; S02 retake variant1
no_gpu_jobs: true
read_only: true

monitor_fix: install required six dependency, verify live Kaggle kernel states

forensic_20261113: inspect S06 failure both attempts; logs only, no GPU

forensic: worker eb01-v7-s06-tail29-3a0a1002da34 ERROR under 2 minutes; inspect actual Kaggle logs

monitor_exact_gpu_job: wolf19loner92/eb01-v7-s06-tail29-17f5716049d7 from run 43443472742

read_only_2026_11_13_ltx_s02: wolf19loner92/eb01-s02-ltx2b-real-5ec733d0369c
reason: check whether live real 125-frame I2V LTX CUDA GPU inference has reached COMPLETE or ERROR, no GPU restart.
