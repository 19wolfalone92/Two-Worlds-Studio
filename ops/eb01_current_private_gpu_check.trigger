inspect: S05 variant1/2; S06 first; S02 retake variant1
no_gpu_jobs: true
read_only: true

monitor_fix: install required six dependency, verify live Kaggle kernel states

forensic_20261113: inspect S06 failure both attempts; logs only, no GPU

forensic: worker eb01-v7-s06-tail29-3a0a1002da34 ERROR under 2 minutes; inspect actual Kaggle logs

monitor_exact_gpu_job: wolf19loner92/eb01-v7-s06-tail29-17f5716049d7 from run 43443472742
