"""Private CPU-only E01 7-reference preview; appended after storyboard utilities."""
from __future__ import annotations
import json
import traceback
from pathlib import Path

def main():
    receipt = Path("/kaggle/working/EB01_V7_FRAMES_RECEIPT.json")
    try:
        folder = Path("/kaggle/temp/eb01-v7-review-private")
        lock = gather_private(Path("/kaggle/input"), folder)
        sheet = Path("/kaggle/working/EB01_V7_ALL_SEVEN_REFERENCES_REVIEW_ONLY.jpg")
        contact_sheet(folder, sheet)
        report = {
            "schema": 7,
            "status": "PRIVATE_SEVEN_SHOT_REFERENCE_REVIEW_PENDING",
            "canon_sources": lock["source_sha256"],
            "frames": {k:v["reference_sha256"] for k,v in lock["scenes"].items()},
            "voices": lock["voice_sha256"],
            "human_approved": False,
            "video_generated": False,
            "gpu_executed": False,
            "published": False,
        }
        receipt.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print("SEVEN_REAL_CANON_REFERENCE_CROPS_CREATED_CPU_ONLY", flush=True)
    except BaseException as e:
        receipt.write_text(json.dumps({
            "status": "FAILED", "reason": str(e)[:250],
            "gpu_executed": False, "published": False}, indent=2))
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
