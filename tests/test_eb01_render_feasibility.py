"""Real incident-based GPU feasibility: no 125f T4 retry without true pilot."""
import unittest

from studio.eb01_render_feasibility import (
    RenderRequest,check_candidate,MODEL_WAN,MODEL_LTX_LEGACY,MODEL_LTX_NEW)

def q(model=MODEL_WAN, vram=14.56,width=320,height=576,frames=125,
      steps=21,mode="production",license_confirmed=True,paid_service=False):
    return RenderRequest(model=model,gpu_vram_gib=vram,width=width,
       height=height,frames=frames,steps=steps,mode=mode,
       license_confirmed=license_confirmed,paid_service=paid_service)

def proof(model=MODEL_WAN,seconds=75):
    return {"model":model,"source_neural_inference_performed":True,
            "clip_sha256":"a"*64,"human_artistic_review":"PASS",
            "rgb_artifact_gate":"PASS","actual_gpu_seconds":seconds,
            "actual_neural_frames":33}


class ResourceAwareCinematicTests(unittest.TestCase):
    def test_actual_t4_wan_lowres_125f_production_blocked(self):
        report=check_candidate(q())
        self.assertEqual(report["status"],"BLOCKED")
        self.assertIn("WAN_720P_OFFLOAD_REQUIRES_AT_LEAST_24GIB_REVIEWED_GPU",report["reasons"])
        self.assertIn("WAN_PREVIOUS_LOW_RES_320X576_WAS_ARTISTICALLY_REJECTED",report["reasons"])

    def test_actual_legacy_ltx_full_t4_timeout_not_repeated(self):
        result=check_candidate(q(model=MODEL_LTX_LEGACY))
        self.assertEqual(result["status"],"BLOCKED")
        self.assertIn("REPEATING_KNOWN_TIMED_OUT_125F_T4_JOB_FORBIDDEN",result["reasons"])

    def test_t4_33frame_diagnostic_is_pilot_only_not_quality(self):
        result=check_candidate(q(model=MODEL_LTX_LEGACY,frames=33,mode="pilot"))
        self.assertEqual(result["status"],"PILOT_ALLOWED_NOT_QUALITY_APPROVED")
        self.assertFalse(result["production_approved"])
        self.assertFalse(result["quality_is_proven"])
        self.assertFalse(result["gpu_launched"])

    def test_bad_legacy_ltx_pilot_size_rejected(self):
        result=check_candidate(q(model=MODEL_LTX_LEGACY,frames=125,mode="pilot"))
        self.assertIn("LEGACY_LTX_UNBENCHMARKED_GPU_PILOT_MAX_33_FRAMES",result["reasons"])

    def test_successful_pilot_not_automatically_approved_cinema(self):
        request=q(vram=24,width=704,height=1280,frames=125,steps=40)
        result=check_candidate(request,pilot=proof())
        self.assertEqual(result["status"],"PRECHECK_PASSED_FINAL_MOVIE_STILL_NEEDS_ARTISTIC_QA")
        self.assertTrue(result["production_approved"])
        self.assertFalse(result["quality_is_proven"])

    def test_unreviewed_pilot_is_not_a_production_pass(self):
        sample=proof()
        sample["human_artistic_review"]="NOT_REVIEWED"
        result=check_candidate(q(vram=24,width=704,height=1280),pilot=sample)
        self.assertIn("PILOT_NOT_HUMAN_APPROVED_FOR_IDENTITY_AND_MOTION",result["reasons"])

    def test_measured_too_slow_gpu_gets_stopped_before_long_job(self):
        result=check_candidate(q(vram=24,width=704,height=1280),
                               pilot=proof(seconds=1000))
        self.assertIn("ESTIMATED_JOB_EXCEEDS_GPU_TIME_BUDGET_NO_LONG_RETRY",result["reasons"])

    def test_paid_service_never_sneaks_into_gpu_backend(self):
        result=check_candidate(q(vram=48,width=704,height=1280,paid_service=True),
                               pilot=proof())
        self.assertIn("PAID_SERVICE_NOT_AUTHORIZED",result["reasons"])

    def test_lt_x_new_big_model_blocked_on_t4(self):
        result=check_candidate(q(model=MODEL_LTX_NEW))
        self.assertIn("LTX25_PRODUCTION_HARDWARE_BELOW_PUBLISHED_DESKTOP_FLOOR",result["reasons"])

    def test_invalid_gpu_and_dimensions_are_fail_closed_not_type_errors(self):
        result=check_candidate(q(vram=float("nan"),width="bad",height=None,frames="bad"))
        self.assertEqual(result["status"],"BLOCKED")
        self.assertIn("INVALID_GPU_CAPACITY",result["reasons"])
        self.assertIn("INVALID_JOB_GEOMETRY_OR_TIMEOUT",result["reasons"])

    def test_wan_5b_16pixel_only_geometry_is_rejected(self):
        result=check_candidate(q(vram=24,width=592,height=1008,frames=17,mode="pilot"))
        self.assertIn("WAN_5B_REQUIRES_32PX_SPATIAL_AND_4N_PLUS_1_TEMPORAL_ALIGNMENT",
                      result["reasons"])

    def test_no_pilot_cannot_count_as_full_movie(self):
        result=check_candidate(q(vram=24,width=704,height=1280))
        self.assertIn("NO_PREVIOUS_VERIFIED_TRUE_GPU_SHORT_PILOT",result["reasons"])

if __name__=="__main__":
    unittest.main()
