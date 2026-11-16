"""Real incident-based GPU feasibility: no 125f T4 retry without true pilot."""
import unittest

from studio.eb01_render_feasibility import (
    RenderRequest,check_candidate,check_native_detail_pilot,
    MODEL_WAN,MODEL_LTX_LEGACY,MODEL_LTX_NEW)

def q(model=MODEL_WAN, vram=14.56,width=320,height=576,frames=125,
      steps=21,mode="production",license_confirmed=True,paid_service=False):
    if model == MODEL_LTX_LEGACY and steps == 21:
        steps = 8
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

    def test_distilled_actual_eight_steps_allowed_but_misleading_twenty_blocked(self):
        eight=check_candidate(q(model=MODEL_LTX_LEGACY,frames=33,mode="pilot",steps=8))
        self.assertEqual(eight["status"],"PILOT_ALLOWED_NOT_QUALITY_APPROVED")
        wrong=check_candidate(q(model=MODEL_LTX_LEGACY,frames=33,mode="pilot",steps=20))
        self.assertIn("LTX_2B_DISTILLED_REQUIRES_ACTUAL_8_STEPS",wrong["reasons"])
        self.assertEqual(wrong["status"],"BLOCKED")
        bad=check_candidate(q(model=MODEL_WAN,frames=33,mode="pilot",steps=8))
        self.assertIn("NON_DISTILLED_MODEL_REQUIRES_MINIMUM_10_STEPS",bad["reasons"])

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

    def test_boolean_gpu_capacity_cannot_bypass_verified_numerical_hardware_gate(self):
        for fake in (True,False):
            report=check_candidate(q(vram=fake,mode="pilot",frames=33))
            self.assertIn("INVALID_GPU_CAPACITY",report["reasons"])
            self.assertEqual(report["status"],"BLOCKED")

    def test_wan_5b_16pixel_only_geometry_is_rejected(self):
        result=check_candidate(q(vram=24,width=592,height=1008,frames=17,mode="pilot"))
        self.assertIn("WAN_5B_REQUIRES_32PX_SPATIAL_AND_4N_PLUS_1_TEMPORAL_ALIGNMENT",
                      result["reasons"])

    def test_no_pilot_cannot_count_as_full_movie(self):
        result=check_candidate(q(vram=24,width=704,height=1280))
        self.assertIn("NO_PREVIOUS_VERIFIED_TRUE_GPU_SHORT_PILOT",result["reasons"])

if __name__=="__main__":
    unittest.main()

class NativeDetailPilotFeasibilityTests(unittest.TestCase):
    def request(self, model=MODEL_LTX_LEGACY, vram=14.56, width=576,
                height=1024, frames=49, steps=8, **kwargs):
        return RenderRequest(
            model=model, gpu_vram_gib=vram, width=width, height=height,
            frames=frames, steps=steps, mode="pilot",
            license_confirmed=True, paid_service=False, **kwargs)

    def test_free_t4_cannot_run_unbenchmarked_native_576x1024_ltx49_pilot(self):
        result=check_native_detail_pilot(self.request())
        self.assertEqual(result["status"],"BLOCKED")
        self.assertIn("NATIVE_DETAIL_PILOT_REQUIRES_24GIB_REVIEWED_GPU",result["reasons"])
        self.assertIn("LEGACY_LTX_UNBENCHMARKED_GPU_PILOT_MAX_33_FRAMES",result["reasons"])
        self.assertFalse(result["gpu_launched"])
        self.assertFalse(result["paid_service_used"])

    def test_free_t4_cannot_run_native_detail_wan49_pilot_on_unreviewed_profile(self):
        result=check_native_detail_pilot(self.request(
            model=MODEL_WAN,steps=20))
        self.assertEqual(result["status"],"BLOCKED")
        self.assertIn("NATIVE_DETAIL_PILOT_REQUIRES_24GIB_REVIEWED_GPU",result["reasons"])
        self.assertFalse(result["gpu_launched"])

    def test_free_t4_cannot_run_ltx25_22b_native_detail_pilot(self):
        result=check_native_detail_pilot(self.request(
            model=MODEL_LTX_NEW,steps=20))
        self.assertIn("NATIVE_DETAIL_PILOT_REQUIRES_24GIB_REVIEWED_GPU",result["reasons"])
        self.assertIn("LTX25_PRODUCTION_HARDWARE_BELOW_PUBLISHED_DESKTOP_FLOOR",result["reasons"])

    def test_native_detail_pilot_requires_portrait_geometry_and_24fps(self):
        result=check_native_detail_pilot(self.request(
            model=MODEL_WAN,steps=20,width=704,height=1280,frames=47,fps=30))
        self.assertIn("NATIVE_DETAIL_PILOT_REQUIRES_24GIB_REVIEWED_GPU",result["reasons"])
        self.assertIn("NATIVE_DETAIL_PILOT_MUST_DECODE_48_TO_96_FRAMES",result["reasons"])
        self.assertIn("NATIVE_DETAIL_PILOT_REQUIRES_24_FPS",result["reasons"])

    def test_reviewed_24gib_wan49_profile_is_only_precheck_not_quality_approval(self):
        result=check_native_detail_pilot(self.request(
            model=MODEL_WAN,vram=24.0,steps=20))
        self.assertEqual(result["status"],"PILOT_PRECHECK_PASS_NOT_QUALITY_APPROVED")
        self.assertFalse(result["gpu_launched"])
        self.assertFalse(result["quality_is_proven"])
        self.assertFalse(result["published"])

    def test_bad_aspect_ratio_and_unaligned_native_size_are_rejected(self):
        result=check_native_detail_pilot(self.request(
            model=MODEL_WAN,vram=24.0,width=600,height=1024,steps=20))
        self.assertIn("NATIVE_DETAIL_PILOT_MUST_REMAIN_9_16_PORTRAIT",result["reasons"])
        self.assertIn("NATIVE_DETAIL_PILOT_REQUIRES_32PX_SPATIAL_ALIGNMENT",result["reasons"])

