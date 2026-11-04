"""Drawing pipeline regression tests using synthetic images, no private art/GPU."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

from studio import character_art as art
from studio.art_director import load_style, panel_prompt
from studio.kaggle_flux import plan_job, run_gpu


def synthetic(path, size=(800, 1200), *, alpha=False, variant=0):
    im = Image.new("RGBA" if alpha else "RGB", size,
                   (0, 0, 0, 0) if alpha else (40, 45, 52))
    draw = ImageDraw.Draw(im)
    draw.ellipse((size[0]//4, size[1]//8, size[0]*3//4, size[1]*7//8),
                 fill=(180+variant, 100, 90, 255) if alpha else (180+variant, 100, 90))
    draw.rectangle((size[0]*2//5, size[1]//3, size[0]*3//5, size[1]*2//3),
                   fill=(40, 200, 240, 255) if alpha else (40, 200, 240))
    im.save(path)


class CharacterDrawingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.handoff = self.base / "work"
        assets = self.handoff / "assets/identities"
        assets.mkdir(parents=True)
        refs = []
        for i, view in enumerate(("front", "profile", "three_quarter")):
            path = assets / f"{view}.png"
            synthetic(path, variant=i)
            refs.append({"file": "identities/"+path.name, "view": view,
                         "sha256": art.digest(path), "review_status": "unreviewed",
                         "rights": "original", "rights_note": "Synthetic testing artwork.",
                         **({"source_crop_pixels": [160, 400]} if view == "front" else {})})
        self.registry = {"schema": 1, "bodies": {"earth_mira": {"version": "SYNTHETIC_v1", "references": refs}}}
        art.write_json(self.handoff / "story/identity_registry.json", self.registry)
        self.spec = {"schema": 1, "style": "Synthetic illustration for technical tests only.",
                     "bodies": {"earth_mira": {"identity_version": "SYNTHETIC_v1",
                         "description": "Synthetic fictional adult character with a blue jacket.",
                         "anchors": {"wardrobe": "blue jacket, exact same in every drawing"},
                         "avoid": ["obsolete oatmeal fixture costume"]}}}
        self.spec_path = self.base / "appearance.json"
        art.write_json(self.spec_path, self.spec)
        self.root = self.base / "drawing"
        art.create_project(self.handoff, self.spec_path, self.root)
        art.build_jobs(self.root)
        self.source = self.base / "candidate.png"
        synthetic(self.source)

    def register(self, job="earth_mira__front", source=None):
        return art.register_candidate(self.root, job, source or self.source,
                                      generator="synthetic_test", note="Synthetic test; not production art.")

    def review(self, item, decision="accept"):
        return {"decision": decision, "reviewer": "Synthetic human test fixture",
                "criteria": {k: True for k in art.CRITERIA},
                "notes": "Synthetic criteria for testing review bookkeeping only.",
                "image_sha256": item["technical"]["sha256"]}

    def test_import_discloses_native_crop_and_prefers_detailed_reference(self):
        project = art.load_project(self.root)
        refs = project["bodies"]["earth_mira"]["references"]
        self.assertEqual(refs[0]["source_pixels"], [160, 400])
        self.assertTrue(refs[0]["quality_warning"])
        self.assertEqual(refs[0]["review_status"], "unreviewed")
        job = art.get_job(self.root, "earth_mira__front")
        self.assertNotEqual(job["references"][0]["view"], "front")
        self.assertEqual(len(art.build_jobs(self.root)["jobs"]), 9)
        self.assertFalse(project["publication_approved"])

    def test_version_drift_fails_before_project_write(self):
        self.spec["bodies"]["earth_mira"]["identity_version"] = "OTHER"
        art.write_json(self.spec_path, self.spec)
        out = self.base / "bad"
        with self.assertRaisesRegex(ValueError, "version mismatch"):
            art.create_project(self.handoff, self.spec_path, out)
        self.assertFalse(out.exists())

    def test_missing_or_changed_reference_never_becomes_a_text_only_identity(self):
        source = self.handoff / "assets/identities/front.png"
        source.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            art.create_project(self.handoff, self.spec_path, self.base/"bad")
        self.registry["bodies"]["earth_mira"]["references"] = []
        art.write_json(self.handoff/"story/identity_registry.json", self.registry)
        with self.assertRaisesRegex(ValueError, "no actual Work"):
            art.create_project(self.handoff, self.spec_path, self.base/"bad")

    def test_reference_escape_is_rejected(self):
        self.registry["bodies"]["earth_mira"]["references"][0]["file"] = "../../appearance.png"
        art.write_json(self.handoff/"story/identity_registry.json", self.registry)
        with self.assertRaisesRegex(ValueError, "Invalid artwork path"):
            art.create_project(self.handoff, self.spec_path, self.base/"bad")

    def test_changed_imported_reference_blocks_next_generation_import(self):
        project = art.load_project(self.root)
        path = self.root/project["bodies"]["earth_mira"]["references"][0]["file"]
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "reference bytes changed"):
            self.register()

    def test_changed_job_is_not_silently_reused(self):
        path = self.root/"jobs.json"
        plan = art.read_json(path)
        plan["jobs"][0]["prompt"] += " Changed character."
        art.write_json(path, plan)
        with self.assertRaisesRegex(ValueError, "Drawing job"):
            self.register()

    def test_import_is_idempotent_and_preserves_review_and_other_versions(self):
        first = self.register()
        reviewed = art.record_review(self.root, first["job_id"], first["id"], self.review(first))
        self.assertEqual(self.register(), reviewed)
        synthetic(self.source, variant=10)
        other = self.register()
        self.assertNotEqual(first["id"], other["id"])
        self.assertEqual(len(art.candidates(self.root)), 2)
        self.assertIsNone(other["human_review"])
        self.assertFalse(other["identity_verified"])

    def test_opaque_rgba_is_not_counted_as_real_transparency(self):
        im = Image.open(self.source).convert("RGBA")
        im.save(self.source)
        candidate = self.register("earth_mira__rig_parts")
        self.assertTrue(candidate["technical"]["alpha"])
        self.assertFalse(candidate["technical"]["real_transparency"])
        self.assertFalse(candidate["technical"]["pass"])
        with self.assertRaisesRegex(ValueError, "Resolve technical"):
            art.record_review(self.root, candidate["job_id"], candidate["id"], self.review(candidate))

    def test_true_alpha_import_does_not_claim_rig_readiness(self):
        synthetic(self.source, alpha=True)
        candidate = self.register("earth_mira__rig_parts")
        self.assertTrue(candidate["technical"]["real_transparency"])
        self.assertTrue(candidate["technical"]["pass"])
        self.assertFalse(candidate["rig_ready"])

    def test_review_requires_actual_pixels_and_explicit_visual_findings(self):
        item = self.register()
        review = self.review(item)
        review["image_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "actual candidate image"):
            art.record_review(self.root, item["job_id"], item["id"], review)
        review = self.review(item)
        review["criteria"]["eyes"] = False
        with self.assertRaisesRegex(ValueError, "Resolve technical and visual"):
            art.record_review(self.root, item["job_id"], item["id"], review)
        self.assertIsNone(art.candidates(self.root)[0][1]["human_review"])

    def test_revision_preserves_identity_and_is_importable_after_replanning(self):
        item = self.register()
        review = self.review(item, "revise")
        review["criteria"]["pose"] = False
        review["notes"] = "Draw the back of the head; preserve the blue jacket."
        art.record_review(self.root, item["job_id"], item["id"], review)
        job = art.revision_job(self.root, item["job_id"], item["id"])
        self.assertEqual(job["edit_target"]["sha256"], item["technical"]["sha256"])
        self.assertIn(review["notes"], job["prompt"])
        art.build_jobs(self.root)
        self.assertEqual(art.get_job(self.root, job["id"]), job)
        revised = self.register(job["id"])
        self.assertIsNone(revised["human_review"])

    def test_gallery_is_offline_and_escapes_embedded_user_text(self):
        art.register_candidate(self.root, "earth_mira__front", self.source,
            generator="synthetic </script><script>bad()</script>", note="Test provenance.")
        path = art.export_gallery(self.root, self.base/"review.html")
        text = path.read_text()
        self.assertNotIn("</script><script>bad()", text)
        self.assertIn("data:image/png;base64,", text)
        self.assertIn("character_review_drafts.json", text)
        self.assertNotIn('<script src=', text)

    def test_review_batch_validates_every_item_before_writing(self):
        one = self.register()
        two = self.register("earth_mira__back")
        reviews = [{**self.review(c), "job_id": c["job_id"], "candidate_id": c["id"]} for c in (one,two)]
        reviews[1]["image_sha256"] = "bad"
        bundle = {"kind": "human_review_drafts", "reviews": reviews}
        with self.assertRaises(ValueError):
            art.import_reviews(self.root, bundle)
        self.assertTrue(all(c["human_review"] is None for _, c in art.candidates(self.root)))
        reviews[1]["image_sha256"] = two["technical"]["sha256"]
        self.assertEqual(art.import_reviews(self.root,bundle)["reviews_recorded"],2)
        self.assertTrue(all(not c["publication_approved"] for _, c in art.candidates(self.root)))

    def test_scene_refers_to_physical_body_and_preserves_authored_direction(self):
        scene = {"id":"S10", "shot":"Actor is frightened inside the existing armour.",
                 "subjects":["earth_mira"],"continuity":{"body":"earth_mira","occupant":"Lyra"}}
        ep = {"id":"E01", "scenes":[scene]}
        saved = copy.deepcopy(ep)
        job = art.scene_job(self.root,ep,"S10")
        art.save_job(self.root,job)
        self.assertEqual(ep,saved)
        self.assertEqual(job["body_ids"],["earth_mira"])
        self.assertIn(scene["shot"],job["prompt"])
        self.assertIn("blue jacket",job["prompt"])
        self.assertEqual(art.get_job(self.root,job["id"]),job)
        ep["scenes"][0]["subjects"]=["noctaris_lyra"]
        with self.assertRaisesRegex(ValueError,"Every depicted physical body"):
            art.scene_job(self.root,ep,"S10")

    def test_existing_art_director_uses_private_current_appearance(self):
        scene = {"id":"S01", "shot":"Synthetic camera direction for character.", "subjects":["earth_mira"]}
        item = panel_prompt(scene,load_style(),"E01",character_project=self.root)
        self.assertIn("blue jacket",item["prompt"])
        self.assertNotIn("oatmeal knitted sweater",item["prompt"])
        self.assertTrue(Path(item["references"]["earth_mira"]["file"]).is_file())
        self.assertFalse(item["consistency_verified"])

    def test_explicit_rear_staging_is_part_of_the_job_not_a_screenplay_edit(self):
        ep={"id":"E01","scenes":[{"id":"S03","subjects":["earth_mira"],
              "shot":"The character approaches the existing mirror."}]}
        saved=copy.deepcopy(ep)
        staging={"earth_mira":{"view":"back","facing":"mirror","plane":"foreground"}}
        job=art.scene_job(self.root,ep,"S03",staging=staging)
        self.assertIn("back view; faces mirror",job["prompt"])
        self.assertEqual(ep,saved)
        with self.assertRaisesRegex(ValueError,"physical bodies"):
            art.scene_job(self.root,ep,"S03",staging={"unknown":staging["earth_mira"]})

    def test_unconnected_comfy_reference_does_not_count_as_conditioning(self):
        graph={"1":{"class_type":"LoadImage","inputs":{"image":"__REFERENCE_IMAGE__"}},
               "2":{"class_type":"CheckpointLoaderSimple","inputs":{}},
               "3":{"class_type":"KSampler","inputs":{"model":["2",0]}},
               "8":{"class_type":"VAEDecode","inputs":{"samples":["3",0]}},
               "9":{"class_type":"SaveImage","inputs":{"images":["8",0]}}}
        self.assertFalse(art.reference_conditioned(graph))
        graph["4"]={"class_type":"IPAdapterAdvanced","inputs":{"image":["1",0],"model":["2",0]}}
        graph["3"]["inputs"]["model"]=["4",0]
        self.assertTrue(art.reference_conditioned(graph))
        graph["9"]["inputs"]["images"]=["1",0]
        self.assertFalse(art.reference_conditioned(graph))

    def test_gpu_adapter_uses_current_spec_and_changed_ref_fails_before_cuda_import(self):
        args=SimpleNamespace(character_project=self.root,job="earth_mira__back")
        job=plan_job(args)
        self.assertIn("blue jacket",job["prompt"])
        Path(job["reference"]["path"]).write_bytes(self.source.read_bytes())
        # Change bytes rather than availability of CUDA, so identity validation
        # must reject this before any torch/diffusers import or model download.
        synthetic(Path(job["reference"]["path"]),variant=25)
        with self.assertRaisesRegex(ValueError,"reference changed"):
            run_gpu(job,self.base/"unused.png")

    def test_status_reports_missing_drawings_without_claiming_model_skill(self):
        self.register()
        status=art.project_status(self.root)
        self.assertEqual(status["candidate_drawings"],1)
        self.assertEqual(status["human_accepted_drawings"],0)
        self.assertIn("blink",status["coverage"]["earth_mira"]["missing_views"])
        self.assertFalse(status["rig_ready"])

    def test_assistant_can_report_a_defect_without_fabricating_human_approval(self):
        item=self.register()
        marked=art.request_revision(self.root,item["job_id"],item["id"],
                  reported_by="assistant visual inspection",notes="Residual coloured halo outside silhouette.")
        self.assertEqual(marked["review_status"],"needs_revision")
        self.assertIsNone(marked["human_review"])
        self.assertFalse(marked["publication_approved"])
        job=art.revision_job(self.root,item["job_id"],item["id"])
        self.assertIn("coloured halo",job["prompt"])


if __name__ == "__main__":
    unittest.main()
