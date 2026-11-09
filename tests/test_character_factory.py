"""From-scratch dependency/execution tests with labelled synthetic pixels only."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image,ImageDraw

from studio import character_factory as f
from studio.character_art import write_json,read_json,fingerprint,load_project
from studio.kaggle_flux import run_gpu,configure_gpu_memory
from studio.kaggle_api import build_job


def spec():
    return {"schema":1,"style":"Synthetic painted technical fixtures; not actual production artwork.",
            "bodies":{
                "actor_a":{"identity_version":"SYNTHETIC_A_v1","identity_group":"twins",
                           "description":"Synthetic adult character in a blue technical fixture jacket.",
                           "anchors":{"costume":"blue fixture jacket"}},
                "actor_b":{"identity_version":"SYNTHETIC_B_v1","identity_group":"twins",
                           "description":"The same fictional adult face with a distinct dark armour fixture.",
                           "anchors":{"costume":"dark fixture armour"}}}}


def pixels(path,size=(768,1152),offset=0):
    image=Image.new("RGB",size,(30,40,60))
    draw=ImageDraw.Draw(image)
    draw.rectangle((100,100,size[0]-100,size[1]-100),fill=(160+offset,100,90))
    draw.rectangle((250,250,size[0]-250,size[1]-250),fill=(20,160,210))
    image.save(path,"PNG")


def evidence(job):
    return {"prompt_sha256":fingerprint({"prompt":job["prompt"]}),
            "reference_sha256":[r["sha256"] for r in job["references"]],
            "test_fixture":True}


class FactoryTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.base=Path(temp.name);self.root=self.base/"factory"
        self.spec=spec();self.source=self.base/"design.json"
        write_json(self.source,self.spec);f.create_design(self.source,self.root)
        self.image=self.base/"synthetic.png";pixels(self.image)

    def register(self,job_id):
        job=f.export_job(self.root,job_id)
        return f.register_result(self.root,job,self.image,backend="SYNTHETIC_TEST_BACKEND",execution=evidence(job))

    def fake_backend(self,job,output):
        pixels(output)
        return {"backend":"SYNTHETIC_TEST_BACKEND","execution":evidence(job)}

    def test_initial_job_is_text_only_and_does_not_pretend_to_have_generated_art(self):
        jobs=f.ready_jobs(self.root)
        self.assertEqual([j["id"] for j in jobs],["actor_a__front"])
        self.assertEqual(jobs[0]["references"],[])
        self.assertEqual(jobs[0]["stage"],"text_only_bootstrap")
        self.assertEqual(f.status(self.root)["actual_drawings"],0)
        self.assertEqual(list(self.root.rglob("*.png")),[])

    def test_old_art_cannot_be_smuggled_into_the_bootstrap_spec(self):
        self.spec["bodies"]["actor_a"]["references"]=["old_work.png"]
        write_json(self.source,self.spec)
        with self.assertRaisesRegex(ValueError,"must not contain existing"):
            f.create_design(self.source,self.base/"other")
        self.assertFalse((self.base/"other").exists())

    def test_secondary_body_waits_for_the_new_seed_and_keeps_its_own_costume(self):
        with self.assertRaisesRegex(ValueError,"blocked waiting"):
            f.export_job(self.root,"actor_b__front")
        first=self.register("actor_a__front")
        second=f.export_job(self.root,"actor_b__front")
        self.assertEqual(second["references"][0]["sha256"],first["sha256"])
        self.assertEqual(second["stage"],"shared_face_new_body_design")
        self.assertIn("dark fixture armour",second["prompt"])
        self.assertIn("Do not copy the other body's outfit",second["prompt"])

    def test_each_body_uses_its_own_generated_front_for_later_views(self):
        self.register("actor_a__front")
        pixels(self.image,offset=10)
        second=self.register("actor_b__front")
        variant=f.export_job(self.root,"actor_b__profile")
        self.assertEqual(variant["references"][0]["body_id"],"actor_b")
        self.assertEqual(variant["references"][0]["sha256"],second["sha256"])

    def test_job_or_actual_input_fingerprints_must_match(self):
        job=f.export_job(self.root,"actor_a__front")
        bad=copy.deepcopy(job);bad["prompt"]+=" Changed source."
        with self.assertRaisesRegex(ValueError,"differ from"):
            f.register_result(self.root,bad,self.image,backend="test",execution=evidence(bad))
        ev=evidence(job);ev["reference_sha256"]=["unrelated"]
        with self.assertRaisesRegex(ValueError,"consumed references"):
            f.register_result(self.root,job,self.image,backend="test",execution=ev)
        self.assertEqual(f.status(self.root)["actual_drawings"],0)

    def test_changed_generated_seed_blocks_all_dependent_jobs(self):
        first=self.register("actor_a__front")
        pixels(self.root/first["file"],offset=15)
        with self.assertRaisesRegex(ValueError,"artwork changed"):
            f.ready_jobs(self.root)

    def test_ledger_cannot_silently_reassign_generated_pixels_to_another_body(self):
        self.register("actor_a__front")
        path=self.root/'ledger.json';ledger=read_json(path)
        ledger['results'][0]['body_id']='actor_b'
        write_json(path,ledger)
        with self.assertRaisesRegex(ValueError,'different physical body'):
            f.ready_jobs(self.root)

    def test_receipt_prompt_drift_blocks_recovery(self):
        self.register('actor_a__front')
        path=self.root/'receipts/actor_a__front.json';receipt=read_json(path)
        receipt['job']['prompt']+=' Unrecorded identity change.'
        write_json(path,receipt)
        with self.assertRaisesRegex(ValueError,'execution receipt disagree'):
            f.ready_jobs(self.root)

    def test_register_is_idempotent_and_new_pixels_do_not_overwrite_the_old_design(self):
        job=f.export_job(self.root,"actor_a__front")
        first=f.register_result(self.root,job,self.image,backend="test",execution=evidence(job))
        self.assertEqual(f.register_result(self.root,job,self.image,backend="test",execution=evidence(job)),first)
        pixels(self.image,offset=5)
        with self.assertRaises(FileExistsError):
            f.register_result(self.root,job,self.image,backend="test",execution=evidence(job))

    def test_resume_does_not_regenerate_completed_bodies(self):
        first=f.run_batch(self.root,self.fake_backend,max_images=2,views=("front","three_quarter","profile"))
        self.assertEqual(first["generated_this_run"],2)
        original=[r["sha256"] for r in f.load_results(self.root)]
        second=f.run_batch(self.root,self.fake_backend,max_images=4,views=("front","three_quarter","profile"))
        self.assertEqual(second["generated_this_run"],4)
        self.assertEqual([r["sha256"] for r in f.load_results(self.root)][:2],original)
        self.assertEqual(f.run_batch(self.root,self.fake_backend,max_images=2,views=("front","three_quarter","profile"))["generated_this_run"],0)

    def test_a_backend_success_message_without_pixels_is_not_completion(self):
        def empty(job,output):return {"backend":"fake","execution":evidence(job)}
        with self.assertRaisesRegex(RuntimeError,"actual drawing"):
            f.run_batch(self.root,empty,max_images=1)
        self.assertEqual(f.status(self.root)["actual_drawings"],0)

    def test_blank_or_broken_png_does_not_complete_the_job(self):
        job=f.export_job(self.root,"actor_a__front")
        self.image.write_bytes(b"not PNG")
        with self.assertRaises(ValueError):
            f.register_result(self.root,job,self.image,backend="test",execution=evidence(job))
        Image.new("RGB",(768,1152),(20,20,20)).save(self.image)
        with self.assertRaisesRegex(ValueError,"almost blank"):
            f.register_result(self.root,job,self.image,backend="test",execution=evidence(job))

    def test_body_order_is_explicit_even_when_json_keys_are_reordered(self):
        p=self.root/"design.json";d=read_json(p)
        d["spec"]["bodies"]={k:d["spec"]["bodies"][k] for k in reversed(d["spec"]["bodies"])}
        write_json(p,d)
        self.assertEqual(f.ready_jobs(self.root)[0]["body_id"],"actor_a")

    def test_different_identity_groups_bootstrap_independently(self):
        self.spec["bodies"]["actor_b"]["identity_group"]="other_face"
        write_json(self.source,self.spec);root=self.base/"independent"
        f.create_design(self.source,root)
        self.assertEqual(len(f.ready_jobs(root)),2)
        self.assertTrue(all(not j["references"] for j in f.ready_jobs(root)))

    def test_export_reuses_actual_generated_pixels_without_approving_them(self):
        f.run_batch(self.root,self.fake_backend,max_images=6,views=("front","three_quarter","profile"))
        target=self.base/"art_project"
        f.export_art_project(self.root,target)
        project=load_project(target)
        for body in project["bodies"].values():
            self.assertEqual({r["view"] for r in body["references"]},{"front","three_quarter","profile"})
            self.assertTrue(all(r["review_status"]=="unreviewed" for r in body["references"]))
        self.assertFalse(project["publication_approved"])

    def test_one_gpu_session_is_shared_by_all_drawings(self):
        calls=[];sessions=[]
        class StubSession:
            def __init__(self):sessions.append(self)
            def render(self,job,output,**kwargs):
                calls.append(job)
                pixels(output,size=(kwargs["width"],kwargs["height"]))
                return {"model":"SYNTHETIC_MODEL","gpu_name":"NOT_A_GPU_TEST","precision":"stub"}
        with patch('studio.kaggle_flux.GPUImageSession',StubSession):
            result=f.run_batch(self.root,f.FluxBackend(self.root),max_images=6,
                               views=("front","three_quarter","profile"))
        self.assertEqual(len(sessions),1)
        self.assertEqual(len(calls),6)
        self.assertIsNone(calls[0]["reference"])
        self.assertTrue(all(c["reference"] for c in calls[1:]))
        self.assertEqual(result["actual_drawings"],6)

    def test_existing_output_is_rejected_before_loading_a_model(self):
        with patch('studio.kaggle_flux.GPUImageSession') as session:
            with self.assertRaises(FileExistsError):
                run_gpu({"prompt":"A long enough synthetic drawing request","seed":1,"reference":None},self.image)
            session.assert_not_called()

    def test_small_gpu_uses_lower_memory_offload_without_resizing_the_drawing(self):
        from types import SimpleNamespace
        calls=[]
        pipe=SimpleNamespace(vae=SimpleNamespace(enable_tiling=lambda:calls.append('tiling'),
                           enable_slicing=lambda:calls.append('slicing')),
                           enable_sequential_cpu_offload=lambda:calls.append('sequential'),
                           enable_model_cpu_offload=lambda:calls.append('model'))
        self.assertEqual(configure_gpu_memory(pipe,15.0),'sequential_cpu_offload')
        self.assertEqual(calls,['tiling','slicing','sequential'])
        calls.clear()
        self.assertEqual(configure_gpu_memory(pipe,40.0),'model_cpu_offload')
        self.assertEqual(calls,['tiling','slicing','model'])

    def test_private_kaggle_job_contains_text_spec_and_factory_but_no_input_art(self):
        out=self.base/"kernel"
        result=build_job("examplekaggle","factory-test-1",mode="design",output_dir=out,
                         design_spec=self.spec,max_images=6)
        source=(out/"kernel_main.py").read_text()
        compile(source,"kernel_main.py","exec")
        self.assertFalse(result["credential_included"])
        self.assertTrue(result['kernel'].startswith('examplekaggle/two-worlds-factory-'))
        self.assertNotIn("KAGGLE_API_TOKEN",source)
        self.assertEqual(list(out.glob("*.png")),[])
        with self.assertRaisesRegex(ValueError,"explicit text-only"):
            build_job("examplekaggle","factory-test-2",mode="design",output_dir=self.base/"bad")


if __name__=="__main__":unittest.main()
