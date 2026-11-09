import unittest
from studio.neural_motion import SIGMAS,distilled_scheduler


class OriginalScheduleTests(unittest.TestCase):
    def test_uniform_pipeline_sigmas_cannot_replace_the_distilled_schedule(self):
        class Backend:
            def set_timesteps(self,*,sigmas,device):self.used_sigmas=sigmas;self.device=device
        scheduler=distilled_scheduler(Backend)()
        scheduler.set_timesteps(sigmas=[1,.875,.75,.625,.5,.375,.25,.125],device='cuda:0',mu=1.1)
        self.assertEqual(scheduler.used_sigmas,SIGMAS)
        self.assertEqual(scheduler.device,'cuda:0')
        with self.assertRaises(ValueError):scheduler.set_timesteps(num_inference_steps=1)
