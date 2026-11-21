"""Fail closed when board RAM overstates usable CUDA RAM or measurements fail."""
import copy,unittest
from tools.eb01_prepare_echo_production import capacity
from studio.eb01_echo_parallel_profile import PROFILE


def measurement():
    return {'account_alias':'owner','profile':PROFILE,'run_id':'123456789',
        'status':'TWO_SINGLE_CARD_GENERATORS_AND_CONTINUATIONS_TECHNICAL_PASS',
        'host_and_both_device_measurement':{'both_gpus_compute_busy_samples':20,
            'host_available_min_gib':9.0,'per_device':[{'physical_gpu_index':i,'name':'Tesla T4',
            'physical_gib':15.,'memory_used_peak_gib':13.6} for i in range(2)]},
        'workers':[{'technical_pass':True,'physical_gpu_index':i,'native_inference_seconds':700.,
            'probe':{'frames':65},'resources':{'finished':True,'per_device':[
            {'index':0,'name':'Tesla T4','physical_gib':14.5622,'device_memory_used_peak_gib':13.6}]}}
            for i in range(2)]}


class MeasuredCapacityTests(unittest.TestCase):
    def test_production_reserve_uses_cuda_usable_not_nvml_board_total(self):
        report=measurement();cap=capacity(report,'owner')
        self.assertEqual(cap['actual_usable_per_card_budget'][0]['remaining_reserve_gib'],.9622)
        # 13.9 + .75 fits the advertised 15 GiB but not the usable 14.5622.
        report['host_and_both_device_measurement']['per_device'][0]['memory_used_peak_gib']=13.9
        with self.assertRaises(ValueError):capacity(report,'owner')

    def test_missing_nonfinite_and_same_card_observations_never_unlock_two_workers(self):
        for change in ('nan_busy','nan_cuda','unfinished','same_gpu','nan_ram'):
            with self.subTest(change=change):
                report=copy.deepcopy(measurement())
                if change=='nan_busy':report['host_and_both_device_measurement']['both_gpus_compute_busy_samples']=float('nan')
                if change=='nan_cuda':report['workers'][1]['resources']['per_device'][0]['physical_gib']=float('nan')
                if change=='unfinished':report['workers'][1]['resources']['finished']=False
                if change=='same_gpu':report['workers'][1]['physical_gpu_index']=0
                if change=='nan_ram':report['host_and_both_device_measurement']['host_available_min_gib']=float('nan')
                with self.assertRaises(ValueError):capacity(report,'owner')
