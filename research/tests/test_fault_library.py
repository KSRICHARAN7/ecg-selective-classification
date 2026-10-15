import hashlib,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fault_library import assignment,DevelopmentFaultLibrary
from signal_tools import add_noise


class FaultLibraryChecks(unittest.TestCase):
    def library(self):
        instance=DevelopmentFaultLibrary.__new__(DevelopmentFaultLibrary)
        noise=np.cos(np.arange(1000)*.071)+.1*np.sin(np.arange(1000)*.013)
        instance.bank={family:np.tile(noise,(2,58,1))for family in ['bw','ma']}
        return instance

    def test_assignment_repeatable_and_severity_independent(self):
        self.assertEqual(assignment('123','bw'),assignment('123','bw'))
        instance=self.library();x=np.sin(np.arange(1000)*.03)
        a=instance.make_pair(x,'123','bw',24)[3];b=instance.make_pair(x,'123','bw',0)[3]
        self.assertEqual(a['channel'],b['channel']);self.assertEqual(a['slot'],b['slot'])

    def test_source_windows_stay_inside_separated_roles(self):
        for identifier in range(200):
            for role,limits in [('train',(0,580)),('ranker',(0,580)),('validation',(600,1180)),('test',(1200,1780))]:
                a=assignment(identifier,'ma',role)
                self.assertGreaterEqual(a['source_start_second'],limits[0]);self.assertLessEqual(a['source_stop_second'],limits[1])
                self.assertEqual(a['source_stop_second']-a['source_start_second'],10)

    def test_withheld_source_rejected(self):
        with self.assertRaises(PermissionError):assignment('123','em')

    def test_reserved_noise_regions_cannot_be_decoded_by_development_library(self):
        for role in ['validation','test']:
            with self.assertRaises(PermissionError):self.library().make_pair(np.arange(1000),'123','bw',6,role)

    def test_pair_exactly_reproducible_and_severity_ordered(self):
        library=self.library();x=np.sin(np.arange(1000)*.03)
        a=library.make_pair(x,'123','ma',6);b=library.make_pair(x,'123','ma',6);strong=library.make_pair(x,'123','ma',0)
        self.assertTrue(np.array_equal(a[1],b[1]));self.assertTrue(np.allclose(a[1]-a[0],a[2],rtol=0,atol=1e-14))
        self.assertAlmostEqual(np.sqrt(np.mean(strong[2]**2))/np.sqrt(np.mean(a[2]**2)),10**(.3))

    def test_positive_noise_gain_cancels_under_rms_scaling(self):
        x=np.sin(np.arange(1000)*.03);noise=np.cos(np.arange(1000)*.071)
        a,_=add_noise(x,noise,6);b,_=add_noise(x,noise*200,6)
        self.assertTrue(np.allclose(a,b,rtol=0,atol=1e-14))

    def test_invalid_severity_and_shape_rejected(self):
        with self.assertRaises(ValueError):self.library().make_pair(np.arange(1000),'123','bw',18)
        with self.assertRaises(ValueError):self.library().make_pair(np.arange(900),'123','bw',6)

    def test_modified_noise_source_rejected_before_decode(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);(root/'outputs').mkdir();(root/'data/pilot/noise').mkdir(parents=True)
            (root/'data/pilot/noise/bw.hea').write_bytes(b'changed')
            (root/'outputs/pilot_download_manifest.json').write_text(json.dumps([{'path':'data/pilot/noise/bw.hea','sha256':hashlib.sha256(b'original').hexdigest()}]))
            with self.assertRaisesRegex(ValueError,'Noise source changed'):DevelopmentFaultLibrary(root)


if __name__=='__main__':unittest.main()
