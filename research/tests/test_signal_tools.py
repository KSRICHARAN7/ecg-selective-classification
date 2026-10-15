import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from signal_tools import add_noise,quality_features,resample_noise


class SignalChecks(unittest.TestCase):
    def test_injected_power_matches_requested_snr(self):
        t=np.arange(1000)/100;x=np.sin(2*np.pi*t)
        n=np.random.default_rng(42).normal(size=1000)
        for level in [24,12,6,0]:
            mixed,injected=add_noise(x,n,level)
            measured=10*np.log10(np.mean((x-x.mean())**2)/np.mean(injected**2))
            self.assertAlmostEqual(measured,level,places=10)

    def test_flat_waveform_is_flagged_not_infinite(self):
        f=quality_features(np.zeros(1000))
        self.assertEqual(f[2],1)
        self.assertTrue(np.isnan(f[3]))
        self.assertFalse(np.isinf(f).any())

    def test_no_silent_zero_power_noise(self):
        with self.assertRaises(ValueError):add_noise(np.arange(1000),np.zeros(1000),6)

    def test_resample_length_and_amplitude(self):
        t=np.arange(3600)/360
        y=resample_noise(np.sin(2*np.pi*2*t),360)
        self.assertEqual(y.shape,(1000,))
        self.assertLess(np.max(np.abs(y[50:-50]-np.sin(2*np.pi*2*np.arange(1000)[50:-50]/100))),.01)


if __name__=='__main__':unittest.main()
