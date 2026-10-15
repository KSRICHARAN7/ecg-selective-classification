import unittest,sys,tempfile,shutil
from pathlib import Path
import numpy as np
from scipy.io import savemat
from scipy.signal import resample_poly
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from external_ecg import ROOT,dictionary,header_metadata,physical_lead_II
class ExternalECGTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)/'sample';self.raw=np.tile(np.round(200*np.sin(np.arange(5000)*.02)),(12,1)).astype('int16');self.raw[2]*=2
  self.names=['III','I','II','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
 def tearDown(self):self.temp.cleanup()
 def write(self,age='40',dx='164889003',gain=1000,unit='mV',bad_checksum=False):
  header=['sample 12 500 5000']
  for i,name in enumerate(self.names):
   checksum=int(self.raw[i].astype('int64').sum()%65536)+(1 if bad_checksum and i==2 else 0)
   header.append(f'sample.mat 16+24 {gain}(0)/{unit} 16 0 {self.raw[i,0]} {checksum} 0 {name}')
  header+=['#Age: '+age,'#Sex: Male','#Dx: '+dx];self.base.with_suffix('.hea').write_text('\n'.join(header)+'\n');savemat(self.base.with_suffix('.mat'),{'val':self.raw})
 def test_fibrillation_is_not_flutter_acronym_AF(self):
  self.write(dx='164890007');m=header_metadata(self.base.with_suffix('.hea'));self.assertTrue(m['eligible']);self.assertEqual(m['reference_AFIB'],0);self.assertEqual(m['rhythm_acronyms'],['AF'])
  self.write();self.assertEqual(header_metadata(self.base.with_suffix('.hea'))['reference_AFIB'],1)
 def test_minor_and_unknown_age_excluded(self):
  for age in ['17','NaN','unknown']:
   self.write(age=age);self.assertFalse(header_metadata(self.base.with_suffix('.hea'))['eligible'])
 def test_nonrhythm_only_excluded(self):
  self.write(dx='59118001');self.assertIn('no_explicit_eligible_rhythm_code',header_metadata(self.base.with_suffix('.hea'))['exclusion_reasons'])
 def test_named_lead_and_physical_gain(self):
  self.write();wave,a=physical_lead_II(self.base);expected=resample_poly(self.raw[2].astype(float)/1000,1,5,window=('kaiser',5.),padtype='constant').astype('float32');np.testing.assert_array_equal(wave,expected);self.assertTrue(a['native_checksum_verified'])
 def test_microvolt_converts_to_millivolt(self):
  self.write(gain=1,unit='uV');a,_=physical_lead_II(self.base);self.write(gain=1000,unit='mV');b,_=physical_lead_II(self.base);np.testing.assert_array_equal(a,b)
 def test_checksum_mismatch_stops(self):
  self.write(bad_checksum=True)
  with self.assertRaises(ValueError):physical_lead_II(self.base)
 def test_flat_source_stops(self):
  self.raw[:]=0;self.write()
  with self.assertRaises(ValueError):physical_lead_II(self.base)
if __name__=='__main__':unittest.main()
