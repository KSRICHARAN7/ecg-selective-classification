"""Deterministic development noise pairs. Withheld sources are not exposed.

The source noise gain is not treated as a calibrated physical amplitude:
positive linear gain cancels under per-window AC-RMS scaling.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import wfdb
from cohort_loader import ROOT
from signal_tools import add_noise,resample_noise

SEED=20261015
FAMILIES=('bw','ma')
SNR_LEVELS=(24,12,6,0)
REGIONS={'train':(0,580),'checkpoint':(0,580),'ranker':(0,580),'validation':(600,1180),'test':(1200,1780)}


def assignment(ecg_id,family,role='train'):
    if family not in FAMILIES:raise PermissionError('Only declared development noise sources are available; em remains reserved')
    if role not in REGIONS:raise ValueError('Unknown role')
    start,stop=REGIONS[role]
    digest=hashlib.sha256(f'{SEED}|{ecg_id}|{family}'.encode()).digest()
    channel=int.from_bytes(digest[:8],'big')%2
    slot=int.from_bytes(digest[8:16],'big')%((stop-start)//10)
    return {'ecg_id':str(ecg_id),'family':family,'role':role,'channel':channel,'slot':slot,'source_start_second':start+10*slot,'source_stop_second':start+10*(slot+1),'seed':SEED}


class DevelopmentFaultLibrary:
    def __init__(self,root=ROOT):
        self.root=Path(root);self.bank={};self.source_hashes={};self.segment_audit=[]
        provenance=json.loads((self.root/'outputs/pilot_download_manifest.json').read_text())
        expected={r['path']:r['sha256']for r in provenance}
        for family in FAMILIES:
            base=self.root/'data/pilot/noise'/family
            for suffix in ['.hea','.dat']:
                path=base.with_suffix(suffix);rel=path.relative_to(self.root).as_posix()
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                if digest!=expected[rel]:raise ValueError('Noise source changed: '+rel)
                self.source_hashes[rel]=digest
            header=wfdb.rdheader(str(base))
            if header.fs!=360 or header.n_sig!=2 or header.sig_len<580*360:raise ValueError('Unexpected source noise layout')
            # Decode development seconds only. Reserved source time is not read.
            source=wfdb.rdrecord(str(base),sampfrom=0,sampto=580*360,physical=True).p_signal
            if source.shape!=(580*360,2)or not np.isfinite(source).all():raise ValueError('Invalid development noise source')
            bank=np.empty((2,58,1000),dtype='float64')
            for channel in range(2):
                for slot in range(58):
                    raw=source[slot*3600:(slot+1)*3600,channel]
                    resampled=resample_noise(raw,360)
                    if len(resampled)!=1000 or not np.isfinite(resampled).all():raise ValueError('Invalid resampled segment')
                    ac=resampled-resampled.mean();rms=float(np.sqrt(np.mean(ac*ac)))
                    if rms<=1e-12:raise ValueError('Zero-power source segment; amend prospectively, do not skip')
                    bank[channel,slot]=resampled
                    self.segment_audit.append({'family':family,'channel':channel,'slot':slot,'start_second':slot*10,'stop_second':(slot+1)*10,'AC_RMS_in_WFDB_source_units':rms,'samples':1000,'resampled_float64_sha256':hashlib.sha256(resampled.tobytes()).hexdigest()})
            self.bank[family]=bank

    def make_pair(self,x,ecg_id,family,snr_db,role='train'):
        if role not in ['train','checkpoint','ranker']:raise PermissionError('Development library never decodes reserved validation or test source regions')
        if snr_db not in SNR_LEVELS:raise ValueError('Severity outside the declared development grid')
        meta=assignment(ecg_id,family,role)
        x=np.asarray(x,dtype='float64')
        if x.shape!=(1000,):raise ValueError('Expected 1000-sample lead-II waveform')
        noise=self.bank[family][meta['channel'],meta['slot']]
        corrupted,injected=add_noise(x,noise,snr_db)
        clean=x-x.mean()
        return clean,corrupted,injected,{**meta,'requested_snr_db':snr_db}
