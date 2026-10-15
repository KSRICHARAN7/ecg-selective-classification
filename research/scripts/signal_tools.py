"""Pilot numerical implementation of protocol v0.3 signal operations."""
import numpy as np
from scipy.signal import welch,resample_poly
from scipy.integrate import trapezoid
from scipy.stats import kurtosis
from math import gcd


def resample_noise(x,original_hz,target_hz=100):
    divisor=gcd(int(original_hz),int(target_hz))
    return resample_poly(np.asarray(x,dtype=float),target_hz//divisor,original_hz//divisor,axis=0,window=('kaiser',5.0),padtype='constant')


def add_noise(x,noise,snr_db):
    x=np.asarray(x,dtype=float);noise=np.asarray(noise,dtype=float)
    if x.shape!=noise.shape or not np.isfinite(x).all() or not np.isfinite(noise).all():
        raise ValueError('Finite matching source and noise are required')
    x=x-x.mean();noise=noise-noise.mean()
    sx=np.sqrt(np.mean(x*x));sn=np.sqrt(np.mean(noise*noise))
    if sx<=1e-12 or sn<=1e-12:raise ValueError('Zero-power source or noise')
    injected=noise*(sx/sn)*10**(-snr_db/20)
    return x+injected,injected


def quality_features(x,fs=100):
    x=np.asarray(x,dtype=float)
    if x.ndim!=1 or len(x)!=1000 or not np.isfinite(x).all() or fs!=100:
        raise ValueError('Expected one finite1000-sample100Hz recording')
    x=x-x.mean();sd=x.std();diff=np.diff(x)
    freq,power=welch(x,fs=fs,window='hann',nperseg=256,noverlap=128,detrend='constant',scaling='density')
    def band(low,high):
        mask=(freq>low)&(freq<high)
        f=np.r_[low,freq[mask],high]
        p=np.r_[np.interp(low,freq,power),power[mask],np.interp(high,freq,power)]
        return trapezoid(p,f)
    def ratio(a,b):return float(a/b) if b>1e-12 else float('nan')
    return np.asarray([sd,np.ptp(x),np.mean(np.abs(diff)<=1e-6),ratio(np.sqrt(np.mean(diff*diff)),sd),ratio(band(.05,.5),band(.05,40)),ratio(band(20,40),band(.5,40)),float(kurtosis(x,fisher=False,bias=False)) if sd>1e-12 else float('nan')])
