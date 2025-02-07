from analytic import write_fascicle_signals
from combine import combine_signals
import numpy as np
from scipy.io import loadmat
from scipy.stats import wasserstein_distance
import multiprocessing as mp
from functools import partial

def get_error(signal):

    
    rawData = loadmat('Data/eCAPSdata_220303.mat')
    a500 = list(rawData['eCAPSdata_220328'][0][-5])
    
    rawSignal = a500[1][-1]-a500[1][0]
    rawTime = a500[2][0]

    timeLimits = np.intersect1d( np.where(rawTime < 9 )[0], np.where(rawTime>1)[0] )

    rawTime = rawTime[timeLimits]
    rawSignal = rawSignal[timeLimits]

    rawSignal /= np.max(np.abs(rawSignal))
    
    tmin=-3 # In s
    tmax=3 # In s
    nx=500000
    tphi=np.arange(tmin,tmax,(tmax-tmin)/(nx-1))
    
    time = tphi[1:-1]*1e3
    
    indices = []
    for t in rawTime:
        indices.append(np.argmin(np.abs(time-t)))

    signal = signal[0,indices]

    signal /= np.max(np.abs(signal))

    return wasserstein_distance(rawSignal,signal) #np.min(np.sum(np.abs(rawSignal-signal)))

def run_vagus_nerve(analytic_input):

    numcores = mp.cpu_count()

    with mp.Pool(numcores-4) as p:
        p.map(partial(write_fascicle_signals,distribution_params=analytic_input),np.arange(39))

    #write_fascicle_signals(analytic_input)

    signal = combine_signals()

    error = get_error(signal)
    #allError = np.load('allError.npy')
    #allError = np.vstack((allError,error))
    #np.save('allError.npy',allError)

    return error
    
