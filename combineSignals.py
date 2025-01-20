import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import wasserstein_distance

rawData = loadmat('Data/eCAPSdata_220303.mat')
a500 = list(rawData['eCAPSdata_220328'][0][-5])

rawSignal = a500[1][-1]-a500[1][0]
rawTime = a500[2][0]

timeLimits = np.intersect1d( np.where(rawTime < 9 )[0], np.where(rawTime>1)[0] )

rawTime = rawTime[timeLimits]
rawSignal = rawSignal[timeLimits]

scaledSignal = rawSignal / np.max(np.abs(rawSignal))

    
tmin=-3 # In s
tmax=3 # In s
nx=500000
tphi=np.arange(tmin,tmax,(tmax-tmin)/(nx-1))

time = tphi[1:-1]*1e3

indices = []
for t in rawTime:
    indices.append(np.argmin(np.abs(time-t)))

def get_error_scaled(signal):

    
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

def get_error(signal):

    
    rawData = loadmat('Data/eCAPSdata_220303.mat')
    a500 = list(rawData['eCAPSdata_220328'][0][-5])
    
    rawSignal = a500[1][-1]-a500[1][0]
    rawTime = a500[2][0]

    timeLimits = np.intersect1d( np.where(rawTime < 9 )[0], np.where(rawTime>1)[0] )

    rawTime = rawTime[timeLimits]
    rawSignal = rawSignal[timeLimits]
    
    tmin=-3 # In s
    tmax=3 # In s
    nx=500000
    tphi=np.arange(tmin,tmax,(tmax-tmin)/(nx-1))
    
    time = tphi[1:-1]*1e3
    
    indices = []
    for t in rawTime:
        indices.append(np.argmin(np.abs(time-t)))

    signal = signal[0,indices]

    return wasserstein_distance(rawSignal,signal) #np.min(np.sum(np.abs(rawSignal-signal)))

if __name__=='__main__':

    x0s = np.linspace(5,7,num=6)
    y0s = np.linspace(0.5,0.7,num=6)
    
    x1s = np.linspace(7.5,9.5,num=6) 
    y1s = np.linspace(0.4,0.6,num=6)
    
    allSignals = []
    allErrors = []
    allErrors_Unscaled = []
    
    valsY0s = []
    valsX0s = []
    
    valsY1s = []
    valsX1s = []
    
    for j, x0 in enumerate(x0s):
    
        index = 0
        
        for y0 in y0s:
            for x1 in x1s:
                for y1 in y1s:
    
                    signal = 0
    
                    try:
    
                        for i in range(39):
                            signal += np.load('signals/maff/'+str(j)+'/'+str(index)+'/signals_'+str(i)+'.npy')
                            signal += np.load('signals/meff/'+str(j)+'/'+str(index)+'/signals_'+str(i)+'.npy')
        
                    
                        error = get_error(signal)
                        error_scaled = get_error_scaled(signal)
        
                        allErrors.append(error_scaled)
                        allErrors_Unscaled.append(error)
        
                        allSignals.append(signal)
    
                    except:
    
                        allErrors.append(100)
                        allErrors_Unscaled.append(100)
                        allSignals.append([])
    
                    valsY1s.append(y1)
                    valsX1s.append(x1)
    
                    valsY0s.append(y0)
                    valsX0s.append(x0)
    
                    index += 1

    np.save('allSignals.npy',allSignals)
    np.save('allErrors.npy',allErrors)
    np.save('allErrors_Unscaled.npy',allErrors_Unscaled)
