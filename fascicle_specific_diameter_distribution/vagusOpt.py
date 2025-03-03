from vagusNerve.runSim import runSim
from combine import combine_signals
import numpy as np
from scipy.io import loadmat
from scipy.stats import wasserstein_distance
import multiprocessing as mp
from functools import partial

def get_error(signalList):

    totalDistance = 0

    for simulation in range(10):
  
        rawSignal = np.load('groundTruth/Signals_Stim'+str(simulation)+'.npy')

        #rawSignal /= np.max(np.abs(rawSignal))

        signal = signalList[simulation]
    
        #signal /= np.max(np.abs(signal))

        totalDistance += wasserstein_distance(rawSignal[0,0],signal[0,0])

    return totalDistance #np.min(np.sum(np.abs(rawSignal-signal)))

def get_error_raw(signal):

    
    rawData = loadmat('../Data/eCAPSdata_220303.mat')
    a500 = list(rawData['eCAPSdata_220328'][0][-5])
    
    rawSignal = a500[1][-1]-a500[1][0]
    rawTime = a500[2][0]

    timeLimits = np.intersect1d( np.where(rawTime < 9 )[0], np.where(rawTime>1)[0] )

    rawTime = rawTime[timeLimits]
    rawSignal = rawSignal[timeLimits]

#    rawSignal /= np.max(np.abs(rawSignal))
    
    tmin=-.5 # In s
    tmax=.5 # In s
    nx=50000
    tphi=np.arange(tmin,tmax,(tmax-tmin)/(nx-1))
    
    time = tphi[1:-1]*1e3
    
    indices = []
    for t in rawTime:
        indices.append(np.argmin(np.abs(time-t)))

    print(signal[0].shape)
    signal = signal[0][0,0,indices]

 #   signal /= np.max(np.abs(signal))

    return wasserstein_distance(rawSignal,signal) #np.min(np.sum(np.abs(rawSignal-signal)))

def runSim_wrapper(fascIdx, stim, rec, params):
    return runSim(0, stim, rec, fascIdx,params, 2000)  # Pass correct arguments

def run_vagus_nerve(analytic_input):

    currents = [24.38, 23.07, 23.79, 23.19, 22.1, 23.97, 22.7, 22.95, 20.84, 23.89]

    numcores = mp.cpu_count()

    signalList = []

    for simulation in range(1):

        stim = {'current':[500*10/173],
                'stimulusDirectory':{
                    "myelinated":'../Data/TitrationBetterConductivity_Standoff_HighConductivity.xlsx'
                }
               }

        # stim = {
        #         'current': [500 / currents[simulation]],  # Convert NumPy array to float
        #         'stimulusDirectory': {
        #         "myelinated": r"D:\vagusOptimization\multiSimulation\Titration\Titration_Sim" + str(simulation) + ".xlsx"
        #          }
        #         }

        rec = {
                'recordingCurrent': 509e-6,
                'recordingDirectory': '../Data/PhiConductivity_Bipolar_Corrected/'
                }

        with mp.Pool(numcores-4) as p:
            signals = p.starmap(runSim_wrapper, [(i, stim, rec, analytic_input) for i in np.arange(39)])

        signal = combine_signals(signals)
        signalList.append(signal)

    error = get_error_raw(signalList)
    signalList = np.array(signalList)
    
    try:
        allSignals = np.load('allSignal.npy')
        allError = np.load('allError.npy')
        allInputs = np.load('allInputs.npy')

        allSignals = np.vstack((allSignals,signalList))
        allError = np.vstack((allError,error))
        allInputs = np.vstack((allInputs,[analytic_input]))
        
    except:
        allSignals = signalList
        allError = error
        allInputs = [analytic_input]

    np.save('allSignal.npy',allSignals)

    np.save('allError.npy',allError)
    np.save('allInputs.npy',allInputs)

    return error
