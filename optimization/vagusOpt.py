from vagusNerve.runSim import runSim
from combine import combine_signals
import numpy as np
from scipy.io import loadmat
from scipy.stats import wasserstein_distance
import multiprocessing as mp
from functools import partial
import os

def get_error(signalList):

    totalDistance = 0

    for simulation in range(1):
  
        rawSignal = np.load('../groundTruth/Signals_Stim_500.npy')

        #rawSignal /= np.max(np.abs(rawSignal))

        signal = signalList[simulation]
    
        #signal /= np.max(np.abs(signal))

        totalDistance += np.sum(np.abs(rawSignal-signal)) #wasserstein_distance(rawSignal[0,0],signal[0,0])

    return totalDistance #np.min(np.sum(np.abs(rawSignal-signal)))

def runSim_wrapper(fascIdx, stim, rec, params):
    return runSim(0, stim, rec, fascIdx,params, 2000)  # Pass correct arguments

def run_vagus_nerve(analytic_input):

    currents = [24.38, 23.07, 23.79, 23.19, 22.1, 23.97, 22.7, 22.95, 20.84, 23.89]

    numcores = mp.cpu_count()

    signalList = []

    for simulation in range(1):

        stim = {'current':[500*10/173],
                'stimulusDirectory':{
                    "myelinated":'path/to/titration.xlsx'
                }
               }


        rec = {
                'recordingCurrent': 509e-6,
                'recordingDirectory': 'path/to/folder/with/sensitivityFiles/'
                }

        with mp.Pool(numcores-4) as p:
            signals = p.starmap(runSim_wrapper, [(i, stim, rec, analytic_input) for i in np.arange(39)])

        signal = combine_signals(signals)
        signalList.append(signal)

    error = get_error(signalList)
    signalList = np.array(signalList)
    
    if os.path.isfile('allSignal.npy'):
        allSignals = np.load('allSignal.npy')
        allError = np.load('allError.npy')
        allInputs = np.load('allInputs.npy',allow_pickle=True)

        allSignals = np.vstack((allSignals,signalList))
        allError = np.vstack((allError,error))
        allInputs = np.vstack((allInputs,[analytic_input]))
        
    else:
        allSignals = signalList
        allError = error
        allInputs = [analytic_input]

    np.save('allSignal.npy',allSignals)

    np.save('allError.npy',allError)
    np.save('allInputs.npy',allInputs)

    return error
