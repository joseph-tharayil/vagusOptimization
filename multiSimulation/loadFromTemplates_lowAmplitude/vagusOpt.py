import numpy as np
from scipy.io import loadmat
from scipy.stats import wasserstein_distance


def get_error(signalList):

    totalDistance = 0

    time = np.linspace(-.5,.5,49997)*1e3
    timeIdx = np.intersect1d(np.where(time>0),np.where(time<10))

    for simulation in range(10):
  
        rawSignal = np.load('../groundTruth_lowAmplitude/Signals_Stim'+str(simulation)+'.npy')

        #rawSignal /= np.max(np.abs(rawSignal))

        signal = signalList[simulation]
    
        #signal /= np.max(np.abs(signal))

        totalDistance += wasserstein_distance(rawSignal[0,0][timeIdx],signal[timeIdx])

    return totalDistance #np.min(np.sum(np.abs(rawSignal-signal)))

def runSim_wrapper(fascIdx, stim, rec, params):
    return runSim(0, stim, rec, fascIdx,params, 2000)  # Pass correct arguments

def run_vagus_nerve(analytic_input):

    signalList = []

    for simulation in range(10):

        for fasc in range(39):
            maffFrac = analytic_input[fasc][0]*.01
            meffFrac = analytic_input[fasc][1]*.01

            sigMaff = np.load('../templates_lowAmplitude/Signals_Stim'+str(simulation)+'_'+str(fasc)+'_maff.npy')*maffFrac
            sigMeff = np.load('../templates_lowAmplitude/Signals_Stim'+str(simulation)+'_'+str(fasc)+'_meff.npy')*meffFrac

            if fasc == 0:
                signal = sigMaff + sigMeff
            else:
                signal += sigMaff + sigMeff

        signalList.append(signal)

    error = get_error(signalList)
    signalList = np.array(signalList)
    try:
        allSignals = np.load('allSignal_unconstrained.npy')
        allError = np.load('allError_unconstrained.npy')
        allInputs = np.load('allInputs_unconstrained.npy')

        allSignals = np.vstack((allSignals,[signalList]))
        allError = np.vstack((allError,error))
        allInputs = np.vstack((allInputs,[analytic_input]))
        
    except:
        allSignals = [signalList]
        allError = error
        allInputs = [analytic_input]

    np.save('allSignal_unconstrained.npy',allSignals)

    np.save('allError_unconstrained.npy',allError)
    np.save('allInputs_unconstrained.npy',allInputs)

    return error
