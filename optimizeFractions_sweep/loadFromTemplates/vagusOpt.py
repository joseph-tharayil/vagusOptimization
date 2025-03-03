import os
import numpy as np
from scipy.io import loadmat
from scipy.stats import wasserstein_distance
from vagusNerve.nerveSetup import getFibersPerFascicle,getFascicleTypes

def get_error(signalList):

    totalDistance = 0

    time = np.linspace(-.5,.5,49997)*1e3
    timeIdx = np.intersect1d(np.where(time>0),np.where(time<10))

    for simulation in range(10):

        if simulation==0:
            rawSignal = np.load('../groundTruth/Signals_Stim_a0'+'.npy')
        else:
  
            rawSignal = np.load('../groundTruth/Signals_Stim'+str(simulation-1)+'.npy')

        #rawSignal /= np.max(np.abs(rawSignal))

        signal = signalList[simulation]
    
        #signal /= np.max(np.abs(signal))

        totalDistance += wasserstein_distance(rawSignal[0,0][timeIdx],signal[timeIdx])

    return totalDistance #np.min(np.sum(np.abs(rawSignal-signal)))

def runSim_wrapper(fascIdx, stim, rec, params):
    return runSim(0, stim, rec, fascIdx,params, 2000)  # Pass correct arguments

def run_vagus_nerve(analytic_input):

    distribution_params_onlyMaff= {'maff':{'diameterParams':None, 'fiberTypeFractions':np.ones(39)*100},'meff':{'diameterParams':None, 'fiberTypeFractions':np.zeros(39)*100}}

    signalList = []

    numFibersIfOnlyMaff = []
    numFibers = []

    fascTypes = getFascicleTypes()

    for fasc in range(39):

        numFibersIfOnlyMaff.append(getFibersPerFascicle(fasc,fascTypes,distribution_params_onlyMaff))

        maffFrac = analytic_input[fasc][0]*.01

        distribution_params = {'maff':{'diameterParams':None, 'fiberTypeFractions':np.ones(39)*100*maffFrac},'meff':{'diameterParams':None, 'fiberTypeFractions':np.zeros(39)*100}}

        numFibers.append(getFibersPerFascicle(fasc,fascTypes,distribution_params))
    
    for simulation in range(10):

        for fasc in range(39):
            maffFrac = analytic_input[fasc][0]*.01

            distribution_params = {'maff':{'diameterParams':None, 'fiberTypeFractions':np.ones(39)*100*maffFrac},'meff':{'diameterParams':None, 'fiberTypeFractions':np.zeros(39)*100}}

            if simulation == 0:
                sigMaff = np.load('../templates/Signals_Stim_a0_'+str(fasc)+'.npy')*maffFrac*numFibers[fasc]/numFibersIfOnlyMaff[fasc]
            else:
                sigMaff = np.load('../templates/Signals_Stim'+str(simulation-1)+'_'+str(fasc)+'.npy')*maffFrac*numFibers[fasc]/numFibersIfOnlyMaff[fasc]

            if fasc == 0:
                signal = sigMaff
            else:
                signal += sigMaff

        signalList.append(signal)

    error = get_error(signalList)
    signalList = np.array(signalList)
    
    if os.path.exists('allError_unconstrained.npy'):
        #allSignals = np.load('allSignal_unconstrained.npy')
        allError = np.load('allError_unconstrained.npy')
        allInputs = np.load('allInputs_unconstrained.npy')

        #allSignals = np.vstack((allSignals,[signalList]))
        allError = np.vstack((allError,error))
        allInputs = np.vstack((allInputs,[analytic_input]))
        
    else:
        #allSignals = [signalList]
        allError = error
        allInputs = [analytic_input]

    #np.save('allSignal_unconstrained.npy',allSignals)

    np.save('allError_unconstrained.npy',allError)
    np.save('allInputs_unconstrained.npy',allInputs)

    return error
