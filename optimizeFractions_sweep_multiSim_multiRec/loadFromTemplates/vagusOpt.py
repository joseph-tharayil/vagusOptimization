import os
import numpy as np
from scipy.io import loadmat
from scipy.stats import wasserstein_distance
from vagusNerve.nerveSetup import getFibersPerFascicle,getFascicleTypes

def calculateResult(fascParams):

    numFibersIfOnlyMaff = []
    numFibers = []

    fascTypes = getFascicleTypes()

    distribution_params_onlyMaff= {'maff':{'diameterParams':None, 'fiberTypeFractions':np.ones(39)*100},'meff':{'diameterParams':None, 'fiberTypeFractions':np.zeros(39)*100}}

    for fasc in range(39):

        numFibersIfOnlyMaff.append(getFibersPerFascicle(fasc,fascTypes,distribution_params_onlyMaff))

        maffFrac = fascParams[fasc]*.01

        distribution_params = {'maff':{'diameterParams':None, 'fiberTypeFractions':np.ones(39)*100*maffFrac},'meff':{'diameterParams':None, 'fiberTypeFractions':np.zeros(39)*100}}

        numFibers.append(getFibersPerFascicle(fasc,fascTypes,distribution_params))

    signalList = []

    for simulation in range(10):

        for recording in range(4):

            for fasc in range(39):
                maffFrac = fascParams[fasc]*.01
    
                distribution_params = {'maff':{'diameterParams':None, 'fiberTypeFractions':np.ones(39)*100*maffFrac},'meff':{'diameterParams':None, 'fiberTypeFractions':np.zeros(39)*100}}
    
                sigMaff = np.load('../templates/Signals_Stim_'+str(simulation)+'_'+str(recording)+'_'+str(fasc)+'.npy')*maffFrac*numFibers[fasc]/numFibersIfOnlyMaff[fasc]
    
                if fasc == 0:
                    signal = sigMaff
                else:
                    signal += sigMaff
    
            signalList.append(signal)

    return np.array(signalList)

def get_error(signalList):

    totalDistance = 0

    time = np.linspace(-.5,.5,49997)*1e3
    timeIdx = np.intersect1d(np.where(time>0),np.where(time<10))

    trueProbs = []
    for i in range(39):
        trueProbs.append(np.load('../../optimizeFractions_sweep_multiSim/groundTruth/fiberProbs_'+str(i)+'.npy'))
    trueProbs = np.array(trueProbs)[:,:,0]*100

    groundTruths = calculateResult(np.sum(trueProbs,axis=-1))


    totalDistance = np.sum(np.abs(groundTruths-signalList)) #wasserstein_distance(rawSignal[0,0][timeIdx],signal[timeIdx])

    return totalDistance 

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

        for recording in range(4):

            for fasc in range(39):
                maffFrac = analytic_input[fasc][0]*.01
    
                distribution_params = {'maff':{'diameterParams':None, 'fiberTypeFractions':np.ones(39)*100*maffFrac},'meff':{'diameterParams':None, 'fiberTypeFractions':np.zeros(39)*100}}
    
                sigMaff = np.load('../templates/Signals_Stim_'+str(simulation)+'_'+str(recording)+'_'+str(fasc)+'.npy')*maffFrac*numFibers[fasc]/numFibersIfOnlyMaff[fasc]
    
                if fasc == 0:
                    signal = sigMaff
                else:
                    signal += sigMaff

            signalList.append(signal)

    error = get_error(signalList)
    signalList = np.array(signalList)
    
    if os.path.exists('allError_unconstrained.npy'):

        allError = np.load('allError_unconstrained.npy')
        allInputs = np.load('allInputs_unconstrained.npy')

        #allSignals = np.vstack((allSignals,[signalList]))
        allError = np.vstack((allError,error))
        allInputs = np.vstack((allInputs,[analytic_input]))
        
    else:

        allError = error
        allInputs = [analytic_input]

    np.save('allError_unconstrained.npy',allError)
    np.save('allInputs_unconstrained.npy',allInputs)

    return error
