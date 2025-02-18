import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

def load_exposures():
    currents = [24.38, 23.07, 23.79, 23.19, 22.1, 23.97, 22.7, 22.95, 20.84, 23.89]
    exposureFunctions = []
    for stim in range(10):
        exposureFunctions.append([])
        for fasc in range(39):
            phi = -1*pd.read_excel(r'D:\vagusNerve\VerticalElectrode\StimulusExposures\Sim_'+str(stim)+'Fasc_'+str(fasc)+'.xlsx')['Phi [V] [real]']/currents[stim]
            deriv = np.diff(phi)
            smoothedDeriv = np.convolve(np.ones(100),deriv,mode='same')
            exposureFunctions[stim].append(np.convolve(np.ones(100),np.diff(smoothedDeriv)))

    exposureFunctions = np.array(exposureFunctions)
    return exposureFunctions

def targetFascicleActivation(currents,exposureFunctions):

    activation = exposureFunctions.T@currents
    maxActivation = np.max(activation)
    return -maxActivation # Negative sign because we want to maximize it, but Python needs to minimize it

def create_bounds_list():
    bounds_list = []
    for i in range(10):
        bounds_list.append((-500,500))

    return bounds_list

def offTargetActivation_Rel(currents,exposureFunctions,targetExposures):

    activation = exposureFunctions.T @ currents
    maxActivation = np.max(activation,axis=0)

    targetActivation = targetExposures.T@currents
    maxTargetActivation = np.max(targetActivation)

    constraintsMet = maxTargetActivation - 2*maxActivation # We expect this to be non-negative
    return constraintsMet

def offTargetActivation_Abs(currents,exposureFunctions,maxDesiredActivation):

    activation = exposureFunctions.T @ currents
    maxActivation = np.max(activation,axis=0)

    constraintsMet = maxDesiredActivation - maxActivation # We expect this to be non-negative
    return constraintsMet

fascicleList = np.arange(39)
targetFascicle = 0
offTarget = np.delete(fascicleList,targetFascicle)

exposures = load_exposures()
targetExposures = exposures[:,targetFascicle] 

bounds_list = create_bounds_list()

offTargetExposures = exposures[:,offTarget]

constraintDict_Rel = {'type':'ineq','fun':offTargetActivation_Rel,'args':(offTargetExposures,targetExposures)}
constraintDict_Abs = {'type':'ineq','fun':offTargetActivation_Abs,'args':(offTargetExposures,np.ones(38)*1e-8)}
constraintList = [constraintDict_Rel, constraintDict_Abs]

optimizedResult = minimize(targetFascicleActivation, np.zeros(10), args=targetExposures, method='COBYLA',bounds=bounds_list,constraints=constraintList)

#optimizedResult = linprog(optimizationCoefficients,A_ub=boundMatrix.T,b_ub=MaxOffTargetActivation,bounds=(-500,500))

np.save('optimizedResult.npy',optimizedResult)
