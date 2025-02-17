import numpy as np
import pandas as pd
from scipy.optimize import linprog

def load_exposures():
    currents = [24.38, 23.07, 23.79, 23.19, 22.1, 23.97, 22.7, 22.95, 20.84, 23.89]
    exposureFunctions = []
    for stim in range(10):
        exposureFunctions.append([])
        for fasc in range(39):
            phi = -1*pd.read_excel(r'D:\vagusNerve\VerticalElectrode\StimulusExposures\Sim_'+str(stim)+'Fasc_'+str(fasc)+'.xlsx')['Phi [V] [real]']/currents[stim]
            deriv = np.diff(phi)
            smoothedDeriv = np.convolve(np.ones(100),deriv,mode='same')
            exposureFunctions[stim].append(np.max(np.diff(smoothedDeriv)))

    exposureFunctions = np.array(exposureFunctions)
    return exposureFunctions

fascicleList = np.arange(39)
targetFascicle = 0
offTarget = np.delete(fascicleList,targetFascicle)

exposures = load_exposures()
optimizationCoefficients = -exposures[:,targetFascicle]
boundMatrix = exposures[:,offTarget]
MaxOffTargetActivation = 1e-7*np.ones(38)

optimizedResult = linprog(optimizationCoefficients,A_ub=boundMatrix.T,b_ub=MaxOffTargetActivation,bounds=(-500,500))

np.save('optimizedResult.npy',optimizedResult)
