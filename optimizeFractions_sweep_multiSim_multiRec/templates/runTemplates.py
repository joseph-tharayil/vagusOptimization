import multiprocessing as mp
from vagusNerve.runSim import runSim
import numpy as np
from functools import partial
import os

numcores = mp.cpu_count()
num_workers = max(1, numcores - 3)  # Ensure at least one worker

def runSim_wrapper(fascIdx, stim, rec):
    distribution_params = {'maff':{'diameterParams':None, 'fiberTypeFractions':np.ones(39)*100},'meff':{'diameterParams':None, 'fiberTypeFractions':np.zeros(39)*100}}
    return runSim(0,stim, rec, fascIdx,distribution_params,2000)  # Pass correct arguments

if __name__ == "__main__":  # ✅ Prevent multiprocessing issues

    currents = np.logspace(1,3,5)
    simCurrents = [24.38, 23.07, 23.79, 23.19, 22.1, 23.97, 22.7, 22.95, 20.84, 23.89]

    recordingArrays = ['PhiConductivity_Bipolar_Corrected/','PhiConductivity_Small_20240213/','PhiConductivity_Small_Otherside_20240215/','PhiConductivity_Close_20240213/']
    recCurrents = [509e-6,250e-6,273e-6,514e-6]
    cutoffs = [1e-4,1e-6,1e-6,1e-4]

    for simulation in np.arange(0,len(simCurrents)):
        stim = {'current':currents/simCurrents[simulation],
                'stimulusDirectory':{
                   "myelinated":r"D:\vagusNerve\VerticalElectrode\Titration_Sim" + str(simulation) + ".xlsx"
                   }
                }
        for recIdx in np.arange(0,len(recCurrents)):
            rec = {
                   'recordingCurrent': recCurrents[recIdx],
                   'recordingDirectory': '../../Data/'+recordingArrays[recIdx],
                   'cutoff':cutoffs[recIdx]
                  }

            if not os.path.isfile('Signals_Stim_' + str(simulation)+'_'+str(recIdx)+"_0.npy"):

                with mp.Pool(num_workers) as p:
                    signals = p.starmap(runSim_wrapper, [(i, stim, rec) for i in np.arange(39)])  # ✅ Use starmap for multiple arguments
            
                for fasc, s in enumerate(signals):
                    np.save('Signals_Stim_' + str(simulation)+'_'+str(recIdx)+'_'+str(fasc) + ".npy",s)  # Ensure it's a NumPy array

