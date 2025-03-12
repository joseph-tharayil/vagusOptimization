import multiprocessing as mp
from vagusNerve.runSim import runSim
import numpy as np
from functools import partial

numcores = mp.cpu_count()
num_workers = max(1, numcores - 4)  # Ensure at least one worker

def runSim_wrapper(fascIdx, stim, rec):
    distribution_params = {'maff':{'diameterParams':None, 'fiberTypeFractions':np.ones(39)*100},'meff':{'diameterParams':None, 'fiberTypeFractions':np.zeros(39)*100}}
    return runSim(0,stim, rec, fascIdx,distribution_params,2000)  # Pass correct arguments

if __name__ == "__main__":  # ✅ Prevent multiprocessing issues
        currents = [2,4,6,8,10]

        for simulation in np.arange(1,len(currents)):
            stim = {'current':[currents[simulation]*10/173],
                    'stimulusDirectory':{
                       "myelinated":'../../Data/TitrationBetterConductivity_Standoff_HighConductivity.xlsx'
                       }
                    }
            rec = {
                   'recordingCurrent': 509e-6,
                   'recordingDirectory': '../../Data/PhiConductivity_Bipolar_Corrected/'
                  }

            with mp.Pool(num_workers) as p:
                signals = p.starmap(runSim_wrapper, [(i, stim, rec) for i in np.arange(39)])  # ✅ Use starmap for multiple arguments
            
            for fasc, s in enumerate(signals):
                np.save('Signals_Stim_a' + str(simulation)+'_'+str(fasc) + ".npy",s[0,0])  # Ensure it's a NumPy array

