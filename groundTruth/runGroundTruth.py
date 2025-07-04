import multiprocessing as mp
from vagusNerve.runSim import runSim
from combine import combine_signals
import numpy as np
from functools import partial

numcores = mp.cpu_count()
num_workers = max(1, numcores - 4)  # Ensure at least one worker

def runSim_wrapper(fascIdx, stim, rec):
    distribution_params = {'maff':{'diameterParams':None, 'fiberTypeFractions':None},'meff':{'diameterParams':None, 'fiberTypeFractions':None}}
    return runSim(0,stim, rec, fascIdx,distribution_params,2000)  # Pass correct arguments

if __name__ == "__main__":  # ✅ Prevent multiprocessing issues

    currents = [500 ]   
    for simulation in np.arange(len(currents)):
        stim = {'current':[currents[simulation]*10/173],
                    'stimulusDirectory':{
                    "myelinated":'path/to/titration.xlsx'
                     }
                   }
        rec = {
                   'recordingCurrent': 509e-6,
                   'recordingDirectory': 'path/to/folder/with/sensitivityFiles/'
                  }

        with mp.Pool(num_workers) as p:
            signals = p.starmap(runSim_wrapper, [(i, stim, rec) for i in np.arange(39)])  # ✅ Use starmap for multiple arguments
        total = combine_signals(signals)
        np.save('Signals_Stim_' + str(currents[simulation]) + ".npy", np.array(total))  # Ensure it's a NumPy array

