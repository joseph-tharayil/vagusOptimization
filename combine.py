import numpy as np
from mpi4py import MPI
import sys

def combine_signals():

    rank = MPI.COMM_WORLD.Get_rank()

    iteration = 0

    names = ['maff','meff']

    totalSignal = 0

    for typeIdx in range(2):

        for fascIdx in range(39):

            totalSignal += np.load('signals/'+names[typeIdx]+'/signals_'+str(fascIdx)+'.npy')

    return totalSignal

if __name__=='__main__':
    
    outputfolder = sys.argv[1]
    distanceIdx = int(sys.argv[2])
    
    main(outputfolder,distanceIdx)
    
    
