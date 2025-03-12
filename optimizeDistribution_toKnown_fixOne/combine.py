import numpy as np

import sys

def combine_signals(signals):

    iteration = 0

    names = ['maff','meff']

    for i, s in enumerate(signals):
        if i == 0:
            totalSignal = s
        else:
            totalSignal += s

    #for typeIdx in range(2):

        #for fascIdx in range(39):

            #totalSignal += np.load('signals/'+names[typeIdx]+'/signals_'+str(fascIdx)+'.npy')

    return totalSignal

if __name__=='__main__':
    
    outputfolder = sys.argv[1]
    distanceIdx = int(sys.argv[2])
    
    main(outputfolder,distanceIdx)
    
    
