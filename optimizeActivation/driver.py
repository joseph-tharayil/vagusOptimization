import dakota.interfacing as di
import numpy as np
import pandas as pd
import time

def get_input(params):

    currents = [24.38, 23.07, 23.79, 23.19, 22.1, 23.97, 22.7, 22.95, 20.84, 23.89]

    stimulusCurrents = []

    for i in range(10):

        stimulusCurrents.append( params['x'+str(i)]/currents[i] )

    return stimulusCurrents

def pack_dakota_results(analytic_output,results):

    targetFascicle = 0
    offTargetFascicles = np.delete(np.arange(39),targetFascicle)

    results[0].function = -analytic_output[targetFascicle] #Now we try to amximize activation in target fascicle
    for fasc in offTargetFascicles:
        results[int(fasc)].function = analytic_output[fasc]/((3*1e-7)/25)-1 # Need the constraint to be less than 0 -> off-target activation (3*1e7)/25

    return results


def getActivationFunctions(stimulusCurrents):

    fascActivations = []
    t = time.time()
    for fascicle in range(39):

        for i, stim in enumerate(stimulusCurrents):

            phi = pd.read_excel(r'D:\vagusNerve\VerticalElectrode\StimulusExposures\Sim_'+str(i)+'Fasc_'+str(fascicle)+'.xlsx')
            
            if i == 0:
                activation = -stim * np.diff(phi['Phi [V] [real]'],2)
            else:
                activation += -stim * np.diff(phi['Phi [V] [real]'],2)
        maxActivation = np.max(activation)
        fascActivations.append(maxActivation)

    return fascActivations

def getActivation(stimulusCurrents):

    fascActivations = getActivationFunctions(stimulusCurrents)

    targetFascicle = 0

    offTargetActivation = 0
    for fasc in range(39):
        if fasc != targetFascicle:
            offTargetActivation += fascActivations[fasc]
        else:
            targetActivation = fascActivations[fasc]

    return [offTargetActivation, targetActivation]

def main():

    params, results = di.read_parameters_file(parameters_file='params.in',results_file='results.out')

    analytic_input = get_input(params)

    t = time.time()

    try:
        stimuli = np.load('stimuli.npy')
        stimuli = np.vstack((stimuli,np.array(analytic_input)))
        np.save('stimuli.npy',stimuli)
    except:
        np.save('stimuli.npy',np.array(analytic_input))

    
    t = time.time()
    analytic_output = getActivationFunctions(analytic_input)

    t = time.time()

    try:
        output = np.load('output.npy')
        output = np.vstack((output,np.array(analytic_output)))
        np.save('output.npy',output)
    except:
        np.save('output.npy',np.array(analytic_output))

    t = time.time()
    
    results = pack_dakota_results(analytic_output,results)
    results.write()

if __name__=='__main__':
    main()
