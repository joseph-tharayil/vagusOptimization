import dakota.interfacing as di
import numpy as np
from vagusOpt import run_vagus_nerve

def get_input(params):

    fascicleParams = []

    for fascicle in range(39):

        fascicleParams.append( params['maffM'+str(fascicle),'maffS'+str(fascicle),'meffM'+str(fascicle),'meffS'+str(fascicle)] )

    return np.array(fascicleParams)

def pack_dakota_results(analytic_output,results):

    for i, label in enumerate(results):
        if results[label].asv.function:
            results[label].function = analytic_output

    return results

def main():

    params, results = di.read_parameters_file(parameters_file='params.in',results_file='results.out')

    analytic_input = get_input(params)

    distribution_params = {'maff':{'diameterParams':analytic_input[:,:2], 'fiberTypeFractions':None},'meff':{'diameterParams':analytic_input[:,2:], 'fiberTypeFractions':None}}

    analytic_output = run_vagus_nerve(distribution_params)

    results = pack_dakota_results(analytic_output,results)
    results.write()

if __name__=='__main__':
    main()
