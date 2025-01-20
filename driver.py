import dakota.interfacing as di
import numpy as np
from vagusNerve import run_vagus_nerve
import sys

def get_input(params):

    fiberTypeParams = []

    for fiberType in range(2):

        fiberTypeParams.append( [params['x'+str(fiberType)],params['y'+str(fiberType)]] )

    return fiberTypeParams

def pack_dakota_results(analytic_output,results):

    for i, label in enumerate(results):
        if results[label].asv.function:
            results[label].function = analytic_output

    return results

def main(input,output):

    params, results = di.read_parameters_file(parameters_file=input,results_file=output)

    analytic_input = get_input(params)

    analytic_output = run_vagus_nerve(analytic_input)

    results = pack_dakota_results(analytic_output,results)
    results.write()

if __name__=='__main__':
    
    input = sys.argv[1]
    output = sys.argv[2]
    main(input,output)
