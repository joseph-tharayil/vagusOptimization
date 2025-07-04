# Inference of vagus nerve parameters through optimization

This repository provides a proof-of-concept for the inference of vagus nerve structure through the use of optimization of model parameters to minimize the error between a ground truth and *in silico* evoked compoint action potential. The model of the eCAP is taken from the paper [Simulation Insights on the Compound Action Potential](https://www.biorxiv.org/content/10.1101/2024.10.16.618681v1), and this repository will allow the replication of the inference results from that paper (which are not yet published in the current preprint). 

# Instructions

## System requirements
This code is configured to run on Windows. In principle it can be run on Linux or MacOS with some configuration, but we have not tested it on those systems.

## Dependencies
This code depends on [Dakota](https://dakota.sandia.gov/), as well as on the vagus nerve models from <https://github.com/joseph-tharayil/vagusNerve>

## Replicating results from the paper
Our goal is to infer the shape and scale parameters of the distribution of myelinated afferent fibers in the vagus nerve. First we generate a ``reference" eCAP, by running the script `groundTruth/runGroundTruth.py`. This script runs the vagus nerve model with known parameters. You must configure this script by providing the path to the inputs to the vagus nerve model, namely the titration output file and the sensitivity files, both of which are described in the vagus nerve model [Readme](https://github.com/joseph-tharayil/vagusNerve/blob/main/README.md).

Then, in the `optimization` folder, run `dakota dakota_vagus.in`. This iteratively calls the vagus nerve model from the script `vagusOpt.py`. As above, you will need to set the paths to the model inputs in this folder.

To replicate the figure in the paper, run the notebook `plotDakotaSignals.ipynb` in this folder.

# Citation
If you use this software, we kindly ask you to cite the following publication:
[Tharayil et al. Simulation Insights on the Compound Action Potential. *bioRxiv, (2024)*](https://www.biorxiv.org/content/10.1101/2024.10.16.618681v1.abstract)
