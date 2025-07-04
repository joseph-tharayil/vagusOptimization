# Inference of vagus nerve parameters through optimization

This repository provides a proof-of-concept for the inference of vagus nerve structure through the use of optimization of model parameters to minimize the error between a ground truth and *in silico* evoked compoint action potential. The model of the eCAP is taken from the paper [Simulation Insights on the Compound Action Potential](https://www.biorxiv.org/content/10.1101/2024.10.16.618681v1), and this repository will allow the replication of the inference results from that paper (which are not yet published in the current preprint). 

# Instructions

## System requirements
This code is configured to run on Windows. In principle it can be run on Linux or MacOS with some configuration, but we have not tested it on those systems.

## Dependencies
This code depends on [Dakota](https://dakota.sandia.gov/), as well as on the vagus nerve models from <https://github.com/joseph-tharayil/vagusNerve>

## Replicating results from the paper
Our goal is to infer the shape and scale parameters of the 


# Citation
If you use this software, we kindly ask you to cite the following publication:
[Tharayil et al. Simulation Insights on the Compound Action Potential. *bioRxiv, (2024)*](https://www.biorxiv.org/content/10.1101/2024.10.16.618681v1.abstract)
