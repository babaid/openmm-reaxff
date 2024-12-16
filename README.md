[![GH Actions Status](https://github.com/openmm/openmm/workflows/CI/badge.svg)](https://github.com/openmm/openmm/actions?query=branch%3Amaster+workflow%3ACI)
[![Conda](https://img.shields.io/conda/v/conda-forge/openmm.svg)](https://anaconda.org/conda-forge/openmm)
[![Anaconda Cloud Badge](https://anaconda.org/conda-forge/openmm/badges/downloads.svg)](https://anaconda.org/conda-forge/openmm)

## OpenMM: A High Performance Molecular Dynamics Library

Introduction
------------

[OpenMM](http://openmm.org) is a toolkit for molecular simulation. It can be used either as a stand-alone application for running simulations, or as a library you call from your own code. It
provides a combination of extreme flexibility (through custom forces and integrators), openness, and high performance (especially on recent GPUs) that make it truly unique among simulation codes.  

Getting Help
------------

Need Help? Check out the [documentation](http://docs.openmm.org/) and [discussion forums](https://simtk.org/forums/viewforum.php?f=161).


## My contribution to OpenMM - A ReaxFF Plugin

This repository includes the reaxff plugin, which I developed to extend OpenMM's capabilities to be able to run Hybrid ReaxFF/MM simulations.
Due to limitations of my time (this project is part of my masters thesis) I used the already available PuReMD implementation of ReaxFF.


PuReMD-ReaxFF is limited to CPU computation only, some parts allow to use OpenMPI, and I also parallelized most of the code that processes data between OpenMM and ReaxFF. The useage of the plugin is straightforward, ReaxFF is implemented as an extra force. The user has to define reactive and non-reactive atoms for it to work and pay close attention to remove any not wanted classical MM forces and constraints from the ReaxFF region.

Once the system has been set up the simulation workflow is as usual, but the time step should be kept below 0.25fs, as recommended by ReaxFF.


For this initial version of the plugin I did not implement link atoms yet.



