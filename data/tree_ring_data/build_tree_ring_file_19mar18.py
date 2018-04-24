#!/usr/bin/env python
#Author: Craig Lage, UC Davis;
#Date: 19-Mar-18

# This program builds a file containing a set of tree ring models for the
# detectors used in an imSim image simulation.  This file will be read into imSim
# This guarantees that the tree ring model is the same each time imSim is run
import numpy as np
import sys, time, subprocess
#****************SUBROUTINES*****************

def LSST_DC2_Tree_Ring_Model(seed):
    # Craig Lage UC Davis 16-Mar-18; cslage@ucdavis.edu
    # This function returns a tree ring model drawn from an anlytical function that was
    # derived based on tree ring data collected by Hye-Yun Park at BNL.  The data
    # used is in imSim/data/tree_ring_data, and a description of the method is in
    # imSim/data/tree_ring_data/Tree_Rings_13Feb18.pdf
    # Based on the data, 40% of the sensors are assumed to have 'bad' tree rings, with an
    # amplitude 10X greater that the 60% of the sensors that have 'good' tree rings.

    rng = np.random.RandomState(seed)

    bad_tree_ring_fraction = 0.40 # Fraction of sensors that have 'bad' tree rings.
    A = 2E-3 # Baseline level of pixel deviation
    numfreqs1 = 15 # Number of spatial frequencies
    meank1 = 60.0 # Mean spatial freqency in pixels
    sigmak1 = 10.0 # Standard deviation of spatial freqency in pixels
    numfreqs2 = 5 # Number of spatial frequencies
    meank2 = 35.0 # Mean spatial freqency in pixels
    sigmak2 = 10.0 # Standard deviation of spatial freqency in pixels

    # This generates the parameters for radial tree ring function.
    # Tree ring amplitude in pixels is given by A + B * r^4
    # a list of spatial frequencies and phases is generated for each detector
    if rng.rand() < bad_tree_ring_fraction:
        B = rng.uniform(4.0E-17, 8.0E-17, 1)[0] # Amplitude factor in pixels^-3 
    else:
        B = rng.uniform(4.0E-18, 8.0E-18, 1)[0] # Amplitude factor in pixels^-3 
    cfreqs = np.concatenate((rng.normal(meank1, sigmak1, numfreqs1), rng.normal(meank2, sigmak2, numfreqs2)))
    cphases = rng.uniform(0.0, 2*np.pi, numfreqs1+numfreqs2)
    sfreqs = np.concatenate((rng.normal(meank1, sigmak1, numfreqs1), rng.normal(meank2, sigmak2, numfreqs2)))
    sphases = rng.uniform(0.0, 2*np.pi, numfreqs1+numfreqs2)

    # This part generates the tree ring origin.  Since the CCD can come from either of four positions,
    # the center is placed 1000 pixels away from one of the four corners, with a 'dither'
    # radius of 100 pixels.  The origin is assumed in the center of the detector, and it is
    # re-centered after being called.
    xy_dist_to_center = 3000.0 # Distance from center of detector to tree ring center
    dither_radius = 100.0
    corner = [rng.choice([-1.0,1.0]), rng.choice([-1.0,1.0])]
    origin = rng.uniform(xy_dist_to_center - dither_radius, xy_dist_to_center + dither_radius, 2)
    center = np.array(corner) * np.array(origin)

    return (A, B, center, cfreqs, cphases, sfreqs, sphases)

if __name__ == '__main__':
    rafts = [(1,0), (2,0), (3,0),\
             (0,1), (1,1), (2,1), (3,1), (4,1),\
             (0,2), (1,2), (2,2), (3,2), (4,2),\
             (0,3), (1,3), (2,3), (3,3), (4,3),\
             (1,4), (2,4), (3,4)]
    sensors = [(0,0), (0,1), (0,2),\
               (1,0), (1,1), (1,2),\
               (2,0), (2,1), (2,2)]


    filename = 'tree_ring_parameters_2018-04-24.txt'
    seed = 18419
    file = open(filename, 'w')
    for raft in rafts:
        for sensor in sensors:
            (A, B, center, cfreqs, cphases, sfreqs, sphases) = LSST_DC2_Tree_Ring_Model(seed)
            file.write('Rx      Ry      Sx      Sy           Cx       Cy       A           B\n')
            file.write('%d\t%d\t%d\t%d\t%9.1f%9.1f%8.3g\t%8.3g\n'%(raft[0],raft[1],sensor[0],sensor[1],center[0],center[1],A,B))
            file.write('   CosFreq         CosPhase        SinFreq         SinPhase\n')
            for i in range(len(cfreqs)):
                file.write('%9.1f\t%9.1f\t%9.1f\t%9.1f\n'%(cfreqs[i],cphases[i],sfreqs[i],sphases[i]))

    file.close()
