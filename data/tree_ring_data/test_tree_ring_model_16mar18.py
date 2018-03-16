#!/usr/bin/env python
#Author: Craig Lage, UC Davis;
#Date: 16-Mar-18

# This program runs tests the GalSim tree ring model proposed for DC2
#from pylab import *
import numpy as np
import sys, time, subprocess
import galsim
#****************SUBROUTINES*****************

def LSST_DC2_Tree_Ring_Model():
    # Craig Lage UC Davis 16-Mar-18; cslage@ucdavis.edu
    # This function returns a tree ring model drawn from an anlytical function that was
    # derived based on tree ring data collected by Hye-Yun Park at BNL.  The data
    # used is in imSim/data/tree_ring_data, and a description of the method is in
    # imSim/data/tree_ring_data/Tree_Rings_13Feb18.pdf
    # Based on the data, 40% of the sensors are assumed to have 'bad' tree rings, with an
    # amplitude 10X greater that the 60% of the sensors that have 'good' tree rings.

    bad_tree_ring_fraction = 0.40 # Fraction of sensors that have 'bad' tree rings.
    A = 2E-3 # Baseline level of pixel deviation
    numfreqs1 = 15 # Number of spatial frequencies
    meank1 = 60.0 # Mean spatial freqency in pixels
    sigmak1 = 10.0 # Standard deviation of spatial freqency in pixels
    numfreqs2 = 5 # Number of spatial frequencies
    meank2 = 35.0 # Mean spatial freqency in pixels
    sigmak2 = 10.0 # Standard deviation of spatial freqency in pixels
    r_max = 8000.0 # Maximum extent of tree ring function in pixels
    dr = 3.0 # Step size of tree ring function in pixels
    npoints = int(r_max / dr) + 1 # Number of points in tree ring function

    # This generates the radial tree ring function.
    # Tree ring amplitude in pixels is given by A + B * r^4
    if np.random.rand() < bad_tree_ring_fraction:
        B = np.random.uniform(4.0E-17, 8.0E-17, 1)[0] # Amplitude factor in pixels^-3 
    else:
        B = np.random.uniform(4.0E-18, 8.0E-18, 1)[0] # Amplitude factor in pixels^-3 
    cfreqs = np.concatenate((np.random.normal(meank1, sigmak1, numfreqs1), np.random.normal(meank2, sigmak2, numfreqs2)))
    cphases = np.random.uniform(0.0, 2*np.pi, numfreqs1+numfreqs2)
    sfreqs = np.concatenate((np.random.normal(meank1, sigmak1, numfreqs1), np.random.normal(meank2, sigmak2, numfreqs2)))
    sphases = np.random.uniform(0.0, 2*np.pi, numfreqs1+numfreqs2)
    def tree_ring_radial_function(r):
        # This function is the integral of the data deviation function
        centroid_shift = 0.0
        for j, fval in enumerate(cfreqs):
            centroid_shift += np.sin(2*np.pi*(r/fval)+cphases[j]) * fval / (2.0*np.pi)
        for j, fval in enumerate(sfreqs):
            centroid_shift += -np.cos(2*np.pi*(r/fval)+sphases[j]) * fval / (2.0*np.pi)
        centroid_shift *= (A + B * r**4) * .01 # 0.01 factor is because data is in percent
        return centroid_shift
    tr_function = galsim.LookupTable.from_func(tree_ring_radial_function, x_min=0.0, x_max=r_max, npoints=npoints)

    # This part generates the tree ring origin.  Since the CCD can come from either of four positions,
    # the center is placed 1000 pixels away from one of the four corners, with a 'dither'
    # radius of 100 pixels.  The origin is assumed in the center of the detector, and it is
    # re-centered after being called.
    xy_dist_to_center = 3000.0 # Distance from center of detector to tree ring center
    dither_radius = 100.0
    corner = [np.random.choice([-1.0,1.0]), np.random.choice([-1.0,1.0])]
    origin = np.random.uniform(xy_dist_to_center - dither_radius, xy_dist_to_center + dither_radius, 2)
    center = np.array(corner) * np.array(origin)
    tr_center = galsim.PositionD(center[0], center[1])
    
    return (tr_center, tr_function)


#****************MAIN PROGRAM*****************

test = 1002
N_Per_Pixel = 1000
Photon_Repeats = 10
Nx = 509
Nsubx = 1
Ny = 2000
Nsuby = 1
TotalElec = Nx * Nsubx * Ny * Nsuby * N_Per_Pixel
pixel_scale = 0.2

newdir = subprocess.Popen('mkdir -p data/testrun_%d'%test, shell=True) 
subprocess.Popen.wait(newdir)

starttime = time.time()

im = galsim.ImageF(Nx, Ny, init_value=0, scale=pixel_scale)
for subx in [7]:
    for suby in [1]:

        xmin = float(Nx * subx)
        xmax = float(Nx * (subx + 1) - 1)
        ymin = float(Ny * suby)
        ymax = float(Ny * (suby + 1) - 1)
        NumElec = N_Per_Pixel * Nx * Ny / Photon_Repeats
        photons = galsim.PhotonArray(NumElec)
        photons.flux = 1.
        im.setCenter((xmin+xmax)/2.0, (ymin+ymax)/2.0)
        (tr_center, tr_function) = LSST_DC2_Tree_Ring_Model()
        new_center = galsim.PositionD(tr_center.x + im.center.x, tr_center.y + im.center.y)
        sensor = galsim.SiliconSensor(treering_func=tr_function,
                                   treering_center=new_center, nrecalc=2*TotalElec)

        elapsed = time.time() - starttime
        print "Setup took %f seconds"%elapsed
        sys.stdout.flush()
        starttime = time.time()

        for n in range(Photon_Repeats):
            #photons.x = galsim.DistDeviate(rng, lambda x:1, x_min=xmin, x_max=xmax, npoints=NumElec)
            # I tried using this, but it is more than 100X slower than the numpy function.
            photons.x = np.random.uniform(xmin, xmax, NumElec)
            photons.y = np.random.uniform(ymin, ymax, NumElec)

            print "Finished building photon list (%d, %d, %d)"%(subx, suby, n)
            elapsed = time.time() - starttime
            print "Photon list took %f seconds"%elapsed
            sys.stdout.flush()
            starttime = time.time()

            sensor.accumulate(photons, im.view())
            print "Finished accumulating photons (%d, %d, %d)"%(subx, suby, n)
            elapsed = time.time() - starttime
            print "Accumulation took %f seconds"%elapsed
            sys.stdout.flush()
            starttime = time.time()            
im.write('data/testrun_%d/image.fits'%test) # Write out a fits file with the spot data
print "Finished writing FITS file"

