"""
Code to add tree ring info to the CCD detectors
"""
import numpy as np
import galsim

__all__ = ['TreeRings']

class TreeRingsError(Exception):
    pass

class TreeRings():
    """
    # Craig Lage UC Davis 16-Mar-18; cslage@ucdavis.edu
    # This function returns a tree ring model drawn from an analytical function that was
    # derived based on tree ring data collected by Hye-Yun Park at BNL.  The data
    # used is in imSim/data/tree_ring_data, and a description of the method is in
    # imSim/data/tree_ring_data/Tree_Rings_13Feb18.pdf
    # Based on the data, 40% of the sensors are assumed to have 'bad' tree rings, with an
    # amplitude 10X greater that the 60% of the sensors that have 'good' tree rings.
    """

    def __init__(self, tr_filename):
        """
        Constructor.
        Craig Lage UC Davis 19-Mar-18; cslage@ucdavis.edu
        This code reads in a file with tree ring parameters from tr_filename
        and assigns a tree ring model to each sensor.
        """
        self.numfreqs = 20 # Number of spatial frequencies
        self.cfreqs = np.zeros([self.numfreqs]) # Cosine frequencies
        self.cphases = np.zeros([self.numfreqs])
        self.sfreqs = np.zeros([self.numfreqs]) # Sine frequencies
        self.sphases = np.zeros([self.numfreqs])
        self.A = 0.0
        self.B = 0.0
        self.r_max = 8000.0 # Maximum extent of tree ring function in pixels
        dr = 3.0 # Step size of tree ring function in pixels
        self.npoints = int(self.r_max / dr) + 1 # Number of points in tree ring function

        with open(tr_filename, 'r') as input_:
            self.lines = input_.readlines() # Contents of tree_ring_parameters file

        return
    
    def Read_DC2_Tree_Ring_Model(self, Rx, Ry, Sx, Sy):
        """
        This function finds the tree ring parameters for a given sensor
        and assigns a tree ring model to that sensor.
        """
        try:
            for i, line in enumerate(self.lines):
                if line.split()[0] == 'Rx':
                    items = self.lines[i+1].split()
                    if int(items[0]) == Rx and int(items[1]) == Ry and int(items[2]) == Sx and int(items[3]) == Sy:
                        Cx = float(items[4])
                        Cy = float(items[5])
                        self.A = float(items[6])
                        self.B = float(items[7])                    
                        for j in range(self.numfreqs):
                            freqitems = self.lines[i + 3 + j].split()
                            self.cfreqs[j] = float(freqitems[0])
                            self.cphases[j] = float(freqitems[1])                        
                            self.sfreqs[j] = float(freqitems[2])
                            self.sphases[j] = float(freqitems[3])                        
                        tr_function = galsim.LookupTable.from_func(self.tree_ring_radial_function, x_min=0.0,\
                                                                   x_max=self.r_max, npoints=self.npoints)
                        tr_center = galsim.PositionD(Cx, Cy)
                        return (tr_center, tr_function)
                    else:
                        continue
                else:
                    continue
            # If we reach here, the (Rx, Ry, Sx, Sy) combination was not found.
            raise TreeRingsError("Failed to read tree ring parameters for Rx=%d, Ry=%d, Sx=%d, Sy=%d"%(Rx, Ry, Sx, Sy))
        except:
            raise TreeRingsError("Failed to read tree ring parameters for Rx=%d, Ry=%d, Sx=%d, Sy=%d"%(Rx, Ry, Sx, Sy))
        return ()

    def tree_ring_radial_function(self, r):
        # This function defines the tree ring radial function
        centroid_shift = 0.0
        for j, fval in enumerate(self.cfreqs):
            centroid_shift += np.sin(2*np.pi*(r/fval)+self.cphases[j]) * fval / (2.0*np.pi)
        for j, fval in enumerate(self.sfreqs):
            centroid_shift += -np.cos(2*np.pi*(r/fval)+self.sphases[j]) * fval / (2.0*np.pi)
        centroid_shift *= (self.A + self.B * r**4) * .01 # 0.01 factor is because data is in percent
        return centroid_shift

    
