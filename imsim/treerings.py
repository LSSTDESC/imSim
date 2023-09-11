
import os
import numpy as np
import galsim
from galsim.config import InputLoader, RegisterInputType, RegisterValueType
from .meta_data import data_dir
import warnings

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
    _req_params = {'file_name' : str}
    _opt_params = {'only_dets' : list}

    def __init__(self, file_name, only_dets=None, logger=None, defer_load=True):
        """
        Constructor.
        Craig Lage UC Davis 19-Mar-18; cslage@ucdavis.edu
        This code reads in a file with tree ring parameters from file_name
        and assigns a tree ring model to each sensor.
        """
        self.logger = galsim.config.LoggerWrapper(logger)
        self.file_name = file_name
        if not os.path.isfile(self.file_name):
            # Then check for this name in the imsim data_dir
            self.file_name = os.path.join(data_dir, 'tree_ring_data', file_name)
        if not os.path.isfile(self.file_name):
            raise OSError("TreeRing file %s not found"%file_name)
        self.only_dets = only_dets
        self.numfreqs = 20 # Number of spatial frequencies
        self.cfreqs = np.zeros([self.numfreqs]) # Cosine frequencies
        self.cphases = np.zeros([self.numfreqs])
        self.sfreqs = np.zeros([self.numfreqs]) # Sine frequencies
        self.sphases = np.zeros([self.numfreqs])
        self.r_max = 8000.0 # Maximum extent of tree ring function in pixels
        dr = 3.0 # Step size of tree ring function in pixels
        self.npoints = int(self.r_max / dr) + 1 # Number of points in tree ring function

        # Make a dict indexed by det_name (a string that looks like Rxx_Syy)
        self.info = {}
        if not defer_load:
            self.fill_dict(only_dets=only_dets)

    def fill_dict(self, only_dets=None):
        """Read the tree rings file and save the information therein to a dict, self.info.
        """
        self.logger.warning("Reading TreeRing file %s", self.file_name)
        if only_dets:
            self.logger.info("Reading in det_names: %s", only_dets)
        with open(self.file_name, 'r') as input_:
            lines = input_.readlines()

        xCenterPix = 2048.5
        yCenterPix = 2048.5

        for i, line in enumerate(lines):
            if not line.startswith('Rx'): continue
            items = lines[i+1].split()

            det_name = "R%s%s_S%s%s"%(tuple(items[:4]))
            if only_dets and det_name not in only_dets: continue
            self.logger.info(det_name)

            Cx = float(items[4]) + xCenterPix
            Cy = float(items[5]) + yCenterPix
            center = galsim.PositionD(Cx, Cy)

            # Temporary instance variables used by tree_ring_radial_function
            self.A = float(items[6])
            self.B = float(items[7])
            for j in range(self.numfreqs):
                freqitems = lines[i + 3 + j].split()
                self.cfreqs[j] = float(freqitems[0])
                self.cphases[j] = float(freqitems[1])
                self.sfreqs[j] = float(freqitems[2])
                self.sphases[j] = float(freqitems[3])
            func = galsim.LookupTable.from_func(self.tree_ring_radial_function,
                                                x_min=0.0, x_max=self.r_max,
                                                npoints=self.npoints)
            self.info[det_name] = (center, func)

    def get_center(self, det_name):
        if det_name not in self.info:
            self.fill_dict((det_name,))
        if det_name in self.info:
            return self.info[det_name][0]
        else:
            warnings.warn("No treering information available for %s.  Setting treering_center to PositionD(0, 0)." % det_name)
            return galsim.PositionD(0,0)

    def get_func(self, det_name):
        if det_name not in self.info:
            self.fill_dict((det_name,))
        if det_name in self.info:
            return self.info[det_name][1]
        else:
            warnings.warn("No treering information available for %s.  Setting treering_func to None." % det_name)
            return None

    def tree_ring_radial_function(self, r):
        """
        This function defines the tree ring distortion of the pixels as
        a radial function.

        Parameters
        ----------
        r: float
            Radial coordinate from the center of the tree ring structure
            in units of pixels.
        """
        centroid_shift = 0.0
        for j, fval in enumerate(self.cfreqs):
            centroid_shift += np.sin(2*np.pi*(r/fval)+self.cphases[j]) * fval / (2.0*np.pi)
        for j, fval in enumerate(self.sfreqs):
            centroid_shift += -np.cos(2*np.pi*(r/fval)+self.sphases[j]) * fval / (2.0*np.pi)
        centroid_shift *= (self.A + self.B * r**4) * .01 # 0.01 factor is because data is in percent
        return centroid_shift

def TreeRingCenter(config, base, value_type):
    """Return the tree ring center for the current det_name.
    """
    tree_rings = galsim.config.GetInputObj('tree_rings', config, base, 'TreeRingCenter')
    req = { 'det_name' : str }
    kwargs, safe = galsim.config.GetAllParams(config, base, req=req)
    det_name = kwargs['det_name']
    center = tree_rings.get_center(det_name)
    return center, safe

def TreeRingFunc(config, base, value_type):
    """Return the tree ring func for the current det_name.
    """
    tree_rings = galsim.config.GetInputObj('tree_rings', config, base, 'TreeRingCenter')
    req = { 'det_name' : str }
    kwargs, safe = galsim.config.GetAllParams(config, base, req=req)
    det_name = kwargs['det_name']
    func = tree_rings.get_func(det_name)
    return func, safe


RegisterInputType('tree_rings', InputLoader(TreeRings, takes_logger=True))
RegisterValueType('TreeRingCenter', TreeRingCenter, [galsim.PositionD], input_type='tree_rings')
RegisterValueType('TreeRingFunc', TreeRingFunc, [galsim.LookupTable], input_type='tree_rings')
