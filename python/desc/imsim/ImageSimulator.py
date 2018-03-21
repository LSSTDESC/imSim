"""
Code to manage the parallel simulation of multiple sensors
using the multiprocessing module.
"""
import os
import multiprocessing
from lsst.afw.cameraGeom import WAVEFRONT, GUIDER
from lsst.sims.photUtils import BandpassDict
from lsst.sims.GalSimInterface import make_galsim_detector
from lsst.sims.GalSimInterface import SNRdocumentPSF
from lsst.sims.GalSimInterface import Kolmogorov_and_Gaussian_PSF
from lsst.sims.GalSimInterface import LSSTCameraWrapper
from lsst.sims.GalSimInterface import GalSimInterpreter
from desc.imsim.skyModel import ESOSkyModel
import desc.imsim

__all__ = ['ImageSimulator']

class ImageSimulator:
    """
    Class to manage the parallel simulation of multiple sensors
    using the multiprocessing module.
    """
    def __init__(self, instcat, psf, numRows=None, config=None, seed=267,
                 outdir='fits'):
        """
        Parameters
        ----------
        instcat: str
            The instance catalog for the desired visit.
        psf: lsst.sims.GalSimInterface.PSFbase subclass PSF to use for
            drawing objects.  A single instance is used by all of the
            GalSimInterpreters so that memory for any atmospheric
            screen data can be shared among the processes.
        numRows: int [None]
            The number of rows to read in from the instance catalog.
            If None, then all rows will be read in.
        config: str [None]
            Filename of config file to use.  If None, then the default
            config will be used.
        seed: int [267]
            Random number seed to pass to the GalSimInterpreter objects.
        """
        self.config = desc.imsim.read_config(config)
        self.psf = psf
        self.outdir = outdir
        self.obs_md, self.phot_params, sources \
            = desc.imsim.parsePhoSimInstanceFile(instcat, numRows=numRows)
        self.gs_obj_arr = sources[0]
        self.gs_obj_dict = sources[1]
        self.camera_wrapper = LSSTCameraWrapper()
        self._make_gs_interpreters(seed)

    def _make_gs_interpreters(self, seed):
        """
        Create a separate GalSimInterpreter for each chip so that they
        can be run in parallel.

        TODO: Find a good way to pass a different seed to each
        gs_interpreter or have them share the random number generator.
        """
        bp_dict = BandpassDict.loadTotalBandpassesFromFiles(bandpassNames=self.obs_md.bandpass)
        noise_and_background \
            = ESOSkyModel(self.obs_md, addNoise=True, addBackground=True)
        self.gs_interpreters = dict()
        for det in self.camera_wrapper.camera:
            det_type = det.getType()
            det_name = det.getName()
            if det_type == WAVEFRONT or det_type == GUIDER:
                continue
            gs_det = make_galsim_detector(self.camera_wrapper, det_name,
                                          self.phot_params, self.obs_md)
            self.gs_interpreters[det_name] \
                = GalSimInterpreter(obs_metadata=self.obs_md,
                                    epoch=2000.0,
                                    detectors=[gs_det],
                                    bandpassDict=bp_dict,
                                    noiseWrapper=noise_and_background,
                                    seed=seed)
            self.gs_interpreters[det_name].setPSF(PSF=self.psf)

    def run(self, processes=1):
        """
        Use multiprocessing module to run chips in parallel.
        """
        if processes == 1:
            # Run serially
            for det_name, gs_interpreter in self.gs_interpreters.items():
                simulate_sensor = SimulateSensor(gs_interpreter, self)
                gs_objects = self.gs_obj_dict[det_name]
                simulate_sensor(gs_objects)
        else:
            # Use multiprocessing
            pool = multiprocessing.Pool(processes=processes)
            results = []
            for det_name, gs_interpreter in self.gs_interpreters.items():
                simulate_sensor = SimulateSensor(gs_interpreter, self)
                gs_objects = self.gs_obj_dict[det_name]
                results.append(pool.apply_async(simulate_sensor, (gs_objects,)))
            pool.close()
            pool.join()
            for res in results:
                res.get()

class SimulateSensor:
    """
    Functor class to serve as the callback for simulating sensors
    in parallel using the multiprocessing module.
    """
    def __init__(self, gs_interpreter, image_simulator):
        """
        Parameters
        ----------
        gs_interpreter: lsst.sims.GalSimInterface.GalSimInterpreter
            The interpreter object for a given sensor.
        """
        self.gs_interpreter = gs_interpreter
        self.image_simulator = image_simulator

    def __call__(self, gs_objects):
        """
        Draw the objects using self.gs_interpreter.

        Parameters
        ----------
        gs_objects: list of lsst.sims.GalSimInterface.GalSimDetector objects
            This list should be restricted to the objects for the
            corresponding sensor.
        """
        for gs_obj in gs_objects:
            if gs_obj.uniqueId in self.gs_interpreter.drawn_objects:
                continue
            self.gs_interpreter.drawObject(gs_obj)
        desc.imsim.add_cosmic_rays(self.gs_interpreter,
                                   self.image_simulator.phot_params)
        outdir = self.image_simulator.outdir
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        prefix = self.image_simulator.config['persistence']['eimage_prefix']
        self.gs_interpreter.writeImages(nameRoot=os.path.join(outdir, prefix)
                                        + str(self.image_simulator.obs_md.OpsimMetaData['obshistID']))
