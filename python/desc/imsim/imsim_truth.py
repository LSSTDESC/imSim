"""
Tools to compute the apparent magnitudes from the SED, mag_norm,
redshift, and reddening info in the input imsim object catalogs.
"""
from __future__ import absolute_import, print_function
import os
import copy
from collections import OrderedDict
import lsst.sims.photUtils as photUtils
import lsst.utils as lsstUtils

__all__ = ['ApparentMagnitudes']

# Create class-level attributes.
_bandpasses = dict()
for band_name in 'ugrizy':
    throughput_dir = os.path.join(lsstUtils.getPackageDir('throughputs'),
                                  'baseline', 'total_%s.dat' % band_name)
    _bandpasses[band_name] = photUtils.Bandpass()
    _bandpasses[band_name].readThroughput(throughput_dir)
_control_bandpass = photUtils.Bandpass()
_control_bandpass.imsimBandpass()

class ApparentMagnitudes(object):
    """
    Class to compute apparent magnitudes for a given rest-frame SED.
    """
    bps = _bandpasses
    control_bandpass = _control_bandpass
    def __init__(self, sed_name, max_mag=1000.):
        """
        Read in the unnormalized SED.
        """
        sed_dir = lsstUtils.getPackageDir('sims_sed_library')
        self.sed_unnormed = photUtils.Sed()
        self.sed_unnormed.readSED_flambda(os.path.join(sed_dir, sed_name))
        self.max_mag = max_mag

    def __call__(self, obj_pars, bands='ugrizy'):
        sed = copy.deepcopy(self.sed_unnormed)
        fnorm = sed.calcFluxNorm(obj_pars.magNorm, self.control_bandpass)
        sed.multiplyFluxNorm(fnorm)

        a_int, b_int = sed.setupCCMab()
        if obj_pars.internalAv != 0 or obj_pars.internalRv != 0:
            # Apply internal dust extinction.
            sed.addCCMDust(a_int, b_int, A_v=obj_pars.internalAv,
                           R_v=obj_pars.internalRv)

        if obj_pars.redshift > 0:
            sed.redshiftSED(obj_pars.redshift, dimming=True)

        if obj_pars.galacticAv != 0 or obj_pars.galacticRv != 0:
            # Apply Galactic extinction.
            sed.addCCMDust(a_int, b_int, A_v=obj_pars.galacticAv,
                           R_v=obj_pars.galacticRv)

        mags = OrderedDict()
        for band in bands:
            try:
                mags[band] = sed.calcMag(self.bps[band])
            except Exception as eObj:
                if str(eObj).startswith('This SED has no flux'):
                    mags[band] = self.max_mag
                else:
                    raise eObj

        return mags
