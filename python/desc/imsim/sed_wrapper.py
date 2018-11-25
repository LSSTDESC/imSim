"""
Wrapper code for lsst.sims.photUtils.Sed to defer reading of SED data and
related calculations until they are needed in order to save memory.
"""
import copy
import numpy as np
import lsst.sims.photUtils as sims_photUtils

__all__ = ['SedWrapper']


class CCMmodel:
    """
    Helper class to cache a(x) and b(x) arrays evaluated on wavelength
    grids for intrinsic and Galactic extinction calculations.
    """
    def __init__(self):
        self.wavelen = dict()
        self.a = dict()
        self.b = dict()
        for ext_type in ('intrinsic', 'Galactic'):
            self.wavelen[ext_type] = None

    def add_dust(self, sed_obj, Av, Rv, ext_type):
        """
        Add dust reddening to the SED object.

        Parameters
        ----------
        sed_obj: lsst.sims.photUtils.Sed
            SED object to which to add the reddening.
        Av: float
            Extinction coefficient.
        Rv: float
            Extinction coefficient.
        ext_type: str
            Extinction type: 'intrinsic' or 'Galactic'
        """
        if (self.wavelen[ext_type] is None or
            not np.array_equal(sed_obj.wavelen, self.wavelen[ext_type])):
            self.a[ext_type], self.b[ext_type] = sed_obj.setupCCM_ab()
            self.wavelen[ext_type] = copy.deepcopy(sed_obj.wavelen)
        sed_obj.addDust(self.a[ext_type], self.b[ext_type],
                        A_v=Av, R_v=Rv)


class SedWrapper:
    """
    Wrapper class to defer reading of SED data and related calculations
    until they are needed in order to avoid excess memory usage.
    """
    shared_resources = dict(ccm_model=CCMmodel())
    def __init__(self, sed_file, mag_norm, redshift, iAv, iRv, gAv, gRv,
                 bp_dict):
        self.sed_file = sed_file
        self.mag_norm = mag_norm
        self.redshift = redshift
        self.iAv = iAv
        self.iRv = iRv
        self.gAv = gAv
        self.gRv = gRv
        self.bp_dict = bp_dict
        self._sed_obj = None

    @property
    def sed_obj(self):
        if self._sed_obj is None:
            self._compute_SED()
        return self._sed_obj

    @property
    def wavelen(self):
        return self.sed_obj.wavelen

    @property
    def flambda(self):
        return self.sed_obj.flambda

    def delete_sed_obj(self):
        "Delete the Sed object to release resources."
        del self._sed_obj
        self._sed_obj = None

    def calcADU(self, bandpass, photParams):
        "Calculate the ADU for the specified bandpass."
        return self.sed_obj.calcADU(bandpass, photParams)

    def _compute_SED(self):
        self._sed_obj = sims_photUtils.Sed()
        self._sed_obj.readSED_flambda(self.sed_file)
        fnorm = sims_photUtils.getImsimFluxNorm(self._sed_obj, self.mag_norm)
        self._sed_obj.multiplyFluxNorm(fnorm)
        if self.iAv != 0:
            self.shared_resources['ccm_model'].add_dust(self._sed_obj, self.iAv,
                                                        self.iRv, 'intrinsic')
        if self.redshift != 0:
            self._sed_obj.redshiftSED(self.redshift, dimming=True)
        self._sed_obj.resampleSED(wavelen_match=self.bp_dict.wavelenMatch)
        if self.gAv != 0:
            self.shared_resources['ccm_model'].add_dust(self._sed_obj, self.gAv,
                                                        self.gRv, 'Galactic')
