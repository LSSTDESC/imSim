"""
This file defines the model classes that wrap PSFs from
galsim into the CatSim interface
"""

from builtins import object
import numpy
import galsim

__all__ = ["PSFbase", "DoubleGaussianPSF", "SNRdocumentPSF",
           "Kolmogorov_and_Gaussian_PSF"]

class PSFbase(object):
    """
    This is the base class for wrappers of GalSim's PSF classes.  To apply a PSF to GalSim images
    using the GalSim Instance Catalog and GalSim Interpreter, the user must define a daughter
    class of this class and instantiate it as the member variable self.PSF in the GalSim Instance Catalog.

    Any Daughter class of this class must have a member method _getPSF which accepts the coordinates
    xPupil and yPupil in arcseconds as kwargs.  This method will instantiate a psf object at those
    coordinates and return it.

    The method applyPSF is defined in this class and should not be overwritten.  It handles the task of actually
    convolving the PSF returned by _getPSF.

    Consult GalSim's documentation to see what kinds of PSFs are available.

    See the classes DoubleGaussianPSF and SNRdocumentPSF below for example implementations.

    See galSimCompoundGenerator.py and galSimStarGenerator.py for example usages.
    """

    def _getPSF(self, xPupil=None, yPupil=None):
        """
        If it had been implemented, this would return a GalSim PSF instantiation at the
        coordinates and wavelength specified and returned it to applyPSF.  As it is, this
        class has not been implemented and is left to the user to implement in Daughter
        classes of PSFbase.

        @param [in] xPupil the x coordinate on the pupil in arc seconds

        @param [in] yPupil the y coordinate on the pupil in arc seconds
        """

        raise NotImplementedError("There is not _getPSF for PSFbase; define a daughter class and define your own")

    def applyPSF(self, xPupil=None, yPupil=None, obj=None, **kwargs):
        """
        Apply the PSF to a GalSim GSObject

        This method accepts the x and y pupil coordinates in arc seconds as well
        as a GalSim GSObject.  The method calculates the PSF parameters based on xPupil
        and yPupil, constructs a Galsim GSObject corresponding to the PSF function, and convolves
        the PSF with the GSObject, returning the result of the convolution.

        In the case of point sources, this object returns the raw PSF, rather than attempting
        a convolution (since there is nothing to convolve with).

        @param [in] xPupil the x pupil coordinate in arc seconds

        @param [in] yPupil the y pupil coordinate in arc seconds

        @param [in] obj is a GalSim GSObject (an astronomical object) with which
        to convolve the PSF (optional)
        """

        #use the user-defined _getPSF method to calculate the PSF at these specific
        #coordinates and (optionally) wavelength
        psf = self._getPSF(xPupil=xPupil, yPupil=yPupil, **kwargs)

        if obj is None:
            #if there is no object, use a DeltaFunction as a point source
            obj = galsim.DeltaFunction()

        #convolve obj with the psf
        if isinstance(psf, galsim.Convolution):
            # If the psf is itself a Convolution object, convolve obj
            # with the individual components to ensure that the
            # obj_list of the returned obj lists those components
            # separately.
            return galsim.Convolution([obj] + psf.obj_list)
        else:
            return galsim.Convolve(obj, psf)

    def __eq__(self, rhs):
        """
        Compare types and underlying galsim ._cached_psf attributes for
        equality test.
        """
        return (type(self) == type(rhs)
                and self._cached_psf == rhs._cached_psf)

class DoubleGaussianPSF(PSFbase):
    """
    This is an example implementation of a wavelength- and position-independent
    Double Gaussian PSF.  See the documentation in PSFbase to learn how it is used.
    """

    def __init__(self, fwhm1=0.6, fwhm2=0.12, wgt1=1.0, wgt2=0.1):
        """
        @param [in] fwhm1 is the Full Width at Half Max of the first Gaussian in arcseconds

        @param [in] fwhm2 is the Full Width at Half Max of the second Gaussian in arcseconds

        @param [in] wgt1 is the dimensionless coefficient normalizing the first Gaussian

        @param [in] wgt2 is the dimensionless coefficient normalizing the second Gaussian

        The total PSF will be

        (wgt1 * G(sig1) + wgt2 * G(sig2))/(wgt1 + wgt2)

        where G(sigN) denotes a normalized Gaussian with a standard deviation that gives
        a Full Width at Half Max of fwhmN.  (Integrating a two-dimensional Gaussian, we find
        that sig = fwhm/2.355)

        Because this PSF depends on neither position nor wavelength, this __init__ method
        will instantiate a PSF and cache it.  It is this cached psf that will be returned
        whenever _getPSF is called in this class.
        """

        r1 = fwhm1/2.355
        r2 = fwhm2/2.355
        norm = 1.0/(wgt1 + wgt2)

        gaussian1 = galsim.Gaussian(sigma=r1)
        gaussian2 = galsim.Gaussian(sigma=r2)

        self._cached_psf = norm*(wgt1*gaussian1 + wgt2*gaussian2)

    def _getPSF(self, xPupil=None, yPupil=None, **kwargs):
        """
        Return a the PSF to be convolved with sources.

        @param [in] xPupil the x coordinate on the pupil in arc seconds

        @param [in] yPupil the y coordinate on the pupil in arc seconds

        Because this specific PSF depends on neither wavelength nor position,
        it will just return the cached PSF function.
        """
        return self._cached_psf



class SNRdocumentPSF(DoubleGaussianPSF):
    """
    This is an example implementation of a wavelength- and position-independent
    Double Gaussian PSF.  See the documentation in PSFbase to learn how it is used.

    This specific PSF comes from equation(30) of the signal-to-noise document (LSE-40),
    which can be found at

    www.astro.washington.edu/users/ivezic/Astr511/LSST_SNRdoc.pdf
    """

    def __init__(self, fwhm=0.6, pixel_scale=0.2, gsparams=None):
        """
        @param [in] fwhm is the Full Width at Half Max of the total PSF.  This is given in
        arcseconds.  The default value of 0.6 comes from a FWHM of 3 pixels with a pixel scale
        of 0.2 arcseconds per pixel.

        Because this PSF depends on neither position nor wavelength, this __init__ method
        will instantiate a PSF and cache it.  It is this cached psf that will be returned
        whenever _getPSF is called in this class.
        """

        #the expression below is derived by solving equation (30) of the signal-to-noise
        #document (www.astro.washington.edu/uses/ivezic/Astr511/LSST_SNRdoc.pdf)
        #for r at half the maximum of the PSF
        alpha = fwhm/2.3835

        eff_pixel_sigma_sq = pixel_scale*pixel_scale/12.0

        sigma = numpy.sqrt(alpha*alpha - eff_pixel_sigma_sq)
        gaussian1 = galsim.Gaussian(sigma=sigma, gsparams=gsparams)

        sigma = numpy.sqrt(4.0*alpha*alpha - eff_pixel_sigma_sq)
        gaussian2 = galsim.Gaussian(sigma=sigma, gsparams=gsparams)

        self._cached_psf = 0.909*(gaussian1 + 0.1*gaussian2)


class Kolmogorov_and_Gaussian_PSF(PSFbase):
    """
    This PSF class is based on David Kirkby's presentation to the DESC Survey Simulations
    working group on 23 March 2017.

    https://confluence.slac.stanford.edu/pages/viewpage.action?spaceKey=LSSTDESC&title=SSim+2017-03-23

    (you will need a SLAC Confluence account to access that link)
    """

    def __init__(self, airmass=1.2, rawSeeing=0.7, band='r', gsparams=None):
        """
        Parameters
        ----------
        airmass

        rawSeeing is the FWHM seeing at zenith at 500 nm in arc seconds
        (provided by OpSim)

        band is the bandpass of the observation [u,g,r,i,z,y]
        """
        # This code was provided by David Kirkby in a private communication

        wlen_eff = dict(u=365.49, g=480.03, r=622.20, i=754.06, z=868.21, y=991.66)[band]
        # wlen_eff is from Table 2 of LSE-40 (y=y2)

        FWHMatm = rawSeeing * (wlen_eff / 500.) ** -0.3 * airmass ** 0.6
        # From LSST-20160 eqn (4.1)

        FWHMsys = numpy.sqrt(0.25**2 + 0.3**2 + 0.08**2) * airmass ** 0.6
        # From LSST-20160 eqn (4.2)

        atm = galsim.Kolmogorov(fwhm=FWHMatm, gsparams=gsparams)
        sys = galsim.Gaussian(fwhm=FWHMsys, gsparams=gsparams)
        psf = galsim.Convolve((atm, sys))

        self._cached_psf = psf

    def _getPSF(self, xPupil=None, yPupil=None, **kwargs):
        return self._cached_psf
