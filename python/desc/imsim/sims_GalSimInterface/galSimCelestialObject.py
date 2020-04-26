import numpy as np
from lsst.sims.utils import arcsecFromRadians

__all__ = ["GalSimCelestialObject"]

class GalSimCelestialObject:
    """
    This is a class meant to carry around all of the data required by
    the GalSimInterpreter to draw an image of a single object.  The idea
    is that all of the drawing functions in the GalSimInterpreter will
    just take a GalSimCelestialObject as an argument, rather than taking
    a bunch of different arguments, one for each datum.
    """

    def __init__(self, galSimType, xPupil, yPupil,
                 halfLightRadius, minorAxis, majorAxis, positionAngle,
                 sindex, sed, bp_dict, photParams, npoints,
                 fits_image_file, pixel_scale, rotation_angle,
                 gamma1=0, gamma2=0, kappa=0, uniqueId=None):
        """
        @param [in] galSimType is a string, either 'pointSource', 'sersic',
        'RandomWalk', or 'FitsImage' denoting the shape of the object

        @param [in] xPupil is the x pupil coordinate of the object in radians

        @param [in] yPupil is the y pupil coordinate of the object in radians

        @param [in] halfLightRadius is the halfLightRadius of the
        object in radians

        @param [in] minorAxis is the semi-minor axis of the object in radians

        @param [in] majorAxis is the semi-major axis of the object in radians

        @param [in] positionAngle is the position angle of the object
        in radians

        @param [in] sindex is the sersic index of the object

        @param [in] sed is an instantiation of lsst.sims.photUtils.Sed
        representing the SED of the object (with all normalization,
        dust extinction, redshifting, etc. applied)

        @param [in] bp_dict is an instantiation of
        lsst.sims.photUtils.BandpassDict representing the bandpasses
        of this telescope

        @param [in] photParams is an instantioation of
        lsst.sims.photUtils.PhotometricParameters representing the physical
        parameters of the telescope that inform simulated photometry

            Together, sed, bp_dict, and photParams will be used to create
            a dict that maps bandpass name to electron counts for this
            celestial object.

        @param [in] npoints is the number of point sources in a RandomWalk

        @param [in] fits_image_file is the filename for the FitsImage

        @param [in] pixel_scale is the pixel size in arcsec of the FitsImage

        @param [in] rotation_angle is the rotation angle in degrees for
        the FitsImage

        @param [in] gamma1 is the real part of the WL shear parameter

        @param [in] gamma2 is the imaginary part of the WL shear parameter

        @param [in] kappa is the WL convergence parameter

        @param [in] uniqueId is an int storing a unique identifier for
        this object
        """
        self._uniqueId = uniqueId
        self._galSimType = galSimType
        self._fits_image_file = fits_image_file

        # The galsim.lens(...) function wants to be passed reduced
        # shears and magnification, so convert the WL parameters as
        # defined in phosim instance catalogs to these values.  See
        # https://github.com/GalSim-developers/GalSim/blob/releases/1.4/doc/GalSim_Quick_Reference.pdf
        # and Hoekstra, 2013, http://lanl.arxiv.org/abs/1312.5981
        g1 = gamma1/(1. - kappa)   # real part of reduced shear
        g2 = gamma2/(1. - kappa)   # imaginary part of reduced shear
        mu = 1./((1. - kappa)**2 - (gamma1**2 + gamma2**2)) # magnification

        self._fluxDict = {}
        self._sed = sed
        self._bp_dict = bp_dict
        self._photParams = photParams

        # Put all the float values into a numpy array for better
        # memory usage.  (Otherwise, python allocates a shared pointer
        # for each one of these 15 values, which adds up to a
        # significant memory overhead.)
        self._float_values = np.array(
            [
                xPupil,                     # xPupilRadians, 0
                arcsecFromRadians(xPupil),  # xPupilArcsec, 1
                yPupil,                     # yPupilRadians, 2
                arcsecFromRadians(yPupil),  # yPupilArcsec, 3
                halfLightRadius,            # halfLightRadiusRadians, 4
                arcsecFromRadians(halfLightRadius), # halfLightRadiusArcsec, 5
                minorAxis,                  # minorAxisRadians, 6
                majorAxis,                  # majorAxisRadians, 7
                positionAngle,              # positionAngleRadians, 8
                sindex,                     # sindex, 9
                pixel_scale,                # pixel_scale, 10
                rotation_angle,             # rotation_angle, 11
                g1,                         # g1, 12
                g2,                         # g2, 13
                mu,                         # mu, 14
                npoints                     # npoints, 15
            ], dtype=float)

        # XXX: We could probably get away with np.float32 for these,
        #      but the main advantage is from only allocating the
        #      actual memory, and not the python pointers to the
        #      memory.  So not so much more gain to be had from
        #      switching to single precision.

    #
    # First properties for all the non-float values.
    #

    @property
    def sed(self):
        return self._sed

    @property
    def uniqueId(self):
        return self._uniqueId

    # XXX: I'd recommend removing all these setters and just let
    #      python raise an AttributeError if people try to set the
    #      values, rather than raising a RuntimeError manually.  -- MJ
    @uniqueId.setter
    def uniqueId(self, value):
        raise RuntimeError("You should not be setting the unique id on the fly; " \
                           + "just instantiate a new GalSimCelestialObject")

    @property
    def galSimType(self):
        return self._galSimType

    @galSimType.setter
    def galSimType(self, value):
        raise RuntimeError("You should not be setting galSimType on the fly; " \
                           + "just instantiate a new GalSimCelestialObject")

    @property
    def npoints(self):
        return int(self._float_values[15]+0.5)

    @npoints.setter
    def npoints(self, value):
        raise RuntimeError("You should not be setting npoints on the fly; " \
        + "just instantiate a new GalSimCelestialObject")

    @property
    def fits_image_file(self):
        return self._fits_image_file

    @fits_image_file.setter
    def fits_image_file(self, value):
        raise RuntimeError("You should not be setting fits_image_file on the fly; "
                           "just instantiate a new GalSimCelestialObject")


    #
    # The float values are accessed from the numpy array called
    # self._float_values.
    #

    @property
    def xPupilRadians(self):
        return self._float_values[0]

    @xPupilRadians.setter
    def xPupilRadians(self, value):
        raise RuntimeError("You should not be setting xPupilRadians on the fly; " \
        + "just instantiate a new GalSimCelestialObject")


    @property
    def xPupilArcsec(self):
        return self._float_values[1]

    @xPupilArcsec.setter
    def xPupilArcsec(self, value):
        raise RuntimeError("You should not be setting xPupilArcsec on the fly; " \
        + "just instantiate a new GalSimCelestialObject")


    @property
    def yPupilRadians(self):
        return self._float_values[2]

    @yPupilRadians.setter
    def yPupilRadians(self, value):
        raise RuntimeError("You should not be setting yPupilRadians on the fly; " \
        + "just instantiate a new GalSimCelestialObject")


    @property
    def yPupilArcsec(self):
        return self._float_values[3]

    @yPupilArcsec.setter
    def yPupilArcsec(self, value):
        raise RuntimeError("You should not be setting yPupilArcsec on the fly; " \
        + "just instantiate a new GalSimCelestialObject")


    @property
    def halfLightRadiusRadians(self):
        return self._float_values[4]

    @halfLightRadiusRadians.setter
    def halfLightRadiusRadians(self, value):
        raise RuntimeError("You should not be setting halfLightRadiusRadians on the fly; " \
        + "just instantiate a new GalSimCelestialObject")


    @property
    def halfLightRadiusArcsec(self):
        return self._float_values[5]

    @halfLightRadiusArcsec.setter
    def halfLightRadiusArcsec(self, value):
        raise RuntimeError("You should not be setting halfLightRadiusArcsec on the fly; " \
        + "just instantiate a new GalSimCelestialObject")


    @property
    def minorAxisRadians(self):
        return self._float_values[6]

    @minorAxisRadians.setter
    def minorAxisRadians(self, value):
        raise RuntimeError("You should not be setting minorAxisRadians on the fly; " \
        + "just instantiate a new GalSimCelestialObject")


    @property
    def majorAxisRadians(self):
        return self._float_values[7]

    @majorAxisRadians.setter
    def majorAxisRadians(self, value):
        raise RuntimeError("You should not be setting majorAxisRadians on the fly; " \
        + "just instantiate a new GalSimCelestialObject")


    @property
    def positionAngleRadians(self):
        return self._float_values[8]

    @positionAngleRadians.setter
    def positionAngleRadians(self, value):
        raise RuntimeError("You should not be setting positionAngleRadians on the fly; " \
        + "just instantiate a new GalSimCelestialObject")


    @property
    def sindex(self):
        return self._float_values[9]

    @sindex.setter
    def sindex(self, value):
        raise RuntimeError("You should not be setting sindex on the fly; " \
        + "just instantiate a new GalSimCelestialObject")

    @property
    def pixel_scale(self):
        return self._float_values[10]

    @pixel_scale.setter
    def pixel_scale(self, value):
        raise RuntimeError("You should not be setting pixel_scale on the fly; "
                           "just instantiate a new GalSimCelestialObject")

    @property
    def rotation_angle(self):
        return self._float_values[11]

    @rotation_angle.setter
    def rotation_angle(self, value):
        raise RuntimeError("You should not be setting rotation_angle on the fly; "
                           "just instantiate a new GalSimCelestialObject")

    @property
    def g1(self):
        return self._float_values[12]

    @g1.setter
    def g1(self, value):
        raise RuntimeError("You should not be setting g1 on the fly; " \
        + "just instantiate a new GalSimCelestialObject")


    @property
    def g2(self):
        return self._float_values[13]

    @g2.setter
    def g2(self, value):
        raise RuntimeError("You should not be setting g2 on the fly; "
                           "just instantiate a new GalSimCelestialObject")

    @property
    def mu(self):
        return self._float_values[14]

    @mu.setter
    def mu(self, value):
        raise RuntimeError("You should not be setting mu on the fly; "
                           "just instantiate a new GalSimCelestialObject")

    def flux(self, band):
        """
        @param [in] band is the name of a bandpass

        @param [out] the ADU in that bandpass, as stored in self._fluxDict
        """
        if band not in self._bp_dict:
            raise RuntimeError("Asked GalSimCelestialObject for flux"
                               "in %s; that band does not exist" % band)

        if band not in self._fluxDict:
            adu = self.sed.calcADU(self._bp_dict[band], self._photParams)
            self._fluxDict[band] = adu*self._photParams.gain

        return self._fluxDict[band]
