"""
Simple bleed trail implementation to apply to "eimages" just prior to
the camera readout code.
"""
import lsst.afw.image as afw_image
import lsst.afw.detection as afw_detect

__all__ = ['apply_channel_bleeding', 'bleed_eimage',
           'find_channels_with_saturation', 'bleed_channel']


def apply_channel_bleeding(gs_interpreter, full_well):
    """
    Apply charge bleeding to eimages with channels that have pixels
    above full well.

    Parameters
    ----------
    gs_interpreter: lsst.sims.GalSimInterface.GalSimInterpreter
        The object that is actually drawing the images

    full_well: int
        The pixel full well/saturation value in electrons.
    """
    for gs_image in gs_interpreter.detectorImages.values():
        bleed_eimage(afw_image.ImageF(gs_image.array), full_well)


def bleed_eimage(eimage, full_well):
    """
    Apply the bleed charge algorithm to an eimage.

    Parameters
    ----------
    eimage: lsst.afw.image.ImageF
        The "eimage" containing the image data in units of electrons per
        pixel.  This is the image prior to electronic readout (i.e.,
        conversion to ADU, addition of bias, dark current, crosstalk, etc.).
        For LSST CCDs, the eimages have parallel transfer directions along
        the x-axis, hence channels correspond to rows in eimages.
    full_well: int
        The pixel full well/saturation value in electrons.

    Returns
    -------
    afw.image.ImageF: This is the input eimage object with the charge
        bleeding applied.
    """
    channels = find_channels_with_saturation(eimage, full_well)
    imarr = eimage.getArray()
    for ypix in channels:
        imarr[ypix, :] = bleed_channel(imarr[ypix, :], full_well)
    return eimage


def find_channels_with_saturation(eimage, full_well):
    """
    Find all channels in an eimage that have pixel values above full
    well.

    Parameters
    ----------
    eimage: lsst.afw.image.ImageF
        The "eimage" containing the image data in units of electrons per
        pixel.  This is the image prior to electronic readout (i.e.,
        conversion to ADU, addition of bias, dark current, crosstalk, etc.).
        For LSST CCDs, the eimages have parallel transfer directions along
        the x-axis, hence channels correspond to rows in eimages.
    full_well: int
        The pixel full well/saturation value in electrons.

    Returns
    -------
    set of ints:  The y-indices of the channels with saturated pixels.
    """
    threshold = afw_detect.Threshold(full_well)
    fp_set = afw_detect.FootprintSet(eimage, threshold)
    channels = []
    for fp in fp_set.getFootprints():
        for span in fp.spans:
            channels.append(span.getY())
    return set(channels)


def bleed_channel(channel, full_well):
    """
    Redistribute charge along a channel for all pixels above full well.

    Parameters
    ----------
    channel: numpy.array of pixel values
        1D array of pixel values in units of electrons.
    full_well: int
        The pixel full well/saturation value in electrons.

    Returns
    -------

    numpy.array of pixel values: The channel of pixel data with
       bleeding applied.  This is a new object, i.e., the input
       numpy.array is unaltered.
    """
    # Construct an afw_image.ImageF to contain the channel data
    # to enable the use of afw_detect to find the above full-well
    # footprints.
    nx = len(channel)
    image = afw_image.ImageF(nx, 1)
    imarr = image.getArray()
    imarr += channel
    threshold = afw_detect.Threshold(full_well)
    fp_set = afw_detect.FootprintSet(image, threshold)

    # Loop over footprints and bleed the charge in +/- directions.
    # Any remaining charge after applying the BleedCharge functor on
    # all footprints will be discarded.
    for fp in fp_set.getFootprints():
        for span in fp.spans:
            x0 = span.getX0()
            x1 = span.getX1() + 1
            excess_charge = sum(imarr[0, x0:x1]) - (x1 - x0)*full_well
            imarr[0, x0:x1] = full_well
            bleed_charge = BleedCharge(imarr, excess_charge, full_well)
            for dx in range(0, max(x0, nx-x1)):
                if bleed_charge(x0 - dx - 1) or bleed_charge(x1 + dx):
                    break
    return imarr[0, :]


class BleedCharge:
    "Class to manage charge redistribution along a channel."
    def __init__(self, imarr, excess_charge, full_well):
        """
        Parameters
        ----------
        imarr: numpy.array
            1D numpy array containing the channel of pixel data.
        excess_charge: float
            The remaining charge above full-well to be distributed
            to the specified pixels.
        full_well: int
            The full well value, i.e., the maximum charge any pixel
            can contain.
        """
        self.imarr = imarr
        self.excess_charge = excess_charge
        self.full_well = full_well

    def __call__(self, xpix):
        """
        Parameters
        ----------
        xpix: int
            Index of the pixel to which charge will be redistributed.
            If it is already at full_well, do nothing.

        Returns
        -------
        bool: True if all excess charge has been redistributed.
        """
        try:
            bled_charge = min(self.full_well - self.imarr[0, xpix],
                              self.excess_charge)
            self.imarr[0, xpix] += bled_charge
            self.excess_charge -= bled_charge
        except IndexError:
            # Trying to bleed charge past end of the channel, so
            # do nothing.
            pass
        return self.excess_charge == 0
