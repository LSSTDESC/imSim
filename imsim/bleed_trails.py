"""
Simple bleed trail implementation to apply to "eimages" just prior to
the camera readout code.
"""
import copy
import numpy as np

__all__ = ['bleed_eimage', 'bleed_channel']


def bleed_eimage(eimage, full_well, midline_stop=True):
    """
    Apply the bleed charge algorithm to an eimage.

    Parameters
    ----------
    eimage: numpy array
        The "eimage" containing the image data in units of electrons per
        pixel.  This is the image prior to electronic readout (i.e.,
        conversion to ADU, addition of bias, dark current, crosstalk, etc.).
        For LSST CCDs, the eimages have parallel transfer directions along
        the x-axis, hence channels correspond to rows in eimages.
    full_well: int
        The pixel full well/saturation value in electrons.
    midline_stop: bool [True]
        Flag to treat the midline of the sensor as a bleed stop.

    Returns
    -------
    numpy array: This is the input eimage object with the charge
        bleeding applied.
    """
    # Find channels (by x-index) with signal above full_well.
    channels = set(np.where(eimage > full_well)[1])

    # Apply bleeding to each channel.
    for xpix in channels:
        if midline_stop:
            ymid = eimage.shape[0]//2
            eimage[:ymid, xpix] = bleed_channel(eimage[:ymid, xpix], full_well)
            eimage[ymid:, xpix] = bleed_channel(eimage[ymid:, xpix], full_well)
        else:
            eimage[:, xpix] = bleed_channel(eimage[:, xpix], full_well)
    return eimage


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
    # Find contiguous sets of pixels that lie above full well, and
    # build a list of end point pairs identifying each set.
    my_channel = copy.deepcopy(channel)
    padded = np.array([0] + list(my_channel) + [0])
    index = np.where(padded > full_well)
    mask = np.zeros(len(padded), dtype=int)
    mask[index] += 1
    diff = mask[1:] - mask[:-1]
    end_points = []
    for x0, x1 in zip(np.where(diff == 1)[0], np.where(diff == -1)[0]):
        end_points.append((x0, x1))

    # Loop over end point pairs.
    for x0, x1 in end_points:
        excess_charge = sum(my_channel[x0:x1]) - (x1 - x0)*full_well
        my_channel[x0:x1] = full_well
        bleed_charge = BleedCharge(my_channel, excess_charge, full_well)
        for dx in range(0, max(x0, len(my_channel) - x1)):
            if bleed_charge(x0 - dx - 1) or bleed_charge(x1 + dx):
                break
    return my_channel


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
            bled_charge = min(self.full_well - self.imarr[xpix],
                              self.excess_charge)
            if xpix >= 0:
                # Restrict charge redistribution to positive xpix
                # values to avoid wrapping the bleed trail to the
                # other end of the channel.  Charge bled off the end
                # will still be removed from the excess charge pool.
                self.imarr[xpix] += bled_charge
            self.excess_charge -= bled_charge
        except IndexError:
            # Trying to bleed charge past end of the channel, so
            # do nothing.
            pass
        return self.excess_charge == 0
