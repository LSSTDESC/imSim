"""
Simple bleed trail implementation to apply to "eimages" just prior to
the camera readout code.
"""
import numpy as np

__all__ = ['bleed_eimage', 'bleed_channel', 'find_channels_with_saturation']

def find_channels_with_saturation(eimage, full_well):
    """Helper function to find which channels (x values) have at least one saturated pixel.

    Parameters
    ----------
    eimage: numpy array
        The "eimage" containing the image data in units of electrons per pixel.
    full_well: int
        The pixel full well/saturation value in electrons.

    Returns
    -------
    set: A set of integers giving all x values with a saturated pixel.
    """
    # Find channels (by x-index) with signal above full_well.
    return set(np.where(eimage > full_well)[1])

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
    channels = find_channels_with_saturation(eimage, full_well)

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
    my_channel = channel.copy()

    # Add 0 at start and end, so saturated points are known to be all internal.
    padded = np.concatenate([[0], my_channel, [0]])

    # Find places where full well condition changes (either true to false or false to true).
    end_points, = np.diff(padded > full_well).nonzero()

    # Pairs of these are now the first saturated pixel in a run then the
    # first subsequent unsaturated pixel.  Logically, they have to alternate.
    # Reshape these into array of (start, end) pairs.
    end_points = end_points.reshape(-1,2)

    # Loop over end point pairs.
    for y0, y1 in end_points:
        excess_charge = sum(my_channel[y0:y1]) - (y1 - y0)*full_well
        my_channel[y0:y1] = full_well
        bleed_charge = BleedCharge(my_channel, excess_charge, full_well)
        for dy in range(0, max(y0, len(my_channel) - y1)):
            if bleed_charge(y0 - dy - 1) or bleed_charge(y1 + dy):
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

    def __call__(self, ypix):
        """
        Parameters
        ----------
        ypix: int
            Index of the pixel to which charge will be redistributed.
            If it is already at full_well, do nothing.

        Returns
        -------
        bool: True if all excess charge has been redistributed.
        """
        if 0 <= ypix < len(self.imarr):
            # The normal case: Add excess charge up to the full well and reduce this
            # amount from the total excess charge to be redistributed.
            bled_charge = min(self.full_well - self.imarr[ypix],
                              self.excess_charge)
            self.imarr[ypix] += bled_charge
            self.excess_charge -= bled_charge
        elif ypix < 0:
            # Off the bottom end, the charge escapes into the electronics.
            # We can reduce the excess charge by one full-well-worth.
            # These electrons are not added to any pixel though.
            self.excess_charge -= min(self.full_well, self.excess_charge)
        else:
            # Electrons do not escape off the top end, so excess charge is not reduced
            # when trying to bleed past the end of the channel.
            pass

        return self.excess_charge == 0
