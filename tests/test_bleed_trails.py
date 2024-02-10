"""
Unit tests for bleed trail implementation.
"""
import unittest
from pathlib import Path
import pickle
import numpy as np
import galsim
import imsim


class BleedTrailTestCase(unittest.TestCase):
    "Bleed trail TestCase subclass"
    def setUp(self):
        self.nypix = 2000
        self.sky_level = 800
        self.full_well = int(1e5)
        self.rng = galsim.BaseDeviate(1234)

    def tearDown(self):
        pass

    def _make_sky_flat(self, nx=4072, ny=4000):
        "Make an eimage with Poisson counts at the sky background level."
        eimage = galsim.ImageF(nx, ny)
        galsim.PoissonDeviate(self.rng, mean=self.sky_level).generate(eimage.array)
        return eimage

    def _add_bright_object(self, eimage, xpix, ypix, npix=4, flux=None):
        """
        Add a bright object, a 2*npix by 2*npix square with constant flux,
        centered at the (xpix, ypix) location.
        """
        if flux is None:
            flux = 2*self.full_well
        imarr = eimage.array
        imarr[ypix - npix:ypix + npix, xpix - npix:xpix + npix] += flux
        return eimage

    def test_bleed_channel(self):
        "Test the bleed_channel function."
        channel = (np.ones(self.nypix, dtype=int) *
                   np.random.poisson(lam=self.sky_level))
        star_center = 1000
        star_size = 20
        channel[star_center-star_size:star_center+star_size] = 2*self.full_well

        total_count = sum(channel)

        # Run the bleed channel function.
        bled_channel = imsim.bleed_channel(channel, self.full_well)

        # Check that charge is conserved in this case.
        self.assertEqual(total_count, sum(bled_channel))

        # Check that maximum pixel value is full well.
        self.assertEqual(self.full_well, max(bled_channel))

        # Check that an un-interrupted bleed trail is produced
        index = np.where(bled_channel == self.full_well)
        nmin = max(0, min(index[0])-1)
        nmax = max(index[0])
        expected_sum = (nmax*(nmax+1))//2 - (nmin*(nmin+1))//2
        self.assertEqual(sum(index[0]), expected_sum)

        # Check that bleed trail is centered on original star.
        self.assertLessEqual(np.abs(star_center - (nmin + nmax)/2), 1)

    def test_bleed_channel_no_negative_pixels(self):
        """This is a reproducer for a case found from running the
        code for ComCam simulations."""
        # Read in the channel data and full well used.
        DATA_DIR = Path(__file__).parent / 'data'
        neg_pixel_data = str(DATA_DIR / 'neg_pixel_bleed.pickle')
        with open(neg_pixel_data, "rb") as fobj:
            channel_data, full_well = pickle.load(fobj)
        bled_channel = imsim.bleed_channel(channel_data, full_well)
        self.assertTrue(all(bled_channel > 0))

    def test_find_channels_with_saturation(self):
        """
        Test the function to find channels in an image that have pixels
        above full well.
        """
        eimage = self._make_sky_flat()
        ny, nx = eimage.array.shape
        npix = 20
        xpix = np.random.choice(range(nx), npix)
        ypix = np.random.choice(range(ny), npix)
        imarr = eimage.array
        for i, j in zip(xpix, ypix):
            imarr[j, i] = 2*self.full_well
        channels = imsim.find_channels_with_saturation(eimage.array, self.full_well)
        self.assertEqual(set(xpix), channels)

    def test_bleed_eimage(self):
        "Test the function to process bleeding of a full eimage."
        eimage = self._make_sky_flat()
        imarr = eimage.array
        ny, nx = imarr.shape

        # Put a super-saturated object in the middle.
        dxy = 4
        xmid = nx//2
        ymid = ny//2
        self._add_bright_object(eimage, xmid, ymid, npix=dxy)

        # Sum the counts in a region that should contain the bled charge.
        # XXX: I swapped the axis that has the 3x range here from the old code.
        #      I think this is right.  The bleeds are along y, which is the first axis.
        #      I believe the old code was testing a phosim-style e-image, which has the x,y
        #      axes swapped, so I think the new version is correct for normal images.
        #      But I would appreciate someone independently thinking this through to confirm
        #      whether this change and other similar ones in this file are correct.
        total_counts = sum(imarr[ymid - 3*dxy:ymid + 3*dxy,
                                 xmid - dxy:xmid + dxy].ravel())

        # Let it bleed.
        bled_imarr = imsim.bleed_eimage(imarr, self.full_well,
                                         midline_stop=False)
        total_bled_counts = sum(bled_imarr[ymid - 3*dxy:ymid + 3*dxy,
                                           xmid - dxy:xmid + dxy].ravel())

        # Check that the charge in the containment region is unchanged.
        self.assertEqual(total_counts, total_bled_counts)

        # Check that the maximum pixel value in the bled image is
        # equal to full well.
        self.assertEqual(max(bled_imarr.ravel()), self.full_well)

    def test_midline_bleed_stop(self):
        "Test of optional midline bleed stop."
        eimage = self._make_sky_flat()
        imarr = eimage.array
        ny, nx = imarr.shape
        flux = 10*self.full_well

        # Put a super-saturated object just to one side of the midline.
        dxy = 4
        xpix = nx//2
        ypix = ny//2 - dxy
        self._add_bright_object(eimage, xpix, ypix, npix=dxy, flux=flux)

        # Save a copy of the eimage for testing with midline stop enabled.
        eimage_save = eimage.copy()

        # Make sure this image will bleed across the midline without
        # the midline stop.
        imsim.bleed_eimage(imarr, self.full_well, midline_stop=False)
        max_pix_right = max(imarr[ny//2:, :].ravel())
        self.assertEqual(max_pix_right, self.full_well)

        # Test with midline stop.
        imsim.bleed_eimage(eimage_save.array, self.full_well, midline_stop=True)
        max_pix_right = max(eimage_save.array[ny//2:, :].ravel())
        self.assertLess(max_pix_right, self.full_well)

    def test_bleed_trail_wrap(self):
        """
        In issue #153, it was noted that bleed trails extending to
        the sensor edge will wrap around to the midline bleed stop
        and continue bleeding from the opposite end of the amplifier
        segment.  Test against this by placing well above saturated
        signal in a pixel at the 0-index end of the channel and ensure
        that no signal appears at the high-index end of the channel.
        """
        channel = np.zeros(self.nypix, dtype=int)
        # Put an initial signal in the first pixel in this channel
        # so that the bleed trail would try to extend +/-10 pixels
        # in each direction.
        channel[0] = 20*self.full_well

        bled_channel = imsim.bleed_channel(channel, self.full_well)

        # Check that the pixels at the other end of the channel are
        # still empty.
        for i in range(-1, -10, -1):
            self.assertEqual(bled_channel[i], 0)

        # Repeat with non-zero (but non-saturated) values near the other end.
        channel[-20:] = self.full_well/2.
        bled_channel2 = imsim.bleed_channel(channel, self.full_well)
        np.testing.assert_array_equal(bled_channel2[:50], bled_channel[:50])
        np.testing.assert_array_equal(bled_channel2[-50:], channel[-50:])

        # Check the bleed stop end of the channel to be sure charge
        # doesn't wrap the other way.
        channel = np.zeros(self.nypix, dtype=int)
        channel[-1] = 20*self.full_well
        bled_channel = imsim.bleed_channel(channel, self.full_well)
        for i in range(10):
            self.assertEqual(bled_channel[i], 0)

        # Repeat with non-zero (but non-saturated) values near the other end.
        channel[:20] = self.full_well/2.
        bled_channel2 = imsim.bleed_channel(channel, self.full_well)
        np.testing.assert_array_equal(bled_channel2[-50:], bled_channel[-50:])
        np.testing.assert_array_equal(bled_channel2[:50], channel[:50])



if __name__ == '__main__':
    unittest.main()
