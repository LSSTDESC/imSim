"""
Unit tests for bleed trail implementation.
"""
import unittest
import numpy as np
import galsim
import lsst.afw.image as afw_image
import desc.imsim


class BleedTrailTestCase(unittest.TestCase):
    "Bleed trail TestCase subclass"
    def setUp(self):
        self.nypix = 2000
        self.sky_level = 800
        self.full_well = int(1e5)

    def tearDown(self):
        pass

    def _make_sky_flat(self, nx=4000, ny=4072):
        "Make an eimage with Poisson counts at the sky background level."
        eimage = afw_image.ImageF(nx, ny)
        imarr = eimage.getArray()
        imarr += np.random.poisson(self.sky_level, size=(ny, nx))
        return eimage

    def _add_bright_object(self, eimage, xpix, ypix, npix=4, flux=None):
        """
        Add a bright object, a 2*npix by 2*npix square with constant flux,
        centered at the (xpix, ypix) location.
        """
        if flux is None:
            flux = 2*self.full_well
        imarr = eimage.getArray()
        imarr[ypix - npix:ypix + npix, xpix - npix:xpix + npix] += flux
        return eimage

    def test_bleed_channel(self):
        "Test the bleed_channel function."
        channel = (np.ones(self.nypix, dtype=np.int) *
                   np.random.poisson(lam=self.sky_level))
        star_center = 1000
        star_size = 20
        channel[star_center-star_size:star_center+star_size] = 2*self.full_well

        total_count = sum(channel)

        # Run the bleed channel function.
        bled_channel = desc.imsim.bleed_channel(channel, self.full_well)

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

    def test_find_channels_with_saturation(self):
        """
        Test the function to find channels in an image that have pixels
        above full well.
        """
        eimage = self._make_sky_flat()
        ny, nx = eimage.getArray().shape
        npix = 20
        xpix = np.random.choice(range(nx), npix)
        ypix = np.random.choice(range(ny), npix)
        imarr = eimage.getArray()
        for i, j in zip(xpix, ypix):
            imarr[j, i] = 2*self.full_well
        channels \
            = desc.imsim.find_channels_with_saturation(eimage, self.full_well)
        self.assertEqual(set(ypix), channels)

    def test_bleed_eimage(self):
        "Test the function to process bleeding of a full eimage."
        eimage = self._make_sky_flat()
        imarr = eimage.getArray()
        ny, nx = imarr.shape

        # Put a super-saturated object in the middle.
        dxy = 4
        xmid = nx//2
        ymid = ny//2
        self._add_bright_object(eimage, xmid, ymid, npix=dxy)

        # Sum the counts in a region that should contain the bled charge.
        total_counts = sum(imarr[ymid - dxy:ymid + dxy,
                                 xmid - 3*dxy:xmid + 3*dxy].ravel())

        # Let it bleed.
        bled_eimage = desc.imsim.bleed_eimage(eimage, self.full_well,
                                              midline_stop=False)
        bled_imarr = bled_eimage.getArray()
        total_bled_counts = sum(bled_imarr[ymid - dxy:ymid + dxy,
                                           xmid - 3*dxy:xmid + 3*dxy].ravel())

        # Check that the charge in the containment region is unchanged.
        self.assertEqual(total_counts, total_bled_counts)

        # Check that the maximum pixel value in the bled image is
        # equal to full well.
        self.assertEqual(max(bled_imarr.ravel()), self.full_well)

        # Test that one can cast a galsim.ImageF object as an afw_image.ImageF
        # and have the bleeding applied.
        gs_image \
            = galsim.ImageF(np.random.poisson(self.sky_level, size=(ny, nx)))
        gs_image.array[ymid - dxy:ymid + dxy,
                       xmid - dxy:xmid + dxy] += 2*self.full_well
        desc.imsim.bleed_eimage(afw_image.ImageF(gs_image.array),
                                self.full_well)
        self.assertEqual(max(gs_image.array.ravel()), self.full_well)

    def test_midline_bleed_stop(self):
        "Test of optional midline bleed stop."
        eimage = self._make_sky_flat()
        imarr = eimage.getArray()
        ny, nx = imarr.shape
        flux = 10*self.full_well

        # Put a super-saturated object just to one side of the midline.
        dxy = 4
        xpix = nx//2 - dxy
        ypix = ny//2
        self._add_bright_object(eimage, xpix, ypix, npix=dxy, flux=flux)

        # Save a copy of the eimage for testing with midline stop enabled.
        eimage_save = eimage.Factory(eimage, deep=True)

        # Make sure this image will bleed across the midline without
        # the midline stop.
        desc.imsim.bleed_eimage(eimage, self.full_well, midline_stop=False)
        max_pix_right = max(imarr[:, nx//2:].ravel())
        self.assertEqual(max_pix_right, self.full_well)

        # Test with midline stop.
        desc.imsim.bleed_eimage(eimage_save, self.full_well, midline_stop=True)
        max_pix_right = max(eimage_save.getArray()[:, nx//2:].ravel())
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
        channel = np.zeros(self.nypix, dtype=np.int)
        # Put an initial signal in the first pixel in this channel
        # so that the bleed trail would try to extend +/-10 pixels
        # in each direction.
        channel[0] = 20*self.full_well

        bled_channel = desc.imsim.bleed_channel(channel, self.full_well)

        # Check that the pixels at the other end of the channel are
        # still empty.
        for i in range(-1, -10, -1):
            self.assertEqual(bled_channel[i], 0)

        # Check the bleed stop end of the channel to be sure charge
        # doesn't wrap the other way.
        channel = np.zeros(self.nypix, dtype=np.int)
        channel[-1] = 20*self.full_well
        bled_channel = desc.imsim.bleed_channel(channel, self.full_well)
        for i in range(10):
            self.assertEqual(bled_channel[i], 0)


if __name__ == '__main__':
    unittest.main()
