import unittest

import numpy as np
import galsim

from imsim.atmPSF import AtmosphericPSF

class AtmPSF(unittest.TestCase):
    def test_r0_500(self):
        """Test that inversion of the Tokovinin fitting formula for r0_500 works."""
        np.random.seed(57721)
        for _ in range(100):
            airmass = np.random.uniform(1.001, 1.5)
            rawSeeing = np.random.uniform(0.5, 1.5)
            band = 'ugrizy'[np.random.randint(6)]
            boresight = galsim.CelestialCoord(0 * galsim.radians, 0 * galsim.radians)
            rng = galsim.BaseDeviate(np.random.randint(2**32))
            atmPSF = AtmosphericPSF(airmass, rawSeeing, band, boresight, rng, screen_size=6.4)

            wlen = dict(u=365.49, g=480.03, r=622.20, i=754.06, z=868.21, y=991.66)[band]
            targetFWHM = rawSeeing * airmass**0.6 * (wlen/500)**(-0.3)

            r0_500 = atmPSF.atm.r0_500_effective
            L0 = atmPSF.atm[0].L0
            vkFWHM = AtmosphericPSF._vkSeeing(r0_500, wlen, L0)

            np.testing.assert_allclose(targetFWHM, vkFWHM, atol=1e-3, rtol=0)


if __name__ == '__main__':
    unittest.main()
