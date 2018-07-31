import unittest

import numpy as np

from desc.imsim.atmPSF import AtmosphericPSF

class AtmPSF(unittest.TestCase):
    def test_r0_500(self):
        """Test that _seeing_resid has the API fsolve wants."""
        for wavelength in [300.0, 500.0, 1100.0]:
            for L0 in [10.0, 25.0, 100.0]:
                for target_seeing in [0.5, 0.7, 1.0]:
                    r0s = [0.1, 0.2]
                    np.testing.assert_array_equal(
                        np.hstack([
                            AtmosphericPSF._seeing_resid(r0s[0], wavelength, L0, target_seeing),
                            AtmosphericPSF._seeing_resid(r0s[1], wavelength, L0, target_seeing),
                        ]),
                        AtmosphericPSF._seeing_resid(r0s, wavelength, L0, target_seeing)
                    )


if __name__ == '__main__':
    unittest.main()
