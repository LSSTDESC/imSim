import numpy as np
import sys
import pytest
sys.path.append('/hpc/home/zg64/IMSIM/imSim')
from imsim import make_batoid_wcs, CCD_Fringing
import galsim

def test_fringing():
    """
    Test the fringing model. 
    """
    # Set a random center ra/dec
    cra = 54.9348753510528
    cdec = -35.8385705255579
    world_center = galsim.CelestialCoord(cra*galsim.degrees, cdec*galsim.degrees)
    
    mjd = 60232.3635999295
    rottelpos = 350.946271812373
    band = 'y'
    serial_num = 382

    xarr, yarr = np.meshgrid(range(4096), range(4004))
    
    # Testing a CCD with an arbitrary location on the focal plane.
    ra = 54.86
    dec = -35.76
    wcs = make_batoid_wcs(ra, dec, rottelpos, mjd, band, 'LsstCam')
    
    config = {
        'image': {
            'type': 'LSST_Image',
            'xsize': 4096,
            'ysize': 4004,
            'wcs': wcs,
            'nobjects': 0,
        },
    }

    galsim.config.ProcessInput(config)
    image = galsim.config.BuildImage(config)

    ccd_fringing = CCD_Fringing(true_center=image.wcs.toWorld(image.true_center),
                                boresight=world_center,
                                seed=serial_num, spatial_vary=True)
    # Test zero value error
    with pytest.raises(ValueError):
        ccd_fringing.calculate_fringe_amplitude(xarr,yarr,amplitude = 0)

    fringe_map = ccd_fringing.calculate_fringe_amplitude(xarr,yarr)
    
    # Check std of the diagnoal of fringe map.
    np.testing.assert_approx_equal(np.std(np.diag(fringe_map)), 0.0014, significant=2)
    
    # Check the min/max of fringing varaition for the current offset.
    np.testing.assert_approx_equal(fringe_map.max(), 1.00205, significant=4)
    np.testing.assert_approx_equal(fringe_map.min(), 0.99794, significant=4)
    
    
if __name__ == '__main__':
    test_fringing()
