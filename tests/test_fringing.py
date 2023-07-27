import numpy as np
from imsim import make_batoid_wcs, CCD_Fringe

def test_fringing():
    """
    Test the fringing model. 
    """
    # Set a random center ra/dec
    cra = 54.9348753510528
    cdec = -35.8385705255579
    
    mjd = 60232.3635999295
    rottelpos = 350.946271812373
    band = 'y'

    xarr, yarr = np.meshgrid(range(4096), range(4004))
    
    #-------------------------------------------------------------------------
    
    # Testing a CCD within the interpolation range of OH spatial variation
    ra = 54.86
    dec = -35.76
    wcs = make_batoid_wcs(ra, dec, rottelpos, mjd, band, 'LsstCam')
    
    ccd_fringing = CCD_Fringe(img_wcs = wcs,c_wcs=[cra, cdec],spatial_vary= True)

    fringe_map = ccd_fringing.calculate_fringe_amplitude(xarr,yarr)

    # Check the fringing varaition
    np.testing.assert_approx_equal(fringe_map.max(),1.002586,significant=4)
    np.testing.assert_approx_equal(fringe_map.min(),0.99741,significant=4)
    
    #-------------------------------------------------------------------------
    
    # Testing a CCD outside the interpolation range of OH spatial variation
    ra = 53
    dec = -33
    wcs = make_batoid_wcs(ra, dec, rottelpos, mjd, band, 'LsstCam')
    ccd_fringing = CCD_Fringe(img_wcs = wcs,c_wcs=[cra, cdec],spatial_vary= True)
    fringe_map = ccd_fringing.calculate_fringe_amplitude(xarr,yarr)
    
    # Check if the default fringing varaition is within +/- 0.2%
    np.testing.assert_approx_equal(fringe_map.max(),1.002,significant=4)
    np.testing.assert_approx_equal(fringe_map.min(),0.998,significant=4)
    
if __name__ == '__main__':
    test_fringing()