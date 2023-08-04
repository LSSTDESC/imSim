import numpy as np
from imsim import make_batoid_wcs, CCD_Fringing

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
    det_num = 94
    
    xarr, yarr = np.meshgrid(range(4096), range(4004))

    
    # Testing a CCD with an arbitrary location on the focal plane.
    ra = 54.86
    dec = -35.76
    wcs = make_batoid_wcs(ra, dec, rottelpos, mjd, band, 'LsstCam')
    
    ccd_fringing = CCD_Fringing(img_wcs = wcs,c_wcs=[cra, cdec],seed = det_num, spatial_vary= True)

    fringe_map = ccd_fringing.calculate_fringe_amplitude(xarr,yarr)
    
    # Check for zero values in the fringe map.
    if (np.all(fringe_map) != True) or (True in np.isnan(fringe_map)):
        raise ValueError(" 0 or nan value in the fringe map!")
    
    # Check std of the diagnoal of fringe map.
    np.testing.assert_approx_equal(np.std(np.diag(fringe_map)),0.0014,significant=2)
    
    # Check the min/max of fringing varaition for the current offset.
    np.testing.assert_approx_equal(fringe_map.max(),1.00205,significant=4)
    np.testing.assert_approx_equal(fringe_map.min(),0.99794,significant=4)
    
    
if __name__ == '__main__':
    test_fringing()