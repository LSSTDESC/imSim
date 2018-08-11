"""
imSim module to make flats.
"""
import galsim

__all__ = ['make_flat']

def make_flat(gs_det, counts_per_iter, niter, rng, buf=2):
    """
    Create a flat by successively adding lower level flat sky images.
    This is based on
    https://github.com/GalSim-developers/GalSim/blob/releases/2.0/devel/lsst/treering_flat.py

    Full LSST CCDs are assembled one amp at a time to limit the memory
    used by the galsim.SiliconSensor model.

    Parameters
    ----------
    gs_det: GalSimDetector
        The detector in the LSST focalplane to use.  This object
        contains the CCD pixel geometry, WCS, and treering info.
    counts_per_iter: int
        Roughly the number of electrons/pixel to add at each iteration.
    niter: int
        Number of iterations. Final flat will have niter*counts_per_iter
        electrons/pixel.
    rng: galsim.BaseDeviate
        Random number generator.
    buf: int [2]
        Pixel buffer around each to account for charge redistribution
        across pixel boundaries.

    Returns
    -------
    galsim.ImageF
    """
    ncol = gs_det.xMaxPix - gs_det.xMinPix + 1
    nrow = gs_det.yMaxPix - gs_det.yMinPix + 1
    flat = galsim.ImageF(ncol, nrow, wcs=gs_det.wcs)
    sensor = galsim.SiliconSensor(rng=rng,
                                  treering_center=gs_det.tree_rings.center,
                                  treering_func=gs_det.tree_rings.func,
                                  transpose=True)

    # Create a noise-free base image to add at each iteration.
    base_image = galsim.ImageF(ncol, nrow, wcs=gs_det.wcs)
    base_image.wcs.makeSkyImage(base_image, sky_level=1.)
    mean_pixel_area = base_image.array.mean()
    sky_level_per_iter = counts_per_iter/mean_pixel_area
    base_image *= sky_level_per_iter

    noise = galsim.PoissonNoise(rng)

    # Build up the full CCD by amplifier segment to limit the memory
    # used by the silicon model.
    nx, ny = 2, 8
    dx = ncol//nx
    dy = nrow//ny
    for i in range(nx):
        xmin = i*dx + 1
        xmax = (i + 1)*dx
        for j in range(ny):
            ymin = j*dy + 1
            ymax = (j + 1)*dy
            temp_image = galsim.ImageF(ncol, nrow, wcs=gs_det.wcs)
            bounds = (galsim.BoundsI(xmin-buf, xmax+buf, ymin-buf, ymax+buf)
                      & temp_image.bounds)
            temp_amp = temp_image[bounds]
            for _ in range(niter):
                temp = sensor.calculate_pixel_areas(temp_amp)
                temp /= temp.mean()
                temp *= base_image[bounds]
                temp.addNoise(noise)
                temp_amp += temp
            amp_bounds = galsim.BoundsI(xmin, xmax, ymin, ymax)
            flat[amp_bounds] += temp_image[amp_bounds]
    return flat
