from .imSim import metadata_from_file, photometricParameters, get_obs_lsstSim_camera
import lsst.sims.GalSimInterface as GSInterface
from lsst.sims.photUtils import PhotometricParameters
from lsst.sims.utils import ObservationMetaData
import numpy as np
import os, galsim

__all__ = ['make_flats']

def make_flats(obs_md, outdir, phosim_commands=None, counts_per_iter=4e3, counts_total=80e3, treering_amplitude = 0.26, nflats=1, nborder=2, treering_period = 47., treering_center = galsim.PositionD(0,0), seed = 31415):
    
    # get photometric parameters
    if phosim_commands!=None:
        phot_params = photometricParameters(phosim_commands)
    else:
        phot_params = PhotometricParameters(bandpass='r', darkcurrent=0, exptime=30., gain=1, nexp=1, readnoise=0)
    
    # create an lsst camera object
    camera = get_obs_lsstSim_camera()
    
    interpreter = GSInterface.GalSimInterpreter(detectors=list(camera))
    
    # create lsst cameraWrapper
    cameraWrapper = GSInterface.LSSTCameraWrapper()
    
    # create a list of detector objects
    detectors = [GSInterface.make_galsim_detector(cameraWrapper, det.getName(), phot_params, obs_md) for det in list(camera) if str(det.getType())=='DetectorType.SCIENCE']
    
    # use detector objects to make blank images with correct lsst pixel geometry and wcs
    blank_imgs = [interpreter.blankImage(det) for det in detectors] 
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outdir = outdir+'/'
    
    for m,img in enumerate(blank_imgs):
        
        nx = img.xmax
        ny = img.ymax

        rng = galsim.UniformDeviate(seed)

        treering_func = galsim.SiliconSensor.simple_treerings(treering_amplitude, treering_period)

        niter = int(counts_total / counts_per_iter + 0.5)
        counts_per_iter = counts_total / niter  # Recalculate in case not even multiple.
        print('Total counts = {} = {} * {}'.format(counts_total,niter,counts_per_iter))

        # an LSST wcs
        wcs = img.wcs

        base_image = galsim.ImageF(nx+2*nborder, ny+2*nborder, wcs=wcs)
        print('image bounds = ',base_image.bounds)

        # nrecalc is actually irrelevant here, since a recalculation will be forced on each iteration.
        # Which is really the point.  We need to set coundsPerIter appropriately so that the B/F effect
        # doesn't change *too* much between iterations.
        sensor = galsim.SiliconSensor(rng=rng,
                                      treering_func=treering_func, treering_center=treering_center)

        # We also need to account for the distortion of the wcs across the image.  
        # This expects sky_level in ADU/arcsec^2, not ADU/pixel.
        base_image.wcs.makeSkyImage(base_image, sky_level=1.)
        # base_image.write(outdir+'wcs_area.fits')

        # Rescale so that the mean sky level per pixel is skyCounts
        mean_pixel_area = base_image.array.mean()

        sky_level_per_iter = counts_per_iter / mean_pixel_area  # in ADU/arcsec^2 now.
        base_image *= sky_level_per_iter

        # The base_image has the right level to account for the WCS distortion, but not any sensor effects.
        # This is the noise-free level that we want to add each iteration modulated by the sensor.
        noise = galsim.PoissonNoise(rng)
        
        for n in range(nflats):
        # image is the image that we will build up in steps.
        # We add on a border of 2 pixels, since the outer row/col get a little messed up by photons
        # falling off the edge, but not coming on from the other direction.
        # We do 2 rows/cols rather than just 1 to be safe

            image = galsim.ImageF(nx+2*nborder, ny+2*nborder, wcs=wcs)

            for i in range(niter):
                # temp is the additional flux we will add to the image in this iteration.
                # Start with the right area due to the sensor effects.
                temp = sensor.calculate_pixel_areas(image)
                temp = temp
                temp /= np.mean(temp.array)  # Normalize to unit mean.
                # temp.write(outdir+'sensor_area.fits')

                # Multiply by the base image to get the right mean level and wcs effects
                temp *= base_image
                # temp.write(outdir+'nonoise.fits')

                # Finally, add noise.  What we have here so far is the expectation value in each pixel.
                # We need to realize this according to Poisson statistics with these means.
                temp.addNoise(noise)
                # temp.write(outdir+'withnoise.fits')

                # Add this to the image we are building up.
                image += temp

            # Cut off the outer border where things don't work quite right.
            # print('bounds = ',image.bounds)
            image = image.subImage(galsim.BoundsI(1+nborder,nx+nborder,1+nborder,ny+nborder))
            # print('bounds => ',image.bounds)
            image.setOrigin(1,1)
            # print('bounds => ',image.bounds)

            image.write(outdir+'flat{:02d}_{:02d}.fits'.format(m,n))
