from desc.imsim import metadata_from_file, photometricParameters, get_obs_lsstSim_camera, metadata_from_file, phosim_obs_metadata
import lsst.sims.GalSimInterface as GSInterface
from lsst.sims.photUtils import PhotometricParameters
from lsst.sims.utils import ObservationMetaData
import numpy as np
import os, galsim
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('file', help="The instance catalog")
parser.add_argument('-n', type=int, default=1, 
                    help='number of flats to make')
parser.add_argument('--outdir', type=str, default='fits',
                    help='Output directory for eimage file')
parser.add_argument('--counts_total', type=float, default=80e3,
                    help='')
parser.add_argument('--counts_per_iter', type=float, default=4e3,
                    help='determines the fidelity of the correlated noise')
parser.add_argument('--treering_period', type=float, default=47,
                    help='The period of the tree ring distortion pattern, in pixels')
parser.add_argument('--treering_amplitude', type=float, default=0.26,
                    help='The amplitude of the tree ring pattern distortion')
parser.add_argument('--phosim_commands', default=None)
parser.add_argument('--seed', type=int, default=267,
                    help='integer used to seed random number generator')
args = parser.parse_args()

phosim_commands = args.phosim_commands

counts_total = args.counts_total

treering_amplitude = args.treering_amplitude

nflats = args.n

nborder = 2

treering_period = args.treering_period

treering_center = galsim.PositionD(0,0)

seed = args.seed

outdir = args.outdir
if not os.path.exists(outdir):
    os.makedirs(outdir)
outdir = outdir+'/'
    

commands = metadata_from_file(args.file)

obs_md = phosim_obs_metadata(commands)

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

        for _ in range(niter):
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
