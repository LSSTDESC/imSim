import galsim
from galsim.config.extra import ExtraOutputBuilder, RegisterExtraOutput
from galsim.config import ImageBuilder, RegisterImageType, InputLoader, RegisterInputType
from galsim.sensor import Sensor, SiliconSensor
from astropy.io.fits import BinTableHDU

from .lsst_image import LSST_ImageBuilderBase
from .utils import pixel_to_focal, focal_to_pixel
from .camera import get_camera

# An extra output type to enable two-pass photon scattering.
# This output should be used in the first pass to write scattered photons to file,
# while the second pass will read the scattered photons then go over all the images
# again and add the photons.
class ScatteredPhotonsBuilder(ExtraOutputBuilder):
    """Build pickled photon arrays containing photons which were not accumulated
    to the sensor during each image's construction.
    """

    def processImage(self, index, obj_nums, config, base, logger):
        """After each image is drawn, concatenate the list of scattered photon arrays
        and store in data ready to be pickled to file later on.
        """
        # If scattered photons have been stored in the base config, concatenate these
        # photon arrays to a single one and store in data, indexing by image number.
        if 'scattered_photons' in base and len(base['scattered_photons']) > 1:
            self.data[index] = galsim.PhotonArray.concatenate(base['scattered_photons'])
            # We need to store the photons using focal plane coordinates so they
            # can be accumulated in the second pass on an arbitrary sensor.
            detector = get_camera(base['output']['camera'])[base['det_name']]
            self.data[index].x, self.data[index].y = pixel_to_focal(self.data[index].x, self.data[index].y, detector)
        else:
            self.data[index] = galsim.PhotonArray(N=0)
        print("PLACEHOLDER! Seems to be done.")

    def finalize(self, config, base, main_data, logger):
        return self.data

    def writeFile(self, file_name, config, base, logger):
        """Write the photon array to fits file.
        Might it be faster/more space efficient to pickle?
        """
        for photon_array in self.final_data:
            photon_array.write(file_name)

    def writeHdu(self, config, base, logger):
        """We don't want to write the scattered photons to FITS, so return an
        empty BinTable in lieu of something else.
        """
        return BinTableHDU(data=None)


class ScatteredPhotons(object):
    """A class to hold the photons which fell outside the sensor being drawn
    by the task that createed them during the first pass. They were saved to file
    then, and will now be read in this second pass to be accumulated on other sensors.
    """

    def __init__(self, file_name, camera, det_name, xsize=4096, ysize=4096, logger=None):
        """
        Initialize the scattered photons input class.

        Parameters:
            file_name: str
                The name of the file to read.
            camera: str
                The name of the camera containing the detector.
            det_name: str
                The name of the detector to use for photon coordinate transformations.
            xsize: int
                The x size in pixels of the CCD. (default: 4096)
            ysize: int
                The y size in pixels of the CCD. (default: 4096)
            logger: logging.logger
                A logger object. (default: None)
        """
        self.file_name = file_name
        self.det = get_camera(camera)[det_name]
        self.xsize = xsize
        self.ysize = ysize
        self.logger = logger
        self.photons = None
        self.photons = galsim.PhotonArray.read(self.file_name)
        self.photons.x, self.photons.y = focal_to_pixel(self.photons.x, self.photons.y, self.det)

    def read_photons(self):
        """Read the scattered photons from the file.
        """
        # Read the scattered photons from the file then transform them from
        # focal plane coordinates to this detector's pixel coordinates.
        self.photons = galsim.PhotonArray.read(self.file_name)
        self.photons.x, self.photons.y = focal_to_pixel(self.photons.x, self.photons.y, self.det)


class ScatteredPhotonsLoader(InputLoader):
    """
    Class to load scattered photons from file.
    """
    def getKwargs(self, config, base, logger):
        req = {'file_name': str, 'camera': str, 'det_name': str}
        opt = {}
        kwargs, safe = galsim.config.GetAllParams(config, base, req=req,
                                                  opt=opt)
        kwargs['xsize'] = base.get('det_xsize', 4096)
        kwargs['ysize'] = base.get('det_ysize', 4096)
        kwargs['logger'] = logger

        # For now assume the scattered photons on file can be used for all detectors.
        safe = True
        return kwargs, safe


class LSST_ScatteredPhotonsImageBuilder(LSST_ImageBuilderBase):

    # def setup(self, config, base, image_num, obj_num, ignore, logger):
    #     """Set up the scattered photons image type.
    #     """
    #     # We need to set the pixel scale to be the same as the camera's pixel scale
    #     # so that we can convert between focal plane coordinates and pixel coordinates.
    #     self.pixel_scale = base['output']['pixel_scale']
    #     self.camera = get_camera(base['output']['camera'])[base['det_name']]
    #     self.focal_plane = self.camera.get_focal_plane()
    #     self.focal_plane.set_pixel_scale(self.pixel_scale)

    def buildImage(self, config, base, image_num, _obj_num, logger):
        """Draw the scattered photons to the image.
        """
        # Make sure we have an input image on which to draw.
        image = base['current_image']

        # Make sure we have a scattered_photons input.
        if not isinstance(base['_input_objs']['scattered_photons'][0], ScatteredPhotons):
            raise galsim.config.GalSimConfigError(
                "When using LSST_ScatteredPhotonsImage, you must provide a scattered_photons input.",)

        # Get the scattered photons from the base config.
        scattered_photons = base['_input_objs']['scattered_photons'][0].photons

        if len(scattered_photons) > 0:
            # There are photons to be accumulated.
            # They should already have been transformed to pixel coordinates when they
            # were read in from file, so go ahead and accumulate them.

            scattered_photons.flux *= 1.e3

            sensor = base.get('sensor', Sensor())

            sensor.accumulate(scattered_photons, image)

        return image, []


RegisterExtraOutput('scattered_photons', ScatteredPhotonsBuilder())
RegisterInputType('scattered_photons', ScatteredPhotonsLoader(ScatteredPhotons))
RegisterImageType('LSST_ScatteredPhotonsImage', LSST_ScatteredPhotonsImageBuilder())
