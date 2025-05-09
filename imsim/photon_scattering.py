import galsim
from galsim.config.extra import ExtraOutputBuilder, RegisterExtraOutput
from galsim.config import ImageBuilder, RegisterImageType, InputLoader, RegisterInputType
from astropy.io.fits import BinTableHDU

# from .lsst_image import LSST_ImageBuilderBase
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


class ScatteredPhotonsInput(object):
    """A class to read in the scattered photons that were written to file
    during an earlier first pass.
    """

    def __init__(self, file_name, wcs, det, xsize=4096, ysize=4096, logger=None):
        """
        Initialize the scattered photons input class.

        Parameters:
            file_name: str
                The name of the file to read.
            wcs: galsim.WCS
                The WCS object to use for the image.
            det: lsst.afw.cameraGeom.Detector
                The detector for the current sensor.
            xsize: int
                The x size in pixels of the CCD. (default: 4096)
            ysize: int
                The y size in pixels of the CCD. (default: 4096)
            logger: logging.logger
                A logger object. (default: None)
        """
        self.file_name = file_name
        self.wcs = wcs
        self.det = det
        self.xsize = xsize
        self.ysize = ysize
        self.logger = logger
        self._photons = None

    def read_photons(self):
        """Read the scattered photons from the file.
        """
        # Read the scattered photons from the file then convert them from
        # focal plane coordinates to pixel coordinates.
        self.photons = galsim.PhotonArray.read(self.file_name)
        self.photons.x, self.photons.y = focal_to_pixel(self.photons.x, self.photons.y, self.det)

    @property
    def photons(self):
        """Get the scattered photons.
        """
        if self._photons is None:
            self.read_photons()
        return self._photons


class ScatteredPhotonsLoader(InputLoader):
    """
    Class to load scattered photons from file.
    """
    def getKwargs(self, config, base, logger):
        req = {'file_name': str}
        opt = {}
        kwargs, safe = galsim.config.GetAllParams(config, base, req=req,
                                                  opt=opt)
        wcs = galsim.config.BuildWCS(base['image'], 'wcs', base, logger=logger)
        kwargs['wcs'] = wcs
        kwargs['xsize'] = base.get('det_xsize', 4096)
        kwargs['ysize'] = base.get('det_ysize', 4096)
        kwargs['logger'] = logger

        # For now assume the scattered photons on file can be used for all detectors.
        safe = True
        return kwargs, safe


# class LSST_ScatteredPhotonsImageBuilder(LSST_ImageBuilderBase):

#     def setup(self, config, base, logger):
#         """Set up the scattered photons image type.
#         """
#         # We need to set the pixel scale to be the same as the camera's pixel scale
#         # so that we can convert between focal plane coordinates and pixel coordinates.
#         self.pixel_scale = base['output']['pixel_scale']
#         self.camera = get_camera(base['output']['camera'])[base['det_name']]
#         self.focal_plane = self.camera.get_focal_plane()
#         self.focal_plane.set_pixel_scale(self.pixel_scale)

#     def draw(self, config, base, logger):
#         """Draw the scattered photons to the image.
#         """
#         # Get the scattered photons from the base config.
#         if 'scattered_photons' in base and len(base['scattered_photons']) > 1:
#             self.data = PhotonArray.concatenate(base['scattered_photons'])
#             # We need to convert the focal plane coordinates to pixel coordinates
#             # so that we can draw them on the image.
#             self.data.x, self.data.y = self.focal_plane.focal_to_pixel(self.data.x, self.data.y)
#         else:
#             self.data = PhotonArray(N=0)


RegisterExtraOutput('scattered_photons', ScatteredPhotonsBuilder())
RegisterInputType('scattered_photons', ScatteredPhotonsLoader())
# RegisterImageType('LSST_ScatteredPhotonsImage', LSST_ScatteredPhotonsImageBuilder)
