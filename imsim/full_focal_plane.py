import os

import galsim
from galsim.config.extra import ExtraOutputBuilder, RegisterExtraOutput
from galsim.config import RegisterImageType, InputLoader, RegisterInputType
from galsim.sensor import Sensor
from astropy.io.fits import BinTableHDU

from .lsst_image import LSST_ImageBuilderBase
from .utils import pixel_to_focal, focal_to_pixel
from .camera import get_camera

def gather_out_of_bounds_photons(image_bounds, photons):
    """ Given image bounds and a list of photon arrays, gather the photons
    that fall outside the bounds, copy them into a new PhotonArray, and return
    them.

    Parameters:
        image_bounds: galsim.Bounds
            The bounds used to determine which photons fall outside.
        photons: List(galsim.PhotonArray)
            A list of PhotonArrays containing all the photons to check.

    Returns:
        out_of_bounds_photons: galsim.PhotonArray
            A new PhotonArray containing the photons falling outside the bounds.
            If no such photons are found, the PhotonArray will be empty (N=0).
    """
    out_of_bounds_indices = [i for i in range(len(photons))
                             if not image_bounds.includes(photons.x[i], photons.y[i])]
    if len(out_of_bounds_indices) > 0:
        out_of_bounds_photons = galsim.PhotonArray(len(out_of_bounds_indices))
        out_of_bounds_photons.copyFrom(photons,
                                       target_indices=slice(len(out_of_bounds_indices)),
                                       source_indices=out_of_bounds_indices,
                                       do_xy=True,
                                       do_flux=True,
                                       do_other=False)
    else:
        out_of_bounds_photons = galsim.PhotonArray(N=0)
    return out_of_bounds_photons

# An extra output type to enable two-pass drawing of off-detector photons. This
# output should be used in the first pass to write off-detector photons to file,
# while the second pass will read the off-detector photons then go through all
# the images again and draw the photons on top of the first pass image.
class OffDetectorPhotonsBuilder(ExtraOutputBuilder):
    """Build photon arrays containing the off-detector photons found during an
    image's construction and write them to file.
    """

    def processImage(self, index, obj_nums, config, base, logger):
        """After each image is drawn, concatenate the list of off-detector
        photon arrays found from each batch/sub-batch and store in data ready
        to be written to file later on.
        """
        if 'off_detector_photons' in base and len(base['off_detector_photons']) > 0:
            self.data[index] = galsim.PhotonArray.concatenate(base['off_detector_photons'])
            # We need to store the photons using focal plane coordinates so they
            # can be accumulated in the second pass on an arbitrary sensor.
            detector = get_camera(base['output']['camera'])[base['det_name']]
            self.data[index].x, self.data[index].y = pixel_to_focal(self.data[index].x, self.data[index].y, detector)
        else:
            # If we've been told to write off-detector photons but found none,
            # let's at least write an empty PhotonArray.
            self.data[index] = galsim.PhotonArray(N=0)

    def finalize(self, config, base, main_data, logger):
        return self.data

    def writeFile(self, file_name, config, base, logger):
        """Write the photon array to fits.
        """
        for photon_array in self.final_data:
            photon_array.write(file_name)

    def writeHdu(self, config, base, logger):
        """We don't want to write the off-detector photons to FITS, so return an
        empty BinTable in lieu of something else.
        """
        return BinTableHDU(data=None)


class OffDetectorPhotons(object):
    """A class to hold the photons which fell outside the sensor being drawn
    by the task that createed them during the first pass. They were saved to file
    then, and will now be read in this second pass to be accumulated on other sensors.
    """

    def __init__(self, file_name, camera, det_name, dir=None, xsize=4096, ysize=4096, logger=None):
        """
        Initialize the off-detector photons input class.

        Parameters:
            file_name: str
                The name of the file to read.
            camera: str
                The name of the camera containing the detector.
            det_name: str
                The name of the detector to use for photon coordinate transformations.
            dir: str
                The directory where the file is. (default: None)
            xsize: int
                The x size in pixels of the CCD. (default: 4096)
            ysize: int
                The y size in pixels of the CCD. (default: 4096)
            logger: logging.logger
                A logger object. (default: None)
        """
        self.file_name = file_name
        if dir is not None:
            self.file_name = os.path.join(dir, self.file_name)
        self.det = get_camera(camera)[det_name]
        self.xsize = xsize
        self.ysize = ysize
        self.logger = logger
        # self.photons = None
        self.photons = galsim.PhotonArray.read(self.file_name)
        self.photons.x, self.photons.y = focal_to_pixel(self.photons.x, self.photons.y, self.det)


class OffDetectorPhotonsLoader(InputLoader):
    """
    Class to load off-detector photons from file.
    """
    def getKwargs(self, config, base, logger):
        req = {'file_name': str, 'camera': str, 'det_name': str}
        opt = {'dir': str}
        kwargs, safe = galsim.config.GetAllParams(config, base, req=req,
                                                  opt=opt)
        kwargs['xsize'] = base.get('det_xsize', 4096)
        kwargs['ysize'] = base.get('det_ysize', 4096)
        kwargs['logger'] = logger

        # For now assume the off-detector photons on file can be used for
        # all detectors.
        # Change assumption, so now we consider that we have a per-detector
        # input file that cannot be reused for other detectors.
        safe = False
        return kwargs, safe


# A class to build one of the images making up a full focal plane, used in a
# second pass which uses as input images from the first pass along with any
# photons which fell off-detector.
class LSST_FocalPlaneImageBuilder(LSST_ImageBuilderBase):

    def setup(self, config, base, image_num, obj_num, ignore, logger):
        xsize, ysize = super().setup(config, base, image_num, obj_num, ignore, logger)
        # Disable application of the addNoise function in
        # LSST_ImageBuilderBase.addNoise since it was already applied
        # in pass1 and not needed for the added off-detector photons
        self.add_noise = False
        return xsize, ysize

    def buildImage(self, config, base, image_num, _obj_num, logger):
        """Draw the off-detector photons to the image.
        """
        # Make sure we have an input image and off-detector photons to draw.
        # Without them, this type of image won't work.
        image = base['current_image']
        if not isinstance(base['_input_objs']['off_detector_photons'][0], OffDetectorPhotons):
            raise galsim.config.GalSimConfigError(
                "When using LSST_FocalPlaneImage, you must provide an off_detector_photons input.",)
        
        off_detector_photons = base['_input_objs']['off_detector_photons'][0].photons

        if len(off_detector_photons) > 0:
            # There are photons to be accumulated. They should already have been
            # transformed to pixel coordinates when they were read from file, so
            # go ahead and accumulate them.

            sensor = base.get('sensor', Sensor())

            sensor.accumulate(off_detector_photons, image)

        return image, []


RegisterExtraOutput('off_detector_photons', OffDetectorPhotonsBuilder())
RegisterInputType('off_detector_photons', OffDetectorPhotonsLoader(OffDetectorPhotons))
RegisterImageType('LSST_FocalPlaneImage', LSST_FocalPlaneImageBuilder())
