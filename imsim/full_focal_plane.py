import os

import numpy as np

import galsim
from galsim.config.extra import ExtraOutputBuilder, RegisterExtraOutput
from galsim.config import RegisterImageType, InputLoader, RegisterInputType
from galsim.sensor import Sensor
from astropy.io.fits import BinTableHDU

from .lsst_image import LSST_ImageBuilderBase
from .utils import pixel_to_focal, focal_to_pixel
from .camera import get_camera, get_gs_bounds


def gather_photons_by_bounds(image_bounds, photons, inside=True):
    """Given image bounds and a list of photon arrays, gather the photons
    that fall inside (or outside) the bounds, copy them into a new
    PhotonArray, and return them.

    Parameters:
        image_bounds: galsim.Bounds
            The bounds used to determine which photons fall outside.
        photons: List(galsim.PhotonArray)
            A list of PhotonArrays containing all the photons to check.
        inside: bool
            If True, then return photons that are inside the image bounds.
            Otherwise, return the photons outside of the bounds.

    Returns:
        photons: galsim.PhotonArray
            A new PhotonArray containing the photons falling inside
            (or outside) the bounds.  If no such photons are found,
            the PhotonArray will be empty (N=0).
    """
    # Use numpy to determine which photons are inside the image bounds.
    indices = np.where(
        (image_bounds.xmin <= photons.x)
        & (photons.x <= image_bounds.xmax)
        & (image_bounds.ymin <= photons.y)
        & (photons.y <= image_bounds.ymax)
    )[0]
    if not inside:
        # Set indices to the complement of the inside indices.
        indices = sorted(set(range(len(photons))) - set(indices))

    if len(indices) > 0:
        target_photons = galsim.PhotonArray(len(indices))
        target_photons.copyFrom(photons,
                                target_indices=slice(len(indices)),
                                source_indices=indices,
                                do_xy=True,
                                do_flux=True,
                                do_other=False)
    else:
        target_photons = galsim.PhotonArray(N=0)
    return target_photons


# An extra output type to enable two-pass drawing of off-detector photons. This
# output should be used in the first pass to write off-detector photons to file,
# while the second pass will read the off-detector photons then go through all
# the images again and draw the photons on top of the first pass image.
class OffDetectorPhotonsBuilder(ExtraOutputBuilder):
    """Build photon arrays containing the off-detector photons found during an
    image's construction and write them to file. This item will typically be
    included in the config for the first pass of a full focal plane run.
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

    def __init__(self, camera, det_name, file_name=None, file_pattern=None, photons=None, dir=None, logger=None):
        """
        Initialize the off-detector photons input class.

        Parameters:
            camera: str
                The name of the camera containing the detector.
            det_name: str
                The name of the detector to use for photon coordinate transformations.
            file_name: str
                The name of the file to read. (default: None)
            file_pattern: str
                The file pattern to use if up to 189 are to be read. (default: None)
            dir: str
                The directory where the file is. (default: None)
            logger: logging.logger
                A logger object. (default: None)
        """
        logger = galsim.config.LoggerWrapper(logger)
        camera = get_camera(camera)
        self.det_name = det_name
        self.det = camera[det_name]
        # Lazy read the off-detector photons when they're accessed for the first time.
        self.photons = None
        # The Loader should have ensured we have one but not both of file_name
        # or file_pattern from the config, but here we also allow the photons to
        # be set directly during initialisation in case when instantiating in code.
        # So, make sure again that two of these are None.
        if sum([file_name is None, file_pattern is None, photons is None]) != 2:
            raise galsim.GalSimIncompatibleValuesError("OffDetectorPhotons must be initialized with exactly one of file_name, file_pattern, or photons.",values={"file_name": file_name, "file_pattern": file_pattern, "photons": photons})

        if file_name is not None:
            # Only have a single file to read.
            file_names = [file_name]
        elif file_pattern is not None:
            # We want to read up to 188 files: potentially photons from every
            # detector except this one (as those photons were drawn normally in
            # the first pass). We may want change this to allow det_num_start
            # and det_num_end to be set in the config in a similar way to output
            # where it's controlled with det_num.first, nfiles and only_dets.
            det_num_start = 0
            det_num_end = 189
            file_names = [file_pattern.replace("DETNAME", camera[i].getName()).replace("DETNUM", "{:03d}".format(i))
                          for i in range(det_num_start, det_num_end)
                          if camera[i].getName() != det_name
                          ]
        else:
            # An OffDetectorPhotons object can be be directly initialized in
            # code with a photon array instead of reading from file.
            file_names = []
            self.photons = photons
        if dir is not None:
            file_names = [os.path.join(dir, fname) for fname in file_names]
        # Allow for the possibility that not all the 188 files exist.
        self.file_names = [fname for fname in file_names if os.path.isfile(fname)]
        if len(self.file_names) == 0:
            logger.warning(f"Detector {det_name} did not find any expected off detector photon files: {file_names}. No photons can be read from file.")

    @property
    def photons(self):
        if self._photons is None:
            self._photons = self._read_photons()
        return self._photons

    @photons.setter
    def photons(self, photon_array):
        # Allow the photons to be set manually from outside the class if necessary.
        self._photons = photon_array

    def _read_photons(self, logger=None):
        logger = galsim.config.LoggerWrapper(logger)
        det_bounds = get_gs_bounds(self.det.getBBox())
        if len(self.file_names) > 0:
            photon_storage = []
            logger.info(f"Detector {self.det_name} reading {len(self.file_names)} files for off-detector photons.")
            for file_name in self.file_names:
                phot_array = galsim.PhotonArray.read(file_name)
                phot_array.x, phot_array.y = focal_to_pixel(phot_array.x, phot_array.y, self.det)
                phot_array = gather_photons_by_bounds(det_bounds, phot_array, inside=True)
                photon_storage.append(phot_array)
            photons = galsim.PhotonArray.concatenate(photon_storage)
        else:
            # If we've reached this point, we've initialised the class without
            # providing the names of any usable files and have now tried to
            # access the photons without setting them from outside.
            # Raise an error.
            raise galsim.GalSimConfigError(f"No readable files provided for OffDetectorPhotons for {self.det_name}; photons cannot be read.")
        return photons


class OffDetectorPhotonsLoader(InputLoader):
    """Class to load off-detector photons from file.
    These options should be provided in the config for the second pass of a full
    focal plane run.
    """
    def getKwargs(self, config, base, logger):
        req = {'camera': str, 'det_name': str}
        opt = {'file_name': str, 'file_pattern': str, 'dir': str}
        kwargs, safe = galsim.config.GetAllParams(config, base, req=req,
                                                  opt=opt)
        if ('file_name' in kwargs) == ('file_pattern' in kwargs):
            raise galsim.GalSimConfigError('off_detector_photons must contain one of either file_name or file_pattern in config')
        if 'file_pattern' in kwargs and ("DETNAME" not in kwargs['file_pattern'] or "DETNUM" not in kwargs['file_pattern']):
            raise galsim.GalSimConfigError('file_pattern must contain both "DETNAME" and "DETNUM" placeholders to be replaced '
                                           'with the detector name and number when reading off-detector photon files.')
        kwargs['logger'] = logger

        # Base assumption (for now) is that we have a per-detector set of inputs
        # that cannot be reused for other detectors.
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

    def buildImage(self, config, base, image_num, obj_num, logger):
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
