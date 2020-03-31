"""
Code to add cosmic rays to LSST CCDs.  The cosmic ray hits are
harvested from real CCD dark frames taken for Camera electro-optical
testing.
"""
from collections import namedtuple, defaultdict
import hashlib
import numpy as np
import astropy.io.fits as fits

__all__ = ['CosmicRays', 'write_cosmic_ray_catalog']

CR_Span = namedtuple('CR_Span', 'x0 y0 pixel_values'.split())

class CosmicRays(list):
    """
    This is a subclass of the Python list data type.  Each element of
    the list represents a cosmic ray (CR) that was extracted from a
    CCD dark exposure.  Each CR is in turn a list of CR_Span tuples
    derived from lsst.detection.Footprint spans, including starting
    pixel indices and pixel values in the serial direction.

    Attributes
    ----------
    num_pix: int
        Number of pixels for the sensors from which the CRs were extracted.
        Should be approximately 4000**2.
    exptime: float
        Sum of exposure times (seconds) of the input darks.
    ccd_rate: float
        Cosmic rays per second per CCD.
    rng: numpy.random.RandomState
        Random number generator.  If the seed is not set with
        .set_seed(...), this is just set to numpy.random.
    """
    def __init__(self):
        """
        Constructor.
        """
        super(CosmicRays, self).__init__()
        self.num_pix = 0
        self.exptime = 0
        self.ccd_rate = 0
        self.rng = np.random     # Use numpy.random module by default.

    def paint(self, image_array, exptime=30., num_crs=None):
        """
        Paint cosmic rays on an input image.  If num_crs is None, the
        number of CRs to paint is drawn from a predicted number based
        on the input exposure time and the CR rate per pixel inferred
        from the data.  The pixel value units are assumed to be
        electrons.

        Parameters
        ----------
        image_array: numpy.array
            Input image array onto which the CRs are painted.
        exptime: float, optional
            Exposure time (seconds) of the image to use for computing
            the number of CRs to paint. Default: 30
        num_crs: int, optional
            Number of CRs to paint, overriding the computed value.
            Default: None

        Returns
        -------
        numpy.array: The input image array with the CRs added.
        """
        if num_crs is None:
            ccd_frac = float(np.prod(image_array.shape))/self.num_pix
            num_crs = self.rng.poisson(exptime*self.ccd_rate*ccd_frac)
        for i in range(num_crs):
            image_array = self.paint_cr(image_array)
        return image_array

    def paint_cr(self, image_array, index=None, pixel=None):
        """
        Paint a single cosmic ray onto the input image array.

        Parameters
        ----------
        image_array: numpy.array
            Input image array onto which the CRs are painted.
        index: int, optional
            The list index of the CR to paint. If None (default),
            then a random CR will be selected.
        pixel: tuple(int, int), optional
            Pixel coordinates of the starting pixel of the footprint
            used for painting the CR.

        Returns
        -------
        numpy.array: The input image array with the CR added.
        """
        if index is None:
            cr = self.rng.choice(self)
        else:
            cr = self[index]
        if pixel is None:
            pixel = (self.rng.randint(image_array.shape[1]),
                     self.rng.randint(image_array.shape[0]))
        for span in cr:
            for dx, value in enumerate(span.pixel_values):
                try:
                    image_array[pixel[1] + span.y0 - cr[0].y0,
                                pixel[0] + span.x0 - cr[0].x0 + dx] += value
                except IndexError:
                    pass
        return image_array

    @staticmethod
    def read_catalog(catalog_file, ccd_rate=None, extname='COSMIC_RAYS'):
        """
        Read a FITS file containing a cosmic ray catalog.

        Parameters
        ----------
        catalog_file: str
            Filename of the cosmic ray catalog.
        ccd_rate: float, optional
            Mean number of cosmic rays per second per CCD.  If None (default),
            extract the rate from the catalog file.
        extname: str, optional
            Extension name of the cosmic ray catalog.  Default: 'COSMIC_RAYS'

        Returns
        -------
        CosmicRays instance.
        """
        cosmic_rays = CosmicRays()
        with fits.open(catalog_file) as catalog:
            cr_cat = catalog[extname]
            cosmic_rays.num_pix = cr_cat.header['NUM_PIX']
            cosmic_rays.exptime = cr_cat.header['EXPTIME']
            crs = defaultdict(list)
            for i, span in enumerate(cr_cat.data):
                crs[span[0]].append(CR_Span(*tuple(span)[1:]))
        super(CosmicRays, cosmic_rays).extend(crs.values())
        if ccd_rate is None:
            cosmic_rays.ccd_rate = float(len(cosmic_rays))/cosmic_rays.exptime
        else:
            cosmic_rays.ccd_rate = ccd_rate
        return cosmic_rays

    def set_seed(self, seed):
        """
        Set the random number seed for a numpy.random.RandomState
        instance that's held as the self.rng attribute.

        Parameters
        ----------
        seed: int
            The seed must be between 0 and 2**32 - 1
        """
        self.rng = np.random.RandomState(seed)

    @staticmethod
    def generate_seed(visit, det_name):

        """
        Deterministically construct an integer, appropriate for a random
        seed, from visit number and detector name.

        Parameters
        ----------
        visit: int
            Visit (or obsHistID) number.
        det_name: str
            Name of the sensor in the LSST focal plane, e.g., "R:2,2 S:1,1".

        Returns
        -------
        int

        Notes
        -----
        See https://stackoverflow.com/a/42089311
        """
        my_string = "{}{}".format(visit, det_name)
        my_int = int(hashlib.sha256(my_string.encode('utf-8')).hexdigest(), 16)

        # Return a seed between 0 and 2**32-1
        return my_int % (2**32 - 1)


def write_cosmic_ray_catalog(fp_id, x0, y0, pixel_values, exptime, num_pix,
                             outfile='cosmic_ray_catalog.fits', overwrite=True):
    """
    Write cosmic ray pixel footprint data as a binary table to a FITS file.

    Parameters
    ----------
    fp_id: sequence
        Sequence containing the footprint ids for each span.
    x0: sequence
        Sequence containing the starting x-index for each span.
    y0: sequence
        Sequence containing the y-index for each span.
    pixel_values: sequence of sequences
        Sequence containing the pixel values in each span.
    exptime: float
        Total exposure time (seconds) of the CR data.
    num_pix: int
        Number of pixels in the sensor used to detect these CRs.
    outfile: str, optional
        Filename of output catalog FITS file.
        Default: 'cosmic_ray_catalog.fits'
    overwrite: bool, optional
        Flag to overwrite an existing outfile. Default: True
    """
    hdu_list = fits.HDUList([fits.PrimaryHDU()])
    columns = [fits.Column(name='fp_id', format='I', array=fp_id),
               fits.Column(name='x0', format='I', array=x0),
               fits.Column(name='y0', format='I', array=y0),
               fits.Column(name='pixel_values', format='PJ()',
                           array=np.array(pixel_values, dtype=np.object))]
    hdu_list.append(fits.BinTableHDU.from_columns(columns))
    hdu_list[-1].name = 'COSMIC_RAYS'
    hdu_list[-1].header['EXPTIME'] = exptime
    hdu_list[-1].header['NUM_PIX'] = num_pix
    hdu_list.writeto(outfile, overwrite=overwrite)

if __name__ == '__main__':
    import os
    import copy
    import galsim
    import lsst.utils as lsstUtils

    crs = CosmicRays()
    catalog = os.path.join(lsstUtils.getPackageDir('imsim'),
                           'data', 'cosmic_ray_catalog.fits.gz')
    crs.read_catalog(catalog)

    image = galsim.ImageF(np.random.normal(1000., 7., (2000, 509)))
    imarray = copy.deepcopy(image.array)
    imarray = crs.paint(imarray, exptime=500)
    image = galsim.ImageF(imarray)
    image.write('galsim_image.fits')
