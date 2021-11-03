"""
Code to add cosmic rays to LSST CCDs.  The cosmic ray hits are
harvested from real CCD dark frames taken for Camera electro-optical
testing.
"""
import os
from collections import namedtuple, defaultdict
import hashlib
import numpy as np
import astropy.io.fits as fits
import galsim

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
    """
    def __init__(self, ccd_rate, catalog_file):
        """
        Constructor.
        """
        super().__init__()
        self._read_catalog(catalog_file, ccd_rate)

    def paint(self, image_array, rng, exptime=30., num_crs=None):
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
        rng: galsim.BaseDeviate
            Random number generator.
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
            pd = galsim.PoissonDeviate(rng, exptime*self.ccd_rate*ccd_frac)
            num_crs = int(pd())
        for i in range(num_crs):
            image_array = self.paint_cr(image_array, rng)
        return image_array

    def paint_cr(self, image_array, rng, index=None, pixel=None):
        """
        Paint a single cosmic ray onto the input image array.

        Parameters
        ----------
        image_array: numpy.array
            Input image array onto which the CRs are painted.
        rng: galsim.BaseDeviate
            Random number generator.
        index: int [None] The list index of the CR to paint. If None,
            then a random CR will be selected.
        pixel: tuple(int, int) [None]
            Pixel coordinates of the starting pixel of the footprint
            used for painting the CR. If None, then a random pixel in
            the image_array will be selected.

        Returns
        -------
        numpy.array: The input image array with the CR added.
        """
        ud = galsim.UniformDeviate(rng)
        if index is None:
            index = int(ud()*len(self))
        cr = self[index]
        if pixel is None:
            pixel = (int(ud()*image_array.shape[1]),
                     int(ud()*image_array.shape[0]))
        for span in cr:
            for dx, value in enumerate(span.pixel_values):
                try:
                    image_array[pixel[1] + span.y0 - cr[0].y0,
                                pixel[0] + span.x0 - cr[0].x0 + dx] += value
                except IndexError:
                    pass
        return image_array

    def _read_catalog(self, catalog_file, ccd_rate, extname='COSMIC_RAYS'):
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
        with fits.open(catalog_file) as catalog:
            cr_cat = catalog[extname]
            self.num_pix = cr_cat.header['NUM_PIX']
            self.exptime = cr_cat.header['EXPTIME']
            crs = defaultdict(list)
            for i, span in enumerate(cr_cat.data):
                crs[span[0]].append(CR_Span(*tuple(span)[1:]))
        super().extend(crs.values())
        if ccd_rate is None:
            self.ccd_rate = float(len(self))/self.exptime
        else:
            self.ccd_rate = ccd_rate


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
