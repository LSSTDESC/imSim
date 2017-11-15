"""
Code to add cosmic rays to LSST CCDs.  The cosmic ray hits are
harvested from real CCD dark frames taken for Camera electro-optical
testing.
"""
from __future__ import print_function
from collections import namedtuple, defaultdict
import numpy as np
import numpy.random as random
import astropy.io.fits as fits

__all__ = ['CosmicRays', 'write_cosmic_ray_catalog']

CR_Span = namedtuple('CR_Span', 'x0 y0 pixel_values'.split())

class CosmicRays(list):
    """
    List of cosmic rays.  Each CR is a list of CR_Span tuples derived from
    lsst.detection.Footprint spans, including starting pixel indices and
    pixel values in the serial direction.

    Attributes
    ----------
    num_pix: int
        Number of pixels for the sensors from which the CRs were extracted.
        Should be approximately 4000**2
    exptime: float
        Sum of exposure times (seconds) of the input darks.
    """
    def __init__(self, *args, **kwds):
        "Constructor."
        super(CosmicRays, self).__init__(*args, **kwds)
        self.num_pix = 0
        self.exptime = 0

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
            num_crs = random.poisson(len(self)/float(self.num_pix)/self.exptime
                                     *float(np.prod(image_array.shape))*exptime)
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
            cr = random.choice(self)
        else:
            cr = self[index]
        if pixel is None:
            pixel = (random.randint(image_array.shape[1]),
                     random.randint(image_array.shape[0]))
        for span in cr:
            for dx, value in enumerate(span.pixel_values):
                try:
                    image_array[pixel[1] + span.y0 - cr[0].y0,
                                pixel[0] + span.x0 - cr[0].x0 + dx] += value
                except IndexError:
                    pass
        return image_array

    def read_catalog(self, catalog_file, extname='COSMIC_RAYS'):
        """
        Read a FITS file containing a cosmic ray catalog.  New CR data
        will be appended.

        Parameters
        ----------
        catalog_file: str
            Filename of the cosmic ray catalog.
        extname: str, optional
            Extension name of the cosmic ray catalog.  Default: 'COSMIC_RAYS'
        """
        with fits.open(catalog_file) as catalog:
            cr_cat = catalog[extname]
            self.exptime += cr_cat.header['EXPTIME']
            self.num_pix = cr_cat.header['NUM_PIX']
            crs = defaultdict(list)
            for i, span in enumerate(cr_cat.data):
                crs[span[0]].append(CR_Span(*tuple(span)[1:]))
        self.extend(crs.values())

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

    image = galsim.ImageF(random.normal(1000., 7., (2000, 509)))
    imarray = copy.deepcopy(image.array)
    imarray = crs.paint(imarray, exptime=500)
    image = galsim.ImageF(imarray)
    image.write('galsim_image.fits')
