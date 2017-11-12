#!/usr/bin/env python
"""
Script to extract cosmic rays from single sensor dark frames.  This
code uses the Camera team's eotest package.  In addition to darks,
Fe55 gains from the standard eotest suite are required.
"""
import os
import glob
import argparse
import tempfile
import numpy as np
import numpy.random as random
import astropy.io.fits as fits
import lsst.afw.detection as afw_detect
import lsst.afw.image as afw_image
import lsst.afw.math as afw_math
import lsst.eotest.image_utils as imutils
import lsst.eotest.sensor as sensorTest

def bg_image(ccd, amp, nx=10, ny=10):
    """
    Generate a background image using lsst.afw.math.makeBackground.
    """
    bg_ctrl = afw_math.BackgroundControl(nx, ny, ccd.stat_ctrl)
    bg = afw_math.makeBackground(ccd[amp], bg_ctrl)
    return bg.getImageF()

class IsMasked(object):
    """
    Functor class to determine if a candidate CR overlaps with a
    masked region of the sensor.
    """
    def __init__(self, mask_file):
        """
        Parameters
        ----------
        mask_file: str
            Filename of a mask file generated with the eotest code.
        """
        self.ccd = sensorTest.MaskedCCD(mask_file)
    def __call__(self, amp, footprint):
        """
        Return True if any part of the footprint overlaps with a
        masked region.

        Parameters
        ----------
        amp: int
            Amplifier number corresponding to the HDU containing the
            corresponding segment data.
        footprint: lsst.afw.detection.Footprint
            The footprint of the candidate cosmic ray.

        Returns
        -------
        bool: True if footprint overlaps with any masked region.
        """
        mask_imarr = self.ccd[amp].getImage().getArray()
        for span in footprint.getSpans():
            iy = span.getY()
            for ix in range(span.getX0(), span.getX1()+1):
                if mask_imarr[iy][ix] > 0:
                    return True
        return False

def make_mask(med_file, gains, outfile, ethresh=0.1, colthresh=20,
              mask_plane='BAD'):
    """
    Create bright pixel mask from a medianed single sensor dark frame.

    Parameters
    ----------
    med_file: str
        Filename of the medianded dark frame.
    gains: dict
        Gains (e-/ADU) of the sixteen amplifiers in the CCD, keyed by amp.
    outfile: str
        Output filename of the mask file.
    ethresh: float, optional
        Threshold in e- per second per pixel for identifying a bright
        pixel defect. Default: 0.1
    colthresh: float, optional
        Threshold in e- per second per column for identifying a bright
        column.  Default: 20
    mask_plane: str, optional
        Name of the mask plane. Default: 'BAD'

    Returns
    -------
    IsMasked object
    """
    ccd = sensorTest.MaskedCCD(med_file)
    exptime = ccd.md.get('EXPTIME')
    pixels, columns = {}, {}
    for amp in ccd:
        bright_pixels \
            = sensorTest.BrightPixels(ccd, amp, exptime, gains[amp],
                                      ethresh=ethresh, colthresh=colthresh)
        pixels[amp], columns[amp] = bright_pixels.find()
    sensorTest.generate_mask(med_file, outfile, mask_plane, pixels=pixels,
                             columns=columns)
    return IsMasked(outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dark_frame_pattern',
                        help='glob pattern for dark frames')
    parser.add_argument('eotest_results',
                        help='eotest results file with the gains per amp')
    parser.add_argument('--outfile', default='cosmic_ray_catalog.fits',
                        type=str, help='Output file name for CR catalog')
    parser.add_argument('--medianed_dark', default=None, type=str,
                        help='Filename of medianed dark.')
    parser.add_argument('--nsig', default=7, type=float,
                        help='Number of sigma threshold for detecting CRs')
    args = parser.parse_args()

    darks = glob.glob(args.dark_frame_pattern)

    eo_results = sensorTest.EOTestResults(args.eotest_results)
    gains = {amp: gain for amp, gain in
             zip(eo_results['AMP'], eo_results['GAIN'])}

    med_file = args.medianed_dark
    if med_file is None:
        med_file = tempfile.NamedTemporaryFile(prefix='tmp_med_file_',
                                               dir='.', suffix='.fits').name
        imutils.fits_median_file(darks, med_file, bitpix=-32)

    mask_file = tempfile.NamedTemporaryFile(prefix='tmp_mask_', dir='.',
                                            suffix='.fits').name
    is_masked = make_mask(med_file, gains, mask_file)

    fp_id, x0, y0, pixel_values = [], [], [], []
    index = -1
    exptime = 0
    num_pix = 0
    for dark in darks:
        print "processing", dark
        ccd = sensorTest.MaskedCCD(dark, mask_files=(mask_file,))
        exptime += ccd.md.get('EXPTIME')
        for amp in ccd.keys():
            image = ccd[amp]
            image -= bg_image(ccd, amp)
            image = image.Factory(image, ccd.amp_geom.imaging)
            num_pix += np.prod(image.getImage().getArray().shape)

            flags = afw_math.MEDIAN | afw_math.STDEVCLIP
            stats = afw_math.makeStatistics(image, flags, ccd.stat_ctrl)
            median = stats.getValue(afw_math.MEDIAN)
            stdev = stats.getValue(afw_math.STDEVCLIP)
            threshold = afw_detect.Threshold(median + args.nsig*stdev)

            fp_set = afw_detect.FootprintSet(image, threshold)
            for fp in fp_set.getFootprints():
                if is_masked(amp, fp):
                    continue
                index += 1
                for span in fp.getSpans():
                    fp_id.append(index)
                    iy = span.getY()
                    ix0, ix1 = span.getX0(), span.getX1()
                    x0.append(ix0)
                    y0.append(iy)
                    row = gains[amp]*image.getImage().getArray()[iy, ix0:ix1+1]
                    pixel_values.append(np.array(row, dtype=np.int))

    hdu_list = fits.HDUList([fits.PrimaryHDU()])
    columns = [fits.Column(name='fp_id', format='I', array=fp_id),
               fits.Column(name='x0', format='I', array=x0),
               fits.Column(name='y0', format='I', array=y0),
               fits.Column(name='pixel_values', format='PJ()',
                           array=np.array(pixel_values, dtype=np.object))]
    hdu_list.append(fits.BinTableHDU.from_columns(columns))
    hdu_list[-1].name = 'COSMIC_RAYS'
    hdu_list[-1].header['EXPTIME'] = exptime
    hdu_list[-1].header['NUM_PIX'] = int(float(num_pix)/len(darks))
    hdu_list.writeto(args.outfile, overwrite=True)

    for item in glob.glob('tmp_*.fits'):
        os.remove(item)
