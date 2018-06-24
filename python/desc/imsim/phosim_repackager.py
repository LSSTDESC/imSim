#!/usr/bin/env python
"""
Code to repackage phosim amplifier files into single sensor
multi-extension FITS files, with an HDU per amplifier.
"""
import os
import sys
import glob
import time
from collections import defaultdict
import astropy.io.fits as fits
import astropy.time
import lsst.obs.lsstCam as lsstCam

__all__ = ['PhoSimRepackager']


def noao_section_keyword(bbox, flipx=False, flipy=False):
    """
    Convert bounding boxes into NOAO section keywords.

    Parameters
    ----------
    bbox : lsst.afw.geom.Box2I
        Bounding box.
    flipx : bool
        Flag to indicate that data should be flipped in the x-direction.
    flipy : bool
        Flag to indicate that data should be flipped in the y-direction.
    """
    xmin, xmax = bbox.getMinX()+1, bbox.getMaxX()+1
    ymin, ymax = bbox.getMinY()+1, bbox.getMaxY()+1
    if flipx:
        xmin, xmax = xmax, xmin
    if flipy:
        ymin, ymax = ymax, ymin
    return '[%i:%i,%i:%i]' % (xmin, xmax, ymin, ymax)


class PhoSimRepackager:
    """
    Class to repackage phosim amplifier files into single sensor
    MEFs with one HDU per amp.
    """
    def __init__(self):
        self.amp_info_records = list(list(lsstCam.LsstCamMapper().camera)[0])

    def process_visit(self, visit_dir, out_dir=None, verbose=False):
        """
        Parameters
        ----------
        visit_dir: str
            Directory containing the phosim amplifier for a given visit.
        out_dir: str [None]
            Output directory for MEF files. If None, then a directory
            with name v<visit #>-<band> will be created in the cwd.
        verbose: bool [False]
            Set to True to print out time for processing each sensor.
        """
        phosim_amp_files \
            = sorted(glob.glob(os.path.join(visit_dir, 'lsst_a_*')))
        amp_files = defaultdict(list)
        for item in phosim_amp_files:
            sensor_id = '_'.join(item.split('_')[4:6])
            amp_files[sensor_id].append(item)
        if out_dir is None:
            tokens = os.path.basename(phosim_amp_files[0]).split('_')
            out_dir = 'v%s-%s' % (tokens[2], 'ugrizy'[int(tokens[3][1])])
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        for sensor_id in amp_files:
            if verbose:
                sys.stdout.write(sensor_id + '  ')
            t0 = time.time()
            self.repackage(amp_files[sensor_id], out_dir=out_dir)
            if verbose:
                print(time.time() - t0)
                sys.stdout.flush()

    def repackage(self, phosim_amp_files, out_dir='.'):
        """
        Repackage a collection of phosim amplifier files for a
        single sensor into a multi-extension FITS file.

        Parameters
        ----------
        phosim_amp_files: list
            List of phosim amplifier filenames.
        """
        # Create the HDUList to contain the MEF data.
        sensor = fits.HDUList(fits.PrimaryHDU())

        # Extract the data for each segment from the FITS files
        # into a dictionary keyed by channel id.
        segments = dict()
        for fn in phosim_amp_files:
            channel = os.path.basename(fn).split('_')[6][1:]
            segments[channel] = fits.open(fn)[0]

        # Set the NOAO section keywords based on the pixel geometry
        # in the LsstCam object.
        for amp in self.amp_info_records:
            hdu = segments[amp.get('name')[1:]]
            hdu.header['EXTNAME'] = 'Segment%s' % amp.get('name')[1:]
            hdu.header['DATASEC'] = noao_section_keyword(amp.getRawDataBBox())
            hdu.header['DETSEC'] \
                = noao_section_keyword(amp.getBBox(),
                                       flipx=amp.get('raw_flip_x'),
                                       flipy=amp.get('raw_flip_y'))
            # Remove the incorrect BIASSEC keyword that phosim writes.
            try:
                hdu.header.remove('BIASSEC')
            except KeyError:
                pass
            sensor.append(hdu)

        # Set keywords in primary HDU, extracting most of the relevant
        # ones from the first phosim amplifier file.
        chip_id = sensor[1].header['CHIPID']
        raft, ccd = chip_id.split('_')
        sensor[0].header['EXPTIME'] = sensor[1].header['EXPTIME']
        sensor[0].header['DARKTIME'] = sensor[1].header['DARKTIME']
        sensor[0].header['RUNNUM'] = sensor[1].header['OBSID']
        sensor[0].header['MJD-OBS'] = sensor[1].header['MJD-OBS']
        sensor[0].header['DATE-OBS'] \
            = astropy.time.Time(sensor[1].header['MJD-OBS'], format='mjd').isot
        sensor[0].header['FILTER'] = sensor[1].header['FILTER']
        sensor[0].header['LSST_NUM'] = chip_id
        sensor[0].header['CHIPID'] = chip_id
        sensor[0].header['OBSID'] = sensor[1].header['OBSID']
        sensor[0].header['TESTTYPE'] = 'PHOSIM'
        sensor[0].header['IMGTYPE'] = 'SKYEXP'
        sensor[0].header['MONOWL'] = -1
        sensor[0].header['RAFTNAME'] = raft
        sensor[0].header['SENSNAME'] = ccd

        tokens = os.path.basename(phosim_amp_files[0]).split('_')
        outfile = '_'.join(tokens[:6] + tokens[7:])
        outfile = os.path.join(out_dir, outfile)

        # astropy's gzip compression is very slow, so remove any .gz
        # extension from the computed output filename and gzip the
        # files later.
        if outfile.endswith('.gz'):
            outfile = outfile[:-len('.gz')]

        sensor.writeto(outfile, overwrite=True)
