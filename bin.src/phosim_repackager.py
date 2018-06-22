#!/usr/bin/env python
"""
Code to repackage phosim amplifier files into single sensor
multi-extension FITS files, with an HDU per amplifier.
"""
import os
import sys
import glob
import time
from collections import defaultdict, OrderedDict
import astropy.io.fits as fits
import astropy.time
import lsst.utils
import desc.imsim

noao_section_keyword = desc.imsim.ImageSource._noao_section_keyword

class PhoSimRepackager:
    """
    Class to repackage phosim amplifier files into single sensor
    MEFs with one HDU per amp.
    """
    def __init__(self, seg_file=None):
        if seg_file is None:
            seg_file = os.path.join(lsst.utils.getPackageDir('imSim'),
                                    'data', 'segmentation_itl.txt')
        self.fp_props = desc.imsim.FocalPlaneInfo.read_phosim_seg_file(seg_file)

    def __call__(self, visit_dir, out_dir=None):
        """
        Parameters
        ----------
        visit_dir: str
            Directory containing the phosim amplifier for a given visit.
        out_dir: str [None]
            Output directory for MEF files. If None, then a directory
            with name v<visit #>-<band> will be created in the cwd.
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

        # Use this list to write the image extensions in the order
        # specified by LCA-10140 via an OrderedDict:
        channels = '10 11 12 13 14 15 16 17 07 06 05 04 03 02 01 00'.split()
        segments = OrderedDict()
        for channel in channels:
            segments[channel] = None

        for sensor_id in amp_files:
            sys.stdout.write(sensor_id + '  ')
            t0 = time.time()
            sensor = fits.HDUList(fits.PrimaryHDU())
            for fn in amp_files[sensor_id]:
                channel = os.path.basename(fn).split('_')[6][1:]
                segments[channel] = fits.open(fn)[0]
            for channel, hdu in segments.items():
                sensor.append(hdu)
                sensor[-1].header['EXTNAME'] = 'Segment%s' % channel
                amp_name = '_'.join((sensor_id, 'C' + channel))
                amp_props = self.fp_props.get_amp(amp_name)
                sensor[-1].header['DATASEC'] \
                    = noao_section_keyword(amp_props.imaging)
                sensor[-1].header['DETSEC'] \
                    = noao_section_keyword(amp_props.mosaic_section,
                                           flipx=amp_props.flip_x,
                                           flipy=amp_props.flip_y)
                sensor[-1].header['BIASSEC'] \
                    = noao_section_keyword(amp_props.serial_overscan)

            # Set keywords in primary HDU, extracting most of the relevant
            # ones from the first phosim amplifier file.
            raft, ccd = sensor_id.split('_')
            sensor[0].header['EXPTIME'] = sensor[1].header['EXPTIME']
            sensor[0].header['DARKTIME'] = sensor[1].header['DARKTIME']
            sensor[0].header['RUNNUM'] = sensor[1].header['OBSID']
            sensor[0].header['MJD-OBS'] = sensor[1].header['MJD-OBS']
            sensor[0].header['DATE-OBS'] \
                = astropy.time.Time(sensor[1].header['MJD-OBS'],
                                    format='mjd').isot
            sensor[0].header['FILTER'] = sensor[1].header['FILTER']
            sensor[0].header['LSST_NUM'] = sensor_id
            sensor[0].header['CHIPID'] = sensor_id
            sensor[0].header['OBSID'] = sensor[1].header['OBSID']
            sensor[0].header['TESTTYPE'] = 'PHOSIM'
            sensor[0].header['IMGTYPE'] = 'SKYEXP'
            sensor[0].header['MONOWL'] = -1
            sensor[0].header['RAFTNAME'] = raft
            sensor[0].header['SENSNAME'] = ccd
            tokens = os.path.basename(amp_files[sensor_id][0]).split('_')
            outfile = '_'.join(tokens[:6] + tokens[7:]).replace('.gz', '')
            outfile = os.path.join(out_dir, outfile)
            sensor.writeto(outfile, overwrite=True)
            print(time.time() - t0)
            sys.stdout.flush()

if __name__ == '__main__':
    import argparse

    parser \
        = argparse.ArgumentParser(description="Repackager for phosim amp files")
    parser.add_argument('visit_dir', type=str, help="visit directory")
    parser.add_argument('--out_dir', type=str, default=None,
                        help="output directory")
    parser.add_argument('--seg_file', type=str, default=None,
                        help="segmentation.txt file")
    args = parser.parse_args()

    repackage_data = PhoSimRepackager(args.seg_file)
    repackage_data(args.visit_dir, out_dir=args.out_dir)
