import os
import copy
import numpy as np
import scipy
from collections import namedtuple
from astropy.io import fits
from astropy.time import Time
import galsim
from galsim.config import ExtraOutputBuilder, RegisterExtraOutput
from .bleed_trails import bleed_eimage
from .camera import Camera
from ._version import __version__

def section_keyword(bounds, flipx=False, flipy=False):
    """Package image bounds as a NOAO image section keyword value."""
    xmin, xmax = bounds.xmin, bounds.xmax
    ymin, ymax = bounds.ymin, bounds.ymax
    if flipx:
        xmin, xmax = xmax, xmin
    if flipy:
        ymin, ymax = ymax, ymin
    return '[%i:%i,%i:%i]' % (xmin, xmax, ymin, ymax)

# XXX: This is tested, but it's not currenly being called by the main config classes.
#      I also couldn't find where we used to call it in the old code.
#      Should we be running this somewhere?
def set_noao_keywords(hdu, slot_name):
    """
    Update the image header for one of the readout segments.  Adds raft-level
    coordinates (one set in Camera coordinates and one set rotated so the CCD
    orientation has serial direction horizontal).  Adds amplifier and rotated
    CCD coordinates as well.  See LCA-13501.

    Parameters
    ----------
    hdu : fits.ImageHDU
        FITS image whose header is being updated
    slot_name : str
        Name of the slot within the raft

    Returns
    -------
    fits.ImageHDU : The modified hdu.
    """
    hdu.header['SLOT'] = slot_name

    PixelParameters = namedtuple('PixelParameters',
                                 ('''dimv dimh ccdax ccday ccdpx ccdpy gap_inx
                                     gap_iny gap_outx gap_outy preh'''.split()))
    pixel_parameters = {
        'E2V': PixelParameters(2002, 512, 4004, 4096, 4197, 4200, 28,
                               25, 26.5, 25, 10),
        'ITL': PixelParameters(2000, 509, 4000, 4072, 4198, 4198, 27,
                               27, 26.0, 26, 3)}

    vendors = {'[1:4096,1:4004]': 'E2V',
               '[1:4072,1:4000]': 'ITL'}

    # Infer the CCD vendor from the DETSIZE keyword.
    try:
        vendor = vendors[hdu.header['DETSIZE']]
    except KeyError:
        raise RuntimeError("DETSIZE not recognized.")

    # Pixel geometries differ between the two CCD vendors
    pixel_pars = pixel_parameters[vendor]

    # Get the segment 'coordinates' from the extension name, e.g., 'Segment00'.
    extname = hdu.header['EXTNAME']
    sx = int(extname[-2])
    sy = int(extname[-1])

    # For convenience of notation in LCA-13501 these are also defined
    # as 'serial' and 'parallel' indices, with Segment = Sp*10 + Ss.
    sp = sx
    ss = sy

    # Extract the x and y location indexes in the raft from the slot
    # name.  Also define the serial and parallel versions.
    cx = int(slot_name[-2])
    cy = int(slot_name[-1])
    cp = cx
    cs = cy

    # Define the WCS and Mosaic keywords.
    hdu.header['WCSNAMEA'] = 'AMPLIFIER'
    hdu.header['CTYPE1A'] = 'Seg_X   '
    hdu.header['CTYPE2A'] = 'Seg_Y   '

    hdu.header['WCSNAMEC'] = 'CCD     '
    hdu.header['CTYPE1C'] = 'CCD_X   '
    hdu.header['CTYPE2C'] = 'CCD_Y   '

    hdu.header['WCSNAMER'] = 'RAFT    '
    hdu.header['CTYPE1R'] = 'RAFT_X  '
    hdu.header['CTYPE2R'] = 'RAFT_Y  '

    hdu.header['WCSNAMEF'] = 'FOCAL_PLANE'

    hdu.header['WCSNAMEB'] = 'CCD_SERPAR'
    hdu.header['CTYPE1B'] = 'CCD_S   '
    hdu.header['CTYPE2B'] = 'CCD_P   '

    hdu.header['WCSNAMEQ'] = 'RAFT_SERPAR'
    hdu.header['CTYPE1Q'] = 'RAFT_S  '
    hdu.header['CTYPE2Q'] = 'RAFT_P  '

    # Keyword values that are common betweem E2V and ITL CCDs.
    hdu.header['PC1_1A'] = 0
    hdu.header['PC1_2A'] = 1 - 2*sx
    hdu.header['PC2_2A'] = 0
    hdu.header['CDELT1A'] = 1
    hdu.header['CDELT2A'] = 1
    hdu.header['CRPIX1A'] = 0
    hdu.header['CRPIX2A'] = 0
    hdu.header['CRVAL1A'] = sx*(pixel_pars.dimv + 1)

    hdu.header['PC1_1C'] = 0
    hdu.header['PC1_2C'] = 1 - 2*sx
    hdu.header['PC2_2C'] = 0
    hdu.header['CDELT1C'] = 1
    hdu.header['CDELT2C'] = 1
    hdu.header['CRPIX1C'] = 0
    hdu.header['CRPIX2C'] = 0
    hdu.header['CRVAL1C'] = sx*(2*pixel_pars.dimv + 1)

    hdu.header['PC1_1R'] = 0
    hdu.header['PC1_2R'] = 1 - 2*sx
    hdu.header['PC2_2R'] = 0
    hdu.header['CDELT1R'] = 1
    hdu.header['CDELT2R'] = 1
    hdu.header['CRPIX1R'] = 0
    hdu.header['CRPIX2R'] = 0
    hdu.header['CRVAL1R'] = (sx*(2*pixel_pars.dimv + 1) + pixel_pars.gap_outx
                             + (pixel_pars.ccdpx - pixel_pars.ccdax)/2.
                             + cx*(2*pixel_pars.dimv + pixel_pars.gap_inx
                                   + pixel_pars.ccdpx - pixel_pars.ccdax))

    hdu.header['PC1_1B'] = 0
    hdu.header['PC1_2B'] = 0
    hdu.header['PC2_2B'] = 1 - 2*sp
    hdu.header['CDELT1B'] = 1
    hdu.header['CDELT2B'] = 1
    hdu.header['CRPIX1B'] = 0
    hdu.header['CRPIX2B'] = 0

    hdu.header['PC1_1Q'] = 0
    hdu.header['PC1_2Q'] = 0
    hdu.header['PC2_2Q'] = 1 - 2*sp
    hdu.header['CDELT1Q'] = 1
    hdu.header['CDELT2Q'] = 1
    hdu.header['CRPIX1Q'] = 0
    hdu.header['CRPIX2Q'] = 0
    hdu.header['CRVAL2Q'] = (sp*(2*pixel_pars.dimv + 1) + pixel_pars.gap_outx
                             + (pixel_pars.ccdpx - pixel_pars.ccdax)/2.
                             + cp*(2*pixel_pars.dimv + pixel_pars.gap_inx
                                   + pixel_pars.ccdpx - pixel_pars.ccdax))

    if vendor == 'ITL':
        hdu.header['PC2_1A'] = -1
        hdu.header['CRVAL2A'] = pixel_pars.dimh + 1 - pixel_pars.preh

        hdu.header['PC2_1C'] = -1
        hdu.header['CRVAL2C'] \
            = pixel_pars.dimh + 1 + sy*pixel_pars.dimh - pixel_pars.preh

        hdu.header['PC2_1R'] = -1
        hdu.header['CRVAL2R'] = (pixel_pars.dimh + 1 + sy*pixel_pars.dimh
                                 + pixel_pars.gap_outy
                                 + (pixel_pars.ccdpy - pixel_pars.ccday)/2.
                                 + cy*(8*pixel_pars.dimh + pixel_pars.gap_iny
                                       + pixel_pars.ccdpy - pixel_pars.ccday)
                                 - pixel_pars.preh)

        hdu.header['PC1_1B'] = -1
        hdu.header['CRVAL1B'] = (ss + 1)*pixel_pars.dimh + 1 - pixel_pars.preh
        hdu.header['CRVAL2B'] = sp*(2*pixel_pars.dimv + 1)

        hdu.header['PC1_1Q'] = -1
        hdu.header['CRVAL1Q'] \
            = (pixel_pars.gap_outy + (pixel_pars.ccdpy - pixel_pars.ccday)/2.
               + cs*(8*pixel_pars.dimh + pixel_pars.gap_iny
                     + pixel_pars.ccdpy - pixel_pars.ccday)
               + (ss + 1)*pixel_pars.dimh + 1 - pixel_pars.preh)

        hdu.header['DTM1_1'] = -1
        hdu.header['DTV1'] \
            = (pixel_pars.dimh + 1) + sy*pixel_pars.dimh + pixel_pars.preh
        hdu.header['DTV2'] = (2*pixel_pars.dimv + 1)*(1 - sx)
    else:
        # vendor == 'E2V'
        hdu.header['PC2_1A'] = 1 - 2*sx
        hdu.header['CRVAL2A'] \
            = sx*(pixel_pars.dimh + 1) + (2*sx - 1)*pixel_pars.preh

        hdu.header['PC2_1C'] = 1 - 2*sx
        hdu.header['CRVAL2C'] = (sx*(pixel_pars.dimh+1) + sy*pixel_pars.dimh
                                 + (2*sx - 1)*pixel_pars.preh)

        hdu.header['PC2_1R'] = 1 - 2*sx
        hdu.header['CRVAL2R'] \
            = (sx*(pixel_pars.dimh + 1) + sy*pixel_pars.dimh
               + pixel_pars.gap_outy
               + (pixel_pars.ccdpy - pixel_pars.ccday)/2.
               + cy*(8*pixel_pars.dimh + pixel_pars.gap_iny + pixel_pars.ccdpy
                     - pixel_pars.ccday) + (2*sx - 1)*pixel_pars.preh)
        hdu.header['PC1_1B'] = 1 - 2*sp
        hdu.header['CRVAL1B'] = (sp*(pixel_pars.dimh + 1) + ss*pixel_pars.dimh
                                 + (2*sp - 1)*pixel_pars.preh)
        hdu.header['CRVAL2B'] = sp*(2*pixel_pars.dimv + 1)

        hdu.header['PC1_1Q'] = 1 - 2*sp
        hdu.header['CRVAL1Q'] \
            = (pixel_pars.gap_outy + (pixel_pars.ccdpy - pixel_pars.ccday)/2.
               + cs*(8*pixel_pars.dimh + pixel_pars.gap_iny
                     + pixel_pars.ccdpy - pixel_pars.ccday)
               + sp*(pixel_pars.dimh + 1) + ss*pixel_pars.dimh
               + (2*sp - 1)*pixel_pars.preh)

        hdu.header['DTM1_1'] = 1 - 2*sx
        hdu.header['DTV1'] = ((pixel_pars.dimh + 1 + 2*pixel_pars.preh)*sx
                              + sy*pixel_pars.dimh - pixel_pars.preh)
        hdu.header['DTV2'] = (2*pixel_pars.dimv + 1)*(1 - sx)

    return hdu

def cte_matrix(npix, cti, ntransfers=20):
    """
    Compute the CTE matrix so that the apparent charge q_i in the i-th
    pixel is given by
    q_i = Sum_j cte_matrix_ij q0_j
    where q0_j is the initial charge in j-th pixel.  The corresponding
    python code would be
    >>> cte = cte_matrix(npix, cti)
    >>> qout = numpy.dot(cte, qin)
    Parameters
    ----------
    npix : int
        Total number of pixels in either the serial or parallel
        directions.
    cti : float
        The charge transfer inefficiency, i.e., the fraction of
        a pixel's charge left behind for each single pixel transfer.
    ntransfers : int [20]
        Maximum number of transfers to consider as contributing to
        a target pixel.
    Returns
    -------
    numpy.array
        The npix x npix numpy array containing the CTE matrix.

    This implementation is from Mike Jarvis.
    """
    my_matrix = np.zeros((npix, npix), dtype=float)
    for i in range(1, npix+1):
        # On diagonal, there are i transfers of the electrons, so i chances to lose a fraction
        # cti into the later pixels.  Net charge is decreased by (1-cti)**i.
        my_matrix[i-1, i-1] = (1.-cti)**i

        # Off diagonal, there must be (i-j) cti losses of charge among i-1 possible transfers.
        # Then that charge has to survive j additional transfers.
        # So net charge is binom(i-1,i-j) * (1-cti)**j * cti**(i-j)
        # (Indeed this is the same equation as above when i=j, but slightly more efficient to
        # break it out separately above.)
        jmin = max(1, i-ntransfers)
        j = np.arange(jmin, i)
        my_matrix[i-1, jmin-1:i-1] = scipy.special.binom(i-1, i-j) * (1.-cti)**j * cti**(i-j)

    return my_matrix

def get_primary_hdu(opsim_md, det_name, lsst_num='LCA-11021_RTM-000', image_type='SKYEXP',
                    added_keywords={}):
    """Create a primary HDU for the output raw file with the keywords
    needed to process with the LSST Stack."""
    phdu = fits.PrimaryHDU()
    phdu.header['RUNNUM'] = opsim_md.get('obshistid')
    phdu.header['OBSID'] = opsim_md.get('obshistid')
    exp_time = opsim_md.get('exptime')
    phdu.header['EXPTIME'] = exp_time
    phdu.header['DARKTIME'] = exp_time
    phdu.header['FILTER'] = opsim_md.get('band')
    phdu.header['TIMESYS'] = 'TAI'
    phdu.header['LSST_NUM'] = lsst_num
    phdu.header['TESTTYPE'] = 'IMSIM'
    phdu.header['IMGTYPE'] = image_type
    phdu.header['OBSTYPE'] = image_type
    phdu.header['MONOWL'] = -1
    raft, sensor = det_name.split('_')
    phdu.header['RAFTNAME'] = raft
    phdu.header['SENSNAME'] = sensor
    ratel = opsim_md.get('fieldRA')
    phdu.header['RATEL'] = ratel
    phdu.header['DECTEL'] = opsim_md.get('fieldDec')
    phdu.header['ROTANGLE'] = opsim_md.get('rotSkyPos')
    mjd_obs = opsim_md.get('mjd')
    mjd_end = mjd_obs + exp_time/86400.
    phdu.header['MJD-OBS'] = mjd_obs
    phdu.header['HASTART'] = opsim_md.getHourAngle(mjd_obs, ratel)
    phdu.header['HAEND'] = opsim_md.getHourAngle(mjd_end, ratel)
    phdu.header['AMSTART'] = opsim_md.get('airmass')
    phdu.header['AMEND'] = phdu.header['AMSTART']  # XXX: This is not correct. Does anyone care?
    phdu.header['IMSIMVER'] = __version__
    phdu.header['PKG00000'] = 'throughputs'
    phdu.header['VER00000'] = '1.4'
    phdu.header['CHIPID'] = det_name
    phdu.header['DATE-OBS'] = Time(mjd_obs, format='mjd', scale='tai').to_value('isot')
    phdu.header['DATE-END'] = Time(mjd_end, format='mjd', scale='tai').to_value('isot')

    phdu.header.update(added_keywords)
    return phdu


class CcdReadout:
    """Class to apply electronics readout effects to e-images using camera
    parameters from the lsst.obs.lsst package."""
    def __init__(self, config, base):
        # det_name should already be set in base config by the main LSST CCD builder
        self.det_name = base['det_name']
        # camera is given in the output field, but defaults to LsstCam
        camera = Camera(base['output'].get('camera','LsstCam'))
        self.ccd = camera[self.det_name]
        amp = list(self.ccd.values())[0]

        # Parse the required parameters
        # XXX: Should any of these have defaults? Or keep them all as required parameters?
        req = {
            'readout_time': float,
            'dark_current': float,
            'bias_level': float,
            'pcti': float,
            'scti': float,
        }
        params = galsim.config.GetAllParams(config, base, req=req)[0]
        self.readout_time = params['readout_time']
        self.dark_current = params['dark_current']
        self.bias_level = params['bias_level']
        self.pcti = params['pcti']
        self.scti = params['scti']

        # Make the corresponding matrices for implementing the cti.
        self.scte_matrix = (None if self.scti == 0
                            else cte_matrix(amp.raw_bounds.xmax, self.scti))
        self.pcte_matrix = (None if self.pcti == 0
                            else cte_matrix(amp.raw_bounds.ymax, self.pcti))

    def apply_cte(self, amp_images):
        """Apply CTI to a list of amp images."""
        for full_segment in amp_images:
            full_arr = full_segment.array
            if self.pcte_matrix is not None:
                for col in range(full_arr.shape[1]):
                    full_arr[:, col] = self.pcte_matrix @ full_arr[:, col]
            if self.scte_matrix is not None:
                for row in range(full_arr.shape[0]):
                    full_arr[row, :] = self.scte_matrix @ full_arr[row, :]
        return amp_images

    def apply_crosstalk(self, amp_arrays):
        """Apply intra-CCD crosstalk to an array of amp data."""
        if self.ccd.xtalk is None:
            return amp_arrays
        output = []
        for amp_index, xtalk_row in enumerate(self.ccd.xtalk):
            output.append(amp_arrays[amp_index] +
                          sum([x*y for x, y in zip(amp_arrays, xtalk_row)]))
        return output

    def build_images(self, config, base, main_data):
        """Build the amplifier images from the "electron-image".
        The steps are
        * add dark current
        * divide the physical image into amplifier segements
        * apply per-amp gains
        * apply appropriate flips in x- and y-directions to
          get the amp image array in readout order
        * apply intra-CCD crosstalk
        * add prescan and overscan pixels
        * apply charge transfer efficiency effects
        * add bias levels and read noise
        """
        eimage = copy.deepcopy(main_data[0])

        # Bleed trail processing. TODO: Get full_well from the camera.
        eimage.array[:] = bleed_eimage(eimage.array, full_well=1e5)

        # Add dark current.
        rng = galsim.config.GetRNG(config, base)
        dark_time = base['exp_time'] + self.readout_time
        dark_current = self.dark_current
        poisson = galsim.PoissonDeviate(rng, mean=dark_current*dark_time)
        dc_data = np.zeros(np.prod(eimage.array.shape))
        poisson.generate(dc_data)
        eimage += dc_data.reshape(eimage.array.shape)

        # Partition eimage into amp-level imaging segments, convert to ADUs,
        # and apply the readout flips.
        amp_arrays = []
        for amp in self.ccd.values():
            amp_data = eimage[amp.bounds].array/amp.gain
            if amp.raw_flip_x:
                amp_data = amp_data[:, ::-1]
            if amp.raw_flip_y:
                amp_data = amp_data[::-1, :]
            amp_arrays.append(amp_data)

        # Add intra-CCD crosstalk.
        amp_arrays = self.apply_crosstalk(amp_arrays)

        # Construct full segments with prescan and overscan pixels.
        amp_images = []
        for amp_data, amp in zip(amp_arrays, self.ccd.values()):
            full_segment = galsim.Image(amp.raw_bounds)
            full_segment[amp.raw_data_bounds].array[:] += amp_data
            amp_images.append(full_segment)

        # Apply CTI.
        amp_images = self.apply_cte(amp_images)

        # Add bias levels and read noise.
        for full_segment in amp_images:
            full_segment += self.bias_level
            # Setting gain=0 turns off the addition of Poisson noise,
            # which is already in the e-image, so that only the read
            # noise is added.
            read_noise = galsim.CCDNoise(rng, gain=0,
                                         read_noise=amp.read_noise)
            full_segment.addNoise(read_noise)
        return amp_images


class CameraReadout(ExtraOutputBuilder):
    """This is a GalSim "extra output" builder to write out the amplifier file simulating
    the camera readout of the main "e-image".
    """

    def finalize(self, config, base, main_data, logger):
        """
        This function will use the CcdReadout class to divide the physical
        CCD image into amplifier segments and add readout effects.
        This function will also add header keywords with the amp names
        and pixel geometry, and will package everything up as an
        astropy.io.fits.HDUList.

        Parameters:
           config:     The configuration field for this output object.
           base:       The base configuration dict.
           main_data:  The main file data in case it is needed.
           logger:     If given, a logger object to log progress. [default: None]

        Returns:
           An HDUList of the amplifier images in a CCD.
        """
        logger.warning("Making amplifier images")

        if '_input_objs' not in base or 'opsim_meta_dict' not in base['_input_objs']:
            raise galsim.config.GalSimConfigError(
                "The readout extra output requires the opsim_meta_dict input object")
        opsim_md = galsim.config.GetInputObj('opsim_meta_dict', config, base, 'readout')

        ccd_readout = CcdReadout(config, base)
        amps = ccd_readout.build_images(config, base, main_data)
        det_name = base['det_name']
        channels = '10 11 12 13 14 15 16 17 07 06 05 04 03 02 01 00'.split()
        x_seg_offset = (1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1)
        y_seg_offset = (0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2)
        wcs = main_data[0].wcs
        crpix1, crpix2 = wcs.crpix
        hdus = fits.HDUList(get_primary_hdu(opsim_md, det_name))
        for amp_num, amp in enumerate(amps):
            channel = 'C' + channels[amp_num]
            amp_info = ccd_readout.ccd[channel]
            raw_data_bounds = amp_info.raw_data_bounds
            hdu = fits.CompImageHDU(np.array(amp.array, dtype=np.int32),
                                    compression_type='RICE_1')
            wcs.writeToFitsHeader(hdu.header, main_data[0].bounds)
            hdu.header['EXTNAME'] = 'Segment' + channels[amp_num]
            xsign = -1 if amp_info.raw_flip_x else 1
            ysign = -1 if amp_info.raw_flip_y else 1
            height, width = raw_data_bounds.numpyShape()
            hdu.header['CRPIX1'] = xsign*crpix1 + x_seg_offset[amp_num]*width
            hdu.header['CRPIX2'] = ysign*crpix2 + y_seg_offset[amp_num]*height
            hdu.header['CD1_2'] *= -xsign
            hdu.header['CD2_2'] *= -xsign
            hdu.header['CD1_1'] *= -ysign
            hdu.header['CD2_1'] *= -ysign
            hdu.header['DATASEC'] = section_keyword(raw_data_bounds)
            hdu.header['DETSEC'] = section_keyword(amp_info.bounds,
                                                   flipx=amp_info.raw_flip_x,
                                                   flipy=amp_info.raw_flip_y)
            hdus.append(hdu)
            amp_name = '_'.join((det_name, channel))
            logger.info("Amp %s has bounds %s.", amp_name,
                        hdu.header['DETSEC'])
        return hdus

    def writeFile(self, file_name, config, base, logger):
        """Write this output object to a file.

        Parameters:
            file_name:  The file to write to.
            config:     The configuration field for this output object.
            base:       The base configuration dict.
            logger:     If given, a logger object to log progress. [default: None]

        """
        logger.warning("Writing amplifier images to %s",file_name)
        # self.final_data is the output of finalize, which is our list
        # of amp images.
        self.final_data[0].header['OUTFILE'] = os.path.basename(file_name)
        self.final_data.writeto(file_name, overwrite=True)


RegisterExtraOutput('readout', CameraReadout())
