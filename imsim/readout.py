import os
import copy
import numpy as np
import scipy
from astropy.io import fits
import galsim
from galsim.config import ExtraOutputBuilder, RegisterExtraOutput


def section_keyword(bounds, flipx=False, flipy=False):
    """Package image bounds as a NOAO image section keyword value."""
    xmin, xmax = bounds.xmin, bounds.xmax
    ymin, ymax = bounds.ymin, bounds.ymax
    if flipx:
        xmin, xmax = xmax, xmin
    if flipy:
        ymin, ymax = ymax, ymin
    return '[%i:%i,%i:%i]' % (xmin, xmax, ymin, ymax)


def cte_matrix(npix, cti, ntransfers=20, nexact=30):
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
        The charge transfer inefficiency.
    ntransfers : int, optional
        Maximum number of transfers to consider as contributing to
        a target pixel.
    nexact : int, optional
        Number of transfers to use exact the binomial distribution
        expression, otherwise use Poisson's approximation.
    Returns
    -------
    numpy.array
        The npix x npix numpy array containing the CTE matrix.
    Notes
    -----
    This implementation is based on
    Janesick, J. R., 2001, "Scientific Charge-Coupled Devices", Chapter 5,
    eqs. 5.2a,b.
    """
    ntransfers = min(npix, ntransfers)
    nexact = min(nexact, ntransfers)
    my_matrix = np.zeros((npix, npix), dtype=np.float)
    for i in range(1, npix):
        jvals = np.concatenate((np.arange(1, i+1), np.zeros(npix-i)))
        index = np.where(i - nexact < jvals)
        j = jvals[index]
        my_matrix[i-1, :][index] \
            = scipy.special.binom(i, j)*(1 - cti)**i*cti**(i - j)
        if nexact < ntransfers:
            index = np.where((i - nexact >= jvals) & (i - ntransfers < jvals))
            j = jvals[index]
            my_matrix[i-1, :][index] \
                = (j*cti)**(i-j)*np.exp(-j*cti)/scipy.special.factorial(i-j)
    return my_matrix


def get_primary_hdu(config, base, main_data, lsst_num='LCA-11021_RTM-000',
                    image_type='SKYEXP'):
    """Create a primary HDU for the output raw file with the keywords
    needed to process with the LSST Stack."""
    opsim_md = galsim.config.GetInputObj('opsim_meta_dict', config, base,
                                         'OpsimMeta')
    md = opsim_md.meta
    phdu = fits.PrimaryHDU()
    phdu.header['RUNNUM'] = md.get('obshistid')
    phdu.header['OBSID'] = md.get('obshistid')
    exp_time = base['exp_time']
    phdu.header['EXPTIME'] = exp_time
    phdu.header['DARKTIME'] = exp_time
    phdu.header['FILTER'] = 'ugrizy'[md.get('filter')]
    phdu.header['TIMESYS'] = 'TAI'
    phdu.header['LSST_NUM'] = lsst_num
    phdu.header['TESTTYPE'] = 'IMSIM'
    phdu.header['IMGTYPE'] = image_type
    phdu.header['OBSTYPE'] = image_type
    phdu.header['MONOWL'] = -1
    raft, sensor = base['det_name'].split('-')
    phdu.header['RAFTNAME'] = raft
    phdu.header['SENSNAME'] = sensor
    ratel = md.get('rightascension')
    phdu.header['RATEL'] = ratel
    phdu.header['DECTEL'] = md.get('declination')
    phdu.header['ROTANGLE'] = md.get('rotskypos')
    mjd_obs = md.get('mjd')
    mjd_end = mjd_obs + base['exp_time']/86400.
    phdu.header['MJD-OBS'] = mjd_obs
    phdu.header['HASTART'] = opsim_md.getHourAngle(mjd_obs, ratel)
    phdu.header['HAEND'] = opsim_md.getHourAngle(mjd_end, ratel)
    phdu.header['AMSTART'] = md.get('airmass')
    # write hardwired version keywords for now.
    phdu.header['IMSIMVER'] = 'unknown'
    phdu.header['PKG00000'] = 'throughputs'
    phdu.header['VER00000'] = '1.4'
    return phdu


class CcdReadout:
    """Class to apply electronics readout effects to e-images using camera
    parameters from the lsst.obs.lsst package."""
    def __init__(self, config, base, rng=None):
        self.rng = rng
        if self.rng is None:
            seed = galsim.config.SetupConfigRNG(base)
            self.rng = galsim.BaseDeviate(seed)
        self.det_name = base['det_name'].replace('-', '_')
        camera = galsim.config.GetInputObj('camera_geometry', config, base,
                                           'Camera')
        self.ccd = camera[self.det_name]
        amp = list(self.ccd.values())[0]
        scti = config['scti']
        self.scte_matrix = (None if scti == 0
                            else cte_matrix(amp.raw_bounds.xmax, scti))
        pcti = config['pcti']
        self.pcte_matrix = (None if pcti == 0
                            else cte_matrix(amp.raw_bounds.ymax, pcti))

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
        """Build the amplifier images applying readout effects and
        repackaging the pixel ordering in readout order."""
        eimage = copy.deepcopy(main_data[0])

        # Add dark current.
        exp_time = base['exp_time']
        dark_current = config['dark_current']
        rng = galsim.PoissonDeviate(seed=self.rng, mean=dark_current*exp_time)
        dc_data = np.zeros(np.prod(eimage.array.shape))
        rng.generate(dc_data)
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
            full_segment += config['bias_level']
            # Setting gain=0 turns off the addition of Poisson noise,
            # which is already in the e-image, so that only the read
            # noise is added.
            read_noise = galsim.CCDNoise(self.rng, gain=0,
                                         read_noise=amp.read_noise)
            full_segment.addNoise(read_noise)
        return amp_images


class CameraReadout(ExtraOutputBuilder):
    """This is a GalSim "extra output" builder to write out the amplifier file simulating
    the camera readout of the main "e-image".
    """

    def finalize(self, config, base, main_data, logger):
        """Perform any final processing at the end of all the image processing.

        This function will be called after all images have been built.

        It returns some sort of final version of the object.  In the
        base class, it just returns self.data, but depending on the
        meaning of the output object, something else might be more
        appropriate.

        Parameters:
           config:     The configuration field for this output object.
           base:       The base configuration dict.
           main_data:  The main file data in case it is needed.
           logger:     If given, a logger object to log progress. [default: None]

        Returns:
           The final version of the object.
        """
        logger.warning("Making amplifier images")
        ccd_readout = CcdReadout(config, base)
        amps = ccd_readout.build_images(config, base, main_data)
        det_name = base['det_name'].replace('-', '_')
        channels = '10 11 12 13 14 15 16 17 07 06 05 04 03 02 01 00'.split()
        x_seg_offset = (1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1)
        y_seg_offset = (0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2)
        wcs = main_data[0].wcs
        crpix1, crpix2 = wcs.crpix
        hdus = fits.HDUList(get_primary_hdu(config, base, main_data))
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
            logger.warning("Amp %s has bounds %s.", amp_name,
                           hdu.header['DETSEC'])
        return hdus

    def writeFile(self, file_name, config, base, logger):
        """Write this output object to a file.

        The base class implementation is appropriate for the cas that
        the result of finalize is a list of images to be written to a
        FITS file.

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
