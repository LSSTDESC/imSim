import numpy as np
import scipy
import galsim
from galsim.config import ExtraOutputBuilder, RegisterExtraOutput

def section_keyword(bounds, flipx=False, flipy=False):
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


class CcdReadout:
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
        for full_segment in amp_images:
            full_arr = full_segment.array
            if self.pcte_matrix is not None:
                for col in range(full_arr.shape[1]):
                    full_arr[:, col] = self.pcte_matrix @ full_arr[:, col]
            if self.scte_matrix is not None:
                for row in range(full_arr.shape[0]):
                    full_arr[row, :] = self.scte_matrix @ full_arr[row, :]
        return amp_images

    def apply_crosstalk(self, amp_arrays, xtalk_coeffs):
        if xtalk_coeffs is None:
            return amp_arrays
        output = []
        for amp_index, xtalk_row in enumerate(xtalk_coeffs):
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
        amp_arrays = self.apply_crosstalk(amp_arrays, self.ccd.xtalk)

        amp_images = []
        for amp_data, amp in zip(amp_arrays, self.ccd.values()):
            full_segment = galsim.Image(amp.raw_bounds)
            full_segment[amp.raw_data_bounds].array[:] += amp_data
            amp_images.append(full_segment)
        amp_images = self.apply_cte(amp_images)

        for full_segment in amp_images:
            full_segment += config['bias_level']
            read_noise = galsim.CCDNoise(self.rng, gain=amp.gain,
                                         read_noise=amp.read_noise)
            full_segment.addNoise(read_noise)
        return amp_images

class CameraReadout(ExtraOutputBuilder):
    """This is a GalSim "extra output" builder to write out the amplifier
    file simulating
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
        channels = [f'C{_}' for _ in
                    '10 11 12 13 14 15 16 17 07 06 05 04 03 02 01 00'.split()]
        x_pos = (1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1)
        for amp_num, amp in enumerate(amps):
            channel = channels[amp_num]
            amp_info = ccd_readout.ccd[channel]
            amp_name = '_'.join((det_name, channel))
            logger.warning("Amp %s has bounds %s.", amp_name, amp_info.raw_data_bounds)
            amp.header = galsim.FitsHeader()
            amp.header['EXTNAME'] = 'SEGMENT' + channel[1:]
            amp.header['DATASEC'] = section_keyword(amp_info.raw_data_bounds)
            amp.header['DETSEC'] = section_keyword(amp_info.bounds,
                                                   flipx=amp_info.raw_flip_x,
                                                   flipy=amp_info.raw_flip_y)
        return amps

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
        galsim.fits.writeMulti(self.final_data, file_name)


RegisterExtraOutput('readout', CameraReadout())
