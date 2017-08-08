"""
Code to access sensor properties, including pixel layout and electronics
readout information.
"""
from __future__ import print_function, absolute_import, division
import numpy as np
import scipy.special
import lsst.afw.geom as afwGeom

__all__ = ['FocalPlaneInfo', 'cte_matrix']


class FocalPlaneInfo(object):
    """
    Class to serve up electronics readout properties of sensors based
    on a focal plane description, such as the segmentation.txt file
    used by PhoSim.

    Attributes
    ----------
    sensors : dict
        A dictionary of sensor properties, keyed by sensor id in the
        LSST focal plane, e.g., "R22_S11".
    amps : dict
        A diction of amplifier properties, keyed by amplifier id, e.g.,
        "R22_S11_C00".
    """
    def __init__(self):
        "FocalPlaneInfo constructor"
        self.sensors = {}
        self.amps = {}

    def get_sensor(self, sensor_id):
        """
        Access to the specified SensorProperties object.

        Parameters
        ----------
        sensor_id : str
            Sensor ID of the form "Rrr_Sss", e.g., "R22_S11".

        Returns
        -------
        SensorProperties object
            The object containing the sensor-wide properties.
        """
        return self.sensors[sensor_id]

    def get_amp(self, amp_id):
        """
        Access to the specified AmplifierProperties object.

        Parameters
        ----------
        amp_id : str
            Amplifier ID of the form "Rrr_Sss_Ccc", e.g., "R22_S11_C00"

        Returns
        -------
        AmplifierProperties object
            The object containing the amplifier properties.
        """
        return self.amps[amp_id]

    @staticmethod
    def sensor_id(raft, ccd):
        """
        Convert the lsst.afw.cameraGeom specifiers to the name used
        by PhoSim.

        Parameters
        ----------
        raft : str
            Raft id using lsst.afw.cameraGeom syntax, e.g., "R:2,2".
        ccd : str
            Sensor id using lsst.afw.cameraGeom syntax, e.g., "S:1,1".

        Returns
        -------
        str
            e.g., "R22_S11"
        """
        return 'R%s%s_S%s%s' % (raft[2], raft[4], ccd[2], ccd[4])

    @staticmethod
    def amp_id(raft, ccd, chan):
        """
        Convert the lsst.afw.cameraGeom specifiers to the name used
        by PhoSim.

        Parameters
        ----------
        raft : str
            Raft id using lsst.afw.cameraGeom syntax, e.g., "R:2,2".
        ccd : str
            Sensor id using lsst.afw.cameraGeom syntax, e.g., "S:1,1".
        chan : str
            Amplifier channel id using lsst.afw.cameraGeom syntax, e.g.,
            "C:0,0".

        Returns
        -------
        str
            e.g., "R22_S11_C00"
        """
        return 'R%s%s_S%s%s_C%s%s' % (raft[2], raft[4], ccd[2], ccd[4],
                                      chan[2], chan[4])

    @staticmethod
    def read_phosim_seg_file(seg_file):
        """
        Factory method to create a FocalPlaneInfo object which has
        been filled with the data from a PhoSim segmentation.txt file.

        Parameters
        ----------
        seg_file : str
            The PhoSim formatted segmentation.txt file.

        Returns
        -------
        FocalPlaneInfo object
            The filled FocalPlaneInfo object.
        """
        my_self = FocalPlaneInfo()
        with open(seg_file, 'r') as fp:
            lines = [line for line in fp.readlines()
                     if not line.startswith('#')]
        i = -1
        while True:
            try:
                i += 1
                sensor_props = SensorProperties(lines[i])
                my_self.sensors[sensor_props.name] = sensor_props
                for j in range(sensor_props.num_amps):
                    i += 1
                    amp_props = AmplifierProperties(lines[i])
                    my_self.amps[amp_props.name] = amp_props
                    sensor_props.append_amp(amp_props)
            except IndexError:
                break
        return my_self


class SensorProperties(object):
    """
    Class to contain the properties of a sensor.

    Attributes
    ----------
    name : str
        The sensor name, e.g., "R22_S11".
    num_amps : int
        The number of amplifiers in this sensor.
    height : int
        The number of physical sensor pixels in the parallel direction.
    width : int
        The number of physical sensor pixels in the serial direction.
    amp_names : tuple
        The amplifier names in the order in which they were added
        to self.  For data read in from segmentation.txt, this ordering
        is used for the crosstalk matrix column ordering.
    """
    def __init__(self, line):
        """
        SensorProperties constructor.

        Parameters
        ----------
        line : str
            Line from segmentation.txt to parse for sensor properties.
        """
        tokens = line.strip().split()
        self.name = tokens[0]
        self.num_amps = int(tokens[1])
        self.height = int(tokens[3])
        self.width = int(tokens[2])
        self._amp_names = []

    @property
    def amp_names(self):
        """
        Amplifier names for amps associated with this sensor.
        """
        return tuple(self._amp_names)

    def append_amp(self, amp_props):
        """
        Append an amplifier to the ._amp_names list.

        Parameters
        ----------
        amp_props : AmplifierProperties object
            The object containing the amplifier properties.
        """
        self._amp_names.append(amp_props.name)


class AmplifierProperties(object):
    """
    Class to contain the properties of an amplifier.

    Attributes
    ----------
    name : str
        The amplifier name, e.g., "R22_S11_C00".
    mosaic_section: lsst.afw.geom.Box2I
        The bounding box in the fully mosaicked image containing the
        pixel data in the current amplifier.
    imaging : lsst.afw.geom.Box2I
        The imaging region bounding box.
    full_segment : lsst.afw.geom.Box2I
        The bounding box for the full segment.
    prescan : lsst.afw.geom.Box2I
        The bounding box for the (serial) prescan region.
    serial_overscan : lsst.afw.geom.Box2I
        The bounding box for the serial overscan region.
    parallel_overscan : lsst.afw.geom.Box2I
        The bounding box for the parallel overscan region.
    gain : float
        The amplifier gain in units of e-/ADU.
    bias_level : float
        The bias level in units of ADU.
    read_noise : float
        The read noise in units of rms ADU.
    dark_current : float
        The dark current in units of e-/pixel/s
    crosstalk : numpy.array
        The row of the intrasensor crosstalk matrix for this amplifier.
    flip_x : bool
        Flag to indicate that pixel ordering in x-direction should be
        reversed relative to mosaicked image.
    flip_y : bool
        Flag to indicate that pixel ordering in y-direction should be
        reversed relative to mosaicked image.
    scti : float
        Charge transfer inefficiency in serial direction.
    pcti : float
        Charge transfer inefficiency in parallel direction.

    """
    def __init__(self, line, scti=1e-6, pcti=1e-6):
        """
        AmplifierProperties constructor.

        Parameters
        ----------
        line : str
            Line from segmentation.txt to parse for amplifier properties.
        scti : float, optional
            Charge transfer inefficiency in the serial direction.
        pcti : float, optional
            Charge transfer inefficiency in the parallel direction.
        """
        self.scti = scti
        self.pcti = pcti
        tokens = line.strip().split()
        self.name = tokens[0]
        xmin, xmax, ymin, ymax = (int(x) for x in tokens[1:5])
        xsize = np.abs(xmax - xmin) + 1
        ysize = np.abs(ymax - ymin) + 1
        self.mosaic_section = afwGeom.Box2I(afwGeom.Point2I(min(xmin, xmax),
                                                            min(ymin, ymax)),
                                            afwGeom.Extent2I(xsize, ysize))
        parallel_prescan = int(tokens[15])
        serial_overscan = int(tokens[16])
        serial_prescan = int(tokens[17])
        parallel_overscan = int(tokens[18])
        self.imaging = afwGeom.Box2I(afwGeom.Point2I(serial_prescan,
                                                     parallel_prescan),
                                     afwGeom.Extent2I(xsize, ysize))
        self.full_segment \
            = afwGeom.Box2I(afwGeom.Point2I(0, 0),
                            afwGeom.Extent2I(xsize + serial_prescan +
                                             serial_overscan,
                                             ysize + parallel_prescan +
                                             parallel_overscan))
        self.prescan = afwGeom.Box2I(afwGeom.Point2I(0, 0),
                                     afwGeom.Extent2I(serial_prescan, ysize))
        self.serial_overscan \
            = afwGeom.Box2I(afwGeom.Point2I(serial_prescan + xsize, 0),
                            afwGeom.Extent2I(serial_overscan, ysize))
        self.parallel_overscan \
            = afwGeom.Box2I(afwGeom.Point2I(0, ysize),
                            afwGeom.Extent2I(serial_prescan + xsize +
                                             serial_overscan,
                                             parallel_overscan))
        self.gain = float(tokens[7])
        self.bias_level = float(tokens[9])
        self.read_noise = float(tokens[11])
        self.dark_current = float(tokens[13])
        self.crosstalk = np.array([float(x) for x in tokens[21:]])
        self.flip_x = (tokens[5] == '-1')
        self.flip_y = (tokens[6] == '-1')


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
