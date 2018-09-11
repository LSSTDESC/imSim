"""
Function to apply chip-centered acceptance cones on instance catalogs.
"""
import numpy as np
import lsst.sims.coordUtils
from lsst.sims.utils import _angularSeparation
import desc.imsim

__all__ = ['InstCatTrimmer']

def degrees_separation(ra0, dec0, ra, dec):
    """
    Compute angular separation in degrees.

    Parameters
    ----------
    ra0: float
        Right ascension of reference location in degrees.
    dec0: float
        Declination of reference location in degrees.
    ra: float or numpy.array
        Right ascension of object(s) in degrees
    dec: float or numpy.array
        Declination of object(s) in degrees.
    """
    return np.degrees(_angularSeparation(np.radians(ra0), np.radians(dec0),
                                         np.radians(ra), np.radians(dec)))

class Disaggregator:
    """.
    Class to disaggregate instance catalog object lines into per chip
    acceptance cones.
    """
    def __init__(self, object_lines, trimmer):
        """
        Parameters
        ----------
        object_lines: list
            list of object entries from an instance catalog.
        trimmer: InstCatTrimmer
            An instance of the InstCatTrimmer class to provide
            visit-level metadata.
        """
        self.object_lines = object_lines
        self.trimmer = trimmer

        # Extract the ra, dec values for each object.
        self._ra = np.zeros(len(object_lines), dtype=np.float)
        self._dec = np.zeros(len(object_lines), dtype=np.float)
        self._sersic = np.zeros(len(object_lines), dtype=np.int)
        self._magnorm = np.zeros(len(object_lines), dtype=np.float)
        for i, line in enumerate(object_lines):
            tokens = line.strip().split()
            self._ra[i] = np.float(tokens[2])
            self._dec[i] = np.float(tokens[3])
            if 'sersic2d' in line:
                self._sersic[i] = 1
            self._magnorm[i] = np.float(tokens[4])
        self._camera = desc.imsim.get_obs_lsstSim_camera()

    def compute_chip_center(self, chip_name):
        """
        Compute the center of the desired chip in focalplane pixel
        coordinates.

        Parameters
        ----------
        chip_name: str
            Name of the CCD, e.g., "R:2,2 S:1,1".

        Returns
        -------
        (float, float): The RA, Dec in degrees of the center of the CCD.
        """
        center_x, center_y = desc.imsim.get_chip_center(chip_name, self._camera)
        return lsst.sims.coordUtils.raDecFromPixelCoords(
            xPix=center_x, yPix=center_y, chipName=chip_name,
            camera=self._camera, obs_metadata=self.trimmer.obs_md, epoch=2000.0,
            includeDistortion=True)

    def get_object_entries(self, chip_name, radius=0.18, sort_magnorm=True):
        """
        Get the object entries within an acceptance cone centered on
        a specified CCD.

        Parameters
        ----------
        chip_name: str
            Name of the CCD, e.g., "R:2,2 S:1,1".
        radius: float [0.18]
            Radius, in degrees, of the acceptance cone.
        sort_magnorm: bool [True]
            Flag to sort the output list by ascending magnorm value.

        Returns
        -------
        list: list of object entries from the original instance catalog.

        Notes
        -----
        This function applies the 'minsource' criterion to the sersic
        galaxies in the instance catalog if 'minsource' is included in
        the instance catalog commands.
        """
        ra0, dec0 = self.compute_chip_center(chip_name)
        seps = degrees_separation(ra0, dec0, self._ra, self._dec)
        index = np.where(seps < radius)

        if (self.trimmer.minsource is not None and
                sum(self._sersic[index]) < self.trimmer.minsource):
            # Apply the minsource criterion.
            return []

        # Collect the selected objects.
        selected = [self.object_lines[i] for i in index[0]]
        if sort_magnorm:
            # Sort by magnorm.
            sorted_index = np.argsort(self._magnorm[index])
            selected = [selected[i] for i in sorted_index]

        return selected

class InstCatTrimmer(dict):
    """
    Subclass of dict to provide trimmed instance catalogs for
    acceptance cones centered on CCDs in the LSST focalplane.

    Attributes
    ----------
    instcat_file: str
        Instance catalog filename.
    obs_md: ObservationMetadata
        Observation metadata for the visit.
    minsource: int
        Minimum number of sersic objects to require for a sensor-visit
        to be simulated.

    """
    def __init__(self, instcat, sensor_list, chunk_size=int(1e5),
                 radius=0.18, numRows=None):
        """
        Parameters
        ----------
        instcat: str
            Path to input instance catalog.  The file can have includeobj
            entries.
        sensor_list: list
            List of sensors, e.g., "R:2,2 S:1,1", for which to provide
            object lists.
        chunk_size: int [int(1e5)]
            Number of lines to read in at a time from the instance catalogs
            to avoid excess memory usage.
        radius: float [0.18]
            Radius in degrees for the acceptance cone to use for each
            sensor.
        numRows: int [None]
            Maximum number of rows to read in from the instance catalog.
        """
        super(InstCatTrimmer, self).__init__()
        self.instcat_file = instcat
        self._read_commands()
        self._process_objects(sensor_list, chunk_size, radius=radius,
                              numRows=numRows)

    def _process_objects(self, sensor_list, chunk_size, radius=0.18,
                         numRows=None):
        """
        Loop over chunks of lines from the instance catalog
        and disaggregate the entries into the separate object lists
        for each sensor using the Disaggregator class to apply the
        acceptance cone cut centered on each sensor.
        """
        num_lines = self._get_num_lines() if numRows is None else numRows
        self.update({sensor: [] for sensor in sensor_list})
        with desc.imsim.fopen(self.instcat_file, mode='rt') as fd:
            nread = 0
            while nread < num_lines:
                object_lines = []
                for _, line in zip(range(chunk_size), fd):
                    nread += 1
                    if not line.startswith('object'):
                        continue
                    object_lines.append(line)
                disaggregator = Disaggregator(object_lines, self)
                for sensor in self:
                    obj_list = disaggregator.get_object_entries(sensor,
                                                                radius=radius)
                    self[sensor].extend(obj_list)

    def _get_num_lines(self):
        """
        Get the total number of lines in the instance catalog.
        This is needed for the exit condition in the _process_objects
        method.
        """
        num_lines = 0
        with desc.imsim.fopen(self.instcat_file, mode='rt') as fd:
            for _ in fd:
                num_lines += 1
        return num_lines

    def _read_commands(self):
        """Read in the commands from the instance catalog."""
        max_lines = 50  # There should be fewer than 50, but put a hard
                        # limit to avoid suspect catalogs.
        self.command_lines = []
        phosim_commands = dict()
        with desc.imsim.fopen(self.instcat_file, mode='rt') as input_:
            for line, _ in zip(input_, range(max_lines)):
                if line.startswith('object'):
                    break
                if line.startswith('#'):
                    continue
                self.command_lines.append(line)
                tokens = line.strip().split()
                phosim_commands[tokens[0]] = float(tokens[1])
        try:
            self.minsource = phosim_commands['minsource']
        except KeyError:
            self.minsource = None
        phosim_commands['bandpass'] = 'ugrizy'[int(phosim_commands['filter'])]
        self.obs_md = desc.imsim.phosim_obs_metadata(phosim_commands)
