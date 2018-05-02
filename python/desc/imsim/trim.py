"""
Function to apply chip-centered acceptance cones on instance catalogs.
"""
import os
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

class InstCatTrimmer:
    """
    Class to trim instance catalogs for acceptance cones centered
    on CCDs in the LSST focalplane.

    Attributes
    ----------
    instcat_file: str
        Instance catalog filename.
    command_lines: list
        PhoSim command entries.
    object_lines: list
        PhoSim object entries.
    obs_md: ObservationMetadata
        Observation metadata for the visit.
    minsource: int
        Minimum number of sersic objects to require for a sensor-visit
        to be simulated.
    """
    def __init__(self, instcat, numRows=None):
        """
        Parameters
        ----------
        instcat: str
            Path to input instance catalog.  The file can have includeobj
            entries.
        numRows: int [None]
            Number of rows to read from the instance catalog.  If None,
            then read all rows.
        """
        self.instcat_file = instcat

        # Use .fopen to read in the command and object lines from the
        # instance catalog.
        with desc.imsim.fopen(instcat, mode='rt') as input_:
            if numRows is None:
                lines = [x for x in input_ if not x.startswith('#')]
            else:
                lines = [x for _, x in zip(range(numRows), input_)
                         if not x.startswith('#')]

        # Extract the phosim commands and create the
        # ObservationMetadata object.
        self.command_lines = []
        phosim_commands = dict()
        for line in lines:
            if line.startswith('object'):
                break
            tokens = line.strip().split()
            phosim_commands[tokens[0]] = float(tokens[1])
            self.command_lines.append(line)
        try:
            self.minsource = phosim_commands['minsource']
        except KeyError:
            self.minsource = None

        phosim_commands['bandpass'] = 'ugrizy'[int(phosim_commands['filter'])]
        self.obs_md = desc.imsim.phosim_obs_metadata(phosim_commands)

        # Save the object lines separately.
        self.object_lines = lines[len(self.command_lines):]

        # Extract the ra, dec values for each object.
        self._ra = np.zeros(len(self.object_lines), dtype=np.float)
        self._dec = np.zeros(len(self.object_lines), dtype=np.float)
        self._sersic = np.zeros(len(self.object_lines), dtype=np.int)
        self._magnorm = np.zeros(len(self.object_lines), dtype=np.float)
        for i, line in enumerate(self.object_lines):
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
            camera=self._camera, obs_metadata=self.obs_md, epoch=2000.0,
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

        if (self.minsource is not None and
            sum(self._sersic[index]) < self.minsource):
            # Apply the minsource criterion.
            return []

        # Collect the selected objects.
        selected = [self.object_lines[i] for i in index[0]]
        if sort_magnorm:
            # Sort by magnorm.
            sorted_index = np.argsort(self._magnorm[index])
            selected = [selected[i] for i in sorted_index]

        return selected

    def write_instcat(self, chip_name, outfile, radius=0.18):
        """
        Write an instance catalog with entries centered on the desired
        CCD.

        Parameters
        ----------
        chip_name: str
            Name of the CCD, e.g., "R:2,2 S:1,1".
        outfile: str
            Name of the output instance catalog file.
        radius: float [0.18]
            Radius, in degrees, of the acceptance cone.
        """
        with open(outfile, 'w') as output:
            for line in self.command_lines:
                output.write(line)
            for line in self.get_object_entries(chip_name, radius=radius):
                output.write(line)
