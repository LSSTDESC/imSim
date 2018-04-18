"""
Function to apply chip-centered acceptance cones on instance catalogs.
"""
import time
import numpy as np
import lsst.sims.coordUtils
from lsst.sims.utils import _angularSeparation
import desc.imsim

__all__ = ['trim_instcat']

def trim_instcat(instcat, chip_name, outfile, radius=0.18):
    """
    Function to apply chip-centered acceptance cones on instance
    catalogs.

    Parameters
    ----------
    instcat: str
        Path to input instance catalog.  The file can have includeobj
        entries.
    chip_name: str
        The name of the chip in the LSST focalplane on which to center
        the acceptance cone, e.g., "R:2,2 S:1,1".
    outfile: str
        The output file name of the trimmed instance catalog.
    radius: float [0.18]
        The radius of the acceptance cone in degrees.  The default
        will contain a single LSST sensor with a ~0.01 degree buffer.

    Returns
    -------
    int: the number of objects written to the output catalog.
    """
    # Get the center of the desired chip in focalplane pixel coordinates.
    camera = desc.imsim.get_obs_lsstSim_camera()
    center_x, center_y = desc.imsim.get_chip_center(chip_name, camera)

    # Read in all of the lines from the instance catalog. TODO: do
    # this in chunks to save memory.
    with desc.imsim.fopen(instcat, mode='rt') as input_:
        lines = [x for x in input_]

    # Write the trimmed output instance catalog.
    with open(outfile, 'w') as output:
        # Extract the phosim commands, writing them to the output
        # catalog, assuming they are all in the lines preceding the
        # object entries.
        nhdr = 0
        phosim_commands = dict()
        for line in lines:
            if line.startswith('#'):
                continue
            if line.startswith('object'):
                break
            tokens = line.strip().split()
            phosim_commands[tokens[0]] = float(tokens[1])
            output.write(line)
            nhdr += 1

        # Compute the coordinates of the center of the chip.
        phosim_commands['bandpass'] = 'ugrizy'[int(phosim_commands['filter'])]
        obs_md = desc.imsim.phosim_obs_metadata(phosim_commands)
        ra0, dec0 = lsst.sims.coordUtils.raDecFromPixelCoords(
            xPix=center_x, yPix=center_y, chipName=chip_name, camera=camera,
            obs_metadata=obs_md, epoch=2000.0, includeDistortion=True)

        # Extract the ra, dec values for each object, compute the
        # offsets from the chip center and apply the acceptance angle.
        lines = np.array(lines[nhdr:])
        ra = np.zeros(len(lines), dtype=np.float)
        dec = np.zeros(len(lines), dtype=np.float)
        for i, line in enumerate(lines):
            lon, lat = line.strip().split()[2:4]
            ra[i] = np.float(lon)
            dec[i] = np.float(lat)

        seps = _angularSeparation(ra0, dec0, ra, dec)
        index = np.where(seps < radius)
        for line in lines[index]:
            output.write(line)
    return len(lines)

if __name__ == '__main__':
    instcat = 'catalogs/phosim_cat_197356.txt'
    chip_name = "R:2,2 S:1,1"
    radius = 0.1
    outfile = 'imsim_instcat.txt'

    t0 = time.clock()
    num_objects = trim_instcat(instcat, chip_name, radius, outfile)
    print(num_objects, time.clock() - t0)
