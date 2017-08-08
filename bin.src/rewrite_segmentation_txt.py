#!/usr/bin/env python
"""
Script to rewrite a PhoSim segmentation.txt file to have pixel
geometries that correspond to actual vendor devices.
"""
import os
import lsst.utils as lsstUtils

class AmpGeom(object):
    """
    Class to represent the pixel geometry for LSST sensors, based on
    LCA-10140.

    Attributes
    ----------
    channel_ids : list (class level)
        The channel ids in extension order as specified by LCA-10140.
    nx : int, optional
        Number of imaging pixels in serial direction per segment.
    ny : int, optional
        Number of imaging pixels in parallel direction per segment.
    prescan : int, optional
        Number of prescan pixels.
    serial_overscan : int, optional
        Number of serial overscan pixels.
    parallel_overscan : int, optional
        Number of parallel ovescan pixels.
    nsegx : int, optional
        Number of imaging segments in the serial direction.
    nsegy : int, optional
        Number of imaging segments in the parallel direction.
    output_nodes : dict, optional
        Readout at output node direction (+/-1) relative to
        positive x-direction.
        The key values must be the channel IDs, e.g., '10', '11', etc..
    """
    channel_ids = '10 11 12 13 14 15 16 17 07 06 05 04 03 02 01 00'.split()
    def __init__(self, nx=509, ny=2000, prescan=3, serial_overscan=20,
                 parallel_overscan=20, nsegx=8, nsegy=2,
                 output_nodes=None):
        """Constructor.

        Parameters
        ----------
        nx : int, optional
            Number of imaging pixels in serial direction per segment.
        ny : int, optional
            Number of imaging pixels in parallel direction per segment.
        prescan : int, optional
            Number of prescan pixels.
        serial_overscan : int, optional
            Number of serial overscan pixels.
        parallel_overscan : int, optional
            Number of parallel ovescan pixels.
        nsegx : int, optional
            Number of imaging segments in the serial direction.
        nsegy : int, optional
            Number of imaging segments in the parallel direction.
        output_nodes : dict, optional
            Readout at output node direction (+/-1) relative to
            positive x-direction.
            The key values must be the channel IDs, e.g., '10', '11', etc..

        Notes
        -----
        Default values for parameters correspond to ITL-3800C sensors.
        """
        self.nx = nx
        self.ny = ny
        self.prescan = prescan
        self.serial_overscan = serial_overscan
        self.parallel_overscan = parallel_overscan
        self.nsegx = nsegx
        self.nsegy = nsegy
        if output_nodes is None:
            self.output_nodes = dict((self.channel_ids[i], -1)
                                     for i in range(nsegx*nsegy))

    def amp(self, channel_id):
        """
        The amplifier number as a function of channel ID.

        Parameters
        ----------
        channel_id : str
            The channel id of the desired segment, e.g., '10'.

        Returns
        -------
        int
            The amplifier number.
        """

    def xy_bounds(self, channel_id):
        """
        The bounding box corners in CCD pixels for the amplifier
        segment.

        Parameters
        ----------
        channel_id : str
            The channel id of the desired segment, e.g., '10'.

        Returns
        -------
        tuple
            Tuple giving the xmin, xmax, ymin, ymax values
            of the bounding box corners in the full CCD pixel
            coordinate system.
        """
        amp = self.channel_ids.index(channel_id) + 1
        namps = self.nsegx*self.nsegy
        if amp <= self.nsegx:
            x1 = (amp - 1)*self.nx
            x2 = amp*self.nx - 1
            y1, y2 = 0, self.ny - 1
        else:
            # Amps in "top half" of CCD, where the ordering of amps 9
            # to 16 is right-to-left.
            x1 = (namps - amp)*self.nx
            x2 = (namps - amp + 1)*self.nx - 1
            y1, y2 = self.ny, 2*self.ny - 1
        return x1, x2, y1, y2

    def xy_flips(self, channel_id):
        """
        Whether the segment image should be flipped in the x- or
        y-directions relative to the full CCD image mosaic.

        Parameters
        ----------
        channel_id : str
            The channel id of the desired segment, e.g., '10'.

        Returns
        -------
        tuple
            A pair of integers indicating a flip (-1) or not (+1).
        """
        amp = self.channel_ids.index(channel_id) + 1
        if amp <= self.nsegx:
            return self.output_nodes[channel_id], 1
        else:
            return self.output_nodes[channel_id], -1

if __name__ == '__main__':
    seg_file = os.path.join(lsstUtils.getPackageDir('obs_lsstSim'),
                            'description', 'segmentation.txt')
    print seg_file

    outfile = 'segmentation.txt_new'

    # Use default ITL geometry.
    geom = AmpGeom()

#    # e2v geometry
#    geom = AmpGeom(nx=512, ny=2002, prescan=10,
#                   output_nodes=dict((kv for kv in
#                                      zip(AmpGeom.channel_ids,
#                                          [-1]*8 + [1]*8))))

    with open(seg_file, 'r') as f:
        lines = f.readlines()

    def line_format(tokens):
        line_format = "%11s " + 4*"%6s" + 2*"%3s" + 8*" %s" + 4*"%3s"
        line_format +=\
            (len(tokens) - len(line_format.split("%")) + 1)*" %s" + "\n"
        return line_format

    with open(outfile, 'w') as output:
        i = -1
        while True:
            try:
                i += 1
                if lines[i].startswith('#'):
                    output.write(lines[i].strip() + '\n')
                    continue
                tokens = lines[i].split()
                sensor_id, num_amps = tokens[0], int(tokens[1])
                tokens[2], tokens[3] = (str(geom.nsegx*geom.nx),
                                        str(geom.nsegy*geom.ny))
                output.write(' '.join(tokens) + '\n')
                for j in range(num_amps):
                    i += 1
                    tokens = lines[i].split()
                    channel_id = tokens[0][-2:]
                    xmin, xmin, ymin, ymax = geom.xy_bounds(channel_id)
                    tokens[1:5] = [str(value) for value
                                   in geom.xy_bounds(channel_id)]
                    tokens[5:7] = [str(value) for value
                                   in geom.xy_flips(channel_id)]
                    tokens[15] = '0'
                    tokens[16] = str(geom.serial_overscan)
                    tokens[17] = str(geom.prescan)
                    tokens[18] = str(geom.parallel_overscan)
                    output.write(line_format(tokens) % tuple(tokens))
            except IndexError:
                break
