"""
Code to convert an eimage to individual sensor segments, applying
electronics readout effects.

 * Retrieve pixel geometry for each segment
 * Copy imaging segment pixels from eimage
 * Add dark current
 * Add defects (bright defects, dark defects, traps)
 * Apply CTE
 * Apply gain
 * Apply crosstalk
 * Add read noise and bias offset
 * Write FITS file for each amplifier

"""
from __future__ import print_function, absolute_import, division
import os
import warnings
from collections import namedtuple, OrderedDict
import numpy as np
import sqlite3
import astropy.io.fits as fits
import astropy.time
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.utils as lsstUtils
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from lsst.sims.catUtils.utils import ObservationMetaDataGenerator
    from lsst.sims.utils import getRotSkyPos
from .focalplane_info import FocalPlaneInfo, cte_matrix
from .imSim import get_logger

__all__ = ['ImageSource', 'set_itl_bboxes', 'set_e2v_bboxes',
           'set_phosim_bboxes', 'set_noao_keywords']


class ImageSource(object):
    '''
    Class to create single segment images based on the pixel geometry
    described by a Camera object from lsst.afw.cameraGeom.

    Attributes
    ----------
    eimage : astropy.io.fits.HDUList
        The input eimage data.  This is used as a container for both
        the pixel data and image metadata.
    eimage_data : np.array
        The data attribute of the eimage PrimaryHDU.
    exptime : float
        The exposure time of the image in seconds.
    sensor_id : str
        The raft and sensor identifier, e.g., 'R22_S11'.
    amp_images : OrderedDict
        Dictionary of amplifier images.
    fp_props : FocalPlaneInfo object
        Object containing the readout properties of the sensors in the
        focal plane, extracted from the segmentation.txt file.
    '''
    def __init__(self, image_array, exptime, sensor_id, seg_file=None,
                 logger=None):
        """
        Class constructor.

        Parameters
        ----------
        image_array : np.array
            A numpy array containing the pixel data for an eimage.
        exptime : float
            The exposure time of the image in seconds.
        sensor_id : str
            The raft and sensor identifier, e.g., 'R22_S11'.
        seg_file : str [None]
            The segmentation.txt file, the PhoSim-formatted file that
            describes the properties of the sensors in the focal
            plane.  If None, then imSim/data/segmentation_itl.txt will
            be used.
        logger: logging.Logger [None]
            logging.Logger object to use. If None, then a logger with level
            INFO will be used.
        """
        self.eimage = fits.HDUList()
        self.eimage.append(fits.PrimaryHDU(image_array))
        self.eimage_data = self.eimage[0].data.transpose()

        self.exptime = exptime
        self.sensor_id = sensor_id

        self._read_seg_file(seg_file)
        self._make_amp_images()

        if logger is None:
            self.logger = get_logger('INFO')
        else:
            self.logger = logger

    def __del__(self):
        self.eimage.close()

    @staticmethod
    def create_from_eimage(eimage_file, sensor_id=None, seg_file=None,
                           opsim_db=None, logger=None):
        """
        Create an ImageSource object from a PhoSim eimage file.

        Parameters
        ----------
        eimage_file: str
           Filename of the eimage FITS file from which the amplifier
           images will be extracted.
        sensor_id: str [None]
            The raft and sensor identifier, e.g., 'R22_S11'.  If None,
            then extract the CHIPID keyword in the primarey HDU.
        seg_file: str [None]
            Full path of segmentation.txt, the PhoSim-formatted file
            that describes the properties of the sensors in the focal
            plane.  If None, then imSim/data/segmentation_itl.txt
            will be used.
        opsim_db: str [None]
            OpSim db file to use to find pointing information for each
            visit.  This is needed for older imSim eimage files that
            do not have the RATEL, DECTEL, and ROTANGLE keywords
            set in the FITS header.   If None and if an eimage file without
            those keywords is provided, then a RuntimeError will be raised.
        logger: logging.Logger [None]
            logging.Logger object to use. If None, then a logger with level
            INFO will be used.

        Returns
        -------
        ImageSource object
            An ImageSource object with the pixel data and metadata from
            the eimage file.
        """
        eimage = fits.open(eimage_file)
        exptime = eimage[0].header['EXPTIME']
        if sensor_id is None:
            sensor_id = eimage[0].header['CHIPID']

        image_source = ImageSource(eimage[0].data, exptime, sensor_id,
                                   seg_file=seg_file, logger=logger)
        image_source.eimage = eimage
        image_source.eimage_data = eimage[0].data
        image_source._read_pointing_info(opsim_db)
        return image_source

    def _read_pointing_info(self, opsim_db):
        try:
            self.ratel = self.eimage[0].header['RATEL']
            self.dectel = self.eimage[0].header['DECTEL']
            self.rotangle = self.eimage[0].header['ROTANGLE']
            return
        except KeyError:
            if opsim_db is None:
                raise RuntimeError("eimage file does not have pointing info. "
                                   "Need an opsim db file.")
        # Read from the opsim db.
        visit = self.eimage[0].header['OBSID']
        # We need an ObservationMetaData object to use the getRotSkyPos
        # function.
        obs_gen = ObservationMetaDataGenerator(database=opsim_db,
                                               driver="sqlite")
        obs_md = obs_gen.getObservationMetaData(obsHistID=visit,
                                                boundType='circle',
                                                boundLength=0)[0]
        # Extract pointing info from opsim db for desired visit.
        conn = sqlite3.connect(opsim_db)
        query = """select descDitheredRA, descDitheredDec,
        descDitheredRotTelPos from summary where
        obshistid={}""".format(visit)
        curs = conn.execute(query)
        ra, dec, rottelpos = [np.degrees(x) for x in curs][0]
        conn.close()
        self.ratel, self.dectel = ra, dec
        obs_md.pointingRA = ra
        obs_md.pointingDec = dec
        self.rotangle = getRotSkyPos(ra, dec, obs_md, rottelpos)

    def _read_seg_file(self, seg_file):
        if seg_file is None:
            seg_file = os.path.join(lsstUtils.getPackageDir('imSim'),
                                    'data', 'segmentation_itl.txt')
        self.fp_props = FocalPlaneInfo.read_phosim_seg_file(seg_file)

    def get_amp_image(self, amp_info_record, imageFactory=afwImage.ImageI):
        """
        Return an amplifier afwImage.Image object with electronics
        readout effects applied.  This method is only provided so that
        the pixel data and geometry can be displayed using
        lsst.afw.cameraGeom.utils.showAmp.

        Parameters
        ----------
        amp_info_record : lsst.afw.table.tableLib.AmpInfoRecord
            Data structure used by cameraGeom to contain the amplifier
            information such as pixel geometry, gain, noise, etc..
        imageFactory : lsst.afw.image.Image[DFIU], optional
            Image factory to be used for creating the return value.

        Returns
        -------
        lsst.afw.Image[DFIU]
            The image object containing the pixel data.
        """
        amp_name = self.amp_name(amp_info_record)
        float_image = self.amp_images[amp_name]
        if imageFactory == afwImage.ImageF:
            return float_image
        # Return image as the type given by imageFactory.  The
        # following line implicitly assumes that the bounding box for
        # the full segment matches the bounding box read from
        # segmentation.txt in the creation of the .fp_props attribute.
        output_image = imageFactory(amp_info_record.getRawBBox())
        output_image.getArray()[:] = float_image.getArray()
        return output_image

    def amp_name(self, amp_info_record):
        """
        The ampifier name derived from a lsst.afw.table.tableLib.AmpInfoRecord.

        Parameters
        ----------
        amp_info_record : lsst.afw.table.tableLib.AmpInfoRecord

        Returns
        -------
        str
             The amplifier name, e.g., "R22_S22_C00".
        """
        return '_'.join((self.sensor_id,
                         'C%s' % amp_info_record.getName()[::2]))

    def _make_amp_images(self):
        """
        Make the amplifier images for all the amps in the sensor.
        """
        self.amp_images = OrderedDict()
        sensor_props = self.fp_props.get_sensor(self.sensor_id)
        for amp_name in sensor_props.amp_names:
            self._make_amp_image(amp_name)
        self._apply_crosstalk()
        for amp_name in sensor_props.amp_names:
            self._add_read_noise_and_bias(amp_name)

    def _make_amp_image(self, amp_name):
        """
        Create the segment image for the amplier geometry specified in amp.

        Parameters
        ----------
        amp_name : str
            The amplifier name, e.g., "R22_S11_C00".
        """
        amp_props = self.fp_props.get_amp(amp_name)
        bbox = amp_props.mosaic_section
        full_segment = afwImage.ImageF(amp_props.full_segment)

        # Get the imaging segment (i.e., excluding prescan and
        # overscan regions), and fill with data from the eimage.
        imaging_segment = full_segment.Factory(full_segment, amp_props.imaging)
        data = self.eimage_data[bbox.getMinY():bbox.getMaxY()+1,
                                bbox.getMinX():bbox.getMaxX()+1].copy()

        # Apply flips in x and y relative to assembled eimage in order
        # to have the pixels in readout order.
        if amp_props.flip_x:
            data = data[:, ::-1]
        if amp_props.flip_y:
            data = data[::-1, :]

        imaging_segment.getArray()[:] = data
        full_arr = full_segment.getArray()

        # Add dark current.
        full_arr += np.random.poisson(amp_props.dark_current*self.exptime,
                                      size=full_arr.shape)

        # Add defects.

        # Apply CTE.
        pcte_matrix = cte_matrix(full_arr.shape[0], amp_props.pcti)
        for col in range(0, full_arr.shape[1]):
            full_arr[:, col] = np.dot(pcte_matrix, full_arr[:, col])

        scte_matrix = cte_matrix(full_arr.shape[1], amp_props.scti)
        for row in range(0, full_arr.shape[0]):
            full_arr[row, :] = np.dot(scte_matrix, full_arr[row, :])

        # Convert to ADU.
        full_arr /= amp_props.gain

        self.amp_images[amp_name] = full_segment

    def _add_read_noise_and_bias(self, amp_name):
        """
        Add read noise and bias.  This should be done as the final
        step before returning the processed image.

        Parameters
        ----------
        amp_name : str
            The amplifier name, e.g., "R22_S11_C00".
        """
        amp_props = self.fp_props.get_amp(amp_name)
        full_arr = self.amp_images[amp_name].getArray()
        full_arr += np.random.normal(scale=amp_props.read_noise,
                                     size=full_arr.shape)
        full_arr += amp_props.bias_level

    def _apply_crosstalk(self):
        """
        Apply inter-amplifier crosstalk using the cross-talk matrix
        from segmentation.txt.  This should be run only once and
        only after ._make_amp_image has been run for each amplifier.
        """
        sensor_props = self.fp_props.get_sensor(self.sensor_id)
        imarrs = np.array([self.amp_images[amp_name].getArray()
                           for amp_name in sensor_props.amp_names])
        for amp_name in sensor_props.amp_names:
            amp_props = self.fp_props.get_amp(amp_name)
            self.amp_images[amp_name].getArray()[:, :] \
                = sum([x*y for x, y in zip(imarrs, amp_props.crosstalk)])

    def get_amplifier_hdu(self, amp_name):
        """
        Get an astropy.io.fits.HDU for the specified amplifier.

        Parameters
        ----------
        amp_name : str
            The amplifier name, e.g., "R22_S11_C00".

        Returns
        -------
        astropy.io.fits.ImageHDU
            Image HDU with the pixel data and header keywords
            appropriate for the requested sensor segment.
        """
        hdu = fits.ImageHDU(data=self.amp_images[amp_name].getArray().astype(np.int32))

        # Copy keywords from eimage primary header.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for key in self.eimage[0].header.keys():
                try:
                    hdu.header[key] = self.eimage[0].header[key]
                except ValueError as eobj:
                    # eimages produced by phosim contain non-ASCII or
                    # non-printable characters resulting in a ValueError.
                    self.logger.warn("ValueError raised while attempting to "
                                     "read {} from eimage header".format(key))

        # Set NOAO geometry keywords.
        amp_props = self.fp_props.get_amp(amp_name)
        hdu.header['DATASEC'] = self._noao_section_keyword(amp_props.imaging)
        hdu.header['DETSEC'] = \
            self._noao_section_keyword(amp_props.mosaic_section,
                                       flipx=amp_props.flip_x,
                                       flipy=amp_props.flip_y)
        hdu.header['BIASSEC'] = \
            self._noao_section_keyword(amp_props.serial_overscan)
        hdu.header['GAIN'] = amp_props.gain

        return hdu

    def write_amplifier_image(self, amp_name, outfile, overwrite=True):
        """
        Write the pixel data for the specified amplifier as FITS image.

        Parameters
        ----------
        amp_name: str
            Amplifier id, e.g., "R22_S11_C00".
        outfile: str
            Filename of the FITS file to be written.
        overwrite: bool [True]
            Flag whether to overwrite an existing output file.
        """
        output = fits.HDUList(fits.PrimaryHDU())
        output.append(self.get_amplifier_hdu(amp_name))
        output.writeto(outfile, overwrite=overwrite)

    def write_fits_file(self, outfile, overwrite=True, run_number=None,
                        lsst_num='LCA-11021_RTM-000'):
        """
        Write the processed eimage data as a multi-extension FITS file.

        Parameters
        ----------
        outfile : str
            Name of the output FITS file.
        overwrite : bool, optional
            Flag whether to overwrite an existing output file.  Default: True.
        """
        output = fits.HDUList(fits.PrimaryHDU())
        output[0].header = self.eimage[0].header
        if run_number is None:
            run_number = output[0].header['OBSID']
        output[0].header['RUNNUM'] = str(run_number)
        output[0].header['DARKTIME'] = output[0].header['EXPTIME']
        mjd_obs = astropy.time.Time(output[0].header['MJD-OBS'], format='mjd')
        output[0].header['DATE-OBS'] = mjd_obs.isot
        output[0].header['LSST_NUM'] = lsst_num
        output[0].header['TESTTYPE'] = 'IMSIM'
        output[0].header['IMGTYPE'] = 'SKYEXP'
        output[0].header['MONOWL'] = -1
        raft, ccd = output[0].header['CHIPID'].split('_')
        output[0].header['RAFTNAME'] = raft
        output[0].header['SENSNAME'] = ccd
        output[0].header['OUTFILE'] = os.path.basename(outfile)
        # Add boresight pointing angles and rotskypos (angle of sky
        # relative to Camera coordinates) from which obs_lsstCam can
        # infer the CCD-wide WCS.
        output[0].header['RATEL'] = self.ratel
        output[0].header['DECTEL'] = self.dectel
        output[0].header['ROTANGLE'] = self.rotangle
        # Use seg_ids to write the image extensions in the order
        # specified by LCA-10140.
        seg_ids = '10 11 12 13 14 15 16 17 07 06 05 04 03 02 01 00'.split()
        for seg_id in seg_ids:
            amp_name = '_C'.join((self.sensor_id, seg_id))
            output.append(self.get_amplifier_hdu(amp_name))
            output[-1].header['EXTNAME'] = 'Segment%s' % seg_id
        output.writeto(outfile, overwrite=overwrite)

    @staticmethod
    def _noao_section_keyword(bbox, flipx=False, flipy=False):
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


def set_itl_bboxes(amp):
    """
    Function to apply realistic pixel geometry for ITL sensors.

    Parameters
    ----------
    amp : lsst.afw.table.tableLib.AmpInfoRecord
        Data structure containing the amplifier information such as
        pixel geometry, gain, noise, etc..

    Returns
    -------
    lsst.afw.table.tableLib.AmpInfoRecord
        The updated AmpInfoRecord.
    """
    amp.setRawBBox(afwGeom.Box2I(afwGeom.Point2I(0, 0),
                                 afwGeom.Extent2I(532, 2020)))
    amp.setRawDataBBox(afwGeom.Box2I(afwGeom.Point2I(3, 0),
                                     afwGeom.Extent2I(509, 2000)))
    amp.setRawHorizontalOverscanBBox(afwGeom.Box2I(afwGeom.Point2I(512, 0),
                                                   afwGeom.Extent2I(20, 2000)))
    amp.setRawVerticalOverscanBBox(afwGeom.Box2I(afwGeom.Point2I(0, 2000),
                                                 afwGeom.Extent2I(532, 20)))
    amp.setRawPrescanBBox(afwGeom.Box2I(afwGeom.Point2I(0, 0),
                                        afwGeom.Extent2I(3, 2000)))
    return amp


def set_e2v_bboxes(amp):
    """
    Function to apply realistic pixel geometry for e2v sensors.

    Parameters
    ----------
    amp : lsst.afw.table.tableLib.AmpInfoRecord
        Data structure containing the amplifier information such as
        pixel geometry, gain, noise, etc..

    Returns
    -------
    lsst.afw.table.tableLib.AmpInfoRecord
        The updated AmpInfoRecord.
    """
    amp.setRawBBox(afwGeom.Box2I(afwGeom.Point2I(0, 0),
                                 afwGeom.Extent2I(542, 2022)))
    amp.setRawDataBBox(afwGeom.Box2I(afwGeom.Point2I(10, 0),
                                     afwGeom.Extent2I(522, 2002)))
    amp.setRawHorizontalOverscanBBox(afwGeom.Box2I(afwGeom.Point2I(522, 0),
                                                   afwGeom.Extent2I(20, 2002)))
    amp.setRawVerticalOverscanBBox(afwGeom.Box2I(afwGeom.Point2I(0, 2002),
                                                 afwGeom.Extent2I(542, 20)))
    amp.setRawPrescanBBox(afwGeom.Box2I(afwGeom.Point2I(0, 0),
                                        afwGeom.Extent2I(10, 2002)))
    return amp


def set_phosim_bboxes(amp):
    """
    Function to apply the segmentation.txt geometry.

    Parameters
    ----------
    amp : lsst.afw.table.tableLib.AmpInfoRecord
        Data structure containing the amplifier information such as
        pixel geometry, gain, noise, etc..

    Returns
    -------
    lsst.afw.table.tableLib.AmpInfoRecord
        The updated AmpInfoRecord.
    """
    amp.setRawBBox(afwGeom.Box2I(afwGeom.Point2I(0, 0),
                                 afwGeom.Extent2I(519, 2001)))
    amp.setRawDataBBox(afwGeom.Box2I(afwGeom.Point2I(4, 1),
                                     afwGeom.Extent2I(509, 2000)))
    amp.setRawHorizontalOverscanBBox(afwGeom.Box2I(afwGeom.Point2I(513, 1),
                                                   afwGeom.Extent2I(6, 2000)))
    amp.setRawVerticalOverscanBBox(afwGeom.Box2I(afwGeom.Point2I(0, 2001),
                                                 afwGeom.Extent2I(519, 0)))
    amp.setRawPrescanBBox(afwGeom.Box2I(afwGeom.Point2I(0, 1),
                                        afwGeom.Extent2I(4, 2000)))
    return amp

PixelParameters = namedtuple('PixelParameters',
                             ('''dimv dimh ccdax ccday ccdpx ccdpy gap_inx
                                 gap_iny gap_outx gap_outy preh'''.split()))
pixel_parameters \
    = {'E2V': PixelParameters(2002, 512, 4004, 4096, 4197, 4200, 28,
                              25, 26.5, 25, 10),
       'ITL': PixelParameters(2000, 509, 4000, 4072, 4198, 4198, 27,
                              27, 26.0, 26, 3)}

# The image extension headers do not include the CCD vendor so we
# infer that from the values of the DETSIZE keyword.
vendors = {'[1:4096,1:4004]': 'E2V',
           '[1:4072,1:4000]': 'ITL'}

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
