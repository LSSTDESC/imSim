import galsim
from galsim.config import ExtraOutputBuilder, RegisterExtraOutput, GetInputObj
from galsim.config.input import ParseValue, GetAllParams
from batoid.analysis import wavefront, zernike
import numpy as np


class OPDBuilder(ExtraOutputBuilder):
    """Class to process optical-path-difference (OPD) image requests from config
    layer.

    Required config inputs:
        file_name: str
            Name of file to write OPD images to.
        fields: list of Angle
            List of field angles for which to compute OPD.  Field angles are
            specified in the (rotated) coordinate system of the telescope's
            entrance pupil (usually the primary mirror).

    Optional config inputs:
        rotTelPos: Angle
            Rotation applied to field angles.  Default: 0.0 radians.
            This is useful for aligning field angles with particular detectors
            when a camera rotator has been engaged.  If both this keyword and
            the telescope's rotTelPos keyword are equal, then the fields sampled
            will correspond to the same focal plane positions as when both
            rotTelPos keywords are 0.0.
        nx: int
            Size of OPD image in pixels.  Default: 255.
        wavelength: float
            Wavelength of light in nanometers.  Default: None.
            If not specified, then the wavelength will be taken from the
            current bandpass.
        projection: str
            Projection mapping field angles to spherical coordinates.
            Default: 'postel'.
            See batoid documentation for more details.
        sphereRadius: float
            Radius of reference sphere in meters.  Default: None.
            If not specified, then the radius will be taken from the telescope
            object.
        reference: {'chief', 'mean'}
            Reference point for measuring OPD.  Default: 'chief'.
            See batoid documentation for more details.
        eps: float
            Annular Zernike obscuration fraction.  Default: None.
            If not specified, then the value will be taken from the telescope
            object.
        jmax: int
            Maximum Zernike order.  Default: 28.

    Notes:
        The OPD image coordinates are always aligned with the entrance pupil,
        regardless of the value of rotTelPos.  The OPD values are in nm, with
        NaN values corresponding to vignetted regions.  The OPD is always
        computed for the fiducial telescope focal plane height; i.e., it ignores
        any detector-by-detector offsets in focal plane height.
    """
    def initialize(self, data, scratch, config, base, logger):
        req = {
            'file_name': str,
            'fields': list
        }
        opt = {
            'rotTelPos': galsim.Angle,
            'nx': int,
            'wavelength': float,
            'projection': str,
            'sphereRadius': float,
            'reference': str,
            'eps': float,
            'jmax': int
        }
        kwargs, safe = GetAllParams(config, base, req=req, opt=opt)

        # Handle defaults for optional kwargs
        self.rotTelPos = kwargs.pop('rotTelPos', 0.0*galsim.radians)
        self.nx = kwargs.pop('nx', 255)
        self.projection = kwargs.pop('projection', 'postel')
        self.reference = kwargs.pop('reference', 'chief')
        telescope = GetInputObj(
            'telescope',
            config,
            base,
            'opd'
        ).fiducial
        self.sphereRadius = kwargs.pop(
            'sphereRadius',
            telescope.sphereRadius
        )
        self.eps = kwargs.pop('eps', telescope.pupilObscuration)
        self.jmax = kwargs.pop('jmax', 28)

        # Try to get wavelength from bandpass if not specified
        if 'wavelength' not in kwargs:
            if 'bandpass' not in base and 'bandpass' in base.get('image',{}):
                bp = galsim.config.BuildBandpass(
                    base['image'],
                    'bandpass',
                    base,
                    logger
                )[0]
                base['bandpass'] = bp
            bandpass = base.get('bandpass', None)
            if bandpass is None:
                raise ValueError("Must specify either wavelength or bandpass")
            self.wavelength = bandpass.effective_wavelength
        else:
            self.wavelength = kwargs['wavelength']

        # Parse list of field angles
        self.fields = []
        self.rot_fields = []
        for d in config['fields']:
            thx, _ = ParseValue(d, 'thx', base, galsim.Angle)
            thy, _ = ParseValue(d, 'thy', base, galsim.Angle)
            self.fields.append((thx, thy))
            rtp = self.rotTelPos
            rot = np.array(
                [[np.cos(rtp), -np.sin(rtp)],
                 [np.sin(rtp), np.cos(rtp)]]
            )
            r_thx, r_thy = rot @ np.array([thx.rad, thy.rad])

            self.rot_fields.append((
                r_thx * galsim.radians,
                r_thy * galsim.radians
            ))

        self.final_data = None

    def finalize(self, config, base, main_data, logger):
        telescope = GetInputObj(
            'telescope',
            config,
            base,
            'opd'
        ).fiducial
        wavelength = self.wavelength
        self.final_data = []
        for (r_thx, r_thy), (thx, thy) in zip(self.rot_fields, self.fields):
            # Use batoid to compute OPD (= wavefront)
            opd = wavefront(
                telescope, r_thx.rad, r_thy.rad, wavelength*1e-9,
                nx=self.nx,
                sphereRadius=self.sphereRadius,
                projection=self.projection,
                reference=self.reference
            )
            dx = opd.primitiveVectors[0, 0]
            dy = opd.primitiveVectors[1, 1]
            opd_arr = opd.array*wavelength  # convert from waves to nm
            # FITS doesn't know about numpy masked arrays,
            # So just fill in masked values with NaN
            opd_arr.data[opd_arr.mask] = np.nan
            opd_img = galsim.Image(np.array(opd_arr.data), scale=dx)
            # Add some provenance information to header
            opd_img.header = galsim.fits.FitsHeader()
            opd_img.header['units'] = 'nm', 'OPD units'
            opd_img.header['dx'] = dx, 'entrance pupil coord scale (m)'
            opd_img.header['dy'] = dy, 'entrance pupil coord scale (m)'
            opd_img.header['thx'] = thx.deg, 'field angle (deg)'
            opd_img.header['thy'] = thy.deg, 'field angle (deg)'
            opd_img.header['r_thx'] = r_thx.deg, 'rotated field angle (deg)'
            opd_img.header['r_thy'] = r_thy.deg, 'rotated field angle (deg)'
            opd_img.header['wavelen'] = wavelength, '(nm)'
            opd_img.header['prjct'] = self.projection, 'field angle map projection'
            opd_img.header['sph_rad'] = self.sphereRadius, 'reference sphere radius (m)'
            opd_img.header['sph_ref'] = self.reference, 'reference point'
            opd_img.header['eps'] = self.eps, 'Annular Zernike obscuration fraction'
            opd_img.header['jmax'] = self.jmax, 'Max index for annular Zernike coefficients'
            opd_img.header['telescop'] = telescope.name
            # Add Annular Zernike coefficients to header
            zk = zernike(
                telescope, thx.rad, thy.rad, wavelength*1e-9,
                nx=self.nx,
                sphereRadius=self.sphereRadius,
                projection=self.projection,
                reference=self.reference,
                eps=self.eps,
                jmax=self.jmax
            )
            zk *= wavelength  # convert from waves to nm
            for j in range(1, self.jmax+1):
                opd_img.header[f'AZ_{j:03d}'] = zk[j], '(nm)'
            self.final_data.append(opd_img)
        return self.final_data

    def writeFile(self, file_name, config, base, logger):
        galsim.fits.writeMulti(self.final_data, file_name=file_name)


RegisterExtraOutput('opd', OPDBuilder())
