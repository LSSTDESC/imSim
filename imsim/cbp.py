import numpy as np
import batoid
from galsim.zernike import zernikeBasis, Zernike
from galsim import CelestialCoord, degrees, UVFunction
from galsim.config import StampBuilder, RegisterStampType
from galsim.config import WCSBuilder, RegisterWCSType
from galsim.config import GetAllParams
from galsim.config.input import GetInputObj
from .batoid_wcs import BatoidWCSFactory, det_z_offset
from .utils import focal_to_pixel
from .camera import get_camera
from astropy.time import Time


class CBPWCS(WCSBuilder):
    def __init__(self):
        self._camera = None

    @property
    def camera(self):
        if self._camera is None:
            self._camera = get_camera(self._camera_name)
        return self._camera

    def buildWCS(self, config, base, logger):
        rubin = GetInputObj('telescope', config, base, 'telescope', 0).fiducial
        cbp = GetInputObj('telescope', config, base, 'telescope', 1).fiducial

        req = {
            "det_name": str
        }
        opt = {
            "camera": str,
            "order": int
        }

        kwargs, safe = GetAllParams(config, base, req=req, opt=opt)
        kwargs['camera'] = kwargs.get('camera', 'LsstCam')
        if self._camera is None:
            self._camera_name = kwargs['camera']
            self._camera = get_camera(self._camera_name)
        if (self._camera is not None and self._camera_name != kwargs['camera']):
            self._camera_name = kwargs['camera']
            self._camera = get_camera(self._camera_name)

        det_name = kwargs['det_name']
        order = kwargs.get('order', 5)

        # Create a BatoidWCSFactory using arbitrary conditions
        factory = BatoidWCSFactory(
            boresight=CelestialCoord(0*degrees, 0*degrees),
            obstime=Time("J2000"),
            telescope=rubin,
            wavelength=500e-9, # TODO: get wavelength from band
            temperature=273,
            pressure=0,
            H2O_pressure=0,
        )

        det = self.camera[det_name]
        thx, thy = factory.get_field_samples(det)
        # This gives field angles that map to detector when passing through the center
        # of the Rubin entrance pupil.  We need to pick pupil points that additionally
        # map back to the CBP however.  Easiest way to this this is to populate rays
        # using fromFieldAngles, and then offset them to pass through the center of the
        # CBP entrance pupil.
        rv = batoid.RayVector.fromFieldAngles(
            thx, thy, projection='gnomonic',
            optic=rubin, wavelength=500e-9
        )
        rv = rv.toCoordSys(cbp['stop'].coordSys)
        rv.x[:] = 0.0
        rv.y[:] = 0.0
        rv.z[:] = 0.0
        z_offset = det_z_offset(det)
        det_telescope = factory._get_det_telescope(z_offset)
        rv1 = det_telescope.trace(rv.copy())
        # x/y transpose to convert from EDCS to DVCS
        fpx, fpy = rv1.y*1e3, rv1.x*1e3
        x, y = focal_to_pixel(fpx, fpy, det)
        # Now trace backwards to CBP focal plane
        cbp.trace(rv, reverse=True)
        cbpx, cbpy = rv.x*1e3, rv.y*1e3  # Use mm on CBP focal plane

        # Now fit a 2d polynomial to each of x, y
        # We'll use the galsim zernike library for this
        meanx = np.mean(x)
        meany = np.mean(y)
        maxr = np.max(np.hypot(x-meanx, y-meany))

        xt = (x-meanx)/maxr
        yt = (y-meany)/maxr

        jmax = (order+1)*(order+2)//2
        basis = zernikeBasis(jmax, x=xt, y=yt)
        ucoefs, *_ = np.linalg.lstsq(basis.T, cbpx, rcond=None)
        vcoefs, *_ = np.linalg.lstsq(basis.T, cbpy, rcond=None)

        ufunc = lambda x_, y_: Zernike(ucoefs)((x_-meanx)/maxr, (y_-meany)/maxr)
        vfunc = lambda x_, y_: Zernike(vcoefs)((x_-meanx)/maxr, (y_-meany)/maxr)

        return UVFunction(ufunc, vfunc)

        if False:
            rv = batoid.RayVector.fromFieldAngles(
                thx, thy, projection='gnomonic',
                optic=rubin, wavelength=500e-9
            )
            tf1 = det_telescope.traceFull(rv)
            for k, v in tf1.items():
                tf1[k]['out'].vignetted[:] = False

            rv2 = batoid.RayVector.fromFieldAngles(
                thx, thy, projection='gnomonic',
                optic=rubin, wavelength=500e-9
            )
            rv2 = rv2.toCoordSys(cbp['stop'].coordSys)
            rv2.x[:] = 0.0
            rv2.y[:] = 0.0
            rv2.z[:] = 0.0
            tf2 = det_telescope.traceFull(rv2)
            for k, v in tf2.items():
                tf2[k]['out'].vignetted[:] = False

            import ipyvolume as ipv
            import webbrowser
            import tempfile
            fig = ipv.figure(width=1200, height=800)
            ipv.style.set_style_dark()
            ipv.style.box_off()
            rubin.draw3d(ipv, color='cyan')
            cbp.draw3d(ipv, color='magenta')
            batoid.drawTrace3d(ipv, tf1, color='yellow')
            batoid.drawTrace3d(ipv, tf2, color='red')
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmpfile:
                html_file = tmpfile.name
                # Save the figure to the temporary HTML file
                ipv.save(html_file)
            webbrowser.open(f'file://{html_file}')



class CBPStampBuilder(StampBuilder):
    pass


RegisterStampType('cbp', CBPStampBuilder())
RegisterWCSType('cbp', CBPWCS(), input_type="telescope")
