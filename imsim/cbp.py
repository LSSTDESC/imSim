import numpy as np
import batoid
from galsim.zernike import zernikeBasis, Zernike
from galsim import CelestialCoord, degrees, UVFunction
from galsim.config import StampBuilder, RegisterStampType
from galsim.config import WCSBuilder, RegisterWCSType
from galsim.config import GetAllParams
from galsim.config.input import GetInputObj
from galsim.photon_array import PhotonOp
from galsim.config.photon_ops import RegisterPhotonOpType, PhotonOpBuilder
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
        # map back to the CBP however.  Easiest way to do this is to populate rays
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
        # Now trace to CBP focal plane
        cbp.trace(rv)
        cbpx, cbpy = rv.x*1e6, rv.y*1e6  # Use microns on CBP focal plane

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

        if False:
            def colorbar(mappable):
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                import matplotlib.pyplot as plt
                last_axes = plt.gca()
                ax = mappable.axes
                fig = ax.figure
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = fig.colorbar(mappable, cax=cax)
                plt.sca(last_axes)
                return cbar

            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 3, figsize=(12, 9))
            ax = axes[0][0]
            colorbar(ax.scatter(x, y, c=cbpx))
            ax.set_title("x")
            ax = axes[1][0]
            colorbar(ax.scatter(x, y, c=cbpy))
            ax.set_title("y")

            ax = axes[0][1]
            colorbar(ax.scatter(x, y, c=ufunc(x, y)))
            ax.set_title("x fit")
            ax = axes[1][1]
            colorbar(ax.scatter(x, y, c=vfunc(x, y)))
            ax.set_title("y fit")

            ax = axes[0][2]
            colorbar(ax.scatter(x, y, c=ufunc(x, y)-cbpx))
            ax.set_title("x residual")
            ax = axes[1][2]
            colorbar(ax.scatter(x, y, c=vfunc(x, y)-cbpy))
            ax.set_title("y residual")

            for ax in axes.flatten():
                ax.set_aspect('equal')

            fig.tight_layout()
            plt.show()

        if False:
            rv = batoid.RayVector.fromFieldAngles(
                thx, thy, projection='gnomonic',
                optic=rubin, wavelength=500e-9
            )
            rv = rv.toCoordSys(cbp['stop'].coordSys)
            rv.x[:] = 0.0
            rv.y[:] = 0.0
            rv.z[:] = 0.0
            tf = det_telescope.traceFull(rv)
            ctf = cbp.traceFull(rv)
            for k, v in tf.items():
                tf[k]['out'].vignetted[:] = False
            for k, v in ctf.items():
                ctf[k]['out'].vignetted[:] = False

            # Shift rays and optics to CBP frame
            cs = cbp['stop'].coordSys
            # cs = batoid.globalCoordSys

            import ipyvolume as ipv
            import webbrowser
            import tempfile
            fig = ipv.figure(width=1600, height=1000)
            ipv.style.set_style_dark()
            ipv.style.box_off()
            rubin.draw3d(ipv, color='cyan', globalSys=cs)
            cbp.draw3d(ipv, color='magenta', globalSys=cs)
            batoid.drawTrace3d(ipv, tf, color='red', globalSys=cs)
            batoid.drawTrace3d(ipv, ctf, color='blue', globalSys=cs)
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmpfile:
                html_file = tmpfile.name
                # Save the figure to the temporary HTML file
                ipv.save(html_file)
            webbrowser.open(f'file://{html_file}')

        return UVFunction(ufunc, vfunc)



class CBPStampBuilder(StampBuilder):
    def getDrawMethod(self, config, base, logger):
        return 'phot'


class CBPRubinOptics(PhotonOp):
    def __init__(self, rubin, cbp, cbp_pos, image_pos, det):
        self.rubin = rubin
        self.cbp = cbp
        self.cbp_pos = cbp_pos
        self.image_pos = image_pos
        self.det = det

    def applyTo(self, photon_array, local_wcs=None, rng=None):
        print(f"{self.image_pos = } pixels")
        print(f"{self.cbp_pos = } microns")
        print()
        print(f"{np.quantile(photon_array.x, [0,0.5,1]) = }")
        print(f"{np.quantile(photon_array.y, [0,0.5,1]) = }")
        # print(f"{local_wcs = }")
        # Use local WCS to get spot positions on the CBP focal plane
        x, y = local_wcs.toWorld(
            photon_array.x,
            photon_array.y
            # photon_array.x+self.image_pos.x,
            # photon_array.y+self.image_pos.y
        )
        x += self.cbp_pos.x
        y += self.cbp_pos.y
        print("CBP focal plane coordinates [micron]:")
        print(f"{np.quantile(x, [0,0.5,1]) = }")
        print(f"{np.quantile(y, [0,0.5,1]) = }")
        x *= 1e-6  # micron to meters
        y *= 1e-6
        # Now we need to make up an initial velocity distribution.
        # I'll guess that isotropic is a good starting point.
        # Experimented to find that 15 degrees is a reasonable opening angle.
        rng = rng.np
        vz = rng.uniform(np.cos(np.deg2rad(15.0)), 1.0, size=photon_array.size())
        vr = np.sqrt(1 - vz**2)
        phi = rng.uniform(0, 2*np.pi, size=photon_array.size())
        vx = vr * np.cos(phi)
        vy = vr * np.sin(phi)
        rv = batoid.RayVector._directInit(
            x=x,
            y=y,
            z=np.zeros(photon_array.size()),
            vx=vx,
            vy=vy,
            vz=-vz,
            t=np.zeros(photon_array.size()),
            wavelength=np.full(photon_array.size(), 500e-9),
            flux=photon_array.flux,
            vignetted=np.zeros(photon_array.size(), dtype=bool),
            failed=np.zeros(photon_array.size(), dtype=bool),
            coordSys=self.cbp['Detector'].coordSys
        )
        if False:
            rvt = rv
            if len(rvt) > 100:
                rvt = rvt[rng.choice(len(rv), 100, replace=False)]
            # Show the ray trace
            tf1 = self.cbp.traceFull(rvt, reverse=True)
            # for k, v in tf1.items():
            #     v['out'].vignetted[:] = False
            rv2 = tf1['Schmidt_plate_entrance']['out']
            tf2 = self.rubin.traceFull(rv2)
            # for k, v in tf2.items():
            #     v['out'].vignetted[:] = False

            print(f"{np.mean(tf2['Detector']['out'].vignetted) = }")
            cs = batoid.globalCoordSys
            cs = self.cbp['stop'].coordSys

            import ipyvolume as ipv
            import webbrowser
            import tempfile
            fig = ipv.figure(width=1600, height=1000)
            ipv.style.set_style_dark()
            ipv.style.box_off()
            self.rubin.draw3d(ipv, color='cyan', globalSys=cs)
            self.cbp.draw3d(ipv, color='magenta', globalSys=cs)
            batoid.drawTrace3d(ipv, tf1, color='red', globalSys=cs)
            batoid.drawTrace3d(ipv, tf2, color='blue', globalSys=cs)
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmpfile:
                html_file = tmpfile.name
                # Save the figure to the temporary HTML file
                ipv.save(html_file)
            webbrowser.open(f'file://{html_file}')

        # Now trace CBP in reverse, and then through Rubin.
        self.cbp.trace(rv, reverse=True)
        self.rubin.trace(rv)
        # RV is back in Rubin focal plane coordinates.
        # Need to convert to image coordinates.

        # x/y transpose to convert from EDCS to DVCS
        fpx, fpy = rv.y*1e3, rv.x*1e3
        x, y = focal_to_pixel(fpx, fpy, self.det)
        print("Rubin focal plane coordinates [pixels]:")
        print(f"{np.quantile(x, [0,0.5,1]) = }")
        print(f"{np.quantile(y, [0,0.5,1]) = }")
        photon_array.x[:] = x - self.image_pos.x
        photon_array.y[:] = y - self.image_pos.y
        # photon_array.flux[:] = np.ones_like(~rv.vignetted, dtype=float)

        print(f"{np.quantile(photon_array.x, [0,0.5,1]) = }")
        print(f"{np.quantile(photon_array.y, [0,0.5,1]) = }")
        print(f"{np.mean(photon_array.flux)}")
        print("\n"*10)


class CBPRubinOpticsBuilder(PhotonOpBuilder):
    def buildPhotonOp(self, config, base, logger):
        req = {
            "camera": str,
        }
        kwargs, safe = GetAllParams(config, base, req=req)
        camera = get_camera(kwargs['camera'])
        det_name = base['det_name']
        det = camera[det_name]
        z_offset = det_z_offset(det)
        telescope = GetInputObj('telescope', config, base, 'telescope', 0)
        cbp = GetInputObj('telescope', config, base, 'telescope', 1)
        return CBPRubinOptics(
            telescope.get_telescope(z_offset),
            cbp.fiducial,
            cbp_pos=base['world_pos'],
            image_pos=base['image_pos'],
            det=det
        )


RegisterStampType('cbp', CBPStampBuilder())
RegisterWCSType('cbp', CBPWCS(), input_type="telescope")
RegisterPhotonOpType('CBPRubinOptics', CBPRubinOpticsBuilder(), input_type='telescope')
