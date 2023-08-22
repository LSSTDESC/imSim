import galsim
from galsim.config import ExtraOutputBuilder, RegisterExtraOutput, GetInputObj
from galsim.config.input import ParseValue, GetAllParams
import numpy as np
import batoid


class SagBuilder(ExtraOutputBuilder):
    """Build surface sag maps.  The sag is the height above or below a plane
    perpendicular to the optic axis.  It characterizes the shape of the optic.


    Required config inputs:
        file_name: str
            Name of file to write sag images to.

    Optional config inputs:
        nx: int
            Size of sag map images in pixels.  Default: 255.
    """
    def initialize(self, data, scratch, config, base, logger):
        req = {'file_name': str}
        opt = {'nx': int}
        kwargs, safe = GetAllParams(config, base, req=req, opt=opt)
        self.nx = kwargs.pop('nx', 255)
        self.final_data = None

    def finalize(self, config, base, main_data, logger):
        telescope = GetInputObj(
            'telescope',
            config,
            base,
            'opd'
        ).fiducial
        self.final_data = []

        xs = np.linspace(-1, 1, self.nx)

        for name, optic in telescope.itemDict.items():
            if not isinstance(optic, batoid.Interface):
                continue
            outer = optic.R_outer
            inner = optic.R_inner
            xx = xs * outer
            xx, yy = np.meshgrid(xx, xx)
            rr = np.hypot(xx, yy)
            # Only bother evaluating sag where it is potentially defined.
            ww = np.where((rr <= outer) & (rr >= inner))
            out = np.full((self.nx, self.nx), np.nan)
            out[ww] = optic.surface.sag(xx[ww], yy[ww])
            # And now mask out any other non-trivial obscurations.
            out[optic.obscuration.contains(xx, yy)] = np.nan

            dx = (xs[1] - xs[0])*outer
            dy = dx

            x0, y0, z0 = optic.coordSys.origin
            R = optic.coordSys.rot

            sag_img = galsim.Image(np.array(out.data))
            sag_img.setCenter(0, 0)
            if self.nx//2 == self.nx/2:  # if even
                world_origin = galsim.PositionD(dx/2, dx/2)
            else:
                world_origin = galsim.PositionD(0, 0)

            sag_img.wcs = galsim.OffsetWCS(
                scale=dx,
                origin=galsim.PositionI(0, 0),
                world_origin=world_origin
            )
            # Add some provenance information to header
            sag_img.header = galsim.fits.FitsHeader()
            sag_img.header['units'] = 'm', 'sag units'
            sag_img.header['dx'] = dx, 'image scale (m)'
            sag_img.header['dy'] = dy, 'image scale (m)'
            sag_img.header['x0'] = x0, 'surface origin (m)'
            sag_img.header['y0'] = y0, 'surface origin (m)'
            sag_img.header['z0'] = z0, 'surface origin (m)'
            sag_img.header['R00'] = R[0, 0], 'surface orientation matrix'
            sag_img.header['R01'] = R[0, 1], 'surface orientation matrix'
            sag_img.header['R02'] = R[0, 2], 'surface orientation matrix'
            sag_img.header['R10'] = R[1, 0], 'surface orientation matrix'
            sag_img.header['R11'] = R[1, 1], 'surface orientation matrix'
            sag_img.header['R12'] = R[1, 2], 'surface orientation matrix'
            sag_img.header['R20'] = R[2, 0], 'surface orientation matrix'
            sag_img.header['R21'] = R[2, 1], 'surface orientation matrix'
            sag_img.header['R22'] = R[2, 2], 'surface orientation matrix'
            sag_img.header['name'] = name
            sag_img.header['telescop'] = telescope.name
            self.final_data.append(sag_img)
        return self.final_data

    def writeFile(self, file_name, config, base, logger):
        galsim.fits.writeMulti(self.final_data, file_name=file_name)


RegisterExtraOutput('sag', SagBuilder())
