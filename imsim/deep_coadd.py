"""
Interface to Rubin deep_coadd PSFs
"""

import galsim
from galsim.config import InputLoader, RegisterInputType, RegisterObjectType
import lsst.daf.butler as daf_butler
import lsst.geom


class DeepCoadd:
    def __init__(self, butler, band, skymap, dstype="deep_coadd"):
        """
        Parameters
        ----------
        butler : lsst.daf.butler.Butler
            Butler for the data repo and collection containing the deep_coadds.
        band : str
            Band of interest, i.e., in "ugrizy".
        skymap : str
            Name of the skymap to use, e.g., "lsst_cells_v2".
        dstype : str
            The dataset type of the cell-based coadds.  Default: "deep_coadd".
        """
        self.butler = butler
        self.band = band
        self.skymap_name = skymap
        self.skymap = butler.get("skyMap", name=skymap)
        self.dstype = dstype
        self._psf_cache = {}
        self._butler_cache = {}

    def getPSF(self, ra, dec):
        """Return the cell coadd PSF, evaluated at the center of
        the cell containing this sky position.
        """
        # Find the tract, patch for this location.
        sky_coords = lsst.geom.SpherePoint(
            lsst.geom.Angle(ra*lsst.geom.degrees),
            lsst.geom.Angle(dec*lsst.geom.degrees)
        )
        tract_info = self.skymap.findTract(sky_coords)
        tract = tract_info.getId()
        patch = tract_info.findPatch(sky_coords).getSequentialIndex()

        # Cache info from the butler for the corresponding deep_coadd.
        butler_key = tract, patch
        if butler_key not in self._butler_cache:
            dataId = dict(skymap=self.skymap_name, tract=tract, patch=patch,
                          band=self.band)
            deep_coadd = self.butler.get(self.dstype, **dataId)
            if hasattr(deep_coadd, "to_legacy"):
                # This is a `lsst.images` object.  Use the `.to_legacy`
                # function to convert to a `lsst.afw.image.Exposure` so
                # that we can access the functions to generate a PSF
                # image array to pass to galsim.
                deep_coadd = deep_coadd.to_legacy()
            wcs = deep_coadd.getWcs()
            psf_grid = deep_coadd.getPsf()  # This is the grid of coadd psfs.
            self._butler_cache[butler_key] = wcs, psf_grid
        else:
            wcs, psf_grid = self._butler_cache[butler_key]

        # Cache cell PSFs based on tract, patch, psf_grid index.
        pixel_coords = wcs.skyToPixel(sky_coords)
        psf_grid_index = psf_grid.grid.index(lsst.geom.Point2I(pixel_coords))
        psf_key = tract, patch, psf_grid_index
        if psf_key not in self._psf_cache:
            # Evaluate PSF at cell center.
            x0 = ((pixel_coords.x // psf_grid.grid.cell_size.x)
                  * psf_grid.grid.cell_size.x + psf_grid.grid.shape.x/2)
            y0 = ((pixel_coords.y // psf_grid.grid.cell_size.y)
                  * psf_grid.grid.cell_size.y + psf_grid.grid.shape.y/2)
            pixel_cell_center = lsst.geom.Point2D(x0, y0)
            sky_cell_center = wcs.pixelToSky(pixel_cell_center)
            # The following is based on the lsst/source_injection
            # implementation. See https://github.com/lsst/source_injection/blob/w.2026.31/python/lsst/source/injection/inject_engine.py#L705
            mat = wcs.linearizePixelToSky(
                sky_cell_center, lsst.geom.arcseconds).getMatrix()
            galsim_wcs = galsim.JacobianWCS(mat[0, 0], mat[0, 1],
                                            mat[1, 0], mat[1, 1])
            psf_array = psf_grid.computeKernelImage(pixel_cell_center).array
            self._psf_cache[psf_key] = galsim.InterpolatedImage(
                galsim.Image(psf_array), wcs=galsim_wcs)

        return self._psf_cache[psf_key]


class DeepCoaddLoader(InputLoader):
    """
    Load the deep_coadd input object.  Here's an example yaml entry:

    input.deep_coadd:
        repo: dp2
        collection: LSSTCam/runs/DRP/DP2
        band: $band  # This will be obtained from the opsim metadata entry.
    """
    def __init__(self):
        super().__init__(init_func=DeepCoadd, takes_logger=True,
                         use_proxy=False)
        self.butler = None

    def getKwargs(self, config, base, logger):
        logger.debug("Get kwargs for DeepCoadd")
        req = {
            "repo": str,
            "collection": str,
            "band": str
        }
        opt = {
            "skymap": str,
            "dstype": str
        }
        params, _ = galsim.config.GetAllParams(config, base, req=req, opt=opt)
        if self.butler is None:
            self.butler = daf_butler.Butler(params["repo"],
                                            collections=[params["collection"]])

        kwargs = {
            "butler": self.butler,
            "band": params["band"],
            "skymap": params.get("skymap", "lsst_cells_v2"),
            "dstype": params.get("dstype", "deep_coadd"),
        }
        safe = True
        return kwargs, safe


def BuildRubinCoaddPSF(config, base, ignore, gsparams, logger):
    """
    Build PSFs from Rubin deep_coadds.  The deep_coadd image will be
    retrieved from the data repository using the butler for the
    tract-patch combination corresponding to the object's sky position.

    Assuming the input.deep_coadd object is defined, to use this
    PSF, add the following to the config yaml:

    input.atm_psf: ""  # disable the atmospheric PSF
    psf:
        type: RubinCoaddPSF
    """
    deep_coadd = galsim.config.GetInputObj('deep_coadd', config, base,
                                           'RubinCoaddPSF')
    image_pos = base['image_pos']
    celestial_coord = base['wcs'].toWorld(image_pos)
    ra = celestial_coord.ra / galsim.degrees
    dec = celestial_coord.dec / galsim.degrees
    safe = False
    return deep_coadd.getPSF(ra, dec), safe


RegisterInputType("deep_coadd", DeepCoaddLoader())
RegisterObjectType("RubinCoaddPSF", BuildRubinCoaddPSF, input_type="deep_coadd")
