"""
Interface to Rubin deep_coadds
"""
import pandas as pd
import galsim
from galsim.config import (InputLoader, RegisterInputType, RegisterValueType,
                           GetAllParams, GetInputObj)
import lsst.daf.butler as daf_butler
import lsst.geom


__all__ = ["DeepCoadds", "DeepCoaddLoader", "DeepCoaddData"]


class DeepCoadds:
    def __init__(self, butler, skymap_name, data_ids=None, dstype="deep_coadd"):
        """
        Parameters
        ----------
        butler : lsst.daf.butler.Butler
            Butler for the data repo and collection containing the deep_coadds.
        skymap_name : str
            Name of the skymap to use, e.g., "lsst_cells_v2".
        data_ids: list
            List of data_ids (dicts) for identifying deep_coadds to simulate.
        dstype : str
            The dataset type of the cell-based coadds.  Default: "deep_coadd".
        """
        self.butler = butler
        self.skymap_name = skymap_name
        self.skymap = butler.get("skyMap", skymap=skymap_name)
        self.data_ids = data_ids
        self.dstype = dstype
        self._psf_cache = {}
        self._grid_cache = {}
        self._wcs_cache = {}

    def get(self, index=None, data_id=None):
        """Return the deep_coadd using the index of the self.data_ids list,
        or if provided, using the data_id.
        """
        if (data_id, index) == (None, None):
            return None
        if data_id is None and index >= 0 and index < len(self.data_ids):
            data_id = self.data_ids[index]
        elif data_id not in self.data_ids:
            return None
        my_data_id = data_id.copy()
        my_data_id['skymap'] = self.skymap_name
        deep_coadd = self.butler.get(self.dstype, **my_data_id)

        # Ensure this is a lsst.images version:
        assert hasattr(deep_coadd, 'to_legacy')
        return deep_coadd

    def getBandpass(self, band):
        passband = self.butler.get('standard_passband', band=band)
        lut = galsim.LookupTable(passband['wavelength'],
                                 passband['throughput'],
                                 interpolant='linear')
        return galsim.Bandpass(lut, wave_type='nm').thin()

    def getWcs(self, data_id):
        key = data_id['tract'], data_id['patch']
        if key not in self._wcs_cache:
            deep_coadd = self.get(data_id=data_id)
            self._wcs_cache[key] = galsim.AstropyWCS(wcs=deep_coadd.fits_wcs)
        return self._wcs_cache[key]

    def getPSF(self, ra, dec, band):
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
        data_id = dict(tract=tract, patch=patch, band=band)
        wcs = self.getWcs(data_id)

        # Cache the cell grid data for the corresponding deep_coadd,
        # keyed by (tract, patch, band).
        grid_key = tract, patch, band
        if grid_key not in self._grid_cache:
            deep_coadd = self.get(data_id=data_id)
            grid = deep_coadd.grid
            psf = deep_coadd.psf
            self._grid_cache[grid_key] = grid, psf
        else:
            grid, psf = self._grid_cache[grid_key]

        # xy offsets for the current grid.
        x_offset = grid.bbox.start.x
        y_offset = grid.bbox.start.y

        # Cache cell PSFs, keyed by (tract, patch, band, grid_index).
        x, y = wcs.toImage(ra, dec, units='degrees')
        grid_index = grid.index_of(x=x + x_offset, y=y + y_offset)
        psf_key = tract, patch, band, grid_index
        if psf_key not in self._psf_cache:
            # Evaluate PSF at cell center.
            x0 = ( (x // grid.cell_shape.x) * grid.cell_shape.x
                   + grid.cell_shape.x/2 + x_offset )
            y0 = ( (y // grid.cell_shape.y) * grid.cell_shape.y
                   + grid.cell_shape.y/2 + y_offset )
            psf_array = psf.compute_kernel_image(x=x0, y=y0).array
            self._psf_cache[psf_key] = galsim.InterpolatedImage(
                galsim.Image(psf_array), wcs=wcs)
        return self._psf_cache[psf_key]


class DeepCoaddLoader(InputLoader):
    """
    Load the deep_coadds input object.  Here's an example yaml entry:

    input.deep_coadds:
        repo: dp2
        collection: dp2
    """
    def __init__(self):
        super().__init__(init_func=DeepCoadds, takes_logger=True,
                         use_proxy=False)
        self.butler = None
        self.deep_coadd_list = None

    def getKwargs(self, config, base, logger):
        logger.debug("Get kwargs for DeepCoadds")
        req = {
            "repo": str,
            "collection": str,
        }
        opt = {
            "skymap_name": str,
            "dstype": str,
            "deep_coadd_list_file": str,
        }
        params, _ = GetAllParams(config, base, req=req, opt=opt)
        if self.butler is None:
            self.butler = daf_butler.Butler(params["repo"],
                                            collections=[params["collection"]])

        deep_coadd_list_file = params.get("deep_coadd_list_file", None)

        if self.deep_coadd_list is None and deep_coadd_list_file is not None:
            df = pd.read_parquet(deep_coadd_list_file)
            columns = ["band", "tract", "patch"]
            self.deep_coadd_list = [dict(zip(columns, row))
                                    for row in zip(*[df[_] for _ in columns])]

        kwargs = {
            "butler": self.butler,
            "skymap_name": params.get("skymap", "lsst_cells_v2"),
            "data_ids": self.deep_coadd_list,
            "dstype": params.get("dstype", "deep_coadd"),
        }
        safe = True
        return kwargs, safe


def DeepCoaddData(config, base, value_type):
    deep_coadds = GetInputObj('deep_coadds', config, base, 'DeepCoaddData')
    num_coadds = len(deep_coadds.data_ids)

    req = { 'field': str }
    params, safe = GetAllParams(config, base, req=req)
    field = params['field']
    if field == 'ncoadds':
        return num_coadds, safe

    coadd_num = base['coadd_num']
    assert (coadd_num >= 0 and coadd_num < num_coadds)
    data_id = deep_coadds.data_ids[coadd_num]

    val = value_type(data_id.get(field, None))

    return val, safe


RegisterInputType('deep_coadds', DeepCoaddLoader())
RegisterValueType('DeepCoaddData', DeepCoaddData, [int, str],
                  input_type='deep_coadds')
