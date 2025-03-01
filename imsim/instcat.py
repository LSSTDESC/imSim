
import os
import gzip
import numpy as np
import math
import astropy.units as u
import astropy.constants
from dust_extinction.parameter_averages import F19

from contextlib import contextmanager
from galsim.config import InputLoader, RegisterInputType, RegisterValueType, RegisterObjectType
from galsim.config import RegisterSEDType
from galsim import CelestialCoord
import galsim
import pickle
from .utils import RUBIN_AREA


def clarify_radec_limits(
    min_ra, max_ra, min_dec, max_dec, threshold=0.5*galsim.degrees
):
    """Handle RA wrapping and poles in ra/dec limits.

    Parameters
    ----------
    min_ra, max_ra, min_dec, max_dec: galsim.Angle
        RA and dec ranges.
    threshold: galsim.Angle
        The threshold for how close to the pole we are before we set the dec
        limit to the pole itself.

    Returns
    -------
    min_ra, max_ra, min_dec, max_dec: galsim.Angle
        The min and max values for RA and Dec.
    ref_ra: galsim.Angle
        The reference for wrapping RA.
    """

    # Handle wrap-around in RA:
    ref_ra = (min_ra + max_ra.wrap(min_ra))/2.
    min_ra = min_ra.wrap(ref_ra)
    max_ra = max_ra.wrap(ref_ra)

    # Special case if we're close to one of the poles.
    if max(np.abs([min_dec.deg, max_dec.deg])) > 90 - threshold.deg:
        if min_dec.deg < 0:
            min_dec = -91.0*galsim.degrees
        else:
            max_dec = 91.0*galsim.degrees
        min_ra = ref_ra - 181*galsim.degrees
        max_ra = ref_ra + 181*galsim.degrees
    return min_ra, max_ra, min_dec, max_dec, ref_ra


def get_radec_limits(
    wcs, xsize, ysize, logger, edge_pix, threshold=0.5*galsim.degrees
):
    """Min and max values for RA, Dec given the wcs.

    Parameters
    ----------
    wcs: galsim WCS object
        The WCS for the image.
    xsize, ysize: int
        The size of the image.
    logger: galsim.Logger
        A logger for logging debug statements.
    edge_pix: int
        The number of pixels to allow objects to be off the image.
    threshold: galsim.Angle
        The threshold for how close to the pole we are before we set the limit
        to the pole itself.

    Returns
    -------
    min_ra, max_ra, min_dec, max_dec: galsim.Angle
        The min and max values for RA and Dec.
    min_x, min_y, max_x, max_y: float
        The min and max values for x and y on the image.
    ref_ra: galsim.Angle
        The reference for wrapping RA.
    """
    # Allow objects to be centered somewhat off the image.
    min_x = 0 - edge_pix
    min_y = 0 - edge_pix
    max_x = xsize + edge_pix
    max_y = ysize + edge_pix

    # Check the min/max ra and dec to faster remove objects that
    # cannot be on image
    ll = galsim.PositionD(min_x,min_y)
    lr = galsim.PositionD(min_x,max_y)
    ul = galsim.PositionD(max_x,min_y)
    ur = galsim.PositionD(max_x,max_y)
    ll = wcs.toWorld(ll)
    lr = wcs.toWorld(lr)
    ul = wcs.toWorld(ul)
    ur = wcs.toWorld(ur)
    min_ra = min([ll.ra, lr.ra, ul.ra, ur.ra])
    max_ra = max([ll.ra, lr.ra, ul.ra, ur.ra])
    min_dec = min([ll.dec, lr.dec, ul.dec, ur.dec])
    max_dec = max([ll.dec, lr.dec, ul.dec, ur.dec])

    min_ra, max_ra, min_dec, max_dec, ref_ra = clarify_radec_limits(
        min_ra, max_ra, min_dec, max_dec, threshold
    )

    logger.debug("RA range for image is %f .. %f", min_ra.deg, max_ra.deg)
    logger.debug("Dec range for image is %f .. %f", min_dec.deg, max_dec.deg)
    return min_ra, max_ra, min_dec, max_dec, min_x, min_y, max_x, max_y, ref_ra


# Some helpers to read in a file that might be gzipped.
@contextmanager
def fopen(filename, **kwds):
    """
    Return a file descriptor-like object that closes the underlying
    file descriptor when used with the with-statement.

    Parameters
    ----------
    filename: str
        Filename of the instance catalog.
    **kwds: dict
        Keyword arguments to pass to the gzip.open or open functions.

    Returns
    -------
    generator: file descriptor-like generator object that can be iterated
        over to return the lines in a file.
    """
    abspath = os.path.split(os.path.abspath(filename))[0]
    if not os.path.isfile(filename):
        raise OSError("File not found: %s"%filename)
    try:
        if filename.endswith('.gz'):
            fd = gzip.open(filename, **kwds)
        else:
            fd = open(filename, **kwds)
        yield fopen_generator(fd, abspath, **kwds)
    finally:
        if fd: fd.close()


def fopen_generator(fd, abspath, **kwds):
    """
    Return a generator for the provided file descriptor that knows how
    to recursively read in instance catalogs specified by the
    includeobj directive.
    """
    with fd as input_:
        for line in input_:
            if not line.startswith('includeobj'):
                yield line
            else:
                filename = os.path.join(abspath, line.strip().split()[-1])
                with fopen(filename, **kwds) as my_input:
                    for line in my_input:
                        yield line

class InstCatalog(object):
    """This just handles the objects part of the instance catalog.

    The other "phosim commands" are handled by OpsimDataLoader.
    """
    # SED normalization for magnorm=0 at 500 nm to be applied to
    # cached SEDs.
    fnu = (0 * u.ABmag).to(u.erg/u.s/u.cm**2/u.Hz)
    _flux_density = fnu.to_value(u.ph/u.nm/u.s/u.cm**2, u.spectral_density(500*u.nm))
    def __init__(self, file_name, wcs, xsize=4096, ysize=4096, sed_dir=None,
                 edge_pix=100, sort_mag=True, flip_g2=True,
                 pupil_area=RUBIN_AREA, min_source=None, skip_invalid=True,
                 logger=None):
        logger = galsim.config.LoggerWrapper(logger)
        self.file_name = file_name
        self.wcs = wcs
        self.xsize = xsize
        self.ysize = ysize
        self.edge_pix = edge_pix
        self.sort_mag = sort_mag
        self.flip_g2 = flip_g2
        self.min_source = min_source
        self.skip_invalid = skip_invalid
        self.pupil_area = pupil_area
        self._sed_cache = {}

        if sed_dir is None:
            self.sed_dir = os.environ.get('SIMS_SED_LIBRARY_DIR')
        else:
            self.sed_dir = sed_dir
        self.inst_dir = os.path.dirname(file_name)

        self._id = None  # Sentinal that _load hasn't been run yet.

    @property
    def id(self):
        self._load()
        return self._id

    def _load(self, logger=None):
        if self._id is not None:
            return

        logger = galsim.config.LoggerWrapper(logger)
        min_ra, max_ra, min_dec, max_dec, min_x, min_y, max_x, max_y, ref_ra \
            = get_radec_limits(self.wcs, self.xsize, self.ysize, logger, self.edge_pix)

        # What position do the dust parameters start, based on object type.
        dust_index_dict = {
            'point' : 13,
            'sersic2d' : 17,
            'knots' : 17,
            'streak' : 16
        }
        default_dust_index = 15  # For fits files

        id_list = []
        world_pos_list = []
        image_pos_list = []
        magnorm_list = []
        sed_list = []
        lens_list = []
        objinfo_list = []
        dust_list = []
        g2_sign = -1 if self.flip_g2 else 1
        logger.warning('Reading instance catalog %s', self.file_name)
        nuse = 0
        ntot = 0

        with fopen(self.file_name, mode='rt') as _input:
            for line in _input:
                if ' inf ' in line: continue
                if line.startswith('object'):
                    ntot += 1
                    if ntot % 10000 == 0:
                        logger.info('Using %d of %d objects so far.',nuse,ntot)
                    # Check if the object is on our image
                    tokens = line.strip().split()
                    ra = float(tokens[2])*galsim.degrees
                    dec = float(tokens[3])*galsim.degrees
                    #logger.debug('object at %s,%s',ra,dec)
                    if not (
                        min_ra <= ra.wrap(ref_ra) <= max_ra
                        and min_dec <= dec <= max_dec
                    ):
                        continue
                    world_pos = galsim.CelestialCoord(ra, dec)
                    #logger.debug('world_pos = %s',world_pos)
                    try:
                        image_pos = self.wcs.toImage(world_pos)
                    except RuntimeError as e:
                        # Inverse direction can fail for objects off the image.
                        logger.debug('%s',e)
                        logger.debug('failed to determine image_pos')
                        continue
                    #logger.debug('image_pos = %s',image_pos)
                    if not (min_x <= image_pos.x <= max_x and min_y <= image_pos.y <= max_y):
                        continue
                    #logger.debug('on image')

                    # OK, probably keep this object.  Finish parsing it.
                    objid = tokens[1]
                    magnorm = float(tokens[4])
                    sed = ((tokens[5], float(tokens[6])))
                    # gamma1,gamma2,kappa
                    lens = (float(tokens[7]), g2_sign*float(tokens[8]), float(tokens[9]))
                    # Columns 10 and 11 are delta_ra, delta_dec, which
                    # are additional offsets in RA, Dec, optionally
                    # used by phosim.  imSim does not use these
                    # columns.
                    dust_index = dust_index_dict.get(tokens[12].lower(), default_dust_index)
                    objinfo = tokens[12:dust_index]
                    dust = tokens[dust_index:]

                    if self.skip_invalid:
                        # Check for some reasons to skip this object.
                        object_is_valid = (magnorm < 50.0 and
                                            not (objinfo[0] == 'sersic2d' and
                                                    float(objinfo[1]) < float(objinfo[2])) and
                                            not (objinfo[0] == 'knots' and
                                                    (float(objinfo[1]) < float(objinfo[2]) or
                                                     int(objinfo[4]) <= 0)))
                        if not object_is_valid:
                            logger.debug("Skipping object %s since not valid.", tokens[1])
                            continue

                    # Object is ok.  Add it to lists.
                    nuse += 1
                    id_list.append(objid)
                    world_pos_list.append(world_pos)
                    image_pos_list.append(image_pos)
                    magnorm_list.append(magnorm)
                    sed_list.append(sed)
                    lens_list.append(lens)
                    objinfo_list.append(objinfo)
                    dust_list.append(dust)

        assert nuse == len(id_list)
        logger.warning("Total objects in file = %d",ntot)
        logger.warning("Found %d objects potentially on image", nuse)
        if nuse == 0:
            logger.warning("No objects found on image")

        # Sort the object lists by mag and convert to numpy arrays.
        self._id = np.array(id_list, dtype=str)
        self.world_pos = np.array(world_pos_list, dtype=object)
        self.image_pos = np.array(image_pos_list, dtype=object)
        self.magnorm = np.array(magnorm_list, dtype=float)
        self.sed = np.array(sed_list, dtype=object)
        self.lens = np.array(lens_list, dtype=object)
        self.objinfo = np.array(objinfo_list, dtype=object)
        self.dust = np.array(dust_list, dtype=object)

        if self.min_source is not None:
            nsersic = np.sum([params[0].lower() == 'sersic2d' for params in self.objinfo])
            if nsersic < self.min_source:
                logger.warning(f"Fewer than {self.min_source} galaxies on sensor.  Skipping.")
                self._id = self._id[:0]
                self.world_pos = self.world_pos[:0]
                self.image_pos = self.image_pos[:0]
                self.magnorm = self.magnorm[:0]
                self.sed = self.sed[:0]
                self.lens = self.lens[:0]
                self.objinfo = self.objinfo[:0]
                self.dust = self.dust[:0]

        if self.sort_mag:
            index = np.argsort(self.magnorm)
            self._id = self._id[index]
            self.world_pos = self.world_pos[index]
            self.image_pos = self.image_pos[index]
            self.magnorm = self.magnorm[index]
            self.sed = self.sed[index]
            self.lens = self.lens[index]
            self.objinfo = self.objinfo[index]
            self.dust = self.dust[index]
            logger.warning("Sorted objects by magnitude (brightest first).")

    def getNObjects(self, logger=None):
        self._load(logger)
        # Note: This method name is required by the config parser.
        return len(self._id)

    def getApproxNObjects(self, logger=None):
        if self._id is None:
            # If we haven't read the file yet, just (over-)estimate the number by
            # quickly counting the lines in the file without doing any processing.
            with fopen(self.file_name, mode='rt') as _input:
                # generators don't implement len(); this is the tricky workaround that doesn't
                # store all the data in memory (like len(list(_input))).
                return sum(1 for _ in _input)
        else:
            return self.getNObjects(logger)

    @property
    def nobjects(self):
        return self.getNObjects()

    # Note: Proxies can call methods, but not access attributes, so it's helpful to have
    #       method versions of things like this, which are maybe more obvious using the
    #       attribute directly.  Since input objects such as this are used via proxy in
    #       multiprocessing contexts, we need to keep the method version around.
    def getID(self, index):
        self._load()
        return self._id[index]

    def getWorldPos(self, index):
        self._load()
        return self.world_pos[index]

    def getImagePos(self, index):
        self._load()
        return self.image_pos[index]

    def getMagNorm(self, index):
        self._load()
        return self.magnorm[index]

    def getSED(self, index):
        self._load()
        # These require reading in an input file.  So cache the raw (unredshifted versions)
        # to try to minimize how much I/O we'll need for these.
        name, redshift = self.sed[index]
        if name in self._sed_cache:
            sed = self._sed_cache[name]
        else:
            # Try both the given sed_dir and the directory of the instance catalog itself.
            full_name = os.path.join(self.sed_dir, name)
            if not os.path.isfile(full_name):
                full_name = os.path.join(self.inst_dir, name)
            if not os.path.isfile(full_name):
                raise OSError("Could not find file %s in either %s or %s"%(
                              name, self.sed_dir, self.inst_dir))
            sed = galsim.SED(full_name, wave_type='nm', flux_type='flambda')

            # Normalize to magnorm=0 at 500 nm.
            sed = sed.withFluxDensity(self._flux_density, 500.*u.nm)

            self._sed_cache[name] = sed

        iAv, iRv, mwAv, mwRv = self.getDust(index)

        # TODO: apply internal extinction here

        # Apply redshift.
        sed = sed.atRedshift(redshift)

        # Apply Milky Way extinction.
        extinction = F19(Rv=mwRv)
        # Use SED wavelengths
        wl = sed.wave_list
        # Restrict to the range where F19 can be evaluated.  F19.x_range
        # is in units of 1/micron so convert to nm.
        wl_min = 1e3/F19.x_range[1]
        wl_max = 1e3/F19.x_range[0]
        wl = wl[np.where((wl_min < wl) & (wl < wl_max))]
        ext = extinction.extinguish(wl*u.nm, Av=mwAv)
        spec = galsim.LookupTable(wl, ext, interpolant='linear')
        mw_ext = galsim.SED(spec, wave_type='nm', flux_type='1')
        sed = sed*mw_ext

        # Not sure why GalSim isn't preserving the LookupTable here.
        # Should fix this is GalSim, but for now, make sure we have a LookupTable, so the
        # sed is fast to integrate (and is pickleable!).
        if (not isinstance(sed._spec, galsim.LookupTable)
            or sed._spec.interpolant != 'linear'):
            new_spec = galsim.LookupTable(wl, sed(wl), interpolant='linear')
            sed = galsim.SED(new_spec, 'nm', 'fphotons')

        return sed

    def getLens(self, index):
        self._load()
        # The galsim.lens(...) function wants to be passed reduced
        # shears and magnification, so convert the WL parameters as
        # defined in phosim instance catalogs to these values.  See
        # https://github.com/GalSim-developers/GalSim/blob/releases/1.4/doc/GalSim_Quick_Reference.pdf
        # and Hoekstra, 2013, http://lanl.arxiv.org/abs/1312.5981
        gamma1, gamma2, kappa = self.lens[index]
        g1 = gamma1/(1. - kappa)   # real part of reduced shear
        g2 = gamma2/(1. - kappa)   # imaginary part of reduced shear
        mu = 1./((1. - kappa)**2 - (gamma1**2 + gamma2**2)) # magnification
        return g1,g2,mu

    def getDust(self, index):
        self._load()
        params = self.dust[index]
        if params[0].lower() != 'none':
            internal_av = float(params[1])
            internal_rv = float(params[2])
            params = params[3:]
        else:
            internal_av = 0.
            internal_rv = 3.1
            params = params[1:]

        if params[0].lower() != 'none':
            galactic_av = float(params[1])
            galactic_rv = float(params[2])
        else:
            galactic_av = 0.
            galactic_rv = 3.1

        return internal_av, internal_rv, galactic_av, galactic_rv

    def getObj(self, index, gsparams=None, rng=None, exptime=30, logger=None):
        self._load(logger)
        if self.objinfo.size == 0:
            raise RuntimeError("Trying to get an object from an empty instance catalog")
        params = self.objinfo[index]

        magnorm = self.getMagNorm(index)
        if magnorm >= 50:
            # Mark of invalid object apparently
            return None

        if gsparams is not None:
            gsparams = galsim.GSParams(**gsparams)

        # Make the object according to the values in the objinfo

        # Note: params here starts at 12, so all indices are 12 less than in previous code.
        if params[0].lower() == 'point':
            obj = galsim.DeltaFunction(gsparams=gsparams)

        elif params[0].lower() == 'streak':
            # Note that satellite streaks are not sources at infinity
            # and so would be out-of-focus compared to stars or
            # galaxies and would therefore have a different PSF.  This
            # implementation does not account for that difference.
            length = float(params[1])
            width = float(params[2])
            obj = galsim.Box(length, width, gsparams=gsparams)
            position_angle = galsim.Angle(float(params[3]), galsim.degrees)
            obj = obj.rotate(position_angle)

        elif params[0].lower() == 'sersic2d':
            a = float(params[1])
            b = float(params[2])
            assert a >= b  # Enforced above.
            pa = float(params[3])
            if self.flip_g2:
                # Previous code first did PA = 360 - params[3]
                # Then beta = 90 + PA
                beta = float(90 - pa) * galsim.degrees
            else:
                beta = float(90 + pa) * galsim.degrees

            n = float(params[4])
            # GalSim can amortize some calculations for Sersics, but only if n is the same
            # as a previous galaxy.  So quantize the n values at 0.05.  There's no way anyone
            # cares about this at higher resolution than that.
            # For now, this is not actually helpful, since n is always either 1 or 4, but if
            # we ever start having more variable n, this will prevent it from redoing Hankel
            # integrals for every galaxy.
            n = round(n * 20.) / 20.

            hlr = (a * b)**0.5  # geometric mean of a and b is close to right.
            # XXX: Note: Previous code had hlr = a, which is wrong. (?)  Galaxies were too large.
            #      Especially when they were more elliptical.  Oops.
            # TODO: Maybe not?  Check if this should be a.
            obj = galsim.Sersic(n=n, half_light_radius=hlr, gsparams=gsparams)
            shear = galsim.Shear(q=b/a, beta=beta)
            obj = obj._shear(shear)
            g1,g2,mu = self.getLens(index)
            obj = obj._lens(g1, g2, mu)

        elif params[0].lower() == 'knots':
            a = float(params[1])
            b = float(params[2])
            assert a >= b
            pa = float(params[3])
            if self.flip_g2:
                beta = float(90 - pa) * galsim.degrees
            else:
                beta = float(90 + pa) * galsim.degrees
            npoints = int(params[4])
            assert npoints > 0
            hlr = (a * b)**0.5
            obj = galsim.RandomKnots(npoints=npoints, half_light_radius=hlr, rng=rng,
                                     gsparams=gsparams)
            shear = galsim.Shear(q=b/a, beta=beta)
            obj = obj._shear(shear)
            # TODO: These look bad in space images (cf. Troxel's talks about Roman sims.)
            #       Should convolve this by a smallish Gaussian *here*:
            #       I'd guess 0.3 arcsec is a good choice for the fwhm of this Gaussian.
            # obj = galsim.Convolve(obj, galsim.Gaussian(fwhm=0.3))
            g1,g2,mu = self.getLens(index)
            obj = obj._lens(g1, g2, mu)

        elif (params[0].endswith('.fits') or params[0].endswith('.fits.gz')):
            # Assume the fits file is given relative to the location of the instance catalog.
            fits_file = os.path.join(self.inst_dir, params[0])
            pixel_scale = float(params[1])
            theta = float(params[2])
            obj = galsim.InterpolatedImage(fits_file, scale=pixel_scale, gsparams=gsparams)
            if theta != 0.:
                obj = obj.rotate(-theta * galsim.degrees)
            g1,g2,mu = self.getLens(index)
            obj = obj._lens(g1, g2, mu)

        else:
            raise RuntimeError("Do not know how to handle object type: %s" % params[0])

        # The seds are normalized to correspond to magnorm=0.
        # The flux for the given magnorm is 10**(-0.4*magnorm)
        # The constant here, 0.9210340371976184 = 0.4 * log(10)
        flux = math.exp(-0.9210340371976184 * magnorm)

        # This gives the normalization in photons/cm^2/sec.
        # Multiply by area and exptime to get photons.
        fAt = flux * self.pupil_area * exptime

        sed = self.getSED(index)
        return obj.withFlux(fAt) * sed


def InstCatObj(config, base, ignore, gsparams, logger):
    """Build an object according to info in instance catalog.
    """
    inst = galsim.config.GetInputObj('instance_catalog', config, base, 'InstCat')

    # Setup the indexing sequence if it hasn't been specified.
    # The normal thing with a catalog is to just use each object in order,
    # so we don't require the user to specify that by hand.  We can do it for them.
    galsim.config.SetDefaultIndex(config, inst.getNObjects(logger))

    req = { 'index' : int }
    opt = { 'num' : int }
    kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
    index = kwargs['index']

    rng = galsim.config.GetRNG(config, base, logger, 'InstCatObj')
    exptime = base.get('exptime', 30)

    obj = inst.getObj(index, gsparams=gsparams, rng=rng, exptime=exptime, logger=logger)
    base['object_id'] = inst.getID(index)

    return obj, safe

def InstCatWorldPos(config, base, value_type):
    """Return a value from the object part of the instance catalog
    """
    inst = galsim.config.GetInputObj('instance_catalog', config, base, 'InstCatWorldPos')

    # Setup the indexing sequence if it hasn't been specified.
    # The normal thing with a catalog is to just use each object in order,
    # so we don't require the user to specify that by hand.  We can do it for them.
    galsim.config.SetDefaultIndex(config, inst.getNObjects())

    req = { 'index' : int }
    opt = { 'num' : int }
    kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
    index = kwargs['index']

    pos = inst.getWorldPos(index)
    return pos, safe

class InstCatSEDBuilder(galsim.config.SEDBuilder):
    """A class for loading an SED from the instance catalog.
    """
    def buildSED(self, config, base, logger):
        """Build the SED based on the specifications in the config dict.
        Parameters:
            config:     The configuration dict for the SED type.
            base:       The base configuration dict.
            logger:     If provided, a logger for logging debug statements.
        Returns:
            the constructed SED object.
        """
        inst = galsim.config.GetInputObj('instance_catalog', config, base, 'InstCatWorldPos')

        galsim.config.SetDefaultIndex(config, inst.getNObjects(logger))

        req = { 'index' : int }
        opt = { 'num' : int }
        kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
        index = kwargs['index']
        sed = inst.getSED(index)
        return sed, safe

# The basic InputLoader almost works.  Just need to handle the wcs.
class InstCatalogLoader(InputLoader):
    def getKwargs(self, config, base, logger):
        import galsim
        req = {
                'file_name' : str,
              }
        opt = {
                'sed_dir' : str,
                'edge_pix' : float,
                'sort_mag' : bool,
                'flip_g2' : bool,
                'pupil_area' : float,
                'min_source' : int,
                'skip_invalid' : bool,
              }
        kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
        wcs = galsim.config.BuildWCS(base['image'], 'wcs', base, logger=logger)
        kwargs['wcs'] = wcs
        kwargs['xsize'] = base.get('det_xsize', 4096)
        kwargs['ysize'] = base.get('det_ysize', 4096)
        kwargs['logger'] = logger
        return kwargs, False

RegisterInputType('instance_catalog', InstCatalogLoader(InstCatalog, has_nobj=True))
RegisterValueType('InstCatWorldPos', InstCatWorldPos, [CelestialCoord],
                  input_type='instance_catalog')
RegisterObjectType('InstCatObj', InstCatObj, input_type='instance_catalog')
RegisterSEDType('InstCatSED', InstCatSEDBuilder(), input_type='instance_catalog')
