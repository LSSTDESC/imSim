
import os
import gzip
import numpy as np
import math
import sqlite3
import astropy
import astropy.coordinates

from contextlib import contextmanager
from galsim.config import InputLoader, RegisterInputType, RegisterValueType, RegisterObjectType
from galsim.config import RegisterSEDType, RegisterBandpassType
from galsim import CelestialCoord
import galsim

from .meta_data import data_dir

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

    The other "phosim commands" are handled by OpsimMetaDict.
    """
    _bp500 = galsim.Bandpass(galsim.LookupTable([499,500,501],[0,1,0]),
                             wave_type='nm').withZeropoint('AB')

    # Using area-weighted effective aperture over FOV
    # from https://confluence.lsstcorp.org/display/LKB/LSST+Key+Numbers
    _rubin_area = 0.25 * np.pi * 649**2  # cm^2

    def __init__(self, file_name, wcs, sed_dir=None, edge_pix=100, sort_mag=True, flip_g2=True,
                 min_source=None, skip_invalid=True, logger=None):
        logger = galsim.config.LoggerWrapper(logger)
        self.file_name = file_name
        self.flip_g2 = flip_g2
        self._sed_cache = {}

        if sed_dir is None:
            self.sed_dir = os.environ.get('SIMS_SED_LIBRARY_DIR')
        else:
            self.sed_dir = sed_dir
        self.inst_dir = os.path.dirname(file_name)

        # Allow objects to be centered somewhat off the image.
        min_x = 0 - edge_pix
        min_y = 0 - edge_pix
        max_x = 4096 + edge_pix  # The image max_x,max_y isn't actually 4096, but close enough.
        max_y = 4096 + edge_pix

        # Check the min/max ra and dec to faster remove objects that cannot be on image
        ll = galsim.PositionD(min_x,min_y)
        lr = galsim.PositionD(min_x,max_y)
        ul = galsim.PositionD(max_x,min_y)
        ur = galsim.PositionD(max_x,max_y)
        ll = wcs.toWorld(ll)
        lr = wcs.toWorld(lr)
        ul = wcs.toWorld(ul)
        ur = wcs.toWorld(ur)
        min_ra = min([ll.ra.deg, lr.ra.deg, ul.ra.deg, ur.ra.deg])
        max_ra = max([ll.ra.deg, lr.ra.deg, ul.ra.deg, ur.ra.deg])
        min_dec = min([ll.dec.deg, lr.dec.deg, ul.dec.deg, ur.dec.deg])
        max_dec = max([ll.dec.deg, lr.dec.deg, ul.dec.deg, ur.dec.deg])
        logger.debug("RA range for image is %f .. %f", min_ra, max_ra)
        logger.debug("Dec range for image is %f .. %f", min_dec, max_dec)

        # What position do the dust parameters start, based on object type.
        dust_index_dict = {
            'point' : 13,
            'sersic2d' : 17,
            'knots' : 17,
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
        g2_sign = -1 if flip_g2 else 1
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
                    ra = float(tokens[2])
                    dec = float(tokens[3])
                    #logger.debug('object at %s,%s',ra,dec)
                    if not (min_ra <= ra <= max_ra and min_dec <= dec <= max_dec):
                        continue
                    world_pos = galsim.CelestialCoord(ra * galsim.degrees, dec * galsim.degrees)
                    #logger.debug('world_pos = %s',world_pos)
                    try:
                        image_pos = wcs.toImage(world_pos)
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
                    # what are 10,11?
                    dust_index = dust_index_dict.get(tokens[12], default_dust_index)
                    objinfo = tokens[12:dust_index]
                    dust = tokens[dust_index:]

                    if skip_invalid:
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

        # Sort the object lists by mag and convert to numpy arrays.
        self.id = np.array(id_list, dtype=str)
        self.world_pos = np.array(world_pos_list, dtype=object)
        self.image_pos = np.array(image_pos_list, dtype=object)
        self.magnorm = np.array(magnorm_list, dtype=float)
        self.sed = np.array(sed_list, dtype=object)
        self.lens = np.array(lens_list, dtype=object)
        self.objinfo = np.array(objinfo_list, dtype=object)
        self.dust = np.array(dust_list, dtype=object)

        if min_source is not None:
            nsersic = np.sum([params[0].lower() == 'sersic2d' for params in self.objinfo])
            if nsersic < min_source:
                logger.warning(f"Fewer than {min_source} galaxies on sensor.  Skipping.")
                self.id = self.id[:0]
                self.world_pos = self.world_pos[:0]
                self.image_pos = self.image_pos[:0]
                self.magnorm = self.magnorm[:0]
                self.sed = self.sed[:0]
                self.lens = self.lens[:0]
                self.objinfo = self.objinfo[:0]
                self.dust = self.dust[:0]

        if sort_mag:
            index = np.argsort(self.magnorm)
            self.id = self.id[index]
            self.world_pos = self.world_pos[index]
            self.image_pos = self.image_pos[index]
            self.magnorm = self.magnorm[index]
            self.sed = self.sed[index]
            self.lens = self.lens[index]
            self.objinfo = self.objinfo[index]
            self.dust = self.dust[index]
            logger.warning("Sorted objects by magnitude (brightest first).")

    def getNObjects(self, logger=None):
        # Note: This method name is required by the config parser.
        return len(self.id)

    @property
    def nobjects(self):
        return self.getNObjects()

    # Note: Proxies can call methods, but not access attributes, so it's helpful to have
    #       method versions of things like this, which are maybe more obvious using the
    #       attribute directly.  Since input objects such as this are used via proxy in
    #       multiprocessing contexts, we need to keep the method version around.
    def getID(self, index):
        return self.id[index]

    def getWorldPos(self, index):
        return self.world_pos[index]

    def getImagePos(self, index):
        return self.image_pos[index]

    def getMagNorm(self, index):
        return self.magnorm[index]

    def getSED(self, index):
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
            sed = sed.withMagnitude(0, self._bp500)  # Normalize to mag 0
            self._sed_cache[name] = sed

        # TODO: Handle the dust effects.  internal, then redshift, then galactic.
        #       Also probably need to think about the normalization and how that interacts with
        #       the flux and/or mag.
        return sed.atRedshift(redshift)

    def getLens(self, index):
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
        params = dust[index]
        if params[0].lower() != 'none':
            internal_av = float(params[1])
            internal_rv = float(params[2])
            params = params[3:]
        else:
            internal_av = 0.
            internal_rv = 1.
            params = params[1:]

        if params[0].lower() != 'none':
            galactic_av = float(params[1])
            galactic_rv = float(params[2])
        else:
            galactic_av = 0.
            galactic_rv = 1.

        return internal_av, internal_rv, galactic_av, galactic_rv

    def getObj(self, index, gsparams=None, rng=None, bandpass=None, chromatic=False, exp_time=30):
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
            assert npoint > 0
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
            fits_file = find_file_path(params[0], get_image_dirs())
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
        fAt = flux * self._rubin_area * exp_time

        sed = self.getSED(index)
        if chromatic:
            return obj.withFlux(fAt) * sed
        else:
            flux = sed.calculateFlux(bandpass) * fAt
            return obj.withFlux(flux)

    def getHourAngle(self, mjd, ra):
        """
        Compute the local hour angle of an object for the specified
        MJD and RA.

        Parameters
        ----------
        mjd: float
            Modified Julian Date of the observation.
        ra: float
            Right Ascension (in degrees) of the object.

        Returns
        -------
        float: hour angle in degrees
        """
        # cf. http://www.ctio.noao.edu/noao/content/coordinates-observatories-cerro-tololo-and-cerro-pachon
        lsst_lat = '-30d 14m 40.68s'
        lsst_long = '-70d 44m 57.90s'
        lsst_elev = '2647m'
        lsst_loc = astropy.coordinates.EarthLocation.from_geodetic(
                        lsst_lat, lsst_long, lsst_elev)

        time = astropy.time.Time(mjd, format='mjd', location=lsst_loc)
        # Get the local apparent sidereal time.
        last = time.sidereal_time('apparent').degree
        ha = last - ra
        return ha


def _is_sqlite3_file(filename):
    """Check if a file is an sqlite3 db file."""
    with open(filename, 'rb') as fd:
        return fd.read(100).startswith(b'SQLite format 3')


class OpsimMetaDict(object):
    """Read the exposure information from the opsim db file.

    The objects are handled by InstCatalog.
    """
    _req_params = {'file_name' : str}
    _opt_params = {'visit' : int,
                   'snap' : int}
    _single_params = []
    _takes_rng = False

    _required_commands = set("""rightascension
                                declination
                                mjd
                                altitude
                                azimuth
                                filter
                                rotskypos
                                rottelpos
                                dist2moon
                                moonalt
                                moondec
                                moonphase
                                moonra
                                nsnap
                                obshistid
                                seed
                                seeing
                                sunalt
                                vistime""".split())

    def __init__(self, file_name, visit=None, snap=0, logger=None):
        self.logger = galsim.config.LoggerWrapper(logger)
        self.file_name = file_name
        self.visit = visit
        self.meta = {'snap' : snap}

        if _is_sqlite3_file(self.file_name):
            self._read_opsim_db()
        else:
            self._read_instcat_header()

        # Set some default values if these aren't present in input file.
        self.meta['gain'] = self.meta.get('gain', 1)
        self.meta['exptime'] = self.meta.get('exptime', 30)
        self.meta['readnoise'] = self.meta.get('readnoise', 0)
        self.meta['darkcurrent'] = self.meta.get('darkcurrent', 0)

    def _read_opsim_db(self):
        """Read visit info from an opsim db file."""
        # Query for the info for the desired visit.
        if self.visit is None:
            raise ValueError('The visit must be set when reading visit info from an opsim db file.')

        self.logger.warning('Reading info from opsim db file %s for visit %s',
                            self.file_name, self.visit)
        with sqlite3.connect(self.file_name) as con:
            table = 'observations'
            # Read the column names for the observations table
            sql = f"select name from pragma_table_info('{table}')"
            columns = [_[0] for _ in con.execute(sql)]
            # Fill the self.meta dict
            sql = f"""select {','.join(columns)} from {table} where
                      observationId={self.visit}"""
            for key, value in zip(columns, list(con.execute(sql))[0]):
                self.meta[key] = value
        self.logger.warning('Done reading visit info from opsim db file')

        if self.meta['snap'] >= self.meta['numExposures']:
            raise ValueError('Invalid snap value: %d. For this visit, snap < %d'
                             % (self.meta['snap'], self.meta['numExposures']))

        if any(key not in self.meta for key in self._required_commands):
            raise ValueError("Some required commands are missing. Required commands: {}".format(
                             str(self._required_commands)))

        # Add a few derived quantities to meta values
        # Note a semantic distinction we make here:
        # "filter" or "band" is a character u,g,r,i,z,y.
        # "bandpass" will be the real constructed galsim.Bandpass object.
        self.meta['band'] = self.meta['filter']
        self.meta['exptime'] = self.meta['visitExposureTime']/self.meta['numExposures']
        readout_time \
            = (self.meta['visitTime'] - self.meta['visitExposureTime'])/self.meta['numExposures']
        # Set "mjd" to be the midpoint of the exposure
        self.meta['mjd'] = (self.meta['observationStartMJD']
                            + (self.meta['snap']*(self.meta['exptime'] + readout_time)
                               + self.meta['exptime']/2)/24./3600.)
        self.meta['HA'] = self.getHourAngle(self.meta['mjd'], self.meta['fieldRA'])
        # Following instance catalog convention, use the visit as the
        # seed.  TODO: Figure out how to make this depend on the snap
        # as well as the visit.
        self.meta['seed'] = self.meta['observationId']
        # For use by the AtmosphericPSF, "rawSeeing" should be set to the
        # atmosphere-only PSF FWHM at 500nm at zenith.  Based on
        # https://smtn-002.lsst.io/, this should be the
        # "seeingFwhm500" column.
        self.meta['rawSeeing'] = self.meta['seeingFwhm500']
        self.meta['FWHMeff'] = self.FWHMeff()
        self.meta['FWHMgeom'] = self.FWHMgeom()
        self.logger.debug("Bandpass = %s",self.meta['band'])
        self.logger.debug("HA = %s",self.meta['HA'])

    def _read_instcat_header(self):
        """Read visit info from the instance catalog header."""
        self.logger.warning('Reading visit info from instance catalog %s',
                            self.file_name)
        with fopen(self.file_name, mode='rt') as _input:
            for line in _input:
                if line.startswith('#'):  # comments
                    continue
                if line.startswith('object'):
                    # Assumes objects are all at the end.  Is this necessarily true?
                    break

                key, value = line.split()
                self.logger.debug('meta value: %s = %s',key,value)
                value = float(value)
                if int(value) == value:
                    self.meta[key] = int(value)
                else:
                    self.meta[key] = float(value)

        self.logger.warning("Done reading meta information from instance catalog")

        # Add a few derived quantities to meta values
        # Note a semantic distinction we make here:
        # "filter" is the number 0,1,2,3,4,5 from the input instance catalog.
        # "band" is the character u,g,r,i,z,y.
        # "bandpass" will be the real constructed galsim.Bandpass object.
        self.meta['band'] = 'ugrizy'[self.meta['filter']]
        self.meta['HA'] = self.getHourAngle(self.meta['mjd'], self.meta['rightascension'])
        self.meta['rawSeeing'] = self.meta.pop('seeing')  # less ambiguous name
        self.meta['airmass'] = self.getAirmass()
        self.meta['FWHMeff'] = self.FWHMeff()
        self.meta['FWHMgeom'] = self.FWHMgeom()
        self.logger.debug("Bandpass = %s",self.meta['band'])
        self.logger.debug("HA = %s",self.meta['HA'])

        # Use the opsim db names for these quantities.
        self.meta['fieldRA'] = self.meta['rightascension']
        self.meta['fieldDec'] = self.meta['declination']
        self.meta['rotTelPos'] = self.meta['rottelpos']
        self.meta['observationId'] = self.meta['obshistid']

    def getAirmass(self, altitude=None):
        """
        Function to compute the airmass from altitude using equation 3
        of Krisciunas and Schaefer 1991.
        Parameters
        ----------
        altitude: float
            Altitude of pointing direction in degrees.
            [default: self.get('altitude')]
        Returns
        -------
        float: the airmass in units of sea-level airmass at the zenith.
        """
        if altitude is None:
            altitude = self.get('altitude')
        altRad = np.radians(altitude)
        return 1.0/np.sqrt(1.0 - 0.96*(np.sin(0.5*np.pi - altRad))**2)

    def FWHMeff(self, rawSeeing=None, band=None, altitude=None):
        """
        Compute the effective FWHM for a single Gaussian describing the PSF.
        Parameters
        ----------
        rawSeeing: float
            The "ideal" seeing in arcsec at zenith and at 500 nm.
            reference: LSST Document-20160
            [default: self.get('rawSeeing')]
        band: str
            The LSST ugrizy band.
            [default: self.get('band')]
        altitude: float
            The altitude in degrees of the pointing.
            [default: self.get('altitude')]
        Returns
        -------
        float: Effective FWHM in arcsec.
        """
        X = self.getAirmass(altitude)

        if band is None:
            band = self.get('band')
        if rawSeeing is None:
            rawSeeing = self.get('rawSeeing')

        # Find the effective wavelength for the band.
        wl = dict(u=365.49, g=480.03, r=622.20, i=754.06, z=868.21, y=991.66)[band]

        # Compute the atmospheric contribution.
        FWHMatm = rawSeeing*(wl/500)**(-0.3)*X**(0.6)

        # The worst case instrument contribution (see LSE-30).
        FWHMsys = 0.4*X**(0.6)

        # From LSST Document-20160, p. 8.
        return 1.16*np.sqrt(FWHMsys**2 + 1.04*FWHMatm**2)

    def FWHMgeom(self, rawSeeing=None, band=None, altitude=None):
        """
        FWHM of the "combined PSF".  This is FWHMtot from
        LSST Document-20160, p. 8.
        Parameters
        ----------
        rawSeeing: float
            The "ideal" seeing in arcsec at zenith and at 500 nm.
            reference: LSST Document-20160
            [default: self.get('rawSeeing')]
        band: str
            The LSST ugrizy band.
            [default: self.get('band')]
        altitude: float
            The altitude in degrees of the pointing.
            [default: self.get('altitude')]
        Returns
        -------
        float: FWHM of the combined PSF in arcsec.
        """
        return 0.822*self.FWHMeff(rawSeeing, band, altitude) + 0.052

    @classmethod
    def from_dict(cls, d):
        """Build an OpsimMetaDict directly from the provided dict.

        (Mostly used for unit tests.)
        """
        ret = cls.__new__(cls)
        ret.file_name = ''
        ret.meta = d
        return ret

    def getAirmass(self, altitude=None):
        """
        Function to compute the airmass from altitude using equation 3
        of Krisciunas and Schaefer 1991.

        Parameters
        ----------
        altitude: float
            Altitude of pointing direction in degrees.
            [default: self.get('altitude')]

        Returns
        -------
        float: the airmass in units of sea-level airmass at the zenith.
        """
        if altitude is None:
            altitude = self.get('altitude')
        altRad = np.radians(altitude)
        return 1.0/np.sqrt(1.0 - 0.96*(np.sin(0.5*np.pi - altRad))**2)

    def FWHMeff(self, rawSeeing=None, band=None, altitude=None):
        """
        Compute the effective FWHM for a single Gaussian describing the PSF.

        Parameters
        ----------
        rawSeeing: float
            The "ideal" seeing in arcsec at zenith and at 500 nm.
            reference: LSST Document-20160
            [default: self.get('rawSeeing')]
        band: str
            The LSST ugrizy band.
            [default: self.get('band')]
        altitude: float
            The altitude in degrees of the pointing.
            [default: self.get('altitude')]

        Returns
        -------
        float: Effective FWHM in arcsec.
        """
        X = self.getAirmass(altitude)

        if band is None:
            band = self.get('band')
        if rawSeeing is None:
            rawSeeing = self.get('rawSeeing')

        # Find the effective wavelength for the band.
        wl = dict(u=365.49, g=480.03, r=622.20, i=754.06, z=868.21, y=991.66)[band]

        # Compute the atmospheric contribution.
        FWHMatm = rawSeeing*(wl/500)**(-0.3)*X**(0.6)

        # The worst case instrument contribution (see LSE-30).
        FWHMsys = 0.4*X**(0.6)

        # From LSST Document-20160, p. 8.
        return 1.16*np.sqrt(FWHMsys**2 + 1.04*FWHMatm**2)


    def FWHMgeom(self, rawSeeing=None, band=None, altitude=None):
        """
        FWHM of the "combined PSF".  This is FWHMtot from
        LSST Document-20160, p. 8.

        Parameters
        ----------
        rawSeeing: float
            The "ideal" seeing in arcsec at zenith and at 500 nm.
            reference: LSST Document-20160
            [default: self.get('rawSeeing')]
        band: str
            The LSST ugrizy band.
            [default: self.get('band')]
        altitude: float
            The altitude in degrees of the pointing.
            [default: self.get('altitude')]

        Returns
        -------
        float: FWHM of the combined PSF in arcsec.
        """
        return 0.822*self.FWHMeff(rawSeeing, band, altitude) + 0.052

    def __getitem__(self, field):
        return self.get(field)

    def get(self, field):
        if field not in self.meta:
            raise ValueError("OpsimMeta field %s not present in instance catalog"%field)
        return self.meta[field]

    def getHourAngle(self, mjd, ra):
        """
        Compute the local hour angle of an object for the specified
        MJD and RA.

        Parameters
        ----------
        mjd: float
            Modified Julian Date of the observation.
        ra: float
            Right Ascension (in degrees) of the object.

        Returns
        -------
        float: hour angle in degrees
        """
        # cf. http://www.ctio.noao.edu/noao/content/coordinates-observatories-cerro-tololo-and-cerro-pachon
        lsst_lat = '-30d 14m 40.68s'
        lsst_long = '-70d 44m 57.90s'
        lsst_elev = '2647m'
        lsst_loc = astropy.coordinates.EarthLocation.from_geodetic(
                        lsst_lat, lsst_long, lsst_elev)

        time = astropy.time.Time(mjd, format='mjd', location=lsst_loc)
        # Get the local apparent sidereal time.
        last = time.sidereal_time('apparent').degree
        ha = last - ra
        return ha


def OpsimMeta(config, base, value_type):
    """Return one of the meta values stored in the instance catalog.
    """
    meta = galsim.config.GetInputObj('opsim_meta_dict', config, base, 'OpsimMeta')

    req = { 'field' : str }
    opt = { 'num' : int }  # num, if present, was used by GetInputObj
    kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
    field = kwargs['field']

    val = value_type(meta.get(field))
    return val, safe

def InstCatObj(config, base, ignore, gsparams, logger):
    """Build an object according to info in instance catalog.
    """
    inst = galsim.config.GetInputObj('instance_catalog', config, base, 'InstCat')

    # Setup the indexing sequence if it hasn't been specified.
    # The normal thing with a catalog is to just use each object in order,
    # so we don't require the user to specify that by hand.  We can do it for them.
    galsim.config.SetDefaultIndex(config, inst.getNObjects())

    req = { 'index' : int }
    opt = { 'num' : int }
    kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
    index = kwargs['index']

    rng = galsim.config.GetRNG(config, base, logger, 'InstCatObj')
    bp = base['bandpass']
    exp_time = base.get('exp_time',None)

    obj = inst.getObj(index, gsparams=gsparams, rng=rng, bandpass=bp, exp_time=exp_time)
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

        galsim.config.SetDefaultIndex(config, inst.getNObjects())

        req = { 'index' : int }
        opt = { 'num' : int }
        kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
        index = kwargs['index']
        sed = inst.getSED(index)
        return sed, safe

class OpsimMetaBandpass(galsim.config.BandpassBuilder):
    """A class for loading a Bandpass for a given instcat
    """
    def buildBandpass(self, config, base, logger):
        """Build the Bandpass object based on the LSST filter name.

        Parameters:
            config:     The configuration dict for the bandpass type.
            base:       The base configuration dict.
            logger:     If provided, a logger for logging debug statements.

        Returns:
            the constructed Bandpass object.
        """
        meta = galsim.config.GetInputObj('opsim_meta_dict', config, base, 'InstCatWorldPos')

        # Note: Previous code used the lsst.sims versions of these.  Here we just use the ones in
        #       the GalSim share directory.  Not sure whether those are current, but probably
        #       good enough for now.
        band = meta.get('band')
        bandpass = galsim.Bandpass('LSST_%s.dat'%band, wave_type='nm')
        bandpass = bandpass.withZeropoint('AB')
        logger.debug('bandpass = %s',bandpass)
        return bandpass, False

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
                'min_source' : int,
                'skip_invalid' : bool,
              }
        kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
        wcs = galsim.config.BuildWCS(base['image'], 'wcs', base, logger=logger)
        kwargs['wcs'] = wcs
        kwargs['logger'] = galsim.config.GetLoggerProxy(logger)
        return kwargs, safe

RegisterInputType('opsim_meta_dict', InputLoader(OpsimMetaDict, file_scope=True, takes_logger=True))
RegisterValueType('OpsimMeta', OpsimMeta, [float, int, str], input_type='opsim_meta_dict')
RegisterBandpassType('OpsimMetaBandpass', OpsimMetaBandpass(), input_type='opsim_meta_dict')

RegisterInputType('instance_catalog', InstCatalogLoader(InstCatalog, has_nobj=True))
RegisterValueType('InstCatWorldPos', InstCatWorldPos, [CelestialCoord],
                  input_type='instance_catalog')
RegisterObjectType('InstCatObj', InstCatObj, input_type='instance_catalog')
RegisterSEDType('InstCatSED', InstCatSEDBuilder(), input_type='instance_catalog')
