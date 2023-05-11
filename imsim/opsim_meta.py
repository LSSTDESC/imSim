
import warnings
import sqlite3
import numpy as np
import astropy
import astropy.coordinates
import pandas as pd
from galsim.config import InputLoader, RegisterInputType, RegisterValueType, RegisterBandpassType
import galsim
from lsst.obs.lsst.translators.lsst import SIMONYI_LOCATION as RUBIN_LOC


def get_opsim_md(config, base):
    """
    If we don't have an OpsimMeta, then skip some header items.
    E.g. when reading out flat field images, most of these don't apply.
    The only exception is filter, which we look for in config and use
    that if present.
    """
    try:
        opsim_md = galsim.config.GetInputObj('opsim_meta_dict', config,
                                             base, 'get_opsim_md')
    except galsim.GalSimConfigError:
        if 'filter' in config:
            filt = galsim.config.ParseValue(config, 'filter', base, str)[0]
        else:
            filt = 'N/A'
        opsim_md = OpsimMetaDict.from_dict(
            dict(band=filt,
                 exptime = base['exp_time']
            )
        )
    return opsim_md


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
                   'snap' : int,
                   'image_type' : str,
                   'reason' : str}
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
                                seqnum
                                obshistid
                                seed
                                seeing
                                sunalt
                                vistime""".split())

    def __init__(self, file_name, visit=None, snap=0, image_type='SKYEXP',
                 reason='survey', logger=None):
        self.logger = galsim.config.LoggerWrapper(logger)
        self.file_name = file_name
        self.visit = visit
        self.meta = {'snap' : snap,
                     'image_type': image_type,
                     'reason': reason}

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
        if self.visit is None:
            raise ValueError('The visit must be set when reading visit info from an opsim db file.')

        self.logger.warning('Reading info from opsim db file %s for visit %s',
                            self.file_name, self.visit)

        # Query for the info for the desired visit.
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

            # Determine the daily sequence number for this exposure by
            # counting the number of snaps since int(observationStartMJD).
            t0 = int(self.meta['observationStartMJD'])
            sql = f'''select numExposures from observations where
                      {t0} <= observationStartMJD and
                      observationId < {self.visit}'''
            df = pd.read_sql(sql, con)
            self.meta['seqnum'] = sum(df['numExposures']) + self.meta['snap']
        self.logger.warning('Done reading visit info from opsim db file')

        if self.meta['snap'] >= self.meta['numExposures']:
            raise ValueError('Invalid snap value: %d. For this visit, must have snap < %d'
                             % (self.meta['snap'], self.meta['numExposures']))

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
        self.meta['FWHMeff'] = self.meta['seeingFwhmEff']
        self.meta['FWHMgeom'] = self.meta['seeingFwhmGeom']
        self.logger.debug("Bandpass = %s",self.meta['band'])
        self.logger.debug("HA = %s",self.meta['HA'])

    def _read_instcat_header(self):
        """Read visit info from the instance catalog header."""
        self.logger.warning('Reading visit info from instance catalog %s',
                            self.file_name)
        with open(self.file_name, mode='rt') as _input:
            for line in _input:
                if line.startswith('#'):  # comments
                    continue
                if line.startswith('object') or line.startswith('includeobj'):
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

        if any(key not in self.meta for key in self._required_commands):
            raise ValueError("Some required commands are missing. Required commands: {}".format(
                             str(self._required_commands)))

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

        self.set_defaults()

        # Use the opsim db names for these quantities.
        self.meta['fieldRA'] = self.meta['rightascension']
        self.meta['fieldDec'] = self.meta['declination']
        self.meta['rotTelPos'] = self.meta['rottelpos']
        self.meta['rotSkyPos'] = self.meta['rotskypos']
        self.meta['observationId'] = self.meta['obshistid']
        self.meta['observationStartMJD'] = self.meta['mjd'] - self.meta['exptime']/2./86400.

    def set_defaults(self):
        # Set some default values if these aren't present in input file.
        if 'exptime' not in self.meta:
            self.meta['exptime'] = self.meta.get('exptime', 30)
        if 'darkcurrent' not in self.meta:
            # TODO: Eventually, get this from Camera object during readout (when we actually need
            #       it), but this value is not currently available from the lsst.camera object.
            self.meta['darkcurrent'] = self.meta.get('darkcurrent', 0)

    @classmethod
    def from_dict(cls, d):
        """Build an OpsimMetaDict directly from the provided dict.

        (Mostly used for unit tests.)
        """
        ret = cls.__new__(cls)
        ret.file_name = ''
        ret.meta = d
        # If possible, add in the derived values.
        if 'band' not in d and 'filter' in d:
            ret.meta['band'] = 'ugrizy'[d['filter']]
        if 'HA' not in d and 'mjd' in d and 'rightascension' in d:
            ret.meta['HA'] = ret.getHourAngle(d['mjd'], d['rightascension'])
        if 'rawSeeing' not in d and 'seeing' in d:
            ret.meta['rawSeeing'] = ret.meta.pop('seeing')
        if 'airmass' not in d and 'altitude' in d:
            ret.meta['airmass'] = ret.getAirmass()
        if 'FWHMeff' not in d and 'band' in d and 'rawSeeing' in d:
            ret.meta['FWHMeff'] = ret.FWHMeff()
        if 'FWHMgeom' not in d and 'band' in d and 'rawSeeing' in d:
            ret.meta['FWHMgeom'] = ret.FWHMgeom()
        ret.set_defaults()
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

    def get(self, field, default=None):
        if field not in self.meta and default is None:
            raise KeyError("OpsimMeta field %s not present in instance catalog"%field)
        return self.meta.get(field, default)

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
        lsst_loc = RUBIN_LOC

        time = astropy.time.Time(mjd, format='mjd', location=lsst_loc)
        # Get the local apparent sidereal time.
        with warnings.catch_warnings():
            # Astropy likes to emit obnoxious warnings about this maybe being slightly inaccurate
            # if the user hasn't updated to the absolute latest IERS data.  Ignore them.
            warnings.simplefilter("ignore")
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

RegisterInputType('opsim_meta_dict', InputLoader(OpsimMetaDict, file_scope=True, takes_logger=True))
RegisterValueType('OpsimMeta', OpsimMeta, [float, int, str], input_type='opsim_meta_dict')
RegisterBandpassType('OpsimMetaBandpass', OpsimMetaBandpass(), input_type='opsim_meta_dict')
