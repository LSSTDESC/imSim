import numpy as np
import os
from pathlib import Path
import galsim
from galsim.config import RegisterBandpassType

__all__ = ['RubinBandpass']


class AtmInterpolator:
    """ Interpolate atmospheric transmission curves from throughputs repo.
    Linear interpolation of log(transmission) independently for every wavelength
    does a good job.  Extrapolation is done by assuming a constant slope in
    log(transmission).
    """
    def __init__(self, Xs, arr):
        self.Xs = Xs
        self.arr = arr
        with np.errstate(all='ignore'):
            self.logarr = np.log(arr)
            self.log_extrapolation_slope = (
                (self.logarr[-1] - self.logarr[-2])/(self.Xs[-1]-self.Xs[-2])
            )

    def __call__(self, X):
        """ Evaluate atmospheric transmission curve at airmass X.
        """
        assert X >= 1.0
        idx = np.searchsorted(self.Xs, X, side='right')
        if idx == len(self.Xs):  # extrapolate
            frac = (X - self.Xs[idx-1])
            out = np.array(self.logarr[idx-1])
            out += frac*self.log_extrapolation_slope
        else:
            frac = (X - self.Xs[idx-1]) / (self.Xs[idx] - self.Xs[idx-1])
            out = (1-frac)*np.array(self.logarr[idx-1])
            out += frac*self.logarr[idx]
        out = np.exp(out)
        out[~np.isfinite(out)] = 0.0  # Clean up exp(log(0)) => 0
        return out


def RubinBandpass(band, airmass=None, logger=None):
    """Return one of the Rubin bandpasses, specified by the single-letter name.

    The zeropoint is automatically set to the AB zeropoint normalization.

    Parameters
    ----------
    band : `str`
        The name of the bandpass.  One of u,g,r,i,z,y.
    logger : logging.Logger
        If provided, a logger for logging debug statements.
    """
    path = Path(os.getenv("RUBIN_SIM_DATA_DIR")) / "throughputs"
    if airmass is None:
        file_name = path / "baseline" / f"total_{band}.dat"
        if not file_name.is_file():
            logger = galsim.config.LoggerWrapper(logger)
            logger.warning("Warning: Using the old bandpass files from GalSim, not lsst.sims")
            file_name = f"LSST_{band}.dat"
        bp = galsim.Bandpass(str(file_name), wave_type='nm')
    else:
        # Could be more efficient by only reading in the bracketing airmasses,
        # but probably doesn't matter much here.
        atmos = {}
        for f in (path / "atmos").glob("atmos_??_aerosol.dat"):
            X = float(f.name[-14:-12])/10.0
            wave, tput = np.genfromtxt(f).T
            atmos[X] = wave, tput
        Xs = np.array(sorted(atmos.keys()))
        arr = np.array([atmos[X][1] for X in Xs])

        interpolator = AtmInterpolator(Xs, arr)
        tput = interpolator(airmass)
        bp_atm = galsim.Bandpass(galsim.LookupTable(wave, tput), wave_type='nm')

        file_name = path / "baseline" / f"hardware_{band}.dat"
        wave, hardware_tput = np.genfromtxt(file_name).T
        bp_hardware = galsim.Bandpass(galsim.LookupTable(wave, hardware_tput), wave_type='nm')
        bp = bp_atm * bp_hardware
    bp = bp.truncate(relative_throughput=1.e-3)
    bp = bp.thin()
    bp = bp.withZeropoint('AB')
    return bp


class RubinBandpassBuilder(galsim.config.BandpassBuilder):
    """A class for building a RubinBandpass in the config file
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
        req = { 'band' : str }
        opt = { 'airmass' : float }
        kwargs, safe = galsim.config.GetAllParams(
            config, base, req=req, opt=opt
        )
        kwargs['logger'] = logger
        bp = RubinBandpass(**kwargs)

        # Also, store the airmass=None version in the base config.
        if 'airmass' in kwargs:
            del kwargs['airmass']
            base['bandpass_fiducial'] = RubinBandpass(**kwargs)
        else:
            base['bandpass_fiducial'] = bp
        logger.debug('bandpass = %s', bp)
        return bp, safe

RegisterBandpassType('RubinBandpass', RubinBandpassBuilder())
