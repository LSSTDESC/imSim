import numpy as np
import os
from pathlib import Path
import galsim
from galsim.config import RegisterBandpassType
from astropy.table import Table

__all__ = ['RubinBandpass']


class AtmInterpolator:
    """ Interpolate atmospheric transmission curves from throughputs repo.
    Linear interpolation of log(transmission) independently for every wavelength
    does a good job.  Extrapolation is done by assuming a constant slope in
    log(transmission).

    Parameters
    ----------
    Xs : `np.array`
        Airmass values at which the transmission curves are tabulated.
    arr : `np.array`
        Transmission curves at the airmass values.  First index is the airmass,
        second index is the wavelength.
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

        Parameters
        ----------
        X : `float`
            Airmass at which to evaluate the transmission curve.

        Returns
        -------
        out : `np.array`
            Transmission curve at the requested airmass.
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


def RubinBandpass(
    band,
    airmass=None,
    camera=None,
    det_name=None,
    logger=None
):
    """Return one of the Rubin bandpasses, specified by the single-letter name.

    The zeropoint is automatically set to the AB zeropoint normalization.

    Parameters
    ----------
    band : `str`
        The name of the bandpass.  One of u,g,r,i,z,y.
    airmass : `float`, optional
        The airmass at which to evaluate the bandpass.  If None, use the
        standard X=1.2 bandpass.
    camera : `str`, optional
        Name of the camera for which to incorporate the detector-specific
        quantum efficiency.  If None, use the standard hardware bandpass.
    det_name : `str`, optional
        Name of the detector for which to incorporate the quantum efficiency.  If
        None, use the standard hardware bandpass.
    logger : logging.Logger
        If provided, a logger for logging debug statements.
    """
    if (camera is None) != (det_name is None):
        raise ValueError("Must provide both camera and det_name if using one.")
    match camera:
        case 'LsstCam':
            camera = 'lsstCam'
        case 'LsstComCamSim':
            camera = 'comCamSim'
        case _:
            camera = camera
    tp_path = Path(os.getenv("RUBIN_SIM_DATA_DIR")) / "throughputs"
    if airmass is None and camera is None:
        file_name = tp_path / "baseline" / f"total_{band}.dat"
        if not file_name.is_file():
            logger = galsim.config.LoggerWrapper(logger)
            logger.warning("Warning: Using the old bandpass files from GalSim, not rubin_sim")
            file_name = f"LSST_{band}.dat"
        bp = galsim.Bandpass(str(file_name), wave_type='nm')
    else:
        if airmass is None:
            airmass = 1.2
        # Could be more efficient by only reading in the bracketing airmasses,
        # but probably doesn't matter much here.
        atmos = {}
        for f in (tp_path / "atmos").glob("atmos_??_aerosol.dat"):
            X = float(f.name[-14:-12])/10.0
            wave, tput = np.genfromtxt(f).T
            atmos[X] = wave, tput
        Xs = np.array(sorted(atmos.keys()))
        arr = np.array([atmos[X][1] for X in Xs])

        interpolator = AtmInterpolator(Xs, arr)
        tput = interpolator(airmass)
        bp_atm = galsim.Bandpass(
            galsim.LookupTable(
                wave, tput, interpolant='linear'
            ),
            wave_type='nm'
        )

        if camera is not None:
            try:
                old_path = Path(os.getenv("OBS_LSST_DATA_DIR"))
            except TypeError:
                raise ValueError(
                    "Unable to find OBS_LSST_DATA; required if using camera or det_name for bandpass."
                )
            old_path = old_path / camera / "transmission_sensor" / det_name.lower()
            det_files = list(old_path.glob("*.ecsv"))
            if len(det_files) != 1:
                raise ValueError(f"Expected 1 detector file, found {len(det_files)}")
            det_file = det_files[0]
            table = Table.read(det_file)
            # Average over amplifiers
            amps = np.unique(table['amp_name'])
            det_wave = table[table['amp_name'] == amps[0]]['wavelength']
            det_tput = table[table['amp_name'] == amps[0]]['efficiency']/100
            for amp in amps[1:]:
                assert np.all(det_wave == table[table['amp_name'] == amp]['wavelength'])
                det_tput += table[table['amp_name'] == amp]['efficiency']/100
            det_tput /= len(amps)
            bp_det = galsim.Bandpass(
                galsim.LookupTable(
                    det_wave, det_tput, interpolant='linear'
                ),
                wave_type='nm'
            )

            # Get the rest of the hardware throughput
            optics_wave, optics_tput = np.genfromtxt(
                tp_path / "baseline" / f"filter_{band}.dat"
            ).T
            for f in ["m1.dat", "m2.dat", "m3.dat", "lens1.dat", "lens2.dat", "lens3.dat"]:
                wave, tput = np.genfromtxt(
                    tp_path / "baseline" / f
                ).T
                assert np.all(wave == optics_wave)
                optics_tput *= tput
            bp_optics = galsim.Bandpass(
                galsim.LookupTable(
                    optics_wave, optics_tput, interpolant='linear'
                ),
                wave_type='nm'
            )
            bp_hardware = bp_det * bp_optics
        else:
            file_name = tp_path / "baseline" / f"hardware_{band}.dat"
            wave, hardware_tput = np.genfromtxt(file_name).T
            bp_hardware = galsim.Bandpass(
                galsim.LookupTable(
                    wave, hardware_tput, interpolant='linear'
                ),
                wave_type='nm'
            )
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
        opt = {
            'airmass' : float,
            'camera' : str,
            'det_name' : str
        }
        kwargs, safe = galsim.config.GetAllParams(
            config, base, req=req, opt=opt
        )
        kwargs['logger'] = logger
        bp = RubinBandpass(**kwargs)

        # Also, store the kwargs=None version in the base config.
        base['fiducial_bandpass'] = RubinBandpass(band=kwargs['band'], logger=logger)
        logger.debug('bandpass = %s', bp)
        return bp, safe

RegisterBandpassType('RubinBandpass', RubinBandpassBuilder())
