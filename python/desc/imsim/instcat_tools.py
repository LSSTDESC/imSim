"""
Utilities to inspect instance catalog object SEDs, magnitudes, and fluxes.
"""
import sys
import copy
from collections import defaultdict
import numpy as np
import pandas as pd
import lsst.sims.photUtils as sims_photUtils
from .imSim import parsePhoSimInstanceFile
from .sed_wrapper import SedWrapper

__all__ = ['make_sed_dataframe', 'reaggregate_galaxies']


class SedInspector(SedWrapper):
    """
    SED handling class to allow for inspection of magnitudes and
    fluxes after various levels of extinction have been applied to the
    instance catalog entries.

    This is implemented as a subclass of SedWrapper in order to keep that
    class as lightweight as possible.
    """
    def __init__(self, *args, gs_obj=None):
        """
        Constructor.  This adds gs_obj as keyword argument to the
        base class arguments.
        """
        super(SedInspector, self).__init__(*args)
        self.gs_obj = gs_obj
        self._intrinsic = None
        self._reddened = None
        self._observed = None

    @staticmethod
    def create_from_gs_object(gs_obj):
        """Factor method to create an SedInspector """
        return SedInspector(gs_obj.sed.sed_file,
                            gs_obj.sed.mag_norm,
                            gs_obj.sed.redshift,
                            gs_obj.sed.iAv,
                            gs_obj.sed.iRv,
                            gs_obj.sed.gAv,
                            gs_obj.sed.gRv,
                            gs_obj.sed.bp_dict,
                            gs_obj=gs_obj)

    @property
    def intrinsic(self):
        """The intrinsic, unreddened SED in the object rest frame."""
        if self._intrinsic is None:
            self._computeSED()
        return self._intrinsic

    @property
    def reddened(self):
        """The object SED with intrinsic reddening and redshift applied."""
        if self._reddened is None:
            self._computeSED()
        return self._reddened

    @property
    def observed(self):
        """
        The observed SED from the Earth.  Intrinsic reddening, redshift,
        and MW reddening are all applied.
        """
        if self._observed is None:
            self._computeSED()
        return self._observed

    def _computeSED(self):
        self._intrinsic = sims_photUtils.Sed()
        self._intrinsic.readSED_flambda(self.sed_file)
        fnorm = sims_photUtils.getImsimFluxNorm(self._intrinsic,
                                                self.mag_norm)
        self._intrinsic.multiplyFluxNorm(fnorm)
        self._reddened = copy.deepcopy(self._intrinsic)
        if self.iAv != 0:
            self.shared_resources['ccm_model'].add_dust(self._reddened,
                                                        self.iAv, self.iRv,
                                                        'intrinsic')
        if self.redshift != 0:
            self._reddened.redshiftSED(self.redshift, dimming=True)

        self._observed = copy.deepcopy(self._reddened)
        if self.gAv != 0:
            self.shared_resources['ccm_model'].add_dust(self._observed,
                                                        self.gAv, self.gRv,
                                                        'Galactic')


def add_mags(mags):
    """
    Combine a list magnitudes by taking the sum of the corresponding fluxes.
    """
    if len(mags) == 1:
        return mags[0]
    return -2.5*np.log10(sum([10.**(-mag/2.5) for mag in mags]))


def get_values(df, rows, column):
    """Get the desired column value from a list of dataframe iloc indexes."""
    return [df.iloc[row][column] for row in rows]


def combine_mags(df, rows):
    """
    Return row of combined magnitudes for the intrinsic, reddened,
    and observed SEDs
    """
    mag_intrinsic = add_mags(get_values(df, rows, 'mag_intrinsic'))
    mag_reddened = add_mags(get_values(df, rows, 'mag_reddened'))
    mag_observed = add_mags(get_values(df, rows, 'mag_observed'))
    return (df.iloc[rows[0]]['galaxy_id'], mag_intrinsic, mag_reddened,
            mag_observed)


def reaggregate_galaxies(df_input):
    """
    Re-aggregate disk and bulge components into a single galaxy and
    return a dataframe with combined magnitudes and galaxy_ids.
    """
    df = df_input.sort_values('uniqueId')

    composites = defaultdict(list)
    for irow, galaxy_id, xpupil in zip(range(len(df)), df['galaxy_id'],
                                       df['xPupilRadians']):
        composites[(galaxy_id, xpupil)].append(irow)

    data = []
    for rows in composites.values():
        if len(rows) > 1:
            data.append(combine_mags(df, rows))

    return pd.DataFrame(data, columns=['galaxy_id', 'mag_intrinsic',
                                       'mag_reddened', 'mag_observed'])


def make_sed_dataframe(instcat, sensor, band):
    """
    Create a DataFrame with magnitudes and fluxes for each of the
    instance catalog object entries.
    """
    _, phot_params, objects = parsePhoSimInstanceFile(instcat)

    print("processing", len(objects[1][sensor]), "SEDs")
    my_seds = [SedInspector.create_from_gs_object(gs_obj)
               for gs_obj in objects[1][sensor]]
    bp_dict = sims_photUtils.BandpassDict.loadTotalBandpassesFromFiles()
    bp = bp_dict[band]

    rows = []
    for ised, sed in enumerate(my_seds):
        if ised % (len(my_seds)//20) == 0:
            sys.stdout.write('.')
            sys.stdout.flush()
        if sed.redshift == 0:
            continue
        rows.append((sed.gs_obj.uniqueId,
                     sed.gs_obj.uniqueId >> 10,
                     sed.gs_obj.xPupilRadians,
                     sed.gs_obj.yPupilRadians,
                     sed.redshift,
                     sed.mag_norm,
                     sed.intrinsic.calcMag(bp),
                     sed.intrinsic.calcADU(bp, phot_params),
                     sed.reddened.calcMag(bp),
                     sed.reddened.calcADU(bp, phot_params),
                     sed.observed.calcMag(bp),
                     sed.observed.calcADU(bp, phot_params)))

    columns = ['uniqueId', 'galaxy_id',
               'xPupilRadians', 'yPupilRadians',
               'redshift', 'mag_norm',
               'mag_intrinsic', 'flux_intrinsic',
               'mag_reddened', 'flux_reddened',
               'mag_observed', 'flux_observed']
    return pd.DataFrame(rows, columns=columns)
