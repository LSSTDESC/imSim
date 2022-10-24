"""Shared functionality for modules using batoid."""

from typing import Optional
from functools import lru_cache
import batoid


@lru_cache
def load_telescope(telescope_id: str, band: str) -> batoid.Optic:
    """Load a telescope.

    Parameters
    ----------
    telescope : str
        The name of the telescope. [default: 'LSST']  Currenly only 'LSST' is functional.
    band : str
        The name of the bandpass
    """

    if telescope_id != "LSST":
        raise NotImplementedError(
            "Batoid WCS only valid for telescope='LSST' currently"
        )
    return batoid.Optic.fromYaml(f"{telescope_id}_{band}.yaml")


def load_telescope_with_shift_optics(
    telescope_id: str, band: str, shift_optics: Optional[dict] = None
) -> batoid.Optic:
    """Load a telescope and applies shifts.

    Parameters
    ----------
    telescope : str
        The name of the telescope. [default: 'LSST']  Currenly only 'LSST' is functional.
    band : str
        The name of the bandpass
    shift_optics : dict[str, list[float]]
        A dict mapping optics keys to shifts represented by a list of 3 floats.
        The corresponding optics will be displaced by the specified corrdinates.
        Example config for perturbed+defocused telescope to obtain a donut:
        -
            type: lsst_optics
            ...
            shift_optics:
              Detector: [0, 0, 1.5e-3]
              M2: [3.0e-3, 0, 0]
    """
    telescope = load_telescope(telescope_id, band)
    if shift_optics is not None:
        for optics_key, shift in shift_optics.items():
            telescope = telescope.withGloballyShiftedOptic(optics_key, shift)
    return telescope
