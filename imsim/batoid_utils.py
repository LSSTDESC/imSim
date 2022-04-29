"""Shared functionality for modules using batoid."""

import batoid


def load_telescope(telescope: str, band: str) -> batoid.Optic:
    """Load a telescope.

    Parameters
    ----------
    telescope : str
        The name of the telescope. [default: 'LSST']  Currenly only 'LSST' is functional.
    band : str
        The name of the bandpass"""

    if telescope != "LSST":
        raise NotImplementedError(
            "Batoid WCS only valid for telescope='LSST' currently"
        )
    return batoid.Optic.fromYaml(f"{telescope}_{band}.yaml")
