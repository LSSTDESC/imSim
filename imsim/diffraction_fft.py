import numpy as np
import scipy.signal

from .diffraction import field_rotation_sin_cos, prepare_e_z, e_equatorial

# Obtained from linear regression of the photon shooting r distribution:
INTERCEPT = 0.0706052627908828
A_SQRT = 1.0 / (2.0 * np.exp(INTERCEPT) * np.pi)


def spike_profile(r: np.ndarray) -> np.ndarray:
    """Spike profile fit to the spikes produced by the statistical approach."""
    return 0.5 * A_SQRT / np.pi / (1 + (A_SQRT * r) ** 2)


def int_spike_profile(r: np.ndarray) -> np.ndarray:
    """Antiderivative of the spike profile."""
    return 0.5 * A_SQRT / np.pi / A_SQRT * np.arctan(A_SQRT * r)


def field_rotation_profile(
    r: np.ndarray, d_alpha: float, resolution: float
) -> np.ndarray:
    r"""Due to field rotation, the spike cross is rotating with time.
    As a consequence outer pixels receive a lower dose of photons than inner pixels.
    This function returns version of `spike_profile` including the field rotation effect.

    Idea: Calculate the photon dose a single pixel in a distance of r to the
    spike center receives.

    Approximations:
    - Assume that the rotation rate is constant over time.
    - A pixel is approximated by an "angular pixel"
      (r0-1/2 <= r <= r0+1/2, 0 <= \phi <= \phi_pix(r0),
      where \phi_pix(r) is chosen such that the arclength r \phi_pix(r) is 1).
    """

    l_pix = resolution  # length of a pixel

    def arclength_dose(r: np.ndarray) -> np.ndarray:
        """If the arclength r * d_alpha is smaller than the length of a pixel,
        return 1.0, otherwise return l_pix / (r * d_alpha)."""
        # The following also allows d_alpha = 0:
        return 1.0 / np.maximum(r * d_alpha / l_pix, 1.0)

    d_pix = 0.5 * l_pix  # Half a pixel
    # use int_spike_profile, to compute the radial integral
    # int(spike_profile(r), r=r-d_pix..r+d_pix):
    return (
        int_spike_profile(r + d_pix) - int_spike_profile(r - d_pix)
    ) * arclength_dose(r)


def antialiased_cross(xy: np.ndarray, alpha: float) -> np.ndarray:
    """Fills a numpy array of same shape as xy with 1. along the x and y axis, rotated by alpha
    with a linear decay with the distance to the axes."""
    rot = np.array(
        [[np.cos(-alpha), -np.sin(-alpha)], [np.sin(-alpha), np.cos(-alpha)]]
    )
    xy_rotated = np.einsum("ij,jkl", rot, xy)
    # Antialiased cross:
    return np.maximum(0.0, 1.0 - np.min(np.abs(xy_rotated), axis=0))


def prepare_psf_field_rotation(
    w: int,
    h: int,
    resolution: float,
    alpha: float,
    d_alpha: float,
) -> np.ndarray:
    """Spike PSF for finite angular spike with (field rotation).
    w: Pixel width of the image
    h: Pixel height of the image
    resolution: length per pixel
    alpha: Rotation angle of the spikes [rad]
    d_alpha: angular spike width (field rotation angle)

    Will return an image of the spike PSF of dimension (2w+1)x(2h+1).
    The angular range of the spike will be [alpha, alpha + d_alpha] (4-fold symmetry).
    """
    x_range = np.arange(-w, w + 1)
    y_range = np.arange(-h, h + 1)
    [x, y] = xy = np.c_[
        np.meshgrid(
            x_range,
            y_range,
            indexing="ij",
            copy=False,
        )
    ]
    # Antialiased cross at alpha - d_alpha / 2:
    psf = antialiased_cross(xy, alpha - d_alpha / 2.0)
    # Interior:
    th = np.arctan2(y, x)
    dth = th - (alpha - d_alpha)  # angular distance to spike boundary
    dth %= np.pi / 2  # 4-fold symmetry
    psf[dth <= d_alpha] = 1.0
    # Radial profile:
    r = np.hypot(x, y) * resolution
    psf *= field_rotation_profile(r, d_alpha, resolution)
    # Normalize, so pixel values add up to 1.
    # Here we actually are neglecting the non-visible tails of the cross.
    psf /= np.sum(psf)
    return psf


def apply_diffraction_psf(
    image: np.ndarray,
    rottelpos: float,
    exptime: np.ndarray,
    latitude: float,
    azimuth: float,
    altitude: float,
    brightness_threshold: float,
    pixel_length: float,
    spike_length_cutoff: int,
):
    """2d convolve, leaving image dimension invariant.
    image: Brightness pixel data
    rottelpos: Telescope rotation angle [rad]
    exptime: Exposure time (used to determine the spike width)
    latitude: Geographic latitude of observators
    azimuth: Azimuth of the telescope
    altitude: Altitude of the telescope
    brightness_threshold: Minimum pixel value onto which to put diffraction spikes
    pixel_length: Scaling factor: The length of 1px in `spike_profile`
    spike_length_cutoff: Size of the PSF (width and height) which will be layed over saturated pixels.
    """

    psf_w = psf_h = spike_length_cutoff  # neglect the outer region (~1/r^2)
    sin_cos = np.empty(2)
    e_z_0, e_z = prepare_e_z(latitude)
    e_focal = e_equatorial(latitude=latitude, azimuth=azimuth, altitude=altitude)
    field_rotation_sin_cos(e_z_0, e_z, e_focal, exptime, sin_cos)
    d_alpha = np.arctan2(sin_cos[1], sin_cos[0])
    rottelpos = np.pi/4. - rottelpos
    spike_per_pixel = prepare_psf_field_rotation(
        psf_w,
        psf_h,
        resolution=pixel_length,
        alpha=rottelpos,
        d_alpha=d_alpha,
    )
    for region_row, region_col in saturated_clusters(image, brightness_threshold):
        img_region = image[region_row, region_col]
        diffracted = scipy.signal.convolve2d(spike_per_pixel, img_region, mode="full")
        img_region[()] = 0.0
        add_image(
            image,
            diffracted,
            row=region_row.start - psf_w,
            col=region_col.start - psf_h,
        )


def add_image(image, overlay, row, col) -> None:
    """Add overlay to image[row:,col:] cropping overlay when needed."""
    row_start = max(0, row)
    row_end = min(row + overlay.shape[0], image.shape[0])
    col_start = max(0, col)
    col_end = min(col + overlay.shape[1], image.shape[1])
    image[row_start:row_end, col_start:col_end] += overlay[
        max(0, -row) : min(overlay.shape[0], image.shape[0] - row),
        max(0, -col) : min(overlay.shape[1], image.shape[1] - col),
    ]


def saturated_clusters(image, brightness_threshold: float):
    """Detect clusters of saturated pixels ( > brightness_threshold) in an image.
    Returns a list of 2-tuples of slices (pixel ranges (x and y) containing one cluster).
    """
    (w, h) = image.shape
    xy = np.c_[
        np.meshgrid(
            np.arange(w),
            np.arange(h),
            indexing="ij",
            copy=False,
        )
    ]
    pixel_coordinates = xy[:, image > brightness_threshold]
    regions = []
    for i, j in pixel_coordinates.T:
        neighbor_indices = [n for n, r in enumerate(regions) if dist_2d(r, (i, j)) <= 1]
        pix_region = (slice(i, i + 1), slice(j, j + 1))
        if not neighbor_indices:
            regions.append(pix_region)
            continue
        neighbor_regions = [regions.pop(n) for n in reversed(neighbor_indices)]
        regions.append(merge_regions(neighbor_regions + [pix_region]))
    return regions


def merge_regions(regions: list):
    """Merge 2d intervals"""
    return tuple(
        slice(min(r[dim].start for r in regions), max(r[dim].stop for r in regions))
        for dim in (0, 1)
    )


def dist_2d(region, p):
    """Distance of pixel p to a 2-slice (2-tuple of slices)."""
    return max(dist_1d(slc, n) for slc, n in zip(region, p))


def dist_1d(slc, n):
    """Distance of a pixel coordinate n to a slice in 1 dimension."""
    if slc.start <= n < slc.stop:
        return 0
    return max(slc.start - n, n - slc.stop + 1)
