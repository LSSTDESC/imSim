Instructions for generating vignetting spline fits.

1. Run `sample_focal_plane.py` to generate instance catalogs of stars
   that randomly sample each CCD in the focal plane.  This will
   produce a file `stars_vignetting.txt`, which will be included in
   main instance catalog file, `instcat_vignetting.txt`.

2. The filter parameter in `instcat_vignetting.txt` is set to i-band
   (filter 3).  This can be changed to any value in [0-5] to get one
   of ugrizy.

3. The `magnorm` value (monochormatic magnitude at 500 nm) may need to
   be adjusted in `sample_focal_plane.py` to get the desired number of
   photons per star.  Currently, `magnorm=22.25` will produce ~100k
   incident photons per star.

4. To run the simulation run `galsim imsim-vignetting.yaml`.

5. To compute the fluxes per cluster and fit the spline function run
   the `fit_vignetting_profile.py` script.

None of these steps have been implemented as completely turn-key
operations, some tweaking by hand may be needed for each.
