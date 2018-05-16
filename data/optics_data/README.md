# Simulation Data for Optical Zernikes

Files in this directory include:

  - **annular_nominal_coeff.txt :** An array of the nominal annular zernike
      coefficients for LSST. These values are interpolated from the zemax
      estimates found in *annular_zemax_estimates.fits* to determine 19
      Zernike coefficients at 35 sampling coordinates.
  - **annular_zemax_estimates.fits :** Zemax estimates for the nominal annular
      zernike coefficients stored as a 32 x 32 array. Note that some of these
      entries are null.
  - **aos_sim_results.txt :** Simulation results for the size of optical
      deviations left over from the active optic system.
  - **regular_zemax_estimates.fits :** Zemax estimates for the nominal zernike
      coefficients. Note that these results are for the regular Zernike
      polynomials and not their annular counterparts.
  - **sensitivity_matrix.txt :** Values of the sensitivity matrix sampled at 35
      positions in the LSST exit pupil.
          