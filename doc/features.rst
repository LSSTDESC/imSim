Features Implemented in imSim
#############################

These summary tables should give a high level overview of effects that have been implemented in imSim. For detailed information please see the linked sub-pages, code, or appropriate reference.

Go directly to:
`Sensors <Sensor Effects_>`_ - `Sky <Sky Model_>`_ - `Throughputs <System Throughputs_>`_ - `Atmospheric PSF <Atmospheric PSF Model_>`_ - `Optics <Optical Model_>`_  - `Calibration Products <Calibration Products_>`_

Sensor Effects
--------------
..
  .. image:: img/features.svg

This table is a list of sensor effects in imSim along with pointers to the techniques used to implement them, and the internal validation tests that have been performed.

.. list-table::
   :widths: 10 10 10 15 15
   :header-rows: 1
   :class: tight-table

   * - Effect
     - Implementation
     - Data / Model Source
     - Short description
     - Validation Page and Notebooks

   * - Brighter Fatter
     - GalSim Feature (Silicon.cpp)
     - Linear scaling of pixel edge vertices displacement derived with Poisson Solver. Pre-computed solutions available for both E2V and ITL sensors, with both 8 and 32 vertices per edge.
     - GalSim reads in vertex data from full electrostatic Poisson solver, scales them linearly with collected charges, and co-adds the effects from all pixels iteratively while collecting the image.
     - :doc:`validation/brighter-fatter`

   * - Diffusion
     - GalSim Feature (sensor.py / Silicon.cpp)
     - Diffusion Parameters estimated from first principles and validated with Fe55
     - GalSim applies random Gaussian displacement for every photon using temperature and voltage dependent amplitude,  See page link
     - :doc:`validation/diffusion`

   * - Tree Rings
     - GalSim Feature (Silicon.cpp) / imSim configuration (tree_rings.py)
     - Analytic model is used to pre-compute 189 unique sensor models with randomized parameters empirically based on BNL acquired data.
     - Radial displacement profile is modelled as a sum of 40 sinusoids modulated by a power law function.
     - :doc:`validation/tree-ring`

   * - CTE
     - readout.py
     - Camera Integration and Testing
     -
     -

   * - Noise Rate
     - readout.py
     - Camera Integration and Testing
     -
     -

   * - Xtalk
     - readout.py
     - Camera Integration and Testing
     -
     - Crosstalk values are read from obs_lsst camera discription

   * - Hot Pixels/Rows
     - **being implemented**
     - Camera Integration and Testing
     -
     -

   * - Fringing
     - **being implemented**
     - Sensor Testing and electromagnetic model
     - Please see: https://inspirehep.net/literature/2183279
     -

   * - Cosmic Rays
     - cosmic_rays.py: ~10K cosmic are randomly added to the exposures.
     - Template data taken from ITL test stands at UofA. We should remeasure on summit.
     -
     -

   * - Edge rolloff
     - Not yet
     -
     -
     -

   * - Bleeding
     - bleed_trails.py called from readout.py
     - Test stand at Davis. Specialized bleed runs.
     -
     -



Sky Model
---------

imSim uses the Rubin project sky model. It is called sims_skybrightness and is located in the rubin-sims package which is an *imSim* dependency.

.. list-table::
   :widths: 10 10 10 15 15
   :header-rows: 1
   :class: tight-table

   * - Effect
     - Implementation
     - Data / Model Source
     - Short description
     - Validation Page and Notebooks

   * - Sky Background
     - See  `here <https://rubin-sim.lsst.io/rs_skybrightness/index.html>`_
     - Based on the `ESO sky brightness model <http://www.eso.org/observing/etc/bin/gen/form?INS.MODE=swspectr+INS.NAME=SKYCALC>`_
       and all-sky camera data from LSST site for twilight sky.
     - The model includes light from twilight (scattered sunlight), zodiacal light (scattered sunlight from SS dust), scattered moonlight, airglow, and emission lines from the upper and lower atmosphere. The model can return SEDs or  magnitude per sq arcsec in LSST filters.
     - Validation plots can be found in the `SPIE paper <https://ui.adsabs.harvard.edu/#abs/2016SPIE.9910E..1AY/abstract>`_.
       Note the model does not include any "weather" (e.g., clouds, variable OH emission). There is an option to change the solar activity, which scales the airglow component.

System Throughputs
-------------------

All of the system throughputs are recorded in the `throughputs baseline <https://github.com/lsst/throughputs/tree/main/baseline>`_.
This information is copied from the System Engineering database.  More information can be found in the `README <https://github.com/lsst/throughputs/blob/main/baseline/README.md>`_ file.
In that directory you can find a graphical representation of the total throughput along with datafile representing each component and the total throughput. The file representing each throughput curve is
referenced below.

.. list-table::
   :widths: 10 10 10 15 15
   :header-rows: 1
   :class: tight-table

   * - Effect
     - Implementation
     - Data / Model Source
     - Short description
     - Validation Page and Notebooks

   * - Camera QE and AR
     - detector.dat
     - SysEngineering 1.7
     - Expected response (QE response + AR coatings) of the CCDs.  Currently, these numbers are joint minimums of the responses of the two vendor's sensors (e2V and ITL).
     -

   * - Lens
     - lens[1,2,3].dat
     - SysEngineering 1.7
     - Combination of fused silicon and BroadBand AntiReflective (BBAR) coatings
     -

   * - Filters
     - filter[u,g,r,i,z,y].dat
     - SysEngineering 1.7
     - Filter throughput in each band. We expect an update with as-built numbers soon.

     -
   * - Mirrors
     - m[1,2,3].dat
     - SysEngineering 1.7
     - Reflectivity curve for each mirror
     -

   * - Atmosphere
     - atmosphere_std.dat and atmosphere_10.dat
     - SysEngineering 1.7
     - MODTRAN based standard US atmosphere with Aerosols added.
     - Both typical (standard) throughput with airmass X=1.2 and optimum X=1.0 files are provided

   * - Total
     - total[u,g,r,i,z,y].dat
     - SysEngineering 1.7
     - The total throughput by band
     -

.. note::

  The hardware[u,g,r,i,z,y].dat files contain everything except atmospheric effects.  Multiplying those with the atmosphere results in a total throughput curve. Atmospheric throughputs for a large set of atmospheres can be found in https://github.com/lsst/throughputs/tree/main/atmos.

Atmospheric PSF model
---------------------

.. note::

   To be added.  See the :ref:`atmospheric psf <AtmosphericPSF-label>` config section on how to configure the atmosphere and in https://iopscience.iop.org/article/10.3847/1538-4365/abd62c for a description of the model.


Optical model
-------------

You have a choice of parametric of fully raytraced optics via *batoid*.

.. list-table::
   :widths: 10 10 10 15 15
   :header-rows: 1
   :class: tight-table

   * - Effect
     - Implementation
     - Data / Model Source
     - Short description
     - Validation Page and Notebooks

   * - Vignetting
     - vignetting.py
     -
     - Either emergent for raytraced objects or via a function for those produced via FFT.  The sky backround is vignetted via a function.
     -

   * - Ghosts
     - Not currently possible.  Planned in the next version of *imSim* when the abilty to handle raytraced light across multiple sensors being processed in parallel will be updated.
     -
     -
     -

   * - Spider Diffraction Spikes
     - diffraction.py, diffraction_fft.py photon_ops.py
     -
     - Statistical Diffraction during batoid ray tracing or parametric model with FFT.
     - Page link :doc:`validation/diffraction`

   * - Aberrated optics
     - optical_system.py telescope_loader.py
     -
     - Either a parametric model built around a sensitivity matrix, or a fully raytraced optical model with FEA and pertubation controls using several degrees of freedom including bending modes and physical actuators.
     - Page link :doc:`validation/aberrated-optics` for the parametric case :doc:`lsst-optical` for the raytraced case.


Calibration Products
--------------------

.. note::

   To be added.  Description of flats, darks etc need to be added.

Detailed Description of Physical Effects Implemented in imSim
-------------------------------------------------------------

Several of the effects listed above have detailed pages descrubing how the model was constructed and what data was used.

.. toctree::
   :maxdepth: 2
   :glob:

   validation/*
