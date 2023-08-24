#################
The Config System
#################

*imSim* utilizes the *GalSim* config system to configure itself.  This choice was made for several reasons.  As *GalSim* is used to render the actual images, using *GalSim* itself is a natural choice for configuration. Perhaps most importantly, by implementing the *imSim* code as a *GalSim* module it becomes possible to use it freely with other *GalSim* based modules.  So, other programs in the *GalSim* ecosystem can also call and configure the *imSim* module and could (for example) utilize the *imSim* electronics readout algorithms as part of their functionality.

The *GalSim* config system is described in the `GalSim documentation <http://galsim-developers.github.io/GalSim/_build/html/config.html>`__. You should start there to understand the basic functionality of the config system.  If you look at the ``config/imsim-config.yaml`` distributed with *imSim* you can see the *imSim's* default configuration.  When you read in a YAML file you will be adding to and modifying these results.



    You should read the `GalSim documentation <http://galsim-developers.github.io/GalSim/_build/html/config.html>`__ for a full description of the Config system.  However, there is one very important concept that is not always immediately obvious to users but is extremely important to understand the section below.

    Consider this seemingly very simple piece of YAML configuration:

    .. code-block:: yaml
        :caption: Variable Use

        input:

            tree_rings:
                file_name: "tree_ring_parameters_2018-04-26.txt"


        image:
            type: LSST_Image

            sensor:
                type: Silicon
                treering_center: { type: TreeRingCenter, det_name: $det_name }
                treering_func: { type: TreeRingFunc, det_name: $det_name }

        output:
            type: LSST_CCD

            camera: LsstCam


    What is happening here is that the ``tree_rings`` command in the ``input`` section is reading a text file included with *imSim*. Upon read in it creates two types of Python dictionaries which are keyed off of the detector name. One dictionary returns the center of the tree rings, one the form of the function.  In the ``sensor`` description, it expects a center and function, and we pass the two dictionaries created earlier with a detector key to them as input. The ``$det_name`` is a variable you can use that was added by the ``LSST_CCD`` keyword in the output section.  It knows what detector is currently being written out.   You will see variations on this pattern repeated many times in the config files.

Each feature in the *imSim* module is implemented as a class which can be configured.  In this section, we list the *imSim* classes along with the config options that you can set.

Input types
===========

These classes define the user configurable parts of the ``input`` section of the configuration YAML files.  Input items are either files that need to be read or opened once at the start (e.g. the sky catalog or tree-ring data file or a checkpoint file), or built once at the start to be used repeatedly (e.g. the telescope configuration or the atmospheric PSF).  Here we list the input types which are added my the *imSim* module.

Instance catalogs
-----------------

Instance catalogs are text files best suited to making small handcrafted inputs. For legacy and compatibility purposes, they follow the format of the *PhoSim* program inputs which are documented on `PhoSim Web Site <https://bitbucket.org/phosim/phosim_release/wiki/Instance%20Catalog>`__.  The file should contain a a header including metadata describing the observation and a list of sources.

Key Name: instance_catalog
^^^^^^^^^^^^^^^^^^^^^^^^^^

Required keywords to set:
"""""""""""""""""""""""""

    * ``file_name`` = *str_value* (default =  None)  The name of the text file containing the instance catalog entries.

Optional keywords to set:
"""""""""""""""""""""""""
    * ``sed_dir`` = *str_value* (default = None)  The directory that contains the template SED files for objects.  Typically this is set via an enviroment variable.
    * ``edge_pix`` =  *float_value* (default = 100.0) How many pixels are objects allowed to be past the edge of the sensor to consider their light.
    * ``sort_mag`` = *bool_value*  (default = True) Whether or not to sort the objects by magnitude and process the brightest objects first.
    * ``flip_g2`` = *bool_value* (default = True) If True, apply a minus sign to the g2 lensing parameter used to shear the objects. *PhoSim* uses a different shear convention, so this default converts the shear values to the convention used by *imSim*, *GalSim*, *TreeCorr*, etc.
    * ``min_source`` = *int_value* (optional) if set, skip simulating any sensor which does not have at least min_sources on it.
    * ``skip_invalid`` = *bool_value* (default = True) Check the objects for some validity conditions and skip those which are invalid.

.. _InstCat-label:

Instance Catalog Object Type
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once an instance catalog has been read in, objects from it can be used as an object type for the top level GalSim ``gal`` field like this:

.. code-block:: yaml

    # Define the galaxy (or delta function) to use
    gal:
        type: InstCatObj

Optional keywords to set:
"""""""""""""""""""""""""

    * ``index`` = *int_value* (optional) by default all of the objects in the file will be processed, but here you can specify some subset of of index numbers yourself of exactly which objects should be read.  E.g. this could be a single integer or a list of integers or a galsim Sequence type, etc.
    * ``num`` =  *int_value* (default = 0) If you have multiple instance_catalog input items defined in the config file, this option will allow you specify which one you should use. The default is the first and usually only one.

.. _InstCatWorld-label:

Instance Catalog World position
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once an instance catalog has been read in, the  world position (i.e. the celestial coordinate on the sky) as defined in the file can be specified to the top level GalSim ``stamp`` field like this:

.. code-block:: yaml

    # Get the world position for the current catalog object
    world_pos:
        type: InstCatWorldPos


Optional keywords to set:
""""""""""""""""""""""""""

    These are the same as for ``InstCatObj`` above.

Instance Catalog SED type
^^^^^^^^^^^^^^^^^^^^^^^^^

An instance catalog contains lines with SED files listed. This allows you to load one of those SEDs from the file.

Required keywords to set:
"""""""""""""""""""""""""

Optional keywords to set:
"""""""""""""""""""""""""
    * ``index`` = *int_value* (optional) by default all of the objects in the file will be processed, but here you can specify some subset of of index numbers yourself of exactly which objects should be read.  E.g. this could be a single integer or a list of integers or a galsim Sequence type, etc.
    * ``num`` = *int_value* (default = 0) If you have multiple instance_catalog input items defined in the config file, this option will allow you specify which one you should use. The default is the first and usually only one.


Sky Catalogs
------------

Instance catalogs are text based and utilize a lot of disk space for the information contained in them. Also, one instance catalog is needed for each visit, even if those visits take place at the exact same position on the sky.  This causes enormous duplication of information.  Instead, for large area simulations, *imSim* utilizes an API based system known as `skyCatalogs <https://github.com/LSSTDESC/skyCatalogs>`__.  The *skyCatalog* presents a unified interface to *imSim* via an API of databases that contain all of the object in the sky.  By configuring *imSim* to use the *skyCatalog* API only metadata for the visits are needed.  *imSim* will retrieve a list of all of the objects it needs to render through the interface.  *skyCatalogs* can contain static and transient information and databases exist both for synthetic skies and true sources of information such as the Gaia catalog.  The *skyCatalog* can also serve as a source of truth information when later analyzing simulated data.

Key Name: sky_catalog
^^^^^^^^^^^^^^^^^^^^^

Required keywords to set:
"""""""""""""""""""""""""

  * ``file_name`` = *str_value* (default =  None)  The name of the yaml text file which specifies sky catalog positions.
  *  ``band`` = *str_value* (default = None)  The name of the LSST band to use.

Optional keywords to set:
"""""""""""""""""""""""""

  * ``edge_pix`` =  *float_value* (default = 100.0) How many pixels is the buffer region were objects are allowed to be past the edge of the sensor.
  * ``obj_types`` : *list*  (optional) List or tuple of object types to render, e.g., ('star', 'galaxy').  If None, then consider all object types.
  * ``max_flux`` = *float_value* (default = None) If object flux exceeds max_flux, the return None for that object. if max_flux == None, then don't apply a maximum flux cut.
  * ``apply_dc2_dilation`` = *bool_value* (default False) Flag to increase object sizes by a factor sqrt(a/b) where a, b are the semi-major and semi-minor axes, respectively. This has the net effect of using the semi-major axis as the sersic half-light radius when building the object, which is how the DESC DC2 simulations defined the half-light radius.  This will only be applied to galaxies.
  * ``approx_nobjects`` = *int_value* (default None) Approximate number of objects per CCD used by galsim to set up the image processing.  If None, then the actual number of objects found by skyCatalogs, via .getNObjects, will be used. If given, this should generally be greater than or equal to the actual number of objects.  It's even ok to be much higher (10x or more is fine) than the actual number of objects.
  * ``mjd`` = *float_vaue*  MJD of the midpoint of the exposure.

Sky Catalog Object Type
^^^^^^^^^^^^^^^^^^^^^^^

    The ``SkyCatObj`` is used as in the :ref:`InstCatObj <InstCat-label>` case above.

Sky Catalog World Position
^^^^^^^^^^^^^^^^^^^^^^^^^^

    The ``SkyCatWorldPos`` is used as in the :ref:`InstCatWorldPos <InstCatWorld-label>` case above.

OpSim Data
----------

Many configuration items require knowledge of various specific details of the observation.  E.g. the time of exposure, the filter being employed, the direction that the telescope is pointing etc.  These can each be specified individually each time they are required, but that can be tedious and prone to consistency errors.  We recommend keeping all of this information in a single place and have everything use value types that read the values from that.  The two most common sources of this information are the top of an instance catalog and the output of a Rubin Operational Simulator simulation, either of which can be read in using the OpSim_data input type.

This input type allows you to specify file inputs which contain this information.

Key Name: opsim_data
^^^^^^^^^^^^^^^^^^^^
Note that several metadata keywords are required to be specified in the file.  They include: *rightascension, declination, mjd, altitude, azimuth, filter, rotskypos, rottelpos, dist2moon, moonalt, moondec, moonphase, moonra, nsnap, seqnum, obshistid, seed, seeing, sunalt, and , vistime.*

Required keywords to set:
""""""""""""""""""""""""""

    * ``file_name`` = *str_value* (default =  None)  Either the name of an instance catalog, the top of which contains the required metadata information, or the name of an OpSim database file. (Note that in the former case, any object information is ignored by this input item.)
    *  ``visit`` = *int_value* (default = None) The visit number. This is only relevant if you are reading an *OpSim* database file.

Optional keywords to set:
"""""""""""""""""""""""""

    * ``snap`` = *int_value* (default = 0) Which snap to use if multiple snaps are being simulated.
    * ``image_type`` = *string_value* (default = 'SKYEXP') The type of exposure to be taken. Other options include 'FLAT' and 'BIAS'.
    * ``reason`` = *string_value* (default='survey') The reason the exposurew was taken. Other options include 'calibration'



OpSim Value Type
^^^^^^^^^^^^^^^^^

Once the opsim data has been specified you can use those values in other parts of the YAML file by specifying keys which have been set. An example is shown below:

.. code-block:: yaml

    atm_psf:
        # This enables the AtmosphericPSF type for the PSF

        airmass: { type: OpsimData, field: airmass }
        rawSeeing:  { type: OpsimData, field: rawSeeing }
        band:  { type: OpsimData, field: band }

The ``field`` key is required.

Telescope Configuration
-----------------------

The optical system of the telescope can be configured including optical aberrations, the state of active optics system, variations due to temperature etc.  Individual actuators and other elements of the optics system can also be configured as an input before the simulation starts.

If the photons are ray-traced through the optics with the `Batoid package  <https://github.com/jmeyers314/batoid>`__ photons will be modified by the changes as they propagate through the optics.  See :ref:`the stamp keyword <stamp-label>` below for details. For more details on the extensive control over the perturbation and FEA parameters of the optical system please refer to :ref:`the optical system section <optical-system-label>`

Key Name: telescope
^^^^^^^^^^^^^^^^^^^^

Required keywords to set:
""""""""""""""""""""""""""

    * ``file_name`` = *str_value* (default =  None)  The name of a yaml file describing the Rubin optics distributed with the batoid package.  The filename can be constructed via the config system in the YAML file as in the following example.

    .. code-block:: yaml

        telescope:
            file_name:
                type: FormattedStr
                format : LSST_%s.yaml
                items:
                    - { type: OpsimData, field: band }


Optional keywords to set:
"""""""""""""""""""""""""

    * ``rotTelPos`` = *angle_value* (default = None) The angle of the camera rotator.
    * ``camera`` = *str_value* (default = 'LSSTCam') The name of the camera to use.
    * ``perturbations:`` = YAML dictionary (default = None)  See :ref:`the optical system section <optical-system-label>` for documentation.
    * ``fea:`` = YAML dictionary (default = None)  See :ref:`the optical system section <optical-system-label>` for documentation


Sky Model
---------

Including the skyModel will load the Rubin Simulation Sky Model from the rubin-sims package.  If you have loaded this module, you will will be able top to refer the ``skyLevel`` variable in the image section to set the brightness of the sky. You can also use the ``apply_sky_gradient`` option in the image section to make the sky level vary over each sensor.

Key Name: sky-model
^^^^^^^^^^^^^^^^^^^^
Required keywords to set:
""""""""""""""""""""""""""

    * ``exp_time`` = *float_value* (default =  None)  The exposure time in seconds.
    * ``mjd`` = *float_value*  THE MJD of the observation.

Optional keywords to set:
"""""""""""""""""""""""""

    * ``eff_area`` = *float_value* (default = RUBIN_AREA) Collecting area of telescope in cm^2. Default: Rubin value from https://confluence.lsstcorp.org/display/LKB/LSST+Key+Numbers


SkyLevel Value Type
^^^^^^^^^^^^^^^^^^^

Once the Rubin sky-model has been specified you can use the calculated sky level in other parts of the YAML file. An example is shown below:

.. code-block:: yaml

    image:
        type: LSST_Image

        sky_level: { type: SkyLevel }  # Computed from input.sky_model.
        apply_sky_gradient: True


Atmospheric PSF
----------------

The class is used to create the PSF which is induced by the atmosphere.  There are two parametric PSFs available: a double Gaussian and a Kolmogorov PSF. The ``AtmosphericPSF`` type is a fully ray-traced turbulent atmosphere with multiple atmospheric layers.  Additionally, you can optionally add a parametric PSF screen which simulates the Rubin Optics.

.. _AtmosphericPSF-label:

Key Name: atmosphericPSF
^^^^^^^^^^^^^^^^^^^^^^^^

This keyword enables an atmospheric PSF with 6 randomly generated atmospheric screens.  Photons are raytraced through this atmosphere to produce a realistic atmospheric PSF.

.. warning::

    You should not attempt to use the option to add parametric optics (through the ``doOpt`` option) if you are using fully ray-traced optics.  Otherwise, you will simulate the optics twice.  See :ref:`the stamp keyword <stamp-label>` below how to activate the ray-traced mode.


Required keywords to set:
""""""""""""""""""""""""""

    * ``airmass`` = *float_value* The aimass in the direction of the pointing.
    * ``rawSeeing`` = *float_value*  The FWHM seeing at zenith at 500 nm in arc seconds
    * ``band`` = *str_value* The filter band of the observation.
    * ``boresight`` = *sky_value* The CelestialCoord of the boresight of the observation.


Optional keywords to set:
"""""""""""""""""""""""""

    * ``t0`` = *float_value* (default = 0.0) Exposure time start in seconds.
    * ``exptime`` = *float_value*  (default = 30.0) Exposure time in seconds.
    * ``kcrit`` = *float_value* (default = 0.2) Critical Fourier mode at which to split first and second kicks.
    *  ``screen_size`` = *float_value* (default = 819.2) Size of the phase screens in meters.
    *  ``screen_scale`` = *float_value* (default = 0.1) Size of phase screen "pixels" in meters.
    *  ``doOpt`` = *bool_value* (default = False) Add in optical phase screens? *SEE WARNING ABOVE*
    *  ``nproc`` = *int_value* (default = None)  Number of processes to use in creating screens. If None (default), then allocate one process per phase screen, of which there are 6, nominally.
    *  ``save_file`` = *str_value* (default = None) A file name to use for saving the built atmosphere.  If the file already exists, then the atmosphere is read in from this file, rather than being rebuilt.


Key Name: DoubleGaussianPSF
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A wavelength and position-independent Double Gaussian PSF. This specific PSF comes from equation(30) of the signal-to-noise document (LSE-40), which can be found at http://www.astro.washington.edu/users/ivezic/Astr511/LSST_SNRdoc.pdf.

Required keywords to set:
""""""""""""""""""""""""""

    * ``fwhm`` = *float_value*  The full width at half max of the total PSF in arc seconds.


Optional keywords to set:
"""""""""""""""""""""""""

    * ``pixel_scale`` = *float_value* (default = 0.2) The pixel scale of the sensor in arc seconds.


Key Name: KolmogorovPSF
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This PSF class is based on David Kirkby's presentation to the DESC Survey Simulations working group on 23 March 2017.

    https://confluence.slac.stanford.edu/pages/viewpage.action?spaceKey=LSSTDESC&title=SSim+2017-03-23

    (you will need a SLAC Confluence account to access that link)

Required keywords to set:
""""""""""""""""""""""""""

    * ``airmass`` = *float_value* The aimass in the direction of the pointing.
    * ``rawSeeing`` = *float_value*  The FWHM seeing at zenith at 500 nm in arc seconds
    * ``band`` = *str_value* The filter band of the observation.



Tree Rings
----------

Tree-rings are a silicon sensor effect induced by internal electric fields in the 3D structure of the silicon CCD.  The fields are created by internal variations in dopant concentration that form while the silicon boule is being grown.  You can find more about *imSim*'s implementation of tree rings in :ref:`the Tree Ring validation section <tree-ring-label>`. This keyword tells imSim where to find the data file which describes the parameters to be used when the effect is turned on. It creates dictionaries that can be used by the LSST sensor description in :ref:`LSST Camera <LSST-Camera-label>` section below.

Key Name: tree_rings
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Required keywords to set:
""""""""""""""""""""""""""

    * ``file_name`` = *str_value*  A file name that contains the parameters of the tree ring model for each sensor.


Optional keywords to set:
"""""""""""""""""""""""""

    * ``only_dets`` = *List* (optional) Only read in the models for the listed sensors in order to save time on startup.



TeeeRingCenter and TreeRingFunction Value Type
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the tree ring models have been read in, you can use them in other parts of the YAML file. The model is keyed on a detector name. An example is shown below:

.. code-block:: yaml

    image:
        type: LSST_Image

        sensor:
            type: Silicon

            treering_center: { type: TreeRingCenter, det_name: $det_name }
            treering_func: { type: TreeRingFunc, det_name: $det_name }


Checkpointing
-------------

As imSim runs, if this option is turned on, it will periodically check-point its progress, writing out its interim output as it runs.  Then, on re-running, it will use this output so as to not redo previous calculations.  This has two main use cases.  The first is the case where you are rerunning several times. This will avoid recreating sensors that have already been simulated. The 2nd main use case is for if a job is stopped before it completes.  This is particularly common when using a batch system with time-limits.  This option allows you to restart your job and pick where you left off.  These keywords tell *imSim* where to find the checkpoint files and how they are named.

.. warning::

    Be careful to manually delete any check-point files if you have made any changes to to the configuration between runs.  Currently, *imSim* only checks if a file for a individual sensor already exists.


Key Name: checkpoint
^^^^^^^^^^^^^^^^^^^^^

Required keywords to set:
""""""""""""""""""""""""""

    * ``file_name`` = *str_value* A file name to store the checkpoint for each sensor.

    Be careful to give a unique name for each possible checkpoint. Here for example is a YAML code block

    .. code-block:: yaml

        file_name:
          type: FormattedStr
          format : checkpoint_%08d-%s.hdf
          items:
              - { type: OpsimData, field: observationId }
              - "$det_name"

Optional keywords to set:
"""""""""""""""""""""""""

    * ``dir`` = *str_value* (default = None) Put the files in a directory with this name.


Image types
===========

These classes define how to draw images.  The basic *GalSim* image types include 'Single', 'Tiled', and 'Scattered'.  *imSim* adds a new type of image that can be used along with a new type of WCS object that uses ray-traced photons to map out a TAN-SIP WCS.

.. _LSST-Camera-label:

LSST Images
-----------

Key name: LSST_Image
^^^^^^^^^^^^^^^^^^^^

The ``LSST_Image`` type is a version of the *GalSim* "Scattered Image" image class that has been modified to understand how to draw the Rubin sky background and how to apply effects such as vignetting to the sky and certain bright objects.  There are extra optional keywords you can use with the ``LSST_Image`` type.

Required keywords to set:
"""""""""""""""""""""""""

    * ``xsize`` = *int_value* The size of the image in the X direction (i.e. the number of columns)
    * ``ysize`` = *int_value* The size of the image in the Y direction (i.e. the number of rows)


Optional keywords to set:
"""""""""""""""""""""""""

    * ``dytpe`` = *str_value* (default = None) allows you to set numpy.dtype  for the underlying data in the image.
    *  ``apply_sky_gradient`` = *bool_value* (default = False) If True vary the sky background level linearly across the sensors to match the expected flux at the four corners of the at each sensor.
    * ``apply_fringing`` = *bool_value* (default = False) If True, apply a fringing pattern to the sky background level.  This is only currently possible for y band observations, so a recommended value for this parameters is ``$band == 'y'``.
    *  ``camera`` = *str_value* (default = 'LsstCam') name of the camera such as ``LsstCam``. Other options include 'LsstComCam' and 'LsstCamImSim'.
    *  ``nbatch`` = *int_value* (default = 10) if checkpointing, otherwise the value is 1. How many batches of objects to run.  If checkpointing, the checkpoint will be written after finishing each batch.
    * ``boresight`` = *sky_value* (required if ``apply_fringing`` is True) The CelestialCoord of the boresight of the observation.

*imSim* also registers a new type of WCS object. When this WCS is chosen the `Batoid ray-tracing package  <https://github.com/jmeyers314/batoid>`__ traces a set of rays through the optics and fits the result to create a WCS which accurately represents the current state of the telescope optics.

.. note::

    In order to use the Batoid WCS you need to have declared a telescope object in the ``input`` section.

Batoid WCS Type
^^^^^^^^^^^^^^^

Required keywords to set:
""""""""""""""""""""""""""
    * ``boresight`` = *sky_value* The CelestialCoord of the boresight of the observation.
    * ``obstime`` = *str_value*  The time of the observation either as a string or a astropy.time.Time instance
    * ``det_name`` = *str_value* The name of the sensor for this WCS.

Optional keywords to set:
"""""""""""""""""""""""""

    * ``camera`` = *str_value* (default = 'LsstCam') The name of the camera to use.
    * ``temperature`` = *float_value* (default = 280K) Ambient temperature in Kelvin.
    * ``pressure`` = *float_value* (default = calculated from Rubin height) Ambient pressure in kPa.
    *  ``H2O_pressure`` = *float_value* (default = 1 kPa) Water vapor pressure in kPa.
    *  ``wavelength`` = *float_value* (default = effective wavelength of the bandpass of the observation) wavelength of photon to use in nanometers.
    *  ``order`` = *int_value* (default = 3) SIP polynomial order for WCS fit.

Rubin Bandpass
^^^^^^^^^^^^^^

*imSim* also registers a new Bandpass type, representing the Rubin filter throughputs for each band pass: u, g, r, i, z, or y.  An example is shown below.

.. code-block:: yaml

    image:
        type: LSST_Image

        bandpass: { type: RubinBandpass, band: r }

Required keywords to set:
""""""""""""""""""""""""""
    * ``band`` = *str_value* The name of the band.  Must be one of {u, g, r, i, z, y}.


Key Name: LSST_Flat
^^^^^^^^^^^^^^^^^^^

 *imSim* also supplies a ```LSST_Flat`` image type.  Calibration flats have extremely high background levels and special file, memory and SED handling are employed in this case in order to optimize computational efficiency.

Required keywords to set:
""""""""""""""""""""""""""
    * ``counts_per_pixel`` = *float_value* Background count level per pixel
    * ``xsize`` = *int_value* The size of the image in the X direction (i.e. the number of columns)
    * ``ysize`` = *int_value* The size of the image in the Y direction (i.e. the number of rows)


Optional keywords to set:
"""""""""""""""""""""""""

    * ``max_count_per_iteration`` = *float_value* (default = 1000) How many photons per pixel to add per iteration.
    * ``buffer_size`` = *int_value* (default = 5) Add a border region with this many pixels in each section being worked on.  A buffer of 5 was found to not introduce noticeable artifacts at the section boundaries.
    * ``nx`` = *int_value* (default = 8)  The number of sections to split the sensor into in X in order to control memory usage.
    * ``ny`` = *int_value* (default = 2)  The number of sections to split the sensor into in Y in order to control memory usage.

.. _stamp-label:

StampTypes
==========

The Stamp drawing code does the main work to actually render the image of an astronomical object.   *imSim* adds the ``LSST_Silicon`` type which understands how to draw objects in the LSSTCam sensors including accounting for absorption in the atmosphere, integrating the SEDs of the objects with the chosen filter, ray-tracing photons through the optical system, adding diffractive spikes from the telescope spider as the camera rotates, automatically using various approximations for both very bright and very dim objects etc.  Those options are set with the parameters below.

If an astronomical object is too bright, by default *imSim* will generate the objects with a FFT instead of via photon shooting in order to save computation time.  This objects are typically saturated and not usable for analysis in any case.

LSST Silicon Sensor Stamps
--------------------------

Stamp Type: LSST_Silicon
^^^^^^^^^^^^^^^^^^^^^^^^^


Required keywords to set:
"""""""""""""""""""""""""

    * ``det_name`` = *str_value* (only required if doing vignetting) The name of the detector.

      .. note::
            If using the output type LSST_CCD, then ``det_name`` will automatically be added
            to the ``eval_variables`` section for you.  In this case, you can simply use
            ``det_name: '$det_name'``.  If not using LSST_CCD, then the value should be of a
            form such as R22_S11.  (This is the central CCD in the focal plane.)

Optional keywords to set:
"""""""""""""""""""""""""
    * ``fft_sb_threshold`` = *float_value* (default = 0) Over this number of counts, use a FFT instead of photon shooting for speed.  If set to 0 don't ever switch to FFT.
    * ``max_flux_simple`` = *float_value* (default = 100) If the flux is less than this value use a simple SED and apply other speed ups.
    * ``method`` = *str_value* (default = 'auto') Choose between automatically deciding whether to use a FFT of photon shooting ('auto') or manually choose between 'fft' and 'phot'.
    * ``maxN`` = *int_value* (detault = 1.0e6) Set limit on the size of photons batches when drawing the image.
    * ``camera`` = *str_value* (default = 'LsstCam') The name of the camera to use.
    * ``diffraction_fft`` = *dict* (optional) Parameters for implementing the diffraction spikes of FFT-rendered objects. See below.
    * ``airmass`` = *float_value* (default = 1.2) The airmass to use when estimating the stamp size to use for FFTs
    * ``rawSeeing`` = *float_value* (default = 0.7) The FWHM seeing at zenith at 500 nm in arc seconds to use when calculating the stamp size to use for FFTs.
    * ``band`` = *str_value* (default = 'r') The filter band of the observation to use when estimating the stamp size to use for FFTs.

Key Name: diffraction_fft:
^^^^^^^^^^^^^^^^^^^^^^^^^^

Required keywords to set:
"""""""""""""""""""""""""

    * ``exptime`` = *float_value*  The time of the exposure.
    * ``azimuth`` = *angle_value*  The azimuth angle of the observation.
    * ``altitude`` = *angle_value*  The altitude angle of the observation.
    * ``rotTelPos`` = *angle_value* (default = None) The angle of the camera rotator.


Optional keywords to set:
"""""""""""""""""""""""""

    * ``enabled`` = *bool_value* (default = True) When doing FFTs, also calculate parametric diffraction spikes from the spider.
    * ``spike_length_cutoff`` = *int_value* (default = 4000) In a FFT the size of the telescope spike length
    * ``brightness threshold`` = *float_value* (default = Set by CCD full well value) In a FFT the value of a pixel that will cause it to be replaced with a diffraction spike.
    * ``latitude`` = *angle_value* (default = Rubin Location) Geographic latitude of the observatory.


Finally, there are a set of operations that can act on photons in *GalSim*.  These are put together in a list and then all of the photons have those operations act on them in turn.  This list of photon-operations are specified in the stamp section.  You can read more about them in the *GalSim* documentation covering `GalSim Photon Ops <http://galsim-developers.github.io/GalSim/_build/html/config_stamp.html#photon-operators-list>`__.  *imSim* adds a new set of Photon Operators to ray-trace the photons through the optical system using the `Batoid package  <https://github.com/jmeyers314/batoid>`__.

If you do not turn these on, you should use the parameterized optics available in the atmospheric PSF instead.  You have three choices for the implementation of the Rubin optics:

Photon Operation Type
---------------------

type: **RubinOptics**
^^^^^^^^^^^^^^^^^^^^^
Ray-trace photons though the Rubin optical system using *batoid*.

Required keywords to set:
""""""""""""""""""""""""""

    * ``boresight`` = *sky_value*  The CelestialCoord of the boresight of the observation.

Optional keywords to set:
""""""""""""""""""""""""""

    * ``camera`` = *str_value* (default = 'LsstCam') The name of the camera to use.

type: **RubinDiffraction**
^^^^^^^^^^^^^^^^^^^^^^^^^^

Apply diffractive effects.

.. warning::

    This only applies to objects rendering using photon shooting.  To be consistent, if you use this, you should also set diffraction_fft as described above.)

Required keywords to set:
"""""""""""""""""""""""""

    * ``azimuth`` = *angle_value*  The azimuth angle of the observation.
    * ``altitude`` = *angle_value*  The altitude angle of the observation.
    * ``latitude`` = *angle_value*  The latitude of the observatory.

Optional keywords to set:
""""""""""""""""""""""""""

    * ``disable_field_rotation`` = *bool_value* (default = False) Do not consider the effect of the rotation of the camera relative to the spiders of the telescope during the exposure.


type: **RubinDiffractionOptics**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
An optimized combination of **RubinOptics** and **RubinDiffraction** that combines shared calculations to for computational efficiency.

Required keywords to set:
"""""""""""""""""""""""""

    * ``boresight`` = *sky_value*  The CelestialCoord of the boresight of the observation.
    * ``azimuth`` = *float_value*  The azimuth angle of the observation.
    * ``altitude`` = *float_value* The altitude angle of the observation.


Optional keywords to set:
""""""""""""""""""""""""""

    * ``camera`` = *str_value* (default = 'LsstCam') The name of the camera to use.
    * ``latitude`` = *angle_value* (default = latitude of Rubin) The latitude of the observatory.
    * ``disable_field_rotation`` = *bool_value* (default = False) Do not consider the effect of the rotation of the camera relative to the spiders of the telescope during the exposure.


Output types
============

The output field is used to specify where to write output files and what format they should be.  There are several possibilities, including FITS files before and after electronics readout, and various types of truth information.  *imSim* adds the ``LSST_CCD`` type, an extra electronics ``readout`` type, and an ``opd`` type for writing out optical path difference images.

*imSim*  understands how to write "eimage" files which are true representations of the electrons in the CCD including signals from the objects and cosmic rays with important physics effects such as the brighter-fatter effect and tree-rings applied.

It can also write "amp" files. These are fully readout electronics files with one amplifier per FITS HDU with all of the proper headers needed to be processed by the Rubin Science Pipelines.  Both of these output formats can be examined with standard tools such as *ds9*.

There are also several extra outputs available to the user including a centroid file containing the true position of the rendered sources generated with the standard *GalSim* "truth" extra output type, a list of optical path differences in the optical system, and a map of surface figure errors.

LSST CCD Sensor output and readout
----------------------------------

Output Type: LSST_CCD
^^^^^^^^^^^^^^^^^^^^^

For modifying the ``LSST_Image`` by adding cosmic rays and relevant headers and optionally writing out the "eimage" FITS files via the standard *GalSim* output.

Optional keywords to set:
""""""""""""""""""""""""""

    * ``cosmic_ray_rate`` = *float_value* (default = 0) The rate of cosmic rays per second in a sensor.
    * ``cosmic_ray_catalog`` = *str_value* (default = Distributed with *imSim*) A file containing cosmic ray images to paint on the sensor.
    * ``header`` = *dict* (optional) Extra items to add to the FITS header in the output files.
    * ``nproc`` = *int_value* (default=1) How many processes to use for working on multiple CCDs at a time.
    * ``nfiles`` = *int_value* (default=189) How many CCDs to build. ("files" here refers to the eimage files that are produced by the standard processing)  The default is to do the full focal plane.
    * ``file_name`` = *str_value* (optional) The name of the eimage file to write out.  If not present, the eimage will not be written to disk.  (In which case, you probably want to include some other output object, such as readout; see below.)
    * ``dir`` = *str_value* (optional) The directory to write the output files to (default is the current directory)
    * ``camera`` = *str_value* (default = 'LsstCam') The name of the camera to use.
    * ``exptime`` = *float_value* (default=30) The exposure time.
    * ``det_num`` = *int_value* (default = sequence from 1 to 189) The detector numbers to use for each output image.


Extra Output Type: readout
^^^^^^^^^^^^^^^^^^^^^^^^^^

Parameters modifying the LSSTCamera electronics readout and writing the amplifier files.

Optional keywords to set:
""""""""""""""""""""""""""

    * ``file_name`` = *str_value* (default = None) The file name for amplifier file output(s).
    * ``camera`` = *str_value* (default = 'LsstCam') The name of the camera to use.
    * ``readout_time`` = *float_value* (default = 2.0) The camera readout time in seconds.
    * ``dark_current`` = *float_value* (default = 0.02) The dark current in electrons per second.
    * ``bias_level`` = *float_value* (default = 1000.0) Bias readout level in ADUs.
    * ``scti`` = *float_value* (default = 1.0e-6) The serial CTI
    * ``pcti`` = *float_value* (default = 1.0e-6) The parallel CTI
    * ``full_well`` = *float_value* (default = 1.0e5) Thu number of electrons needed to fill the sensor well.
    * ``read_noise`` = *float_value* (default given by camera object) The read noise in ADU.

Extra Output Type: opd
^^^^^^^^^^^^^^^^^^^^^^

Write out the optical path differences images to study raytracing behaviour.

.. note::

        The OPD image coordinates are always aligned with the entrance pupil,
        regardless of the value of rotTelPos.  The OPD values are in nm, with
        NaN values corresponding to vignetted regions.  The OPD is always
        computed for the fiducial telescope focal plane height; i.e., it ignores any detector-by-detector offsets in focal plane height.

Required keywords to set:
""""""""""""""""""""""""""

    * ``file_name`` = *str_value* The name of the file to write OPD images to.
    * ``fields`` = *list*  List of field angles for which to compute OPD.  Field angles are specified in the (rotated) coordinate system of the telescope's entrance pupil (usually the primary mirror).

Optional keywords to set:
""""""""""""""""""""""""""

    * ``rotTelPos`` = *angle_value* (default = None) The angle of the camera rotator.
    * ``nx`` = *int_value* (default = 255) Size of the OPD image in pixels.
    * ``wavelength`` = *float_value* (default = None) avelength of light in nanometers.  If not specified, then the wavelength will be taken from the current bandpass.
    * ``projection`` = *str_value* (default = 'postel') Projection mapping field angles to spherical coordinates. See batoid documentation for more details.
    * ``sphereRadius`` = *float_value* (default = None) Radius of reference sphere in meters.  If not specified, then the radius will be taken from the telescope object.
    *  ``reference`` = *str_value* (default = 'chief') Either 'chief' or 'mean'.  See batoid documentation for more details.
    *  ``eps`` = *float_value* (default = None) Annular Zernike obscuration fraction.  If not specified, then the value will be taken from the telescope object.
    *  ``jmax`` = *int_value* (default = 28) Maximum Zernike Order.

