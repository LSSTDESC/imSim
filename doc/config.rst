#################
The Config System
#################

*imSim* utilizes the *imSim* config system to configure itself.  This choice was made for several reasons.  As *GalSim* is used to render the actual images, using *GalSim* itself is a natural choice for configuration. Perhaps most importantly, by implementing the *imSim* code as a *GalSim* module it becomes possible to use it freely with other *GalSim* based modules.  So, other programs in the *GalSim* ecosystem can also call and configure the *imSim* module and could (for example) utilize the *imSim* electronics readout algorithms as part of their functionality.

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

Each feature in the *imSim* module is implemented as a class which can be configured.  In this section, we list list the *imSim* classes along with the config options that you can set.

Input types
===========

These classes define the user configurable parts of the ``input`` section of the configuration YAML files.  They define how you should read the both the metadata and object lists that will be simulated, how the telescope optics should be configured, how the brightness of the night sky is modeled, how the atmospheric PSF is produced, how you should read in data files that describe silicon sensor effects, and how you can read in interim check-pointed files that were written in previous *imSim* runs.  Here we list the input types which are added my the *imSim* module.

Instance catalogs
-----------------

Instance catalogs are text files best suited to making small handcrafted inputs. For legacy and compatibility purposes, They follow the format of the *PhoSim* program inputs which are documented on `PhoSim Web Site <https://bitbucket.org/phosim/phosim_release/wiki/Instance%20Catalog>`__.  The file should contain a a header including metadata describing the observation and a list of sources.

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
    * ``flip_g2`` = *bool_value* (default = True) If True, apply a minus sign to the g2 lensing parameter used to shear the objects
    * ``min_source`` = *int_value* (default = False) if set, skip simulating any sensor which does not have at least min_sources on it.
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

    * ``index`` = *int_value* (default = number of objects in the file) by default all of the objects in the file will be processed, here you can specify an index your self of exactly which objects should be read if you would like by specifying a sequence of which items to process.
    * ``num`` =  *int_value* (default = 1) If you have multiple Random Numbers defined in the config file.  This option will allow you specify which one you should use. The default is the first and usually only one.

.. _InstCatWorld-label:

Instance Catalog World position
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once an instance catalog has been read in, the worlkd position as defined in the file can be specified to the top level GalSim ``stamp`` field like this:

.. code-block:: yaml

    # Define the galaxy (or delta function) to use
    world_pos:
        type: InstCatWorldPos


Optional keywords to set:
""""""""""""""""""""""""""

    These are the same as for ``InstCatObj`` above.


``RegisterSEDType('InstCatSED`` not used?? Check..



Sky Catalogs
------------

Instance catalogs are text based and utilize a lot of disk space for the information contained in them. Also, one instance catalog is needed for each visit, even if those visits take place at the exact same position on the sky.  This causes enormous duplication of information.  Instead, large area simulations, *imSim* utilizes an API based system known as `skyCatalogs <https://github.com/LSSTDESC/skyCatalogs>`__.  The *skyCatalog* presents a unified interface to *imSim* via an API of databases that contain all of the object in the sky.  By configuring *imSim* to use the *skyCatalog* API only metadata for the visits are needed.  *imSim* will retrieve a list of all of the objects it needs to render through the interface.  *skyCatalogs* can contain static and transient information and databases exist both for synthetic skies and true sources of information such as the Gaia catalog.  The *skyCatalog* can also serve as a source of truth information when later analyzing simulated data.

Key Name: sky_catalog
^^^^^^^^^^^^^^^^^^^^^

Required keywords to set:
"""""""""""""""""""""""""

  * ``file_name`` = *str_value* (default =  None)  The name of the yaml text file which specifies sky catalog positions.
  *  ``band`` = *str_value* (default = None)  The name of the LSST band to use.

Optional keywords to set:
"""""""""""""""""""""""""

  * ``edge_pix`` =  *float_value* (default = 100.0) How many pixels is the buffer region were objects are allowed to be past the edge of the sensor.
  * ``obj_types`` : *list*  List or tuple of object types to render, e.g., ('star', 'galaxy').  If None, then consider all object types.
  * ``max_flux`` = *float_value* (default = None) If object flux exceeds max_flux, the return None for that object. if max_flux == None, then don't apply a maximum flux cut.
  * ``apply_dc2_dilation`` = *bool_value* (default False) Flag to increase object sizes by a factor sqrt(a/b) where a, b are the semi-major and semi-minor axes, respectively. This has the net effect of using the semi-major axis as the sersic half-light radius when building the object.  This will only be applied to galaxies.
  * ``approx_nobjects`` = *int_value* (default None) Approximate number of objects per CCD used by galsim to set up the image processing.  If None, then the actual number of objects found by skyCatalogs, via .getNObjects, will be used.
  * ``mjd`` = *float_vaue*  MJD of the midpoint of the exposure.

Sky Catalog Object Type
^^^^^^^^^^^^^^^^^^^^^^^

    The ``SkyCatObj`` is used as in the :ref:`InstCatObj <InstCat-label>` case above.

Sky Catalog World Position
^^^^^^^^^^^^^^^^^^^^^^^^^^

    The ``SkyCatWorldPos`` is used as in the :ref:`InstCatWorldPos <InstCatWorld-label>` case above.

OpSim Data
----------

.. note::

    We need to rationalize this and figure out the best approach. It's hard to explain now. Will also need to say more about other ways too.  Do it here?

Each visit requires metadata that describes the time of exposure, the filter being employed, the direction that the telescope is pointing etc. There are several ways to pass this information to *imSim*.  You can include the information in the top of an instance catalog, you can give it the output of of a Rubin Operational Simulator simulation which will give a list of visits with all of the needed information, or you can manually specify information in the YAML file itself.

This input type allows you to specify file inputs which contain this information.

Key Name: opsim_data
^^^^^^^^^^^^^^^^^^^^
Note that several metadata keywords are required to be specified in the file.  They include: *rightascension, declination, mjd, altitude, azimuth, filter, rotskypos, rottelpos, dist2moon, moonalt, moondec, moonphase, moonra, nsnap, seqnum, obshistid, seed, seeing, sunalt, and , vistime.*

Required keywords to set:
""""""""""""""""""""""""""

    * ``file_name`` = *str_value* (default =  None)  The name of the text file that contains the required metadata information. Note that this data file can also contain object information.

Optional keywords to set:
"""""""""""""""""""""""""

    * ``visit`` = *int_value* (default = None) The visit number.
    * ``snap`` = *int_value* (default = 0) How many exposures should be taken.
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

OpSim Bandpass
^^^^^^^^^^^^^^

Once the metadata information has been specified you can use that information to specify the bandpass in other parts of the YAML file.  Using the LSST band that you specified it will read in the appropriate throughput file amd use it for the bandpass.  An example is shown below.

.. code-block:: yaml

    image:
        type: LSST_Image

        bandpass: { type: OpsimBandpass }


There are no configuration parameters for this class.

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

    * ``rotTelPos`` = *angle_value* (default = None) The angle of the camera rotator in degrees.
    * ``cameraName`` = *string_value* (default = 'LSSTCam') The name of the camera to use.
    * ``perturbations:`` = YAML dictionary (default = 'None')  See :ref:`the optical system section <optical-system-label>` for documentation.
    * ``fea:`` = YAML dictionary (default = 'None')  See :ref:`the optical system section <optical-system-label>` for documentation


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

Key Name: atmosphericPSF
^^^^^^^^^^^^^^^^^^^^^^^^

This keyword enables an atmospheric PSF with 6 randomly generated atmospheric screens.  Photons are raytraced through this atmosphere to produce a realistic atmospheric PSF.  See section (``needs to be added``) for more information on the model.

.. warning::
    You should not attempt to use the option to add parametric optics (through the ``doOpt`` option) if you are using fully ray-traced optics.  Otherwise, you will simulate the optics twice.  See See :ref:`the stamp keyword <stamp-label>` below how to activate the ray-traced mode.


Required keywords to set:
""""""""""""""""""""""""""

    * ``airmass`` = *float_value* (default =  None)  The aimass in the direction of the pointing.
    * ``rawSeeing`` = *float_value*  The FWHM seeing at zenith at 500 nm in arc seconds
    * ``band`` = *str_value* The filter band of the observation.
    * ``boresight`` = *RaDec_value* The CelestialCoord of the boresight of the observation.


Optional keywords to set:
"""""""""""""""""""""""""

    * ``t0`` = *float_value* (default = 0.0) Exposure time start in seconds.
    * ``exptime`` = *float_value*  (default = 30.0) Exposure time in seconds.
    * ``kcrit`` = *float_value* (default = 0.2) Critical Fourier mode at which to split first and second kicks.
    *  ``screen_size`` = *float_value* (default = 819.2) Size of the phase screens in meters.
    *  ``screen_scale`` = *float_value* (default = 0.1) Size of phase screen "pixels" in meters.
    *  ``doOpt`` = *bool_value* (default = True) Add in optical phase screens? *SEE WARNING ABOVE*
    *  ``nproc`` = *int_value* (default = None)  Number of processes to use in creating screens. If None (default), then allocate one process per phase screen, of which there are 6, nominally.
    *  ``save_file`` = *str_value* (default = None) A file name to use for saving the built atmosphere.  If the file already exists, then the atmosphere is read in from this file, rather than being rebuilt.


Key Name: DoubleGaussianPSF
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A wavelength and position-independent Double Gaussian PSF. This specific PSF comes from equation(30) of the signal-to-noise document (LSE-40), which can be found at http://www.astro.washington.edu/users/ivezic/Astr511/LSST_SNRdoc.pdf.

Required keywords to set:
""""""""""""""""""""""""""

    * ``fwhm`` = *float_value* (default =  None)  The full width at half max of the total PSF in arc seconds.


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

    * ``airmass`` = *float_value* (default =  None)  The aimass in the direction of the pointing.
    * ``rawSeeing`` = *float_value*  The FWHM seeing at zenith at 500 nm in arc seconds
    * ``band`` = *str_value* The filter band of the observation.



Tree Rings
----------

Tree-rings are a silicon sensor effect induced by internal electric fields in the 3D structure of the silicon CCD.  The fields are created by internal variations in dopant concentration that form while the silicon boule is being grown.  You can find more about *imSim*'s implementation of tree rings in :ref:`the Tree Ring validation section <tree-ring-label>`. This keyword tells imSim where to find the data file which describes the parameters to be used when the effect is turned on. It creates dictionaries that can be used by the LSST sensor description in :ref:`LSST Camera <LSST-Camera-label>` section below.

- RegisterInputType('tree_rings',
- RegisterValueType('TreeRingCenter'
- RegisterValueType('TreeRingFunc'

Checkpointing
-------------

As imSim runs, if this option is turned on, it will periodically check-point its progress, writing out its interim output as it runs.  Then, on re-running, it will use this output so as to not redo previous calculations.  This has two main use-cases.  The first is the case where you are rerunning several times. This will avoid recreating sensors that have already been simulated. The 2nd main use case is for if a job is stopped before it completes.  This is particularly common when using a batch system with time-limits.  This option allows you to restart your job and pick where you left off.  These keywords tell *imSim* where to find the checkpoint files and how they are named.

.. warning::

    Be careful to manually delete any check-point files if you have made any changes to to the configuration between runs.  Currently, *imSim* only checks if a file for a individual sensor already exists.

- RegisterInputType('checkpoint',


Image types
===========

These classes define how to draw images.  The basic *GalSim* image types include 'Single', 'Tiled', and 'Scattered'.  *imSim* adds a new type of image that can be used along with a new type of WCS object that uses ray-traced photons to map out a TAN-SIP WCS.

.. _LSST-Camera-label:

LSST Camera Images
------------------

The ``LSST_Image`` type is a version of the *GalSim* "Scattered Image" image class that has been modified to understand how to draw the Rubin sky background and how to apply effects such as vignetting to the sky and certain bright objects.

*imSim* also registers a new type of WCS object. When this WCS is chosen the `Batoid ray-tracing package  <https://github.com/jmeyers314/batoid>`__ traces a set of rays through the optics and fits the result to create a WCS which accurately represents the current state of the telescope optics.

- RegisterImageType('LSST_Image', LSST_ImageBuilder())

- RegisterWCSType('Batoid', BatoidWCSBuilder(), input_type="telescope")
- RegisterWCSType('Dict'

Image Calibration Flats
-----------------------

 *imSim* also supplies a ```LSST_Flat`` image type.  Calibration flats have extremely high background levels and special file, memory and SED handling are employed in this case in order to optimize computational efficiency.

 - RegisterImageType('LSST_Flat', LSST_FlatBuilder())

.. _stamp-label:

StampTypes
==========

The Stamp drawing code does the main work to actually render the image of an astronomical object.   *imSim* adds the ``LSST_Silicon`` type which understands how to draw objects in the LSSTCam sensors including accounting for absorption in the atmosphere, integrating the SEDs of the objects with the chosen filter, ray-tracing photons through the optical system, adding diffractive spikes from the telescope spider, automatically using various approximations for both very bright and very dim objects etc.  Those options are set with the parameters below.

- RegisterStampType('LSST_Silicon',


There are a set of operations that can act on photons in *GalSim*.  The are put together in a list and then all of the photons have those operations act on them in turn.  This list of photon-operations are specified in the stamp section.  You can read more about them in the *GalSim* documentation covering `GalSim Photon Ops <http://galsim-developers.github.io/GalSim/_build/html/config_stamp.html#photon-operators-list>`__.  *imSim* adds a new set of Photon Operators to ray-trace the photons through the optical system using the `Batoid package  <https://github.com/jmeyers314/batoid>`__.

If you do not turn these on, you should use the parameterized optics available in the atmospheric PSF instead.  You have three choices:

  - RubinOptics:

    Photons ray-traced though the Rubin optical system.

  - RubinDiffractionOptics:

    Ray-traced photons including the effects of diffraction when passing through edges like the telescope spiders.

  - RubinDiffraction:

    Diffractive effects only.


- RegisterPhotonOpType(identifier,


Output types
============

The output field is used to specify where to write output files and what format they should be.  There are several possibilities, including FITS files before and after electronics readout, and various types of truth information.  *imSim* adds the ``LSST_CCD`` type. It  understands how to write "eimage" files which are true representations of the electrons in the CCD including signals from the objects and cosmic rays with important physics effects such as the brighter-fatter effect and tree-rings applied.

It can also write "amp" files. These are fully readout electronics files with one amplifier per FITS HDU with all of the proper headers needed to be processed by the Rubin Science Pipelines.  Both of these output formats can be examined with standard tools such as *ds9*.

There are also several extra outputs available to the user including a centroid file containing the true position of the rendered sources, a list of optical path differences in the optical system, and a map of surface figure errors.

- RegisterOutputType('LSST_CCD', LSST_CCDBuilder())
- RegisterExtraOutput('readout', CameraReadout())
- RegisterExtraOutput('opd', OPDBuilder())



