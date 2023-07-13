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

- RegisterInputType('instance_catalog',
- RegisterValueType('InstCatWorldPos'
- RegisterObjectType('InstCatObj'
- RegisterSEDType('InstCatSED'


Sky Catalogs
------------

Instance catalogs are text based and utilize a lot of disk space for the information contained in them. Also, one instance catalog is needed for each visit, even if those visits take place at the exact same position on the sky.  This causes enormous duplication of information.  Instead, large area simulations, *imSim* utilizes an API based system known as `skyCatalogs <https://github.com/LSSTDESC/skyCatalogs>`__.  The *skyCatalog* presents a unified interface to *imSim* via an API of databases that contain all of the object in the sky.  By configuring *imSim* to use the *skyCatalog* API only metadata for the visits are needed.  *imSim* will retrieve a list of all of the objects it needs to render through the interface.  *skyCatalogs* can contain static and transient information and databases exist both for synthetic skies and true sources of information such as the Gaia catalog.  The *skyCatalog* can also serve as a source of truth information when later analyzing simulated data.


- RegisterObjectType('SkyCatObj'
- RegisterValueType('SkyCatWorldPos'


OpSim Data
----------

.. note::

    We need to rationalize this and figure out the best approach. It's hard to explain now.

Each visit requires metadata that describes the time of exposure, the filter being employed, the direction that the telescope is pointing etc. There are several ways to pass this information to *imSim*.  You can include the information in the top of an instance catalog, you can give it the output of of a Rubin Operational Simulator simulation which will give a list of visits with all of the needed information, or you can manually specify information in the YAML file itself.

This input type allows you to specify file inputs which contain this information.

    Will need to say more about other ways too.  Do it here?

- RegisterValueType('OpsimData'
- RegisterBandpassType('OpsimBandpass'


Telescope Configuration
-----------------------

The optical system of the telescope can be configured including optical aberrations, the state of active optics system, variations due to temperature etc.  Individual actuators and other elements of the optics system can also be configured as an input before the simulation starts.

If the photons are ray-traced through the optics with the `Batoid package  <https://github.com/jmeyers314/batoid>`__ photons will be modified by the changes as they propagate through the optics.  See :ref:`the stamp keyword <stamp-label>` below for details.

- TelescopeLoader(DetectorTelescope))

Sky Model
---------

Including the skyModel will load the Rubin Simulation Sky Model from the rubin-sims package.  If you have loaded this module, you will will be able top to refer the ``skyLevel`` variable in the image section to set the brightness of the sky. You can also use the ``apply_sky_gradient`` option in the image section to make the sky level vary over each sensor.

- RegisterInputType('sky_model'
- RegisterValueType('SkyLevel'

Atmospheric PSF
----------------

The class is used to create the PSF which is induced by the atmosphere.  There are two parametric PSFs available: a double Gaussian and a Kolmogorov PSF. The ``AtmosphericPSF`` type is a fully ray-traced turbulent atmosphere with multiple atmospheric layers all of which can have their parameters specified.  Additionally, you can optionally add a parametric PSF screen which simulates the Rubin Optics.

.. warning::
    You should not attempt to use the option to add parametric optics if you are using fully ray-traced optics.  Otherwise, you will simulate the optics twice.  See See :ref:`the stamp keyword <stamp-label>` below how to activate the ray-traced mode.

- RegisterObjectType('AtmosphericPSF'
- RegisterObjectType('DoubleGaussianPSF'
- RegisterObjectType('KolmogorovPSF'

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



