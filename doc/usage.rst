Using imSim
===========

imSim is implemented as a set of modules that are executed by the *GalSim* program.  In order to run *imSim* you create a YAML file which imports and configures the modules.  *imSim* builds on basic *GalSim* functionality so when using *imSim* you will configure both *GalSim* and *imSim* functionality.  In this section a simple config file is demonstrated.  You can find a more complete description of the config system option in the :doc:`Config System <config>`.

*imSim* is packaged with some YAML config files for it's own use which you can look at.  You can see some in the "*config*" and also the "*examples*" directory.

To start copy the two files ``imsim-user.yaml`` and ``example_instance_catalog.txt`` to a working directory.  Edit the ``imsim-user.yaml`` file to point to your local copy of the ``example_instance_catalog.txt`` file by removing the two instances of ``$os.environ.get('IMSIM_HOME')+`` from the file.

Now try to run the file with the command:

.. code-block:: sh

    galsim imsim-user.yaml


*GalSim* will process the YAML file and open and read the example instance catalog file generating a fits file which corresponds to the listed sources.

*imSim* has more than one way to specify sources.  Instance catalogs are simple text files best suited to making small handcrafted inputs. For compatibility purposes, They follow the format of the *PhoSim* program which is documented on `PhoSim Web Site <https://bitbucket.org/phosim/phosim_release/wiki/Instance%20Catalog>`__.  For more complex and large area simulations *imSim* utilizes an API based system known as `skyCatalogs <https://github.com/LSSTDESC/skyCatalogs>`__.  By querying the SkyCatalog via it's API it can return a list to *imSim* of all the objects at that position in the the sky at that time. *skyCatalogs* can return both static and time dependent sources and can be configured to serve objects from both synthetic sky maps and true lists of sources such as from *Gaia*.

It is also important to specify *MetaData* to *imSim* which supplies necessary information such as the location, pointing, filters being used and time that the exposure was taken.  This information can be manually specified in the YAML files, given as part of a text based instance catalog, or retrieved from a Rubin Observatory *OpSim* file which is the simulated output of a Rubin Schedular execution.

Almost every element of the system from the atmosphere to optics, to details of the electronics readout can be specified through the config system in the YAML files.   Here, we will give an example of how to construct a very simple instance catalog which will image a singe very bright magnitude 16 star.

Each YAML file begins by importing the imSim code which is located in a module.  It then reads a template yaml file that configures many of the of the imSim classes.  Everything in your user file will modify or add to those values.  The file is located in ``config/imsim-config.yaml`` and is an excellent reference.

The rest of this file configures the input and output options.

.. code-block:: yaml
   :caption: 1-star.yaml

   modules:
    - imsim

   template: imsim-config

   input.instance_catalog.sed_dir: $os.environ.get('SIMS_SED_LIBRARY_DIR')

   input.instance_catalog.file_name: 1-star.txt
   input.opsim_data.file_name: 1-star.txt

   input.tree_rings.only_dets: [R22_S11]

   output.dir: output
   output.det_num:
     type: List
     items: [94]

   output.nfiles: 1


This file contains the needed metadata and a single star.

.. code-block::
   :caption: 1-star.txt

   rightascension 0.0
   declination 0.0
   mjd 59797.2854090
   altitude 0.0
   azimuth 0.0
   filter 2
   rotskypos 0.0
   rottelpos 0.0
   dist2moon 90.0
   moonalt -90.0
   moondec 0.0
   moonra 0.0
   moonphase 0.0
   nsnap 2
   obshistid 1
   seed 57721
   seeing 1.0
   sunalt -50.0
   vistime 33.0
   seqnum 0
   object MS_567_8a 0.0 0.0 16.0 starSED/phoSimMLT/lte033-4.5-1.0a+0.4.BT-Settl.spec.gz 0 0 0 0 0 0 point none CCM 0.0635117705 3.1

After creating these files you can:

.. code-block:: sh

    galsim 1-star.yaml


In the output directory