Usage
=====

To run the example config, change into the *imSim* source directory,
then

.. code-block:: sh

    cd config
    galsim imsim-user.yaml

Here is another minimal scenario, using a one-star catalog:

.. code-block:: yaml
   :caption: 1-star.yaml

   ---
   modules: [imsim]
   template: imsim-config

   input.instance_catalog.file_name: 1-star.txt
   input.opsim_meta_dict.file_name: 1-star.txt
   input.tree_rings.only_dets: [R22_S11]

   output.dir: output
   output.det_num:
     type: List
     items: [94]
   output.nfiles: 1
   output.truth: ""

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
   object MS_567_8a 0.0 0.0 16.0 starSED/phoSimMLT/lte033-4.5-1.0a+0.4.BT-Settl.spec.gz 0 0 0 0 0 0 point none CCM 0.0635117705 3.1
