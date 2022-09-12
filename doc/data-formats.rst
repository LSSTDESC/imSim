Data formats
############

Centroid file description
-------------------------

If the ``--create_centroid_file`` option is specified at the
``imsim.py`` command line, "centroid files" will be created for each
sensor-visit that contain truth-level information on the objects that
were rendered.  Here is some example output::

    SourceID                   Flux   Realized flux       xPix       yPix       flags      GalSimType
    31570294448132  629932434.70918 629946074.00000    1700.25    3859.67          12     pointSource
    31049104327684   97825890.06290  97818025.00000    1189.14    3922.07          12     pointSource
    692619268        48470916.66577  48468612.00000    1770.14     838.55          12     pointSource
    31854532234244   48925216.38775  48932814.00000    3273.89    1682.70          12     pointSource
    32166202499076   20548934.18112  20548613.00000    3982.98    3715.42          12     pointSource
    31854524204036   18176027.36589  18176766.00000     359.81    2372.41          12     pointSource
    32166199366660   14159590.95596  14160210.00000    2553.25    1861.16          12     pointSource
    31854523236356    6576819.16019   6578069.00000     931.90    3958.34           0     pointSource
    31053893695492    8810419.68268   8811371.00000    3040.27    3775.97           0     pointSource

Column Descriptions
~~~~~~~~~~~~~~~~~~~

``SourceID``
    The object ID from the instance catalog.

``Flux``
    The number of photons computed from the source model.

``Realized flux``
    The number of photons to be rendered drawn from a Poisson
    distribution with ``mean=Flux``.

``xPix``
    The x-pixel coordinate on the CCD corresponding to the (RA, Dec)
    position of the object.
    
``yPix``
    The y-pixel coordinate on the CCD corresponding to the (RA, Dec)
    position of the object.
    
``flags``
    Object rendering bit flags cast as an integer with the following
    definitions:

   .. table::
      :widths: auto
   
      =====  =====================
      Bit    Rendering condition
      =====  =====================
      1      Object Skipped
      2      Simple SED
      3      No Silicon Model
      4      Rendered with FFT
      =====  =====================

``GalSimType``
    The morphological type of object: ``pointSource``, ``sersic``,
    ``RandomWalk``, ``FitsImage``.
