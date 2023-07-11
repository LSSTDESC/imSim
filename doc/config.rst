The Config System
=================

*imSim* utilizes the *imSim* config system to configure itself.  This choice was made for several reasons.  As *GalSim* is used to render the actual images, using *GalSim* itself is a natural choice to set its configuration options. Perhaps most importantly, by implementing the *imSim* code as a *GalSim* module it becomes possible to use it freely with other *GalSim* based modules.  So, other programs in the *GalSim* ecosystem can also call and configure the *imSim* module and could (for example) utilize the *imSim* electronics readout algorithms as part of their functionality.

The *GalSim* config system is described in the `GalSim documentation <http://galsim-developers.github.io/GalSim/_build/html/config.html>`__. You should start there to understand the basic functionality of the config system.  If you look at the ``config/imsim-config.yaml`` distributed with *imSim* you can see the *imSim's* default configuration.  When you read in a YAML file you will be adding to and modifying these results.

Each feature in the *imSim* module is implemented as a class which can be configured.  In this section, we list list the *imSim* classes along with the config options that you can set.

