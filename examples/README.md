This directory contains some config files and instance catalogs that will
allow you to test running imSim and show you examples of the config language.

_Simple example of running and imSim config file in this directory:_

To test that the your installed version of imSim is working you can run one the
files in this directory.  If you have setup imSim properly, from any working
area you should be able to:

```
galsim $IMSIM_HOME/imSim/examples/imsim-user.yaml
```

and the program should run to completion without errors.

This directory also contains an example of running using skyCatalogs.  A small
skyCatalog is distributed for testing in the distribution.  To make your own
YAML files:  Copy one of the templates from the "config" directory and add
your commands to the bottom of the file like in these examples.
