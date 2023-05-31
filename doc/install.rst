Installation Instructions
=========================

.. note::

   If you find that parts of this document are out of date or are incorrect, please submit a bug report `here <https://github.com/LSSTDESC/imSim/issues>`_.

   These instructions were last updated June of 2023.


The *imSim* software is *GalSim* based, and it also depends on the `LSST science pipelines <https://pipelines.lsst.io/>`_ and Rubin simulation framework.   It is easiest to work in an environment where someone else has already built the science pipelines, simulation packages and GalSim for you.  Luckily, several such options exist.

We list four options below in increasing levels of complexity.  These options all use either a *conda* environment or a *docker* image.

Method 1: Using the *cvmfs* distribution
---------------------------------------------

If you are working at the USDF (Rubin Project computing)or at NERSC (DESC computing), perhaps the easiest way to setup and use *imSim* is to rely on the prebuilt versions of the pipelines contained in the cvmfs distribution which is installed there.  This solution is also appropriate for personal laptops and university/lab based computing systems if you are able to install the *cvmfs* system.

The `CernVM file system <https://cvmfs.readthedocs.io/>`_  (cvmfs) is a distributed read-only file system developed at CERN for reliable low-maintenance world-wide software distribution.  LSST-France distributes weekly builds of the Rubin science pipelines for both Linux and MacOS.  Details and installation instructions can  be found at `sw.lsst.eu <https://sw.lsst.eu/index.html>`_ .  The distribution includes conda and imSim dependencies from conda-forge along with the science pipelines.

Load and setup the science pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First you need to setup the science pipelines.  This involves sourcing a setup file and then using the Rubin *eups* commands to set them up.

.. note:: 

   You will need at least version  ``w_2023_21-dev`` of the science pipelines to complete these instructions.

Source the appropriate setup script (note the -ext in the name) and then setup the distribution.

.. code-block:: sh

   source /cvmfs/sw.lsst.eu/darwin-x86_64/lsst_distrib/w_2023_21-dev/loadLSST-ext.bash
   setup lsst-distrib


Install needed data files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now go to where you would like to install *imSim* and download a set of data files that are used by the rubin-simulation framework (you will only need to do this once).

.. code-block:: sh

   mkdir -p rubin_sim_data/sims_sed_library
   curl https://s3df.slac.stanford.edu/groups/rubin/static/sim-data/rubin_sim_data/skybrightness_may_2021.tgz | tar -C rubin_sim_data -xz
   curl https://s3df.slac.stanford.edu/groups/rubin/static/sim-data/rubin_sim_data/throughputs_aug_2021.tgz | tar -C rubin_sim_data -xz
   curl https://s3df.slac.stanford.edu/groups/rubin/static/sim-data/sed_library/seds_170124.tar.gz  | tar -C rubin_sim_data/sims_sed_library -xz


Setup imSim itself
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, clone copies of both the imSim and skyCatalog software packages from  GitHub.

.. code-block:: sh

   git clone https://github.com/LSSTDESC/imSim.git
   git clone https://github.com/LSSTDESC/skyCatalogs

at this point if you would only like to use *imSim* you can ``pip install imSim/`` and ``pip install skyCatalog/`` however we instead suggest using the *eups* tool to simply setup the packages for use without installing them. This will allow you to edit the packages in place, use multiple versions, change branches etc. You should definitely do this if you plan to do any *imSim* development.


.. code-block:: sh

   setup -k -r imSim
   setup -k -r skyCatalogs


Setup and Run *imSim*
~~~~~~~~~~~~~~~~~~~~~


This setup step should be repeated for each new session.  Here is an ``imsim-setup.sh`` file you can use before each session (you should first source the cmvfs setup as above):



.. code-block:: sh

   setup lsst_distrib

   export IMSIM_HOME=*PUT YOUR INSTALL DIRECTORY HERE*
   export RUBIN_SIM_DATA_DIR=$IMSIM_HOME/rubin_sim_data
   export SIMS_SED_LIBRARY_DIR=$IMSIM_HOME/rubin_sim_data/sims_sed_library

   setup -k -r $IMSIM_HOME/imSim
   setup -k -r $IMSIM_HOME/skyCatalogs



Now you can run *imSim* with the command 

.. code-block:: sh

   galsim some_imsim_file.yaml



Method 2: *Conda* and the *Stackvana* package
---------------------------------------------

Install *Conda*
~~~~~~~~~~~~~~~

First, install `MiniConda <https://docs.conda.io/en/latest/miniconda.html>`__
(if you do not have a *Conda* installation already):

.. code-block:: sh

   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
   bash /tmp/miniconda.sh -b -u -p ./conda
   rm -rf /tmp/miniconda.sh

Next, create a *Python* 3.10 *Conda* environment and install *Mamba*:

.. code-block:: sh

   conda create -n imSim python=3.10
   conda activate imSim
   conda install -c conda-forge mamba

Clone *imSim* and dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we are ready to install *imSim*:

.. code-block:: sh

   git clone https://github.com/LSSTDESC/imSim.git
   # conda dependencies:
   mamba install -y -c conda-forge --file imSim/etc/standalone_conda_requirements.txt
   # pip dependencies:
   pip install batoid skyCatalogs==1.2.0 gitpython
   pip install git+https://github.com/lsst/rubin_sim.git
   # Install imSim:
   pip install imSim/

Install *rubin_sim_data*
~~~~~~~~~~~~~~~~~~~~~~~~

*imSim* makes use of some external datasets. Here we are putting them in a
directory called ``rubin_sim_data``, but the choice is yours. If you do change
the locations, make sure to update the ``RUBIN_SIM_DATA_DIR`` and
``SIMS_SED_LIBRARY_DIR`` paths we set below.

To download:

.. code-block:: sh

   mkdir -p rubin_sim_data/sims_sed_library
   curl https://s3df.slac.stanford.edu/groups/rubin/static/sim-data/rubin_sim_data/skybrightness_may_2021.tgz | tar -C rubin_sim_data -xz
   curl https://s3df.slac.stanford.edu/groups/rubin/static/sim-data/rubin_sim_data/throughputs_aug_2021.tgz | tar -C rubin_sim_data -xz
   curl https://s3df.slac.stanford.edu/groups/rubin/static/sim-data/sed_library/seds_170124.tar.gz  | tar -C rubin_sim_data/sims_sed_library -xz


Set runtime environment variables for the *Conda* environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: sh

    conda env config vars set RUBIN_SIM_DATA_DIR=$(pwd)/rubin_sim_data
    conda env config vars set SIMS_SED_LIBRARY_DIR=$(pwd)/rubin_sim_data/sims_sed_library

Method 3: Using the pre-built *imSim* Docker image
--------------------------------------------------

Here we assume you have `Docker <https://docs.docker.com/get-docker/>`_
installed.

The easiest method is to pull the latest *imSim* environment *Docker* image
from `DockerHub <https://hub.docker.com/r/lsstdesc/imsim-env>`__. This has the
latest LSST stack and version of *imSim* preinstalled: 

.. code-block:: sh

   docker pull lsstdesc/imsim-env:latest


Running the Docker image
~~~~~~~~~~~~~~~~~~~~~~~~
To then run the image do:

.. code-block:: sh

    docker run -it --privileged --rm lsstdesc/imsim-env:latest

*imSim* is installed (as an editable install) under ``/home/lsst``. The LSST
stack is activated automatically on the startup of the image. 

.. note:: If you have trouble accessing the internet within the container, you
   may have to add ``--network host`` to the ``docker run`` command.

Method 3: Building your own *imSim* Docker image
------------------------------------------------

If the *imSim* *Docker* image doesn't quite meet your needs, perhaps you need
some additional software or dependencies, or you want to develop an *imSim*
project that is stored locally on your machine actively within a container,
then you can use the *imSim* *Docker* image as a starting point to build your
own *Docker* image.

The *imSim* Dockerfile
~~~~~~~~~~~~~~~~~~~~~~

The *imSim* ``Dockerfile`` is located in the root directory of the *imSim*
repository, which you can use as a starting point:

.. literalinclude:: ../Dockerfile
   :language: Docker
   :linenos:


It builds upon the `prebuilt Docker images
<https://hub.docker.com/r/lsstsqre/centos/tags>`_ produced by the Rubin Data
Management team. Standard images are produced on a weekly basis and track the
`on-going development of the LSST Stack
<https://lsst-dm.github.io/lsst_git_changelog/weekly/summary.html>`_. For
``imsim-env:latest`` we build upon the latest LSST stack image, however you can
change this to a specific build if you prefer.

We then install *imSim* and *GalSim* and their dependencies, and download the
``rubin_sim_data``.

If you want to install additional *Python* dependencies on-top of the default
build, do it under the ``RUN`` command on line 24.

If you are installing additional general software, this can be done at the
start of the image.

Mounting a volume
~~~~~~~~~~~~~~~~~
You could for example use a *Docker* image as a clean environment to develop
*imSim*, but keep the active development files locally on your machine. To do
this, remove the *imSim* installation from the ``Dockerfile``. Then, when
running the container, mount your local *imSim* directory like so.

.. code-block:: sh

    docker run -it --privileged --rm -v ${HOME}/imSim:/home/lsst/imSim lsstdesc/my-imsim-env:latest

The ``-v`` option maps your home *imSim* directory on the host system to
``/home/lsst/imSim`` in the Docker container (which is the default working
directory).

Setting user and group ids
~~~~~~~~~~~~~~~~~~~~~~~~~~
The prebuilt Rubin images set the default linux user and group both to ``lsst``
with ``uid=1000`` and ``gid=1000``.   If the desired user and group on the host
system have the same ids, then the ``lsst`` user and group in the Docker image
can be renamed with the following, replacing the line

.. code-block:: sh

    WORKDIR /home/lsst

with

.. code-block:: sh

    USER root
    ARG user=<desired_username>
    ARG group=<desired_group_name>
    RUN /usr/sbin/groupmod -n ${group} lsst
    RUN /usr/sbin/usermod -l ${user} lsst
    USER ${user}
    WORKDIR /home/${user}

Alternatively, if the ``lsst`` user doesn't conflict with the desired
user/group, the latter can be added to the image and set as the default user:

.. code-block:: sh

    USER root
    ARG user=<desired_username>
    ARG group=<desired_group_name>
    ARG uid=<desired_uid>
    ARG gid=<desired_gid>
    RUN /usr/sbin/groupadd -g ${gid} ${group}
    RUN /usr/sbin/useradd -u ${uid} -g ${gid} ${user}
    USER ${user}
    WORKDIR /home/${user}

Building the Docker image
~~~~~~~~~~~~~~~~~~~~~~~~~
Assuming the above ``Dockerfile`` is in the current directory, then the
following command will build the image

.. code-block:: sh

    docker build -t <repository>:<tag> .

where ``<repository>`` and ``<tag>`` are chosen by the user.

The available images can be listed via

.. code-block:: sh

    docker images
