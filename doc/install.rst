Installation instructions
=========================

.. note:: As *imSim* is still under heavy development, parts of this document
   are likely to become outdated at some point. In such cases, please submit a
   bug report `here <https://github.com/LSSTDESC/imSim/issues>`_.

   The `CI directives
   <https://github.com/LSSTDESC/imSim/blob/main/.github/workflows/ci.yml>`_ are
   guaranteed to result in a working setup as they are tested very frequently.

   These instructions were last updated Jan 2023.

Although the *imSim* software is *GalSim* based, which can be installed via
``pip install galsim``, it also depends on the `LSST software stack
<https://pipelines.lsst.io/>`_, which is not so easy to install.

For that reason we recommend working within the *imSim* *Docker* image, where
the LSST software stack is pre-installed. Alternatively, you can use the
*imSim* *Docker* file as a template to build your own *Docker* image, or,
another option is to work within a *Conda* environment that gets the LSST stack
through *Stackvana*. The three installation options are detailed below.

Method 1: Using the pre-built *imSim* Docker image
--------------------------------------------------

Here we assume you have `Docker <https://docs.docker.com/get-docker/>`_
installed.

The easiest method is to pull the latest *imSim* environment *Docker* image
from `DockerHub <https://hub.docker.com/r/lsstdesc/imsim-env>`__: 

.. code-block:: sh

   docker pull lsstdesc/imsim-env:latest


Running the Docker image
~~~~~~~~~~~~~~~~~~~~~~~~
To then run the image do:

.. code-block:: sh

    docker run -it --privileged --rm -v ${HOME}:/home/lsst imsim-env:latest


The ``-v ${HOME}:/home/lsst`` option maps your home directory on the host
system to ``/home/lsst`` in the Docker container (which is the default working
directory). You can omit this, or chose a different directory to mount into the
container.

Once inside the container you will have to activate the LSST stack as normal:

.. code-block:: sh

   source /opt/lsst/software/stack/loadLSST.bash
   setup lsst_distrib


.. note:: If you have trouble accessing the internet within the container, you
   may have to add ``--network host`` to the ``docker run`` command.

Method 2: Building your own *imSim* Docker image
------------------------------------------------

If the *imSim* *Docker* image doesn't quite meet your needs, perhaps you need
some additional software or dependencies, you can use the *imSim* *Docker*
image as a starting point to build your own *Docker* image.

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

We then install *imSim* and *galsim* and their dependencies, and download the
``rubin_sim_data``.

If you want to install additional *Python* dependencies on-top of the default
build, do it under the ``RUN`` command on line 22, on the line before ``conda
clean -afy``.

If you are installing additional general software, this can be done at the
start of the image.

Setting user and group ids
~~~~~~~~~~~~~~~~~~~~~~~~~~
The prebuilt Rubin images set the default linux user and group both to ``lsst`` with ``uid=1000`` and ``gid=1000``.   If the desired user and group on the host system have the same ids, then the ``lsst`` user and group in the Docker image can be renamed with the following, replacing the line

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

Alternatively, if the ``lsst`` user doesn't conflict with the desired user/group, the latter can be added to the image and set as the default user:

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
Assuming the above Dockerfile is in the current directory, then the following command will build the image

.. code-block:: sh

    docker build ./ -t <repository>:<tag>

where ``<repository>`` and ``<tag>`` are chosen by the user.

The available images can be listed via

.. code-block:: sh

    docker images

Method 3: Via Conda / Stackvana
-------------------------------

Another option is to install *imSim* and the LSST stack within a *Conda*
environment.

Install conda
~~~~~~~~~~~~~

First, install `MiniConda <https://docs.conda.io/en/latest/miniconda.html>`__
(if you do not have a *Conda* installation already):

.. code-block:: sh

   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
   bash /tmp/miniconda.sh -b -u -p ./conda
   rm -rf /tmp/miniconda.sh

Next, create a *Python* 3.10 *Conda* environment and install *Mamba*:

.. code-block:: sh

   ./conda/bin/conda create -n imSim python=3.10
   . ./conda/bin/activate
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

Install rubin_sim_data
~~~~~~~~~~~~~~~~~~~~~~

*imSim* makes use of some external datasets. Here we are putting them in a
directory called ``rubin_sim_data``, but the choice is yours. If you do change
the locations, make sure to update the ``RUBIN_SIM_DATA_DIR`` path we set
below.

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
