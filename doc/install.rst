Install
=======

.. note::
   As *imSim* is still under heave development, parts of this document
   might outdate at some point. In such cases, please submit a bug
   report `here <https://github.com/LSSTDESC/imSim/issues>`_.

   The `CI directives
   <https://github.com/LSSTDESC/imSim/blob/main/.github/workflows/ci.yml>`_
   are guaranteed to result in a working setup as they are tested very
   frequently.
  
Although *imSim* is *GalSim* based (which just can be installed via
``pip install galsim``, it also depends on the
`LSST software stack <https://pipelines.lsst.io/>`_,
which is not as easy to install.

Method 1: Via Conda / Stackvana
-------------------------------

Install conda
~~~~~~~~~~~~~

Here, we are going to use miniconda:

.. code-block:: sh

   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
   bash /tmp/miniconda.sh -b -u -p ./conda
   rm -rf /tmp/miniconda.sh

Next, we create a python 3.8 environment and install mamba:

.. code-block:: sh

   ./conda/bin/conda create -n py38 python=3.8
   . ./conda/bin/activate
   conda activate py38
   conda install -c conda-forge mamba


Clone *imSim* and dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: sh

   git clone https://github.com/LSSTDESC/imSim.git
   # conda dependencies:
   mamba install -y -c conda-forge --file imSim/conda_requirements.txt
   # pip dependencies:
   pip install batoid git+https://github.com/LSSTDESC/skyCatalogs.git git+https://github.com/lsst/rubin_sim.git
   # Install imSim:
   pip install imSim/

Install rubin_sim_data
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: sh

   mkdir -p rubin_sim_data
   curl https://s3df.slac.stanford.edu/groups/rubin/static/sim-data/rubin_sim_data/skybrightness_may_2021.tgz | tar -C rubin_sim_data -xz
   curl https://s3df.slac.stanford.edu/groups/rubin/static/sim-data/rubin_sim_data/throughputs_aug_2021.tgz | tar -C rubin_sim_data -xz


Set runtime environment variables for the conda environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: sh

    conda env config vars set RUBIN_SIM_DATA_DIR=$(pwd)/rubin_sim_data


This will export the environment variable ``SIMS_SED_LIBRARY_DIR`` whenever
you activate the ``py38`` conda environment.

To make use of the default *imSim* configuration, you also need a SED dataset.
You can obtain such a dataset from NERSC, e.g.

``/cvmfs/sw.lsst.eu/linux-x86_64/lsst_sims/sims_w_2021_19/stack/current/Linux64/sims_sed_library/2017.01.24-1-g5b328a8``

Once, you have a SED dataset, you also have to point the variable
``SIMS_SED_LIBRARY_DIR`` to its location.

.. code-block:: sh

    conda env config vars set SIMS_SED_LIBRARY_DIR=your/sed/path

Method 2: Using a Docker image
------------------------------

Example Dockerfile
~~~~~~~~~~~~~~~~~~
Assuming you have `Docker <https://docs.docker.com/get-docker/>`_ installed, the following Dockerfile will enable you to build an image with all of components needed to run imSim:

.. code-block:: sh

    from lsstsqre/centos:7-stack-lsst_distrib-w_2022_38

    RUN source /opt/lsst/software/stack/loadLSST.bash &&\
        setup lsst_distrib &&\
        pip install galsim==2.4 &&\
        pip install batoid &&\
        pip install git+https://github.com/LSSTDESC/skyCatalogs.git@master &&\
        pip install dust_extinction &&\
        pip install palpy &&\
        git clone https://github.com/LSSTDESC/imSim.git &&\
        cd imSim &&\
        pip install -e . &&\
        cd .. &&\
        git clone https://github.com/lsst/rubin_sim.git &&\
        cd rubin_sim &&\
        pip install -e . &&\
        cd .. &&\
        mkdir rubin_sim_data &&\
        curl https://s3df.slac.stanford.edu/groups/rubin/static/sim-data/rubin_sim_data/skybrightness_may_2021.tgz | tar -C rubin_sim_data -xz &&\
        curl https://s3df.slac.stanford.edu/groups/rubin/static/sim-data/rubin_sim_data/throughputs_aug_2021.tgz | tar -C rubin_sim_data -xz

    WORKDIR /home/lsst

    CMD source /opt/lsst/software/stack/loadLSST.bash; setup lsst_distrib; export RUBIN_SIM_DATA_DIR=/opt/lsst/software/stack/rubin_sim_data; bash

The final ``CMD`` line sets up the runtime environment in bash.

Note that the file containing these commands should literally be called ``Dockerfile``.

Here we've used one of the `prebuilt Docker images <https://hub.docker.com/r/lsstsqre/centos/tags>`_ produced by Rubin Data Management team that are available from Docker Hub.  Standard images are produced on a weekly basis and track the `on-going development of the LSST Stack <https://lsst-dm.github.io/lsst_git_changelog/weekly/summary.html>`_.  For weekly ``w_2022_22`` and later, python 3.10 is the baseline version provided with the prebuilt Rubin Docker images.

Various components, e.g., GalSim, imSim, etc., can be omitted from the Dockerfile build and installed separately as shown in the conda/stackvana method.

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


Running the Docker image
~~~~~~~~~~~~~~~~~~~~~~~~
To run the image do

.. code-block:: sh

    docker run -it --privileged --rm -v ${HOME}:/home/<user> <repository>:<tag>

The ``-v ${HOME}:/home/<user>`` option maps the user's home directory on the host system to ``/home/<user>`` in the Docker image.
