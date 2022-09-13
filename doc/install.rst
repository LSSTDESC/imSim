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
------------------------------

.. code-block:: sh

   git clone https://github.com/LSSTDESC/imSim.git
   # conda dependencies:
   mamba install -y -c conda-forge --file imSim/conda_requirements.txt
   # pip dependencies:
   pip install batoid git+https://github.com/LSSTDESC/skyCatalogs.git git+https://github.com/lsst/rubin_sim.git
   # Install imSim:
   pip install imSim/
   
Install rubin_sim_data
----------------------

.. code-block:: sh

   mkdir -p rubin_sim_data
   curl https://s3df.slac.stanford.edu/groups/rubin/static/sim-data/rubin_sim_data/skybrightness_may_2021.tgz | tar -C rubin_sim_data -xz
   curl https://s3df.slac.stanford.edu/groups/rubin/static/sim-data/rubin_sim_data/throughputs_aug_2021.tgz | tar -C rubin_sim_data -xz


Set runtime environment variables for the conda environment
-----------------------------------------------------------

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
