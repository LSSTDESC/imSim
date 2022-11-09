# Start from the latest release LSST stack image.
from lsstsqre/centos:7-stack-lsst_distrib-w_latest

# Clone imSim and rubin_sim repos.
RUN git clone https://github.com/LSSTDESC/imSim.git &&\
    git clone https://github.com/lsst/rubin_sim.git

# Install imSim, rubin_sim, and dependencies.
RUN source /opt/lsst/software/stack/loadLSST.bash &&\
    setup lsst_distrib &&\
    pip install --upgrade galsim &&\
    pip install dust_extinction palpy batoid &&\
    pip install git+https://github.com/LSSTDESC/skyCatalogs.git@master &&\
    cd imSim && pip install -e . && cd .. &&\
    cd rubin_sim && pip install -e .

# Download Rubin Sim data.
RUN mkdir rubin_sim_data &&\
    curl https://s3df.slac.stanford.edu/groups/rubin/static/sim-data/rubin_sim_data/skybrightness_may_2021.tgz | tar -C rubin_sim_data -xz &&\
    curl https://s3df.slac.stanford.edu/groups/rubin/static/sim-data/rubin_sim_data/throughputs_aug_2021.tgz | tar -C rubin_sim_data -xz

# Location of Rubin sim data (downloaded in step above).
ENV RUBIN_SIM_DATA_DIR /opt/lsst/software/stack/rubin_sim_data

# SED library at NERSC.
ENV SIMS_SED_LIBRARY_DIR /global/cfs/cdirs/descssim/imSim/lsst/data/sims_sed_library
