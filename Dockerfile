# Start from the latest release LSST stack image.
FROM ghcr.io/lsst/scipipe:al9-w_latest

# Information about image.
ARG BUILD_DATE
LABEL lsst-desc.imsim.maintainer="https://github.com/LSSTDESC/imSim"
LABEL lsst-desc.imsim.description="A Docker image combining the LSST Science Pipelines software stack and imSim (and its dependencies)."
LABEL lsst-desc.imsim.version="latest"
LABEL lsst-desc.imsim.build_date=$BUILD_DATE

WORKDIR /home/lsst

# Clone imSim and rubin_sim repos.
RUN git clone https://github.com/LSSTDESC/imSim.git &&\
    git clone https://github.com/lsst/rubin_sim.git

# 1) Install imSim Conda requirements
# 2) Install imSim pip requirements
# 3) Install rubin_sim
# 4) Install imSim
RUN source /opt/lsst/software/stack/loadLSST.bash &&\
    setup lsst_distrib &&\
    mamba install -y --file imSim/etc/docker_conda_requirements.txt &&\
    python3 -m pip install batoid skyCatalogs==2.0.1 gitpython &&\
    python3 -m pip install rubin_sim/ &&\
    python3 -m pip install imSim/

WORKDIR /opt/lsst/software/stack

# Download Rubin Sim data.
RUN mkdir -p rubin_sim_data/sims_sed_library
RUN curl https://s3df.slac.stanford.edu/groups/rubin/static/sim-data/rubin_sim_data/skybrightness_may_2021.tgz | tar -C rubin_sim_data -xz
RUN curl https://s3df.slac.stanford.edu/groups/rubin/static/sim-data/rubin_sim_data/throughputs_2023_09_07.tgz | tar -C rubin_sim_data -xz
RUN curl https://s3df.slac.stanford.edu/groups/rubin/static/sim-data/sed_library/seds_170124.tar.gz  | tar -C rubin_sim_data/sims_sed_library -xz

# Set location of Rubin sim data (downloaded in step above).
ENV RUBIN_SIM_DATA_DIR=/opt/lsst/software/stack/rubin_sim_data

# Set location of SED library (downloaded in step above).
ENV SIMS_SED_LIBRARY_DIR=/opt/lsst/software/stack/rubin_sim_data/sims_sed_library

WORKDIR /home/lsst

# Make a script to activate the LSST stack
RUN echo "source /opt/lsst/software/stack/loadLSST.bash" >> .bashrc &&\
    echo "setup lsst_distrib" >> .bashrc
