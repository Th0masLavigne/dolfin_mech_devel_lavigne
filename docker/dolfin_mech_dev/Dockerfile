# Base image from FEniCS Legacy Project
ARG FENICS_VERSION=2024-05-30
FROM ghcr.io/scientificcomputing/fenics-gmsh:${FENICS_VERSION}
#
ARG REPO_URL
#
LABEL name="dolfin_mech github branch devel"
LABEL description="Ubuntu environment containing dolfin_mech devel and FEniCS legacy python API"
LABEL version="devel"
LABEL vsc-url="$REPO_URL"
LABEL usage="$REPO_URL/-/wikis/home"
#
ENV PIP_ROOT_USER_ACTION=ignore
# to build doc
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
#
# Install system dependencies (for plotting/GUI/utility)
RUN apt-get update && \
    apt-get install -y nano xvfb libgl1-mesa-dri libglx-mesa0 libglu1-mesa mesa-utils && \
    python3 -m pip install --upgrade pip
#
# Dolfin_mech 
# And Other standard Python packages used by common demos/post-processing
RUN pip3 install "git+https://github.com/mgenet/dolfin_mech.git@devel" && \   
    pip3 install pandas imageio pyvista meshio h5py && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /home/dolfin_mech
