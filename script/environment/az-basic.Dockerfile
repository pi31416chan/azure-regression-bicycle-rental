FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:20231011.v1

WORKDIR /

ENV CONDA_PREFIX=/azureml-envs/venv_az
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX
ENV PATH=$CONDA_PREFIX/bin:$PATH

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Create conda environment
COPY az-basic-conda.yaml .
RUN conda env create -p $CONDA_PREFIX -f az-basic-conda.yaml -q && \
    rm az-basic-conda.yaml && \
    conda run -p $CONDA_PREFIX pip cache purge && \
    conda clean -a -y

