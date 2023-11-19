FROM docker.io/pi31416chan/az-basic:latest

WORKDIR /

# Create conda environment
COPY az-sklearn-conda.yaml .
RUN conda env update -p $CONDA_PREFIX -f az-sklearn-conda.yaml --prune && \
    rm az-sklearn-conda.yaml && \
    conda run -p $CONDA_PREFIX pip cache purge && \
    conda clean -a -y
