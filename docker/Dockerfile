# Multi-Stage Production Dockerfile for spatial-microenvironment-profiler
# Provides Python 3.11 with PyTorch/Scanpy/Squidpy and R 4.3 with spacexr (RCTD)

FROM rocker/r-ver:4.3.3 AS r-base

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for Bioconductor and Arrow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libgit2-dev \
    libarrow-dev \
    libgsl-dev \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Bioconductor manager and core spatial deconvolution packages
RUN R -e "install.packages(c('remotes', 'optparse', 'jsonlite', 'Matrix', 'arrow', 'ggplot2'), repos='https://cloud.r-project.org')" \
    && R -e "remotes::install_bioc(c('SingleCellExperiment', 'SpatialExperiment', 'SummarizedExperiment'))" \
    && R -e "remotes::install_github('dmcable/spacexr', build_vignettes = FALSE)"

# Final Stage: Python 3.11 + R Runtime
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/usr/local/bin:$PATH"

# Install R runtime dependencies and graphics libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base \
    r-base-dev \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    libcurl4 \
    libxml2 \
    libarrow-dev \
    procps \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy R site libraries from the R build stage
COPY --from=r-base /usr/local/lib/R/site-library /usr/local/lib/R/site-library

# Create non-root user
RUN useradd -m -u 1000 bioengineer
WORKDIR /workspace

# Install Python requirements
COPY pyproject.toml /workspace/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel hatchling \
    && pip install --no-cache-dir "torch>=2.2.0" "torchvision>=0.17.0" --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir .

COPY . /workspace/
RUN chown -R bioengineer:bioengineer /workspace

USER bioengineer

# Smoke test Python and R environment parity
RUN python -c "import scanpy, squidpy, torch, pydantic; print('Python Spatial Stack: OK')" \
    && Rscript -e "library(spacexr); library(arrow); cat('R spacexr Stack: OK\n')"

ENTRYPOINT ["spatial-profiler"]
CMD ["--help"]

