FROM nvidia/cuda:12.6.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/usr/local/huggingface
ENV PATH="/opt/miniconda3/bin:${PATH}"
ARG PATH="/opt/miniconda3/bin:${PATH}"
ARG MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py311_24.3.0-0-Linux-x86_64.sh"

RUN apt-get update \
    && apt-get install --no-install-recommends --assume-yes \
    wget

# Install Miniconda on x86 or ARM platforms
RUN wget $MINICONDA_URL -O miniconda.sh && \
    mkdir -p /root/.conda && \
    bash miniconda.sh -b -p /opt/miniconda3 && \
    rm -f miniconda.sh && \
    conda init bash && \
    conda config --set solver classic

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN python -m spacy download en_core_web_lg