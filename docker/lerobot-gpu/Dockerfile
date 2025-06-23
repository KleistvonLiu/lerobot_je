FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# Configure environment variables
ARG PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive
ENV MUJOCO_GL="egl"
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies and set up Python in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget\
    libglib2.0-0 libgl1-mesa-glx libegl1-mesa ffmpeg \
    speech-dispatcher libgeos-dev \
    python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && python -m venv /opt/venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && echo "source /opt/venv/bin/activate" >> /root/.bashrc

COPY environment.yml /lerobot/environment.yml

# Clone repository and install LeRobot in a single layer
COPY . /lerobot
WORKDIR /lerobot
# RUN /opt/venv/bin/pip install --upgrade --no-cache-dir pip \
    # && /opt/venv/bin/pip install --no-cache-dir ".[test, aloha, xarm, pusht, dynamixel]"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*
    
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    chmod +x miniconda.sh && \
    ./miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    echo "export PATH=/opt/conda/bin:$PATH" >> ~/.bashrc
ENV PATH=/opt/conda/bin:$PATH

RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/
RUN conda config --append channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/fastai/
RUN conda config --append channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
RUN conda config --append channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
RUN conda config --set show_channel_urls yes

RUN conda config --set channel_alias https://mirrors.tuna.tsinghua.edu.cn/anaconda
RUN conda config --set ssl_verify False

RUN conda clean --all
RUN conda update conda

RUN conda env create -f environment.yml && \
    conda init bash && \
    . /root/.bashrc && \
    conda activate lerobot && \
    conda install pip && \
    conda run -n lerobot pip install --upgrade pip
