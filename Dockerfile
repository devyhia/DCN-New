#FROM nvidia/cuda:9.0-cudnn7-runtime
FROM nvidia/cuda:9.0-cudnn7-devel


ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda2-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

# Install BLAS / LAPACK for GPU computations
#RUN apt-get install -y libblas-dev liblapack-dev

# Install Vim for on-the-fly edits
RUN apt-get install -y vim

RUN conda install matplotlib scikit-learn pillow theano ipython

# Matplotlib requires Cython
RUN conda install Cython

# Fix problem with matplotlib verison in conda
RUN pip uninstall -y matplotlib
RUN python -m pip install --upgrade pip
RUN pip install matplotlib

#COPY . /usr/local/src/code
WORKDIR /usr/local/src/code
