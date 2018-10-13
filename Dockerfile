FROM nvidia/cuda:9.0-cudnn7-runtime


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
RUN apt-get install -y libblas-dev liblapack-dev

RUN conda install matplotlib scikit-learn pillow
RUN conda install theano
# Matplotlib requires Cython
RUN conda install Cython
#RUN conda install pygpu   # Conda already includes gpu support in their theano formula

# Fix problem with matplotlib verison in conda
RUN pip uninstall -y matplotlib
RUN python -m pip install --upgrade pip
RUN pip install matplotlib

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# move from the weird nvidia versionion in end of .so files
RUN for f in $(ls /usr/local/cuda/lib64/*.so.9.0); do ln -s $f $(echo $f | sed 's/\(.*\)\.9\.0/\1/g'); echo "... Linked $f"; done

COPY . /usr/local/src/dec-new
WORKDIR /usr/local/src/dec-new
