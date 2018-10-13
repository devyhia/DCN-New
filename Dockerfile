FROM python:2.7

RUN pip install matplotlib scikit-learn theano pillow

COPY . /usr/local/src/dec-new
WORKDIR /usr/local/src/dec-new
