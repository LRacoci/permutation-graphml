ARG BASE_IMAGE
FROM ${BASE_IMAGE}
MAINTAINER mipl


RUN pip install \
        imutils \
        keras \
        numpy \
        scikit-image \
        tflearn \
        sklearn \
        graph_nets

RUN apt-get install -y python3-tk

CMD ["/bin/bash"]
