ARG BASE_IMAGE
FROM ${BASE_IMAGE}
MAINTAINER mipl


RUN pip install \
        imutils \
        keras \
        numpy \
        scikit-image \
        tflearn \
        graph_nets

CMD ["/bin/bash"]
