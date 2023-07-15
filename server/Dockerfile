FROM nvcr.io/nvidia/tritonserver:23.06-py3

RUN pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
RUN pip install torch-neuron neuron-cc[tensorflow] torchvision