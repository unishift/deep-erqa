FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

RUN rm /etc/apt/sources.list.d/cuda.list && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    apt-get update && \
    add-apt-repository ppa:jonathonf/ffmpeg-4 && \
    apt-get update && \
    apt-get install -y nasm git ninja-build doxygen xxd libffms2-dev ffmpeg libavcodec-extra zip git vim zsh htop tmux wget

RUN python -m pip install -U pip && \
    pip install -U setuptools wheel && \
    pip install jupyterlab ipywidgets pytorch-lightning mlflow tensorboard albumentations torchvision tqdm opencv-python

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV HOME /home
# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8
RUN chmod a+rwx $HOME
WORKDIR $HOME

ADD configs/* ${HOME}/
RUN wget -q -O - git.io/antibody | sh -s - -b /usr/local/bin

COPY entrypoint.sh ${HOME}/entrypoint.sh
ENTRYPOINT bash ${HOME}/entrypoint.sh
CMD ['zsh']
