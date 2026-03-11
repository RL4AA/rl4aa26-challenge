FROM quay.io/jupyter/scipy-notebook
# nodejs required for the visualisation in intro notebook
ARG NODE_VERSION=20
# Create a script file sourced by both interactive and non-interactive bash shells
ENV BASH_ENV /home/jovyan/.bash_env
RUN touch "${BASH_ENV}"
RUN echo '. "${BASH_ENV}"' >> ~/.bashrc

# Download and install nvm
# curl already installed by base notebook, so it's safe to use
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.4/install.sh | PROFILE="${BASH_ENV}" bash
RUN echo node > .nvmrc
RUN nvm install

# just verify that node and npm are installed correctly
RUN node -v
RUN npm -v
# grab cpu-only torch so we don't need all the nvidia cuda libs
# RL models don't really benefit from the GPU, and setting up GPU support in a docker container is a pain
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu 

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt