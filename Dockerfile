# FROM nvidia/cuda:10.0-base-ubuntu18.04
FROM consol/ubuntu-xfce-vnc
ENV REFRESHED_AT 2019-12-10

# Switch to root user to install additional software.
USER 0

# Install some basic utilities.
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    bzip2 \
    libx11-6 \
    tmux \
    htop \
    gcc \
    xvfb \
    python-opengl \
    x11-xserver-utils \
 && rm -rf /var/lib/apt/lists/*

# Install Minecraft needed libraries.
RUN apt-get update
RUN apt-get install openjdk-8-jdk -y

RUN sudo apt-get install python3 python3-pip python3-tk -y 
# Upgrade pip so it can find tensorflow==2.0.0rc0.
RUN pip3 install -U pip

# Create a working directory.
RUN mkdir /code
WORKDIR /code

COPY requirements.txt requirements.txt
RUN pip3 install --upgrade -r requirements.txt

COPY . . 


CMD ["python3", "assignment.py", "REINFORCE_BASELINE"]
