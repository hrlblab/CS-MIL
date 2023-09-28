## Pull from existing image
FROM nvcr.io/nvidia/pytorch:21.05-py3
  
## Copy requirements
COPY ./requirements.txt .

## Install Python packages in Docker image

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install "opencv-python-headless<4.3"
RUN pip3 install openslide-python

RUN apt-get update \
  && DEBIAN_FRONTEND="noninteractive" apt-get install -y libopenslide0 \
  && rm -rf /var/lib/apt/lists/*

#RUN export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
RUN export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libopenslide.so.0


## Copy all files (here "./src/run_inference.py")
COPY ./ ./


RUN mkdir /myhome/
COPY ./src /myhome
RUN chmod -R 777 /myhome


## Execute the inference command 
CMD ["./src/run_inference.py"]
ENTRYPOINT ["python3"]
