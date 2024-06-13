FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04


WORKDIR /sd3

RUN apt-get update
RUN apt-get install python3.10 -y
RUN apt-get install python-is-python3 -y
RUN apt-get install pip -y
RUN apt-get install git -y
RUN apt-get install curl -y
RUN pip install requests starlette pydantic fastapi uvicorn torch transformers sentencepiece protobuf
RUN pip install -U diffusers


COPY main.py /sd3/main.py
COPY stable-diffusion-3-medium-diffusers /sd3/stable-diffusion-3-medium-diffusers

EXPOSE 5001
ENTRYPOINT ["python3", "main.py"]




