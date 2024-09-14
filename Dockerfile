FROM python:3.12.3-slim-bullseye


WORKDIR .
COPY . .


RUN pip install --upgrade setuptools
RUN pip install -r  requirements.txt