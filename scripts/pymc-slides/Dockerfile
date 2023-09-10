FROM python:3.9-slim-buster

RUN pip install --upgrade pip && pip install latent-calendar rich lyft-bikes IPython

WORKDIR /app
COPY ./pymc-slides.py ./pymc-slides.py

ENTRYPOINT [ "python", "pymc-slides.py" ]