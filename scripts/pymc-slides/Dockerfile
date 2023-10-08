FROM python:3.10

# TODO: Pin the latent-calendar version
RUN pip install --upgrade pip && pip install latent-calendar rich lyft-bikes IPython qrcode

WORKDIR /app
COPY ./pymc-slides.py ./pymc-slides.py

ENTRYPOINT [ "python", "pymc-slides.py" ]