# PyMC Slides

Slides for PyMC Labs talk in October 2023

## Usage

**Create the slides**

Run from scripts/pymc-slides/ directory

```bash
source create-slide-material.sh
```

**Interactive use**

```bash
docker build -t pymc-slides . 

docker run --rm -it --entrypoint bash -v $(pwd):/app -w /app pymc-slides

```

