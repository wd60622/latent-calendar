# Clean up and create images directory
rm -rf images && mkdir images

# Create slide material
docker build -t pymc-slides .
docker run --rm -it -v $(pwd):/app pymc-slides