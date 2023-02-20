# Diffusion Models in Function Space

## Environment Setup

Clone the repo and create a conda environment:
```
conda env create -f environment.yml
```

This environment assumes you have a cuda 11.3 compatible GPU. A workaround for cpu-only machines is to replace `- cudatoolkit=11.3` with `- cpuonly` in `environment.yml`, followed by creating the environment as above.
