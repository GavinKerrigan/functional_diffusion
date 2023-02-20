# Diffusion Generative Models in Infinite Dimensions

This repo contains the code for *Diffusion Generative Models in Infinite Dimensions*, appearing at AISTATS 2023.

See [here](https://arxiv.org/abs/2212.00886) for an arXiv version of our paper.

## Environment and Setup

Clone the repo and create a conda environment:
```
conda env create -f environment.yml
```

This environment assumes you have a cuda 11.3 compatible GPU. A workaround for cpu-only machines is to replace `- cudatoolkit=11.3` with `- cpuonly` in `environment.yml`, followed by creating the environment as above.

## Citation
If you found our paper or code useful, please consider citing our work as follows:

```
@inproceedings{kerrigan2023diffusion,
  author={Kerrigan, Gavin and Ley, Justin and Smyth, Padhraic},
  title={Diffusion Generative Models in Infinite Dimensions},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  year={2023}
}
```
