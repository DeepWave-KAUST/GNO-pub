![LOGO](https://github.com/DeepWave-Kaust/GNO-pub/blob/main/logo/logo.jpg)

<div align="center">
<h3><strong>Seismic wavefield solutions via physics-guided generative neural operator</strong></h2>
<h4>Shijun Cheng, Mohammad H. Taufik, Tariq Alkhalifah</h3>
<h4><em>DeepWave Consortium, King Abdullah University of Science and Technology (KAUST)</em></h4>
<p><em>Corresponding author: Shijun Cheng (<a href="mailto:sjcheng.academic@gmail.com">sjcheng.academic@gmail.com</a>)</em></p>
</div>

## Project structure
This repository is organized as follows:

* :open_file_folder: **gno**: Python library containing routines for the generative neural operator;
* :open_file_folder: **logo**: folder containing logo;
* :open_file_folder: **dataset**: folder to store datasets, including:
  * :open_file_folder: **dataset/traindata_generation**: MATLAB scripts and velocity models for generating the training dataset;
    * :open_file_folder: **dataset/traindata_generation/scripts**: MATLAB scripts for data generation (entry point: `main.m`);
    * :page_facing_up: **dataset/traindata_generation/v_train.mat**: velocity models used during training data generation;

## Supplementary files

To ensure reproducibility, we provide resources for both the training and sampling stages, along with our pre-trained GNO model.

### Training dataset

Due to the large size of the training dataset, we cannot host it directly. Instead, we provide a MATLAB script to reproduce it from scratch. The script and the required velocity models are included directly in this repository.

**Steps to generate the training data:**

1. Navigate to the data generation scripts folder:

```
cd dataset/traindata_generation/scripts
```

2. Open MATLAB and run the entry-point script:
```matlab
   main.m
```
   The script will read the velocity models from `dataset/traindata_generation/v_train.mat` and write the generated `.mat` training files to `dataset/train/`.

> **Note:** Generation time depends on the number of frequency components and shot gathers configured in `main.m`. We recommend running on a machine with sufficient RAM and adjusting the parallelism settings at the top of the script if needed.

### Pre-trained model and test dataset

The pre-trained GNO weights and the test dataset are available for download. *(Add Zenodo / cloud drive link here once available.)*

## Getting started :space_invader: :robot:

To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment. Simply run:

```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. Activate the environment by typing:
```
conda activate gno
```

After that you can simply install your package:
```
pip install .
```
or in developer mode:
```
pip install -e .
```

## Running code :page_facing_up:

Once you have generated (or downloaded) the supplementary files and installed the environment, you can run the training and sampling scripts.

**Training:**
```
python train.py
```

**Sampling / inference:**
```
python sample.py
```

When evaluating the performance of our pre-trained GNO, use the test dataset provided in the supplementary files.

> **Disclaimer:** All experiments were carried out on an Intel(R) Xeon(R) CPU @ 2.10 GHz equipped with a single NVIDIA A100 GPU. Different hardware configurations may require adjustments to the environment setup. If your GPU does not support large batch sizes, reduce the `batch_size` argument in `gno/train.py`.

## Cite us

```bibtex
@article{cheng2025seismic,
  title={Seismic wavefield solutions via physics-guided generative neural operator},
  author={Cheng, Shijun and Taufik, Mohammad H and Alkhalifah, Tariq},
  journal={arXiv preprint arXiv:2503.06488},
  year={2025}
}
@inproceedings{cheng2025generative,
  title={A generative neural operator for seismic wavefield representation},
  author={Cheng, S and Taufik, MH and Alkhalifah, T},
  booktitle={86th EAGE Annual Conference \& Exhibition},
  volume={2025},
  number={1},
  pages={1--5},
  year={2025},
  organization={European Association of Geoscientists \& Engineers}
}
```

