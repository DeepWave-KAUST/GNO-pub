![LOGO](https://github.com/DeepWave-Kaust/GNO-dev/blob/main/asset/logo_new.jpg)

<div align="center">

<h3><strong>Seismic wavefield solutions via physics-guided generative neural operator</strong></h2>

<h4>Shijun Cheng, Mohammad H. Taufik, Tariq Alkhalifah</h3>

<h4><em>DeepWave Consortium, King Abdullah University of Science and Technology (KAUST)</em></h4>

<p><em>Corresponding author: Shijun Cheng (<a href="mailto:sjcheng.academic@gmail.com">sjcheng.academic@gmail.com</a>)</em></p>

</div>

# Project structure
This repository is organized as follows:

* :open_file_folder: **gno**: python library containing routines for generative neural operator;
* :open_file_folder: **asset**: folder containing logo;
* :open_file_folder: **data**: folder to store dataset;

## Supplementary files
To ensure reproducibility, we provide the the data set for both training and sampling stages and our trainined GNO model. 

* **Training data set**
Since the dataset is so large, we provide a matlab script [here](https://kaust.sharepoint.com/:f:/r/sites/M365_Deepwave_Documents/Shared%20Documents/Restricted%20Area/REPORTS/DW0064/traindata_generation/code?csf=1&web=1&e=HeWbDB) to generate the dataset and the corresponding velocity model [here](https://kaust.sharepoint.com/:f:/r/sites/M365_Deepwave_Documents/Shared%20Documents/Restricted%20Area/REPORTS/DW0064/traindata_generation/dataset?csf=1&web=1&e=Sz1M2h) used for training. You can extract the script and velocity model to generate training and test datasets.

## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
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
When you have downloaded the supplementary files and have installed the environment, you can run the training and sampling code. 
For traning, you can directly run:
```
python train.py
```

For samping, you can directly run:
```
python sample.py
```

When you test the performance of our trained GNO, you can use the test data we provide.

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce A100 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU. If your graphics card does not large batch size training, please reduce the configuration value of args (`batch_size`) in the `gno/train.py` file.

## Cite us 
Cheng et al. (2026) Seismic wavefield solutions via physics-guided generative neural operator.

