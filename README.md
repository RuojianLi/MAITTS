# MAITTS (IEEE ITSC 2024 paper)

<font size=5.5>Multi-scale traffic time series imputation and observed reconstruction: A joint multi-task training approach</font>

Our study introduces a multi-task training approach for missing data imputation in traffic time series and observed reconstruction. 

Given the substantial and intricate seasonality inherent in traffic time series, this paper employs Fast Fourier Variation to transform one-dimensional multi-scale time series into two-dimensional data, thereby representing the sequence information. Additionally, the MLP structure is utilized to capture sequence features across both Intraperiod Variation and Interperiod Variation. 

Our study introduces employing Diagonally-Masked Self-Attention mechanism to accomplish the observation reconstruction task. 

## MAITTS: Mit block&Attention-based Imputation for Traffic Time Series
|  ![Figure1](figs\MAITTS_structure.PNG)  |
| :-------------------------------------: |
| *Figure 1. Overall structure of MAITTS* |


## Main Results
![experimental_result](figs\experimental_result.png)


## Get Started

1. All dependencies of our development environment are listed in file [`conda_env_dependencies.yml`](conda_env_dependencies.yml). You can quickly create a usable python environment with an anaconda command `conda env create -f conda_env_dependencies.yml`.
2. Download data. You can download data from the [highD](https://levelxdata.com/highd-dataset/) website. Place the downloaded data in "\data\data_preprocessing".
3. Train the model. We have provided the experimental files of MAITTS in "run_model.py". You can reproduce  experiment results by running the following  code:

```bash
python run_model.py
```


## Citation

If you find this repo useful, please cite our paper. 

```

```




## Contact

If you have any question or want to use the code, please contact liruojian@zju.edu.cn

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base:

https://github.com/WenjieDu/PyPOTS

https://github.com/thuml/TimesNet

https://github.com/thuml/Time-Series-Library

