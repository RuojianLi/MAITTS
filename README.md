
| ![Figure1](/figs/MAITTS_structure.png)  |
| :-------------------------------------: |
| *Figure 1. Overall structure of MAITTS* |


## Main Results
![experimental_result](/figs/experimental_result.png)


## Get Started

1. All dependencies of our development environment are listed in file [`conda_env_dependencies.yml`](conda_env_dependencies.yml). You can quickly create a usable python environment with an anaconda command `conda env create -f conda_env_dependencies.yml`.
2. Download data. You can download data from the [highD](https://levelxdata.com/highd-dataset/) website. Place the downloaded data in "\data_preprocessing".
3. Data preprocessing. Run file [highd_preprocessing.py](data_preprocessing\highd_preprocessing.py) to obtain fixed detector and a certain proportion of floating car scene data, which is stored in folder [experimental_data](data_preprocessing\experimental_data).
4. Train the model. We have provided the experimental files of MAITTS in [run_model.py](run_model.py). You can reproduce  experiment results by running the following  code:

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

