# Diffusion Cross-domain Recommendation (DiffCDR)


## Introduction

This repository provides the implementations of DiffCDR, including several baselines used in the paper.


## Requirements

- Python 3.7
- Pytorch
- Pandas
- Numpy
- Tqdm

## Dependent repositories

PTUPCDR repository:  
https://github.com/easezyc/WSDM2022-PTUPCDR  
The implementation of the data processing and several baselines of our code are based on this repository. We modify the original repository by adding new baselines and removing irrelevant parts.​

DPM-Solver:  
https://github.com/LuChengTHU/dpm-solver
The current version may have varied.

## Dataset

The Amazon datasets we used: 
1. CDs and Vinyl: http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_CDs_and_Vinyl_5.json.gz
2. Movies and TV: http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz  
3. Books: http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz

Put the data files in `./data/raw`.

Data process via:
```python
python entry.py --process_data_mid 1 --process_data_ready 1
```

## Experiments

Parameters settings:

- use_cuda: using GPU `1` or CPU as `0`
- task: different tasks within `1, 2 or 3`, default as `1`
- ratio: train/test ratio within `[0.8, 0.2], [0.5, 0.5] or [0.2, 0.8]`, default as `[0.8, 0.2]`
- exp_part: experiments with options `[None_CDR, CDR, ss_CDR, la_CDR,diff_CDR]`
- epoch: pre-training and CDR mapping training epoches, default as `10`
- seed: random seed, default as `1`
- root: root path, default as `./`
- save_path: path to save model files for base models and load model for CDRs, default as `./model_save_default/model.pth`
- diff_lr: learning rate of DiffCDR,default as `0.001`.


You can run models through:

```powershell
# Run the base models and augment model:
python entry.py --exp_part None_CDR 

# Run EMCDR and PTUPCDR
python entry.py --exp_part CDR

# Run  SSCDR
python entry.py --exp_part ss_CDR

# Run  LACDR
python entry.py --exp_part la_CDR

# Run  DiffCDR
python entry.py --exp_part diff_CDR

```

[CMF] Ajit P Singh and Geoffrey J Gordon. 2008. Relational learning via collective matrix factorization. In Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. 650–658.


[EMCDR] Tong Man, Huawei Shen, Xiaolong Jin, and Xueqi Cheng. 2017. Cross-domain recommendation: An embedding and mapping approach.. In IJCAI, Vol. 17. 2464–2470.


[PTUPCDR] Yongchun Zhu, Zhenwei Tang, Yudan Liu, Fuzhen Zhuang, Ruobing Xie, Xu Zhang, Leyu Lin, and Qing He. 2022. Personalized transfer of user preferences for cross-domain recommendation. In Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining. 1507–1515


[SSCDR] SeongKu Kang, Junyoung Hwang, Dongha Lee, and Hwanjo Yu. 2019. Semi-supervised learning for cross-domain recommendation to cold-start users. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 1563–1572.


[LACDR] Tianxin Wang, Fuzhen Zhuang, Zhiqiang Zhang, Daixin Wang, Jun Zhou, and Qing He. 2021. Low-Dimensional Alignment for Cross-Domain Recommendation. Association for Computing Machinery, New York, NY, USA, 3508–3512. 


More hyper-parameter settings can be made in `./code/config.json`.

