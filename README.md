# Models-for-DPICEN 
Models for DPICEN
The code can be found in the master branch


# [RESS 2024] DPICEN code

This is the source code for "<b>DPICEN: Deep physical information consistency embedded network for bearing fault diagnosis under unknown domain</b>". 

## Abstract
In recent years, intelligent transfer models have focused on narrowing the gap between the source domain and target domain data to improve diagnostic effectiveness. However, collecting unlabelled target domain data in advance is challenging, leading to suboptimal performance of domain adaptation models for unknown target domain data. To address this issue, this paper proposes a deep physical information consistency embedded network (DPICEN) for tackling unknown domain bearing fault diagnosis problems. First, a physical information encoder (PIE) is constructed to encode physical information into tensors with values of 0/1. Second, fault samples and their encoded tensors are embedded into a physically consistent space, and the mean squared error (MSE) is employed to reduce the distance between data feature embeddings and physical information embeddings. Subsequently, to further constrain the distribution differences of unknown domain data, a plug-and-play multiple sparse regularization (MSR) algorithm is proposed. Finally, the embedded features are input into a classifier with MSR to achieve bearing fault diagnosis. The results demonstrate the effectiveness and advancement of DPICEN in comparison with 16 related methods in 13 unknown domain fault diagnosis tasks in three bearing datasets. 

## Proposed Network

![image](https://github.com/user-attachments/assets/0cbbf3df-931e-4003-a6ed-67c7d6c9f88d)




## Dataset Preparation

**You can find the dataset here:
【1】Case Western Reserve University Bearing Data Center Website [Online] Available: http://csegroups.case.edu/bearingdatacenter/home [DB]. 
【2】Huang H, Baddour N, Liang M. Multiple time-frequency curve extraction Matlab code and its application to automatic bearing fault diagnosis under time-varying speed conditions [J]. MethodsX, 2019, 6: 1415-32.https://www.sciencedirect.com/science/article/pii/S2215016119301402.
And the paper can be downloaded from my personal homepage [here](https://john-520.github.io/).**


### unknown domain

For example:

```python
mian.py
---dataloaders  #dataset

ulties.py
    '输入原始数据和标签'  
    # data_raw 
    # label
```

## Contact

If you have any questions, please feel free to contact me:

- **Name:** Feiyu Lu
- **Email:** 21117039@bjtu.edu.cn
- **微信公众号:** 轴承智能故障诊断<img width="300" alt="二维码" src="https://github.com/user-attachments/assets/77a67e89-3214-4ff4-8256-01c75ec49e4b">


## Citation

If you find this paper and repository useful, please cite our paper 😊.

```
@article{LU2024110454,
title = {DPICEN: Deep physical information consistency embedded network for bearing fault diagnosis under unknown domain},
journal = {Reliability Engineering & System Safety},
volume = {252},
pages = {110454},
year = {2024},
issn = {0951-8320},
doi = {https://doi.org/10.1016/j.ress.2024.110454},
url = {https://www.sciencedirect.com/science/article/pii/S095183202400526X},
author = {Feiyu Lu and Qingbin Tong and Xuedong Jiang and Ziwei Feng and Ruifang Liu and Jianjun Xu and Jingyi Huo},
keywords = {Fault diagnosis, Physical information, Domain adaptation, Mean squared error, Unknown domain},
}
```

```
@article{LU2024102536,
title = {Towards multi-scene learning: A novel cross-domain adaptation model based on sparse filter for traction motor bearing fault diagnosis in high-speed EMU},
journal = {Advanced Engineering Informatics},
volume = {60},
pages = {102536},
year = {2024},
issn = {1474-0346},
doi = {https://doi.org/10.1016/j.aei.2024.102536},
url = {https://www.sciencedirect.com/science/article/pii/S1474034624001848},
author = {Feiyu Lu and Qingbin Tong and Jianjun Xu and Ziwei Feng and Xin Wang and Jingyi Huo and Qingzhu Wan},
keywords = {Bearing fault diagnosis, Sparse filter, Cross-domain adaptation},
```
