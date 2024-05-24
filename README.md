# Multi-label Image Recognition with Partial Labels 

[![1](https://img.shields.io/badge/SOTA-Leaderboard_On_Microsoft_COCO-blue)](https://paperswithcode.com/sota/multi-label-image-recognition-with-partial)
[![1](https://img.shields.io/badge/SOTA-Leaderboard_On_Pascal_VOC-blue)](https://paperswithcode.com/sota/multi-label-image-recognition-with-partial-1)
[![1](https://img.shields.io/badge/SOTA-Leaderboard_On_Visual_Genome-blue)](https://paperswithcode.com/sota/multi-label-image-recognition-with-partial-2)

Implementation of papers: 

- [Structured Semantic Transfer for Multi-Label Recognition with Partial Labels](https://aaai-2022.virtualchair.net/poster_aaai1133)  
  36th Association for the Advance of Artificial Intelligence (AAAI), 2022.  
  Tianshui Chen*, Tao Pu*, Hefeng Wu, Yuan Xie, Liang Lin. (* equally contributed) 
- [Semantic-Aware Representation Blending for Multi-Label Image Recognition with Partial Labels](https://aaai-2022.virtualchair.net/poster_aaai1134)  
  36th Association for the Advance of Artificial Intelligence (AAAI), 2022.  
  Tao Pu*, Tianshui Chen*, Hefeng Wu, Liang Lin. (* equally contributed) 
- [Heterogeneous Semantic Transfer for Multi-label Recognition with Partial Labels](https://arxiv.org/pdf/2205.11131.pdf)   
  International Journal of Computer Vision (IJCV), 2024.   
  Tianshui Chen*, Tao Pu*, Lingbo Liu, Yukai Shi, Zhijing Yang, Liang Lin. (* equally contributed)   
- [Dual-Perspective Semantic-Aware Representation Blending for Multi-Label Image Recognition with Partial Labels](https://www.sciencedirect.com/science/article/abs/pii/S0957417424003919)   
  Expert Systems with Applications (ESWA), 2024.   
  Tao Pu, Tianshui Chen, Hefeng Wu, Yukai Shi, Zhijing Yang, Liang Lin.   

## Preliminary
1. Donwload data.zip ([[One Drive](https://1drv.ms/u/s!Auj5G110nTE5gjeEHDh17tf_K0zp?e=GIvTvH)] [[Baidu Drive](https://pan.baidu.com/s/11hwhedvUePdGNvW3DSrqQA?pwd=5bxz)]), and unzip it.
3. Modify the lines 16-19 in config.py.
4. Create servel folders (i.e., "exp/log", "exp/code", "exp/checkpoint", "exp/summary") to record experiment details.

## Usage
```
cd HCP-MLR-PL

# modify experiment settings
# range of <#model>: SST, SARB, HST, DSRB
vim scripts/<#model>.sh

./<#model>.sh
```

## Common Issues
### 1. How to generate the partial labels?
Since all the datasets have complete labels, we randomly drop a certain proportion of positive and negative labels to create partially annotated datasets. To control the remaining labels' proportion, we can modify the variable **'prob'** in each file of the directory **'scripts'**. Specifically, we provide the partial labels generating function in **'datasets/coco2014.py'**, **'datasets/vg.py'**, **'datasets/voc2007.py'**. 

As you can find, in each dataset class, we provide two elements of annotations: (1) **'labels'**: original ground truth annotations whose shape is $N * C$; (2) **'changeLabels'**: generated partial labels whose shape is $N * C$. For ease of reproducibility, we freeze the random seed of generating partial labels.

**Notes:** for convenience, we also provide partial labels of each dataset on all known label proportions. ([[One Drive]](https://1drv.ms/u/s!Auj5G110nTE5gjUkjiTzJkgSHOJ3?e=mXmRqe) [[Baidu Drive]](https://pan.baidu.com/s/19R-tWBtsOTbSUphihLXr_g))

## Citation
```
@inproceedings{Chen2022SST,
  author={Chen, Tianshui and Pu, Tao and Wu, Hefeng and Xie, Yuan and Lin, Liang},
  title={Structured semantic transfer for multi-label recognition with partial labels},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  year={2022},
  volume={36},
  number={1},
  pages={339--346}
}

@inproceedings{Pu2022SARB,
  author={Pu, Tao and Chen, Tianshui and Wu, Hefeng and Lin, Liang},
  title={Semantic-aware representation blending for multi-label image recognition with partial labels},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2022},
  volume={36},
  number={2},
  pages={2091--2098}
}

@article{Chen2024HST,
  title={Heterogeneous Semantic Transfer for Multi-label Recognition with Partial Labels},
  author={Chen, Tianshui and Pu, Tao and Liu, Lingbo and Shi, Yukai and Yang, Zhijing and Lin, Liang},
  journal={International Journal of Computer Vision},
  year={2024}
}

@article{Pu2024DSRB,
  title = {Dual-perspective semantic-aware representation blending for multi-label image recognition with partial labels},
  author = {Tao Pu and Tianshui Chen and Hefeng Wu and Yukai Shi and Zhijing Yang and Liang Lin}
  journal = {Expert Systems with Applications},
  year = {2024},
  pages = {123526},
  issn = {0957-4174},
  doi = {https://doi.org/10.1016/j.eswa.2024.123526},
}
```

## Contributors
For any questions, feel free to open an issue or contact us:    

* tianshuichen@gmail.com
* putao537@gmail.com
