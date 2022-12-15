# Multi-label Image Recognition with Partial Labels 

Implementation of papers: ![](https://visitor-badge.glitch.me/badge?page_id=HCPLab-SYSU.HCP-MLR-PL) 

- [Structured Semantic Transfer for Multi-Label Recognition with Partial Labels](https://www.aaai.org/AAAI22Papers/AAAI-1133.ChenT.pdf)  
  36th Association for the Advance of Artificial Intelligence (AAAI), 2022.  
  Tianshui Chen, Tao Pu, Hefeng Wu, Yuan Xie, Liang Lin.  
- [Semantic-Aware Representation Blending for Multi-Label Image Recognition with Partial Labels](https://www.aaai.org/AAAI22Papers/AAAI-1134.PuT.pdf)  
  36th Association for the Advance of Artificial Intelligence (AAAI), 2022.  
  Tao Pu, Tianshui Chen, Hefeng Wu, Liang Lin.  
- [Heterogeneous Semantic Transfer for Multi-label Recognition with Partial Labels](https://arxiv.org/pdf/2205.11131.pdf)   
  Technical Report.   
  Tianshui Chen, Tao Pu, Lingbo Liu, Yukai Shi, Zhijing Yang, Liang Lin.   
- [Semantic-Aware Representation Blending for Multi-Label Image Recognition with Partial Labels](https://arxiv.org/pdf/2205.13092.pdf)   
  Technical Report.   
  Tao Pu, Tianshui Chen, Hefeng Wu, Yongyi Lu, Liang Lin.   

## Preliminary
1. Donwload data.zip ([[One Drive](https://1drv.ms/u/s!ArFSFaZzVErwgXMvjwsvLad6x3S5?e=hbtbTp)] [[Baidu Drive](https://pan.baidu.com/s/11hwhedvUePdGNvW3DSrqQA?pwd=5bxz)]), and unzip it.
2. Modify the lines 16-19 in config.py.
3. Create servel folders (i.e., "exp/log", "exp/code", "exp/checkpoint", "exp/summary") to record experiment details.

## Usage
```
cd HCP-MLR-PL

# modify experiment settings
# range of <#model>: SST, SARB, HST, SARB-journal
vim scripts/<#model>.sh

./<#model>.sh
```

## Common Issues
### 1. How to generate the partial labels?
Since all the datasets have complete labels, we randomly drop a certain proportion of positive and negative labels to create partially annotated datasets. To control the remaining labels' proportion, we can modify the variable **'prob'** in each file of the directory **'scripts'**. Specifically, we provide the partial labels generating function in **'datasets/coco2014.py'**, **'datasets/vg.py'**, **'datasets/voc2007.py'**. 

As you can find, in each dataset class, we provide two elements of annotations: (1) **'labels'**: original ground truth annotations whose shape is $N * C$; (2) **'changeLabels'**: generated partial labels whose shape is $N * C$. For ease of reproducibility, we freeze the random seed of generating partial labels.

**Notes:** for convenience, we also provide partial labels of each dataset on all known label proportions. (see [partial-labels](https://pan.baidu.com/s/19R-tWBtsOTbSUphihLXr_g), password: mc1e)

## Citation
```
@article{Chen2022SST,
  title={Structured Semantic Transfer for Multi-Label Recognition with Partial Labels},
  author={Chen, Tianshui and Pu, Tao and Wu, Hefeng and Xie, Yuan and Lin, Liang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2022}
}

@article{Pu2022SARB,
  title={Semantic-Aware Representation Blending for Multi-Label Image Recognition with Partial Labels},
  author={Pu, Tao and Chen, Tianshui and Wu, Hefeng and Lin, Liang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2022}
}

@article{Chen2022HST,
  title={Heterogeneous Semantic Transfer for Multi-label Recognition with Partial Labels},
  author={Tianshui Chen, Tao Pu, Lingbo Liu, Yukai Shi, Zhijing Yang, Liang Lin},
  journal={arXiv preprint arXiv:2205.11131},
  year={2022}
}

@article{Pu2022SARB-journal,
  title={Semantic-Aware Representation Blending for Multi-Label Image Recognition with Partial Labels},
  author={Tao Pu, Tianshui Chen, Hefeng Wu, Yongyi Lu, Liang Lin},
  journal={arXiv preprint arXiv:2205.13092},
  year={2022}
}
```

## Contributors
For any questions, feel free to open an issue or contact us:    

* tianshuichen@gmail.com
* putao537@gmail.com
