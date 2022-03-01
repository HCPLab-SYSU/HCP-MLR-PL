# Multi-label Image Recognition with Partial Labels ![](https://visitor-badge.glitch.me/badge?page_id=HCPLab-SYSU.HCP-MLR-PL) 

Implementation of papers: <a href="https://github.com/putao537/Awesome-Multi-label-Image-Recognition"><img src="https://img.shields.io/badge/Awesome-MLR-blue" alt=""></a> 
- [Structured Semantic Transfer for Multi-Label Recognition with Partial Labels](https://www.aaai.org/AAAI22Papers/AAAI-1133.ChenT.pdf)  
  36th Association for the Advance of Artificial Intelligence (AAAI), 2022.  
  Tianshui Chen*, Tao Pu*, Hefeng Wu, Yuan Xie, Liang Lin.  
- [Semantic-Aware Representation Blending for Multi-Label Image Recognition with Partial Labels](aaai.org/AAAI22Papers/AAAI-1134.PuT.pdf)  
  36th Association for the Advance of Artificial Intelligence (AAAI), 2022.  
  Tao Pu*, Tianshui Chen*, Hefeng Wu, Liang Lin.  

## Preliminary
1. Donwload [data.zip](https://1drv.ms/u/s!ArFSFaZzVErwgXMvjwsvLad6x3S5?e=hbtbTp), and unzip it.
2. Modify the lines 16-19 in config.py.
3. Create servel folders (i.e., "exp/log", "exp/code", "exp/checkpoint", "exp/summary") to record experiment details.


## Usage
1. Run SST
   ```bash
   cd HCP-MLR-PL
   vim scripts/SST.sh
   ./scripts/SST.sh
   ```

2. Run SARB
   ```bash
   cd HCP-MLR-PL
   vim scripts/SARB.sh
   ./scripts/SARB.sh
   ```

## Citation
```
Coming soon.
```

## Contributors
For any questions, feel free to open an issue or contact us:    

* tianshuichen@gmail.com
* putao537@gmail.com
