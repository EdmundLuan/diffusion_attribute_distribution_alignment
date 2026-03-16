# Diffusion Attribute Distribution Alignment

This repo code implementations for several problem instances of the diffusion attribute distribution alignment (DADA) problem. 
Different licenses may apply for each of the instances as they are builded upon open-source code from others. 

## Human Face Experiment Implementation

The implemenation of our method is released under `human_face/p2_weighting`. 
The codebase builds upon [P2 weighting (CVPR 2022)](https://github.com/jychoi118/P2-weighting). 

### Pretrained Diffusion Weights

- Pretrained DDIM model weights are from [P2 weighting (CVPR 2022)](https://github.com/jychoi118/P2-weighting). 
- Download the `ffhq_p2.pt` file from the cloud drive links provided therein, and move the file to `human_face/p2_weighting/models/`. 


## Citation

```bibtex
@article{luan2026inference,
    title  ={Inference-Time Attribute Distribution Alignment for Unconditional Diffusion},
    author ={Luan, Hao and Ng, See-Kiong and Ling, Chun Kai}, 
    journal={ICLR 2026 ReALM-GEN Workshop}
    year   ={2026}, 
    url    ={https://openreview.net/forum?id=x06yxrU43a}
}
```
