# X-Part: high fidelity and structure coherent shape decomposition

**Xinhao Yan, Jiachen Xu, Yang Li, Changfeng Ma, Yunhan Yang, Chunshi Wang, Zibo Zhao, Zeqiang Lai, Yunfei Zhao, Zhuo Chen, Chunchao Guo**

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2509.08643-red)](https://arxiv.org/abs/2509.08643)
[![Hunyuan3D-Studio](https://img.shields.io/badge/Hunyuan3D-Studio-yellow)](https://3d.hunyuan.tencent.com/studio)
[![Hunyuan3D](https://img.shields.io/badge/Hunyuan-3D-blue)](https://3d.hunyuan.tencent.com)

</div>

Generating 3D shapes at part level is pivotal for downstream applications such as mesh retopology, UV mapping, and 3D printing. However, existing part-based generation methods often lack sufficient controllability and suffer from poor semantically meaningful decomposition. To this end, we introduce X-Part, a controllable generative model designed to decompose a holistic 3D object into semantically meaningful and structurally coherent parts with high geometric fidelity. X-Part exploits the bounding box as prompts for the part generation and injects point-wise semantic features for meaningful decomposition. Furthermore, we design an editable pipeline for interactive part generation. Extensive experimental results show that X-Part achieves state-of-the-art performance in part-level shape generation. This work establishes a new paradigm for creating production-ready, editable, and structurally sound 3D assets.

![Teaser](./assets/teaser.jpg)

### TODO List 
- [X] Realse the paper.
- [X] Realse the code.
- [ ] Realse the pre-trained models.

### Install 
1.  We recommend using a virtual environment to install the required packages. 

2. Our code is tested on Python 3.10, PyTorch 2.4.0+cu121 and CUDA 12.1. You can install them according to your system.

3. Install the required packages of [Sonata](https://github.com/facebookresearch/sonata).

<!-- 4. Then you can install the package by running:
    ```
    pip install viser fpsample trimesh numba
    ``` -->

### Inference
1. Download the pre-trained models (TODO)
2. Run the following command to automantically generate the masks:
    ```
    python demo.py --config partgen/config/infer.yaml --mesh_path ./data/test.glb --output_path ./results/
    ```



### Citation
```
@article{yan2025x,
  title={X-Part: high fidelity and structure coherent shape decomposition},
  author={Yan, Xinhao and Xu, Jiachen and Li, Yang and Ma, Changfeng and Yang, Yunhan and Wang, Chunshi and Zhao, Zibo and Lai, Zeqiang and Zhao, Yunfei and Chen, Zhuo and others},
  journal={arXiv preprint arXiv:2509.08643},
  year={2025}
}
```