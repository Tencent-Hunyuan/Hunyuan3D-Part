### ☯️ **Hunyuan3D Part**

Available on [![Hunyuan3D-Studio](https://img.shields.io/badge/Hunyuan3D-Studio-yellow)](https://3d.hunyuan.tencent.com/studio) [![Hunyuan3D](https://img.shields.io/badge/Hunyuan-3D-blue)](https://3d.hunyuan.tencent.com)  

Pipeline of our image to 3D part generation. It contains two key components, P3-SAM and X-Part. The holistic mesh is fed to part detection module P3-SAM to obtain the semantic features, part segmentations and part bounding boxes. Then X-Part generate the complete parts.  
<img src="P3-SAM/images/HYpart-fullpip.jpg" alt="drawing" width="800"/>

###  P3-SAM： Native 3D part Segmentation.   

<!-- <video src="https://murcherful.github.io/P3-SAM/images/method.webm" controls width="800"></video> -->

- Paper: [ https://arxiv.org/abs/2509.06784](https://arxiv.org/abs/2509.06784).  
- Code: [https://github.com/Tencent-Hunyuan/Hunyuan3D-Part/tree/main/P3-SAM/](P3-SAM/).  
- Project Page: [https://murcherful.github.io/P3-SAM/ ](https://murcherful.github.io/P3-SAM/).
- Weights: [https://huggingface.co/tencent/Hunyuan3D-Part](https://huggingface.co/tencent/Hunyuan3D-Part)  
- HuggingFace Demo: [https://huggingface.co/spaces/tencent/Hunyuan3D-Part](https://huggingface.co/spaces/tencent/Hunyuan3D-Part).   
<img src="P3-SAM/images/teaser.jpg" alt="drawing" width="800"/>



###  X-Part： high-fidelity and structure-coherent shape decomposition  

<video src="https://yanxinhao.github.io/Projects/X-Part/assets/videos/xpart-homepage.mp4" controls width="800"></video>

- Paper: [https://arxiv.org/abs/2509.08643](https://arxiv.org/abs/2509.08643).  
- Code: [https://github.com/Tencent-Hunyuan/Hunyuan3D-Part/tree/main/XPart](XPart/).
- Project Page: [https://yanxinhao.github.io/Projects/X-Part/](https://yanxinhao.github.io/Projects/X-Part/).  
- Weights: [https://huggingface.co/tencent/Hunyuan3D-Part](https://huggingface.co/tencent/Hunyuan3D-Part)
- HuggingFace Demo: [https://huggingface.co/spaces/tencent/Hunyuan3D-Part](https://huggingface.co/spaces/tencent/Hunyuan3D-Part).    
<img src="XPart/assets/teaser.jpg" alt="drawing" width="800"/>





### **Notice**    
- **The current release is a light version of X-Part**. The full version is available on [![Hunyuan3D-Studio](https://img.shields.io/badge/Hunyuan3D-Studio-yellow)](https://3d.hunyuan.tencent.com/studio).  
- For X-Part, we recommend using ​​scanned​​ or ​​AI-generated meshes​​ (e.g., from Hunyuan3D V2.5 or V3.0) as input.
- P3-SAM can handle any input mesh. 



#### 🔗 Citation

```
@article{ma2025p3sam,
  title={P3-sam: Native 3d part segmentation},
  author={Ma, Changfeng and Li, Yang and Yan, Xinhao and Xu, Jiachen and Yang, Yunhan and Wang, Chunshi and Zhao, Zibo and Guo, Yanwen and Chen, Zhuo and Guo, Chunchao},
  journal={arXiv preprint arXiv:2509.06784},
  year={2025}
}

@article{yan2025xpart,
  title={X-Part: high fidelity and structure coherent shape decomposition},
  author={Yan, Xinhao and Xu, Jiachen and Li, Yang and Ma, Changfeng and Yang, Yunhan and Wang, Chunshi and Zhao, Zibo and Lai, Zeqiang and Zhao, Yunfei and Chen, Zhuo and others},
  journal={arXiv preprint arXiv:2509.08643},
  year={2025}
}
```
