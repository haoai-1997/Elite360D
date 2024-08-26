# Elite360D (CVPR2024)

Office source code of paper **Elite360D: Towards Efficient 360 Depth Estimation via Semantic- and Distance-Aware Bi-Projection Fusion**, [Arxiv](https://arxiv.org/abs/2403.16376), [Project]()

# Preparation

#### Installation

Environments

* python 3.10
* Pytorch >= 1.12.0
* CUDA >= 11.3

Install requirements

```bash
pip install -r requirements.txt
```

#### Datasets 

Please download the preferred datasets,  i.e., [Matterport3D](https://niessner.github.io/Matterport/), [Stanford2D3D](http://3dsemantics.stanford.edu/), and [Structured3D](https://structured3d-dataset.org/). For Matterport3D and Stanford2D3D, please preprocess it following [UniFuse](https://github.com/alibaba/UniFuse-Unidirectional-Fusion).
