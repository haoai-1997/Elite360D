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

Please download the preferred datasets,  i.e., [Matterport3D](https://niessner.github.io/Matterport/), [Stanford2D3D](http://3dsemantics.stanford.edu/), and [Structured3D](https://structured3d-dataset.org/). For Matterport3D and Stanford2D3D, please preprocess them following [UniFuse](https://github.com/alibaba/UniFuse-Unidirectional-Fusion).

# Training 

#### ResNet-18 as ERP branch encoder on Matterport3D

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_port 29221 train_elite360d.py --model_name Elite360D_R18 --log_dir ./workdirs --gpu_devices 1 2 --batch_size 4
```

It is similar for other datasets. 

# Evaluation  

```
CUDA_VISIBLE_DEVICES=0 python eval_elite360d.py --model_name $MODEL_NAME --log_dir $LOG_DIR --load_weights_dir $WEIGHTS_DIR --gpu_devices 1
```

## Citation

Please cite our paper if you find our work useful in your research.

```
@inproceedings{ai2024elite360d,
  title={Elite360D: Towards Efficient 360 Depth Estimation via Semantic-and Distance-Aware Bi-Projection Fusion},
  author={Ai, Hao and Wang, Lin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9926--9935},
  year={2024}
}
```
# Acknowledgements

We thank the authors of the projects below:  
*[Unifuse](https://github.com/alibaba/UniFuse-Unidirectional-Fusion)*, *[Panoformer](https://github.com/zhijieshen-bjtu/PanoFormer)*, *[SpherePHD](https://github.com/KAIST-vilab/SpherePHD_public, https://github.com/keevin60907/SpherePHD)*, *[HexRUNet](https://github.com/matsuren/HexRUNet_pytorch)*,
  *[SphereNet](https://github.com/ChiWeiHsiao/SphereNet-pytorch)*,
If you find these works useful, please consider citing:
```
@article{jiang2021unifuse,
      title={UniFuse: Unidirectional Fusion for 360$^{\circ}$ Panorama Depth Estimation}, 
      author={Hualie Jiang and Zhe Sheng and Siyu Zhu and Zilong Dong and Rui Huang},
	  journal={IEEE Robotics and Automation Letters},
	  year={2021},
	  publisher={IEEE}
}
```
```
@inproceedings{shen2022panoformer,
  title={PanoFormer: Panorama Transformer for Indoor 360$$\^{}$\{$$\backslash$circ$\}$ $$ Depth Estimation},
  author={Shen, Zhijie and Lin, Chunyu and Liao, Kang and Nie, Lang and Zheng, Zishuo and Zhao, Yao},
  booktitle={European Conference on Computer Vision},
  pages={195--211},
  year={2022},
  organization={Springer}
}
```
```
@inproceedings{lee2019spherephd,
  title={Spherephd: Applying cnns on a spherical polyhedron representation of 360deg images},
  author={Lee, Yeonkun and Jeong, Jaeseok and Yun, Jongseob and Cho, Wonjune and Yoon, Kuk-Jin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9181--9189},
  year={2019}
}
```
```
@inproceedings{zhang2019orientation,
  title={Orientation-aware semantic segmentation on icosahedron spheres},
  author={Zhang, Chao and Liwicki, Stephan and Smith, William and Cipolla, Roberto},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3533--3541},
  year={2019}
}
```
```
@inproceedings{coors2018spherenet,
  title={Spherenet: Learning spherical representations for detection and classification in omnidirectional images},
  author={Coors, Benjamin and Condurache, Alexandru Paul and Geiger, Andreas},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={518--533},
  year={2018}
}
```