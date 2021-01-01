## Dense Pose Object Detector (DPOD)

[PyTorch](https://pytorch.org/) implementation of the DPOD detector based on ICCV 2019 paper "DPOD: 6D Pose Object Detector and Refiner", cf. [References](#references) below.
[**[Full paper]**](https://arxiv.org/pdf/1902.11020.pdf)

<a href="https://www.siemens.com/" target="_blank">
 <img align="right" src="/media/figs/siemens-logo.png" width="15%"/>
</a>

<a href="https://openaccess.thecvf.com/content_ICCV_2019/html/Zakharov_DPOD_6D_Pose_Object_Detector_and_Refiner_ICCV_2019_paper.html" target="_blank">
<img width="80%" src="/media/figs/dpod-teaser.gif"/>
</a>

### Dependencies
* PyTorch (torch) (BSD License: https://github.com/pytorch/pytorch/blob/master/LICENSE)
* OpenCV (cv2) (BSD License: https://opencv.org/license/)
* NumPy (BSD License: https://numpy.org/doc/stable/license.html)
* SciPy (BSD License: https://www.scipy.org/scipylib/license.html)
* scikit-learn (BSD License: https://github.com/scikit-learn/scikit-learn/blob/master/COPYING)
* pandas (BSD License: https://github.com/pandas-dev/pandas/blob/master/LICENSE)
* plyfile (GNU General Public License v3 or later (GPLv3+): https://github.com/dranjan/python-plyfile/blob/master/COPYING)
* PyYAML (MIT License: https://github.com/yaml/pyyaml/blob/master/LICENSE)


## Setting up the environment
Set up a virtual environment using: 
```
conda env create -n dpod -f environment.yml
conda activate dpod
```

## Usage
To test the code activate the created virtual environment and execute the following command:
```
python main.py config.ini -t
```
For training the model run:
```
python main.py config.ini
```

## Datasets
Mini versions of the training and test datasets as well as the 3D models from the [LineMOD dataset](https://bop.felk.cvut.cz/datasets/) are located
in the *db_mini* folder.
* models - 3D models from the [LineMOD dataset](https://bop.felk.cvut.cz/datasets/)
* models_uv - 3D models with UV texture
* test - RGB test images from the [LineMOD dataset](https://bop.felk.cvut.cz/datasets/) 
* train - rendered train patch images, i.e. rgb, correspondences (uv or uvw), normals, 
and sample backgrounds from [MS COCO](https://cocodataset.org/)

Pretrained networks for LineMOD dataset trained on synthetic renderings can be found under the following [link](https://drive.google.com/drive/folders/1oMWzwBb-OP_caSrHNWyeoS-rDhN2Xf8o?usp=sharing).

## References

#### DPOD: 6D Pose Object Detector and Refiner (ICCV 2019)
*Sergey Zakharov\*, Ivan Shugurov\*, Slobodan Ilic*

```
@inproceedings{dpod,
author = {Sergey Zakharov and Ivan Shugurov and Slobodan Ilic},
title = {DPOD: 6D Pose Object Detector and Refiner},
booktitle = {International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}
```