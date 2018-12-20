# Learning Depth from Monocular Videos using Direct Methods


## 代码修改记录

- master

添加batch以及linklink功能，正在进行baseline训练

- old_master

最初的master，没有什么改动

- add-kpts

之前添加kpts的功能，同时还有可视化图片、val以及保存图片的功能

- obso-multi-batch，obso-re-all-multi-batch，obso-multigpu，obso-linklink，obso-all-multi-batch，obso-dataparallel，batched-depth-pred,obbso-multi-batch-2

都是添加multi-batch时候的中间branch，后来都没有用了

- obso-disable-ddvo

之前尝试把ddvl反传去掉，后来没有意义了

- naive-nonlocal
在add-kpts基础上添加nonlocal，只支持一个batch，暂时的结果并不好

- baseline
在add-kpts后面检测posenet+ddvo的baseline用的

- essential-method
尝试使用直接法的一个branch，具体目的忘了




<img align="center" src="https://github.com/MightyChaos/MightyChaos.github.io/blob/master/projects/cvpr18_chaoyang/demo.gif">

Implementation of the methods in "[Learning Depth from Monocular Videos using Direct Methods](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Learning_Depth_From_CVPR_2018_paper.pdf)".
If you find this code useful, please cite our paper:

```
@InProceedings{Wang_2018_CVPR,
author = {Wang, Chaoyang and Miguel Buenaposada, José and Zhu, Rui and Lucey, Simon},
title = {Learning Depth From Monocular Videos Using Direct Methods},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```
## Dependencies
- Python 3.6
- PyTorch 0.3.1  (latter or eariler version of Pytorch is non-compatible.)

- visdom, dominate


## Training
### data preparation
We refer "[SfMLeaner](https://github.com/tinghuiz/SfMLearner)" to prepare the training data from KITTI. We assume the processed data is put in directory "./data_kitti/".

### training with different pose prediction modules
Start visdom server before for inspecting learning progress before starting the training process.
```
python -m visdom.server -port 8009
```
1. #### train from scratch with PoseNet
```
bash run_train_posenet.sh
```
see [run_train_posenet.sh](https://github.com/MightyChaos/LKVOLearner/blob/master/run_train_posenet.sh) for details.

2. #### finetune with DDVO
Use pretrained posenet to give initialization for DDVO. Corresponds to the results reported as "PoseNet+DDVO" in the paper.
```
bash run_train_finetune.sh
```
see [run_train_finetune.sh](https://github.com/MightyChaos/LKVOLearner/blob/master/run_train_finetune.sh) for details.

## Testing
- Pretrained depth network reported as "Posenet-DDVO(CS+K)" in the paper [[download](https://drive.google.com/file/d/1SJWLfA7kqpERj_U2gYXl7Vuy1eQyOO_K/view?usp=sharing)].
- Depth prediction results on KITTI eigen test split(see Table 1 in the paper):   [[Posenet(K)](https://drive.google.com/open?id=1Wj7ulSimrvrzNx4TRd-JspmX3DJwgPiV)], [[DDVO(K)](https://drive.google.com/open?id=1wiODwgX_Vm_w7fVK1y_X5CNJTtgaPwcN)], [[Posenet+DDVO(K)](https://drive.google.com/open?id=1uUQJLcUOoY2hG6QS_F-wbM3GDAjD-Z5h)],[[Posenet+DDVO(CS+K)](https://drive.google.com/open?id=1hp4zFgK5NSNGdvaQL2ZumeinMQY_-AwK)]

- To test yourself:
```
CUDA_VISIBLE_DEVICES=0 nice -10 python src/testKITTI.py --dataset_root $DATAROOT --ckpt_file $CKPT --output_path $OUTPUT --test_file_list test_files_eigen.txt
```

## Evaluation
We again refer to "[SfMLeaner](https://github.com/tinghuiz/SfMLearner)" for their evaluation code.


## Acknowledgement
Part of the code structure is borrowed from "[Pytorch CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)"
