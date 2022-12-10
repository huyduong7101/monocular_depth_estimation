# Monocular depth estimation with Self-supervised method - Viettel Digital 2022
# 1. Overview
- Problem: Depth Estimation
<p align="center">
  <img src="assets/teaser.gif" alt="example input output gif" width="80%" />
</p>

- Method: Monocular depth estimation with self-supervised, based on [Monodepth2](https://github.com/nianticlabs/monodepth2) and HRDepth. Our proposal methods are detailed in folder [document]("document")

- Dataset: KITTI

# 2. Set up
We ran our experiments with PyTorch 1.10.1, CUDA 11.1, Python 3.6.6 and Ubuntu 18.04

## KITTI training data

You can download the entire [KITTI_raw dataset]() by running:

```shell
wget -i splits/kitti_archives_to_download.txt -P kitti_data/
```

Then unzip with

```shell
cd kitti_data
unzip "*.zip"
cd ..
```

<font color=blue>**Warning:**</font> <font color=white>it weighs about 175GB, so make sure you have enough space to unzip too!</font>

# 3. How to run
We have two versions corresponding to [VDT_Phase1]("document/Report_VDT_2022_Phase1.pdf") and [VDT_Phase2]("document/Report_VDT_2022_Phase2.pdf"). 

To run [VDT_Phase1]("document/Report_VDT_2022_Phase1.pdf")
```shell
CUDA_VISIBLE_DEVICES=0 python train.py --model_name densenet-hr-depth --split eigen_zhou --backbone densenet --depth_decoder hr-depth --png
```

To run [VDT_Phase2]("document/Report_VDT_2022_Phase2.pdf")
```shell
CUDA_VISIBLE_DEVICES=0 python train_v2.py
```

## 📊 KITTI evaluation

To prepare the ground truth depth maps run:
```shell
python export_gt_depth.py --data_path kitti_data --split eigen
python export_gt_depth.py --data_path kitti_data --split eigen_benchmark
```
...assuming that you have placed the KITTI dataset in the default location of `./kitti_data/`.

The following example command evaluates the epoch 19 weights of a model named `densenet`:
```shell
python evaluate_depth.py --load_weights_folder ./densenet/models/weights_19/ --eval_mono --backbone densenet --depth_decoder hr-depth
```





