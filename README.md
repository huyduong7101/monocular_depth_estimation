# Dự đoán độ sâu sử dụng phương pháp học Self-supervised

Đây là phần implementation cho training và evaluation cho bài toán dự đoán độ sâu được đề xuất trong

>[Dự đoán độ sâu sử dụng phương pháp học Self-supervised](./miniproject-report-LeHuyDuong-official.pdf) - Viettel Digital Talent 2022 - Giai đoạn 1
>
>Dương Lê Huy - huyduong7101@gmail.com, Nam Nguyễn Văn - namnv78@viettel.com.vn

<p align="center">
  <img src="assets/kitti_input/26_20.png" alt="example input output gif" width="600" />
</p>
<p align="center">
  <img src="assets/densenet/26_20_disp.jpeg" alt="example input output gif" width="600" />
</p>

**Note:** Phần implementation gốc là từ [Monodepth2](https://github.com/nianticlabs/monodepth2), chúng tôi đã chỉnh sửa lại phần code để phù hợp với mô hình được đề xuất trong [report](./miniproject-report-LeHuyDuong-official.pdf).

## Update
**2022.1.21**
1. Phiên bản chính thức đầu tiên

## Setup
Chúng tôi huấn luyện mô hình trên Ubuntu 18.04, Python 3.6.6, PyTorch 1.10.1 và CUDA 11.1

## KITTI training data

Bạn có thể donwload bộ dữ liệu [KITTI_raw dataset]() bằng câu lệnh:

```shell
wget -i splits/kitti_archives_to_download.txt -P kitti_data/
```

Sau đó unzip với

```shell
cd kitti_data
unzip "*.zip"
cd ..
```

<font color=blue>**Warning:**</font> <font color=white>Bộ dữ liệu KITTI RAW có dung lượng khoảng 175GB</font>

## Training

Tham số model và các tensorboard event files được lưu mặc định trong `history` folder.
Bạn có thể thay đổi bởi --log_dir flag

```shell
CUDA_VISIBLE_DEVICES=0 python train.py --model_name densenet-hr-depth --split eigen_zhou --backbone densenet --depth_decoder hr-depth --png
```

## KITTI evaluation

Xây dựng ground truth cho bộ evaluation
```shell
python export_gt_depth.py --data_path kitti_data --split eigen
python export_gt_depth.py --data_path kitti_data --split eigen_benchmark
```

Evaluation với weight điều chỉnh từ --load_weights_folder flag với --eval_split default là eigen
```shell
python evaluate_depth.py --load_weights_folder ./densenet/models/weights_19/ --eval_mono --backbone densenet --depth_decoder hr-depth
```

## Prediction for a single image

Bạn có thể dự đoán depth map từ một ảnh đơn với câu lệnh

```shell
python test_simple.py --image_path assets/test_image.jpg --model_name densenet-hr
```





