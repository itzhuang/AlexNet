# AlexNet
代码基础来自于B站UP霹雳吧啦Wz，在基础上做了一些注释和修改。
此代码用于初学者学习AlexNet网络，任务为花数据集的划分。
## 数据集准备
* 点击[链接](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)下载数据集
* 运行`split_data.py`文件
* 整体`data_set`目录的数据集结构如下：
```
├─flower_photos
│  ├─daisy
│  ├─dandelion
│  ├─roses
│  ├─sunflowers
│  └─tulips
├─train
│  ├─daisy
│  ├─dandelion
│  ├─roses
│  ├─sunflowers
│  └─tulips
└─val
    ├─daisy
    ├─dandelion
    ├─roses
    ├─sunflowers
    └─tulips
```
## 模型训练
运行`train.py`文件，注意数据集路径的选取。可自己定义epochs等参数。
## 模型推理
运行`predict.py`文件，注意输入图像路径的选取。

---
更多详细内容请参考UP主的[讲解视频](https://www.bilibili.com/video/BV1W7411T7qc/?spm_id_from=333.337.search-card.all.click&vd_source=2f28996b0f3cc4d7c53938d9826081a4)
