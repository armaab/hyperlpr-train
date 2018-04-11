Hyperlpr端到端车牌识别训练脚本
=====

该脚本用于训练[HyperLPR](https://github.com/zeusees/HyperLPR)的端到端识别模型。

依赖
-----
Python3.5+

+ keras
+ tensorflow
+ opencv-python
+ h5py
+ numpy

使用方法
-----

**显示帮助文档**:
```
python main.py --help
```

**将模型导出为h5文件**:
```
python main.py [-num_channels 3] export -m model.h5
```

**训练**:
```
usage: main.py train [-h] -ti TI -tl TL -vi VI -vl VL -b B -img-size IMG_SIZE
                     IMG_SIZE [-pre PRE] [-start-epoch START_EPOCH] -n N
                     [-label-len LABEL_LEN] -c C [-log LOG]

optional arguments:
  -h, --help            show this help message and exit
  -ti TI                训练图片目录
  -tl TL                训练标签文件
  -vi VI                验证图片目录
  -vl VL                验证标签文件
  -b B                  batch size
  -img-size IMG_SIZE IMG_SIZE
                        训练图片宽和高
  -pre PRE              pre trained weight file
  -start-epoch START_EPOCH
  -n N                  number of epochs
  -label-len LABEL_LEN  标签长度
  -c C                  checkpoints format string
  -log LOG              tensorboard 日志目录, 默认为空
```

**训练用法示例**:
```
python main.py train -ti train_imgs -tl train_labels.txt -vi val_imgs -vl val_labels.txt -b 16 -img-size 250 40 -n 100 -c checkpoints/'weights.{epoch:02d}-{val_loss:.2f}.h5' -log log
```
表示训练图片位于`train_imgs`目录下, 训练标签在`train_labels.txt`中。
验证图片位于`val_imgs`目录下, 训练标签在`val_labels.txt`中。
Batch size为16, 图片尺寸为250x40, 到100个epochs时结束训练。
Cehckpoints在`checkpoints`目录下, tensorboard的log目录为`log`。

**注意**: 如果没有修改CNN结构和参数，
训练图片的高度(也就是命令行参数`-img-size`中的第二个参数)必须是40.
如果修改了，需要给出相应高度。

训练数据格式
-------
训练需要准备训练数据集和验证数据集, 每种数据集都包括图片和标签两部分。

### 图片
图片为包含真实车牌的图片（如下图），
所有图片大小必须与命令行参数中指定的相同。

![示例图片](imgs/plate.jpg)

**注意**：1. 车牌不必恰好包含在图片中，最好左右留出一些区域，如上图所示；左右留出区域长度不必相同。
2. 实际上在***一定范围内***，任意放置（只要倾斜角不太大，长度差异不是过大）的车牌都可以识别，但前提是训练数据集要准备得当。

### 标签
标签为UTF-8编码的纯文本文件, 每行的格式如下
```
filename:label
```
其中`filename`是图片名称, 不包含顶级目录<sup>[1](#myfootnote1)</sup>;
label是真实车牌号, 英文字母***必须大写***，label的长度必须与命令行中的参数`-label-len`相同。

<a name="myfootnote1">1</a> 顶级目录在命令行参数中以`-ti`给出
