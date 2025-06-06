Towards Oriented Fisheye Object Detection: Dataset and Baseline
===

FishOBB
===
The dataset can be downloaded from [here](https://pan.baidu.com/s/1NaTZuoIslkxCGQQTKQsQRA?pwd=yndg)<br>
The format of the labels can be describe as: <br>
```
cls, x/img_w, y/img_h, w/img_w, h/imgh, theta
```
`cls` means the ID of the category. `theta` denotes the rotation angle of the bounding box and `[x, y, w, h, theta]` is the bounding box of the target. `img_w` and `img_h` are width and height of the image.

Install
===
1.Refer to [YOLOv7](https://github.com/WongKinYiu/yolov7) for the installation.

2.Install [Rotated_IoU](https://github.com/lilanxiao/Rotated_IoU) under folder utils/.

Get Started
===

Place the dataset in the following location:<br>
root/  <br>
├── FDA-YOLO/  <br>
├── yolov7.pt  <br>
└── VOC(vocformat) <br>


Generate mosaic_linetable.txt and mosaic_igtable.txt and modify the corresponding path(Optional, they are already contained in [here](https://pan.baidu.com/s/1NaTZuoIslkxCGQQTKQsQRA?pwd=yndg)): <br>
```
python tables.py
```

Place [yolov7.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) under the folder FDA-YOLO/. <br>

Place [demo.pt](https://pan.baidu.com/s/1PykpLVPb0aN_YsChi-_f5Q?pwd=rngf) under runs/train/voc/weights/. <br>

Train
===
```
python train-voc.py
```

Test
===
```
python test.py
```
