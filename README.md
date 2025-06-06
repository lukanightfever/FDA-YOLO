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


Generate the table, and modify the corresponding path: <br>
```
python tables.py
```

Place [yolov7](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) under FDA-YOLO.
