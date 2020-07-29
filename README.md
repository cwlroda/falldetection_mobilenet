# Fall Detection using Pose Estimation

## Introduction
Fall Detection model based on a lightweight Pose Estimation model:

https://github.com/PINTO0309/MobileNetV2-PoseEstimation

The model used is MobileNetV2 with OpenVINO.

The detection runs on CPU only, but can be combined with a Neural Compute Stick 2 (NCS2).

## Environment

- Ubuntu 18.04 x86_64
- OpenVINO 2019 R1.0.1
- USB Camera
- Neural Compute Stick 2 (NCS2)
- Google Edge TPU
- Python 3.7.6

## Usage
```console
$ git clone https://github.com/cwlroda/falldetection.git
$ cd falldetection
```
**With Threading (image output only)**
```console
$ python3 main.py -v {VIDEO_PATH}
```
**CPU - Sync Mode (webcam)**  
```console
$ python3 openvino-usbcamera-cpu-ncs2-sync.py -d CPU
```
**CPU - Sync + Boost Mode (webcam)**  
```console
$ python3 openvino-usbcamera-cpu-ncs2-sync.py -d CPU -b True
```
**Help Mode**
```console
$ python3 main.py --help

OR

$ python3 openvino-usbcamera-cpu-ncs2-sync.py --help
```

## Future Plans
1. Model retraining for improved accuracy
2. Compatibility with multiple RTSP streams

