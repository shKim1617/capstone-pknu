from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time

# with로 카메라 장치 확인해서 
# 영상 뽑기 전 세팅까지 되어 있음
class Yolo_camera:
    def __init__(self, pipeline):
        self.nnPath = str((Path(__file__).parent / Path('../models/yolov8n_coco_640x352.blob')).resolve().absolute())
        self.labelMap = [
            "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
            "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
            "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
            "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
            "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
            "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
            "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
            "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
            "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
            "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
            "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
            "teddy bear",     "hair drier", "toothbrush"
        ]
        self.syncNN = True
        
        self.camRgb = pipeline.create(dai.node.ColorCamera)
        self.detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
        self.xoutRgb = pipeline.create(dai.node.XLinkOut)
        self.nnOut = pipeline.create(dai.node.XLinkOut)  
        
    def set_camera(self):
        self.xoutRgb.setStreamName("rgb")
        self.nnOut.setStreamName("nn")

        # Properties
        # 카메라 설정
        # 해상도, 프레임 속도, 색상 순서
        self.camRgb.setPreviewSize(640, 352)
        self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.camRgb.setInterleaved(False)
        self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        self.camRgb.setFps(40)
        
        # Network specific settings
        # 욜로 네트워크 설정
        # 신뢰도 임계값, 객체 클래스 수, IOU 임계값, blop path 설정
        self.detectionNetwork.setConfidenceThreshold(0.5)
        self.detectionNetwork.setNumClasses(80)
        self.detectionNetwork.setCoordinateSize(4)
        self.detectionNetwork.setIouThreshold(0.5)
        self.detectionNetwork.setBlobPath(self.nnPath)
        self.detectionNetwork.setNumInferenceThreads(2)
        self.detectionNetwork.input.setBlocking(False)
        
        # Linking
        # 노드 연결
        self.camRgb.preview.link(self.detectionNetwork.input)
        if self.syncNN:
            self.detectionNetwork.passthrough.link(self.xoutRgb.input)
        else:
            self.camRgb.preview.link(self.xoutRgb.input)

        self.detectionNetwork.out.link(self.nnOut.input)
        

class Depth_camera:
    def __init__(self, pipeline):
        # Closer-in minimum depth, disparity range is doubled (from 95 to 190):
        self.extended_disparity = False
        # Better accuracy for longer distance, fractional disparity 32-levels:
        self.subpixel = False
        # Better handling for occlusions:
        self.lr_check = True

        # Define sources and outputs
        self.monoLeft = pipeline.create(dai.node.MonoCamera)
        self.monoRight = pipeline.create(dai.node.MonoCamera)
        self.depth = pipeline.create(dai.node.StereoDepth)
        self.xout = pipeline.create(dai.node.XLinkOut)

    def set_camera(self):
        self.xout.setStreamName("disparity")

        # Properties
        # 모노카메라는 해상도 설정이 없단다...
        # 이걸 어떻게 해결한담..
        #self.monoLeft.setPreviewSize(640, 352)
        self.monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        self.monoLeft.setCamera("left")
        #self.monoRight.setPreviewSize(640, 352)
        self.monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        self.monoRight.setCamera("right")
        

        # Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
        self.depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
        self.depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        self.depth.setLeftRightCheck(self.lr_check)
        self.depth.setExtendedDisparity(self.extended_disparity)
        self.depth.setSubpixel(self.subpixel)

        # Linking
        self.monoLeft.out.link(self.depth.left)
        self.monoRight.out.link(self.depth.right)
        self.depth.disparity.link(self.xout.input)     
        
def connect_device():
    # Connect to device and start pipeline
    device = dai.Device()
    # Device name
    print('Device name:', device.getDeviceName())
    # Bootloader version
    if device.getBootloaderVersion() is not None:
        print('Bootloader version:', device.getBootloaderVersion())
    # Print out usb speed
    print('Usb speed:', device.getUsbSpeed().name)
    # Connected cameras
    print('Connected cameras:', device.getConnectedCameraFeatures())
    
    return device

def check_device(device):
    if device:
        return True
    else:
        return False

pipeline = dai.Pipeline()

yolo_camera = Yolo_camera(pipeline=pipeline)
yolo_camera.set_camera()

depth_camera = Depth_camera(pipeline=pipeline)
depth_camera.set_camera()