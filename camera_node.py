#!/usr/bin/env python3

import cv2
import depthai as dai
from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time

def create_node():
    # Create pipeline
    pipeline = dai.Pipeline()

    # depth node
    # Define sources and outputs
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    depth = pipeline.create(dai.node.StereoDepth)
    xout = pipeline.create(dai.node.XLinkOut)

    # yolov8_nano node
    camRgb = pipeline.create(dai.node.ColorCamera)
    detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    nnOut = pipeline.create(dai.node.XLinkOut)
    
    depth_node = [monoLeft, monoRight, depth, xout]
    yolo_node = [camRgb, detectionNetwork, xoutRgb, nnOut]
    
    return pipeline, depth_node, yolo_node

def set_yolo(camRgb, detectionNetwork, xoutRgb, nnOut):
    # 욜로 추가 설정
    # Get yolo v8n model blob file path
    nnPath = str((Path(__file__).parent / Path('../models/yolov8n_coco_640x352.blob')).resolve().absolute())
    if not Path(nnPath).exists():
        import sys
        raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

    # yolo v8 abel texts
    # coco 데이터셋 기반으로 총 80개의 객체 유형 검출
    # 여기를 내 마음대로 수정가능?
    labelMap = [
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

    syncNN = True

    xoutRgb.setStreamName("rgb")
    nnOut.setStreamName("nn")

    # Properties
    # 카메라 설정
    # 해상도, 프레임 속도, 색상 순서
    camRgb.setPreviewSize(640, 352)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camRgb.setFps(40)

    # Network specific settings
    # 욜로 네트워크 설정
    # 신뢰도 임계값, 객체 클래스 수, IOU 임계값, blop path 설정
    detectionNetwork.setConfidenceThreshold(0.5)
    detectionNetwork.setNumClasses(80)
    detectionNetwork.setCoordinateSize(4)
    detectionNetwork.setIouThreshold(0.5)
    detectionNetwork.setBlobPath(nnPath)
    detectionNetwork.setNumInferenceThreads(2)
    detectionNetwork.input.setBlocking(False)

    # Linking
    # 노드 연결
    camRgb.preview.link(detectionNetwork.input)
    if syncNN:
        detectionNetwork.passthrough.link(xoutRgb.input)
    else:
        camRgb.preview.link(xoutRgb.input)

    detectionNetwork.out.link(nnOut.input)

with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0
    color2 = (255, 255, 255)

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    # 프레임 정규화 함수
    # 욜로 출력은 0~1 범위이므로 프레임 크기에 맞게 변환
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    # 프레임 표시 함수
    # 검출된 객체를 사각형과 레이블로 표시
    def displayFrame(name, frame):
        color = (255, 0, 0)
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        # Show the frame
        cv2.imshow(name, frame)

    # 메인 루프
    # rgb 프레임과 검출 결과 처리
    # 결과 출력
    # 종료 조건: q
    while True:
        if syncNN:
            inRgb = qRgb.get()
            inDet = qDet.get()
        else:
            inRgb = qRgb.tryGet()
            inDet = qDet.tryGet()

        if inRgb is not None:
            frame = inRgb.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)

        if inDet is not None:
            detections = inDet.detections
            counter += 1

        if frame is not None:
            displayFrame("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break

"""
각각의 노드를 설정하고
가공된 데이터를 받아 그냥.
각 이미지의 이름을 그것에 맞는 이름으로 해서
받아와.
그 다음에 세세한 설정을 들어가자고
알았지?
일단은 욜로부터
"""