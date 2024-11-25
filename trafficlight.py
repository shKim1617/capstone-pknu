#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import time
from pathlib import Path
import sys

# MobilenetSSD label texts
labelMap = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Load neural network blob
nnBlobPath = str((Path(__file__).parent / Path('./mobilenet-ssd_openvino_2021.4_6shave.blob')).resolve().absolute())
if len(sys.argv) > 1:
    nnBlobPath = sys.argv[1]

if not Path(nnBlobPath).exists():
    raise FileNotFoundError(f'Required file not found: {nnBlobPath}')

# Initialize pipeline
pipeline = dai.Pipeline()

# Define nodes
camRgb = pipeline.create(dai.node.ColorCamera)
spatialDetectionNetwork = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutNN = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
xoutNN.setStreamName("detections")
xoutDepth.setStreamName("depth")

# Camera and network properties
camRgb.setPreviewSize(300, 300)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
stereo.setSubpixel(True)

spatialDetectionNetwork.setBlobPath(nnBlobPath)
spatialDetectionNetwork.setConfidenceThreshold(0.5)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(5000)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
camRgb.preview.link(spatialDetectionNetwork.input)
spatialDetectionNetwork.out.link(xoutNN.input)
stereo.depth.link(spatialDetectionNetwork.inputDepth)
spatialDetectionNetwork.passthrough.link(xoutRgb.input)
spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

# Helper function: Normalize frame
def frameNorm(frame, bbox):
    norm_vals = np.clip(np.array([
        int(bbox[0] * frame.shape[1]),
        int(bbox[1] * frame.shape[0]),
        int(bbox[2] * frame.shape[1]),
        int(bbox[3] * frame.shape[0])
    ]), 0, None)
    return norm_vals

# Detect traffic light colors
def detect_traffic_light_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define red color range in HSV
    red_lower1 = np.array([0, 70, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 70, 50])
    red_upper2 = np.array([180, 255, 255])

    # Detect red regions
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Calculate red pixel ratio
    red_pixels = cv2.countNonZero(red_mask)
    total_pixels = roi.shape[0] * roi.shape[1]

    # If red pixels exceed 50% of ROI, consider it as "red light"
    if red_pixels / total_pixels > 0.5:
        return "red"
    return None

# Display frame and detect traffic light
def display_frame(name, frame, detections):
    for detection in detections:
        bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
        x1, y1, x2, y2 = bbox

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = labelMap[detection.label] if detection.label < len(labelMap) else "Unknown"
        confidence = f"{int(detection.confidence * 100)}%"
        cv2.putText(frame, f"{label} {confidence}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # traffic light인 경우
        if label == "traffic light":
            roi = frame[y1:y2, x1:x2]

            # Detect circle (traffic light shape)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=5, maxRadius=30
            )
            # 원형 검출
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    x, y, r = circle
                    center_x, center_y = x1 + x, y1 + y
                    cv2.circle(frame, (center_x, center_y), r, (0, 255, 0), 2)
                    cv2.circle(frame, (center_x, center_y), 2, (0, 0, 255), 3)

                    # Detect light color
                    light_color = detect_traffic_light_color(roi)
                    if light_color:
                        cv2.putText(frame, light_color, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        print(f"Traffic Light Detected: {light_color}")

    # Show frame
    cv2.imshow(name, frame)

# Run pipeline
with dai.Device(pipeline) as device:
    preview_queue = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    detection_queue = device.getOutputQueue("detections", maxSize=4, blocking=False)
    depth_queue = device.getOutputQueue("depth", maxSize=4, blocking=False)

    while True:
        in_rgb = preview_queue.get()
        in_det = detection_queue.get()
        frame = in_rgb.getCvFrame()
        detections = in_det.detections

        display_frame("Preview", frame, detections)

        # Exit on 'q'
        if cv2.waitKey(1) == ord('q'):
            break
