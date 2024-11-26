from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time

# 원본 영상과 욜로 영상 추출
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
        
        # 나중에 추가 세팅 필수
        self.qRgb = None
        self.qDet = None

        self.frame = None
        self.detections = []
        self.startTime = 0
        self.counter = 0
        self.color2 = (255, 255, 255)
        
    def set_camera(self):
        self.xoutRgb.setStreamName("rgb")
        self.nnOut.setStreamName("nn")

        # Properties
        # 카메라 설정
        # 해상도, 프레임 속도, 색상 순서
        self.camRgb.setPreviewSize(640, 352)        # 해상도 설정 불가. detectionNetwork랑 사이즈 안맞음
        self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.camRgb.setInterleaved(False)
        self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        self.camRgb.setFps(40)
        
        # Network specific settings
        # 욜로 네트워크 설정
        # 신뢰도 임계값, 객체 클래스 수, IOU 임계값, blop path 설정
        # 해상도 설정 추가
        # self.detectionNetwork.setPreviewSize(640, 400) 해상도 설정 함수가 존재하지 않음
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
        
    def set2_camera(self, device):
        # 추가 세팅
        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        self.qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        self.qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
        
        self.frame = None
        self.detections = []
        self.startTime = time.monotonic()
        self.counter = 0
        self.color2 = (255, 255, 255)
        
    def frameNorm(self, frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
       
    # 프레임 표시 함수
    # 검출된 객체를 사각형과 레이블로 표시
    # 여기는 나중에 구현 파트에 들어가면 되는 거고
    def displayFrame(self, name, frame):
        color = (255, 0, 0)
        for detection in self.detections:
            bbox = self.frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(frame, self.labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
        return frame
        
    def return_img(self):
        try:
            if self.syncNN:
                inRgb = self.qRgb.get()
                inDet = self.qDet.get()
            else:
                inRgb = self.qRgb.tryGet()
                inDet = self.qDet.tryGet()

            if inRgb is not None:
                frame = inRgb.getCvFrame()
                cv2.putText(frame, "NN fps: {:.2f}".format(self.counter / (time.monotonic() - self.startTime)),
                            (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, self.color2)

            if inDet is not None:
                self.detections = inDet.detections
                self.counter += 1

            if frame is not None:
                frame = self.displayFrame("rgb", frame)
                color_img = inRgb.getCvFrame()
                img = frame.copy()
                return img, color_img, self.detections
            return None, None, []
        except RuntimeError as e:
            print(f"Error in return_img: {e}")
            return None, None, []
        
        

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
        
        self.q = None

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
        
        # 추가함. rgb 영상에 맞게 보정
        # self.depth.setDepthAlign(dai.CameraBoardSocket.RGB)  # RGB 정렬
        # rgb 렌즈 기준으로 영상을 새로 뽑아내는 문제가 발생함. 
        # 적당히 수치를 조정해서 위치를 맞추는 게 좋아 보임

        # Linking
        self.monoLeft.out.link(self.depth.left)
        self.monoRight.out.link(self.depth.right)
        self.depth.disparity.link(self.xout.input)     
        
    def set2_camera(self, device):
        # Output queue will be used to get the disparity frames from the outputs defined above
        self.q = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)   

    def return_img(self):
        inDisparity = self.q.get()  # blocking call, will wait until a new data has arrived
        frame = inDisparity.getFrame()
        # Normalization for better visualization
        frame = (frame * (255 / self.depth.initialConfig.getMaxDisparity())).astype(np.uint8)
        img1 = frame.copy()
        # cv2.imshow("disparity", frame)

        # Available color maps: https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html
        frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        # cv2.imshow("disparity_color", frame)     
        img2 = frame.copy()
        return img1, img2
        
    

        
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
    
def cv2_waitkey():
    if cv2.waitKey(1) == ord('q'):
        return True


# pipeline = dai.Pipeline()

# yolo_camera = Yolo_camera(pipeline=pipeline)
# depth_camera = Depth_camera(pipeline=pipeline)

# yolo_camera.set_camera()
# depth_camera.set_camera()


# device = dai.Device(pipeline)

# yolo_camera.set2_camera(device)
# depth_camera.set2_camera(device)

# # token = 0

# while check_device(device):
#     yolo_img, line_img = yolo_camera.return_img()
#     img1, img2 = depth_camera.return_img()
#     if yolo_img is None:
#         print("no img")
#     else:
#         cv2.imshow("rgb", yolo_img)
#         cv2.imshow("img", img1)
#         cv2.imshow("img2", img2)
#         cv2.imshow("line", line_img)
        
#         # 이미지 해상도 확인
#         # if img1 is not None and token < 3:
#         #     print("Image shape:", img1.shape)  # (Height, Width, Channels)
#         #     print("Height:", img1.shape[0])    # 세로 (Height)
#         #     print("Width:", img1.shape[1])     # 가로 (Width)
#         #     # print("Channels:", img1.shape[2])  # 채널 수 (예: RGB=3, Grayscale=1)
#         #     token += 1
        
#     if cv2.waitKey(1) == ord('q'):
#         break

# device.close()

