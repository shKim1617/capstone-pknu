#!/usr/bin/env python3

import cv2
import depthai as dai
import time

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
    
def create_pipeline(device):
    # Create pipeline
    pipeline = dai.Pipeline()
    cams = device.getConnectedCameraFeatures()
    streams = []
    for cam in cams:
        print(str(cam), str(cam.socket), cam.socket)
        c = pipeline.create(dai.node.Camera)
        x = pipeline.create(dai.node.XLinkOut)
        c.isp.link(x.input)
        c.setBoardSocket(cam.socket)
        stream = str(cam.socket)
        if cam.name:
            stream = f'{cam.name} ({stream})'
        x.setStreamName(stream)
        streams.append(stream)
    return pipeline, streams
    
def start_pipeline(device, pipeline):
    device.startPipeline(pipeline)
       
def get_imgs(device, streams):

    data = {}  # 각 스트림의 메시지를 저장할 딕셔너리

    if not device.isClosed():
        queueNames = device.getQueueEvents(streams)  # 데이터가 준비된 스트림 이름 가져오기
        for stream in queueNames:
            messages = device.getOutputQueue(stream).tryGetAll()  # 스트림에서 모든 메시지 가져오기
            data[stream] = []  # 각 스트림의 메시지를 초기화
            for message in messages:
                if isinstance(message, dai.ImgFrame):  # 메시지가 이미지 프레임일 경우만 처리
                    data[stream].append(message.getCvFrame())  # OpenCV 형식의 프레임 저장


    return data  # 각 스트림의 프레임을 반환       

def calculate_fps(start_time, frame_count):
    elapsed_time = time.time() - start_time
    if elapsed_time > 0:
        fps = frame_count / elapsed_time
        return fps
    return 0.0

device = connect_device()
pipeline, streams = create_pipeline(device)
start_pipeline(device, pipeline)

start_time = time.time()
frame_count = 0
fps = 0

while(True):
    data = get_imgs(device, streams)
    frame_count += 1
    
    # 프레임 계산
    if time.time() - start_time >= 1.0:
        fps = calculate_fps(start_time, frame_count)
        start_time = time.time()
        frame_count = 0
        
    for stream, frames in data.items():
        for frame in frames:
            # 각 프레임에 대해 처리 (예: OpenCV로 출력)
            cv2.putText(frame, "Fps: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255))
            cv2.imshow(stream, frame)
    if cv2.waitKey(1) == ord('q'):
        device.close()
        break


# 욜로에서 사용하는 카메라
# camRgb.setPreviewSize(640, 352)
# camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
# camRgb.setInterleaved(False)
# camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
# camRgb.setFps(40)

"""
괜찮은 뎁스 카메라 코드 모음

stereo_depth_custom_mesh - 흑백 영상. 가까우면 흰색
depth_preview - 상당히 빠르고 괜찮음. 컬러맵과 흑백맵을 같이 사용함
depth_crop_control - 기존에 사용했던 코드같은데 키보드를 이용해서 화면을 이동가능함. 요상하네
depth_colormap - 먼 곳은 파랑. 가까우면 빨강. 속도도 빠르고 괜찮은데 해상 컬러를 구분하기가 애매하네
"""