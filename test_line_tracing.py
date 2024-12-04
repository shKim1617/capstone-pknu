# 라인 트레이싱 테스트

from serial_arduino import open_serial_port, communicate_with_arduino
from drive import set_direction, set_speed
from camera_class import Yolo_camera, Depth_camera, check_device
from line_tracing import detect_road
import depthai as dai
import cv2

# serial port 
py_serial = open_serial_port()

if not py_serial:
    exit()
    
# 카메라 세팅
pipeline = dai.Pipeline()

yolo_camera = Yolo_camera(pipeline=pipeline)
depth_camera = Depth_camera(pipeline=pipeline)

yolo_camera.set_camera()
depth_camera.set_camera()

device = dai.Device(pipeline)

yolo_camera.set2_camera(device)
depth_camera.set2_camera(device)

# 장애물 거리 확인 변수
check_color = 0
count = 0    
    
# 뽈록카메라
# cap = cv2.VideoCapture(0)

# 중앙선 탐지 성공 시 저장한 값
prev_lane_gap = None

# 영상 저장 세팅
# 동영상 저장 설정
fps = 10  # 초당 프레임 수
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 형식
frame_width, frame_height = 640, 352
# VideoWriter 객체 초기화
line = cv2.VideoWriter("line.mp4", fourcc, fps, (frame_width, frame_height))
road = cv2.VideoWriter("road.mp4", fourcc, fps, (frame_width, frame_height//2))
    
speed = 1400

while check_device(device):
    yolo_img, line_img, detections = yolo_camera.return_img()
    depth_img, depth_color_img = depth_camera.return_img()
    # ret, frame = cap.read()
         
    if line_img is None:
        print("no img")
    else:
        road_img, slope, prev_lane_gap, stop_line = detect_road(line_img, prev_lane_gap)
        
        # 이미지 저장
        line_video = cv2.resize(line_img, (frame_width, frame_height))  # 크기 조정
        line.write(line_video)  # 프레임 추가
        
        road_video = cv2.resize(road_img, (frame_width, frame_height)) 
        road.write(road_video)
        
        stop_size = stop_line[2] * stop_line[3]
        if stop_size > 100:
            speed += 10
        
        communicate_with_arduino(py_serial, speed, 90)                  
    if speed == 1500:
        break
    
communicate_with_arduino(py_serial, 1500, 90)  

device.close()