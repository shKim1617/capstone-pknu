from serial_arduino import open_serial_port, communicate_with_arduino
from drive import set_direction, set_speed
from camera_class import Yolo_camera, Depth_camera, check_device
from line_tracing import detect_road
from traffic_light_detection import displayFrame_traffic_light, detect_green_light
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

traffic_count = 0

green_flag = True
    
# 뽈록카메라
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# 중앙선 탐지 성공 시 저장한 값
prev_lane_gap = None
    
while check_device(device):
    yolo_img, line_img, detections = yolo_camera.return_img()
    depth_img, depth_color_img = depth_camera.return_img()
    ret, frame = cap.read()
         
    if line_img is None:
        print("no img")
    else:
        road_img, slope, prev_lane_gap = detect_road(frame, prev_lane_gap)
        
        # 신호등 
        img, cropped_traffic = displayFrame_traffic_light(line_img, detections)
        percent = detect_green_light(cropped_traffic)
        
        if green_flag:
            direction = set_direction(slope) 
            speed = set_speed(slope)
        else:
            # 정지를 해야함
            # 근데 굳이 주황선 앞으로 가서
            direction = set_direction(0)
            speed = set_speed(1500)
        
        communicate_with_arduino(py_serial, speed, direction)
    
    count += 1
    
    if count > 500:
        break

device.close()