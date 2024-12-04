from serial_arduino import open_serial_port, communicate_with_arduino
from drive import set_direction, set_speed
from camera_class import Yolo_camera, Depth_camera, check_device
from line_tracing import detect_road
from obstacle_detection import detection_and_crop_obj, green_obj, red_obj, yellow_obj
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
count_color = 0    

count = 0
    
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
        
        _, cropped_obj = detection_and_crop_obj(line_img, detections, "bottle")
        if check_color == 0:
            green_percentage = green_obj(cropped_obj)
            if green_percentage > 20:
                count_color += 1
            if count_color == 10:
                check_color = 1
        elif check_color == 1:
            yellow_percentage = yellow_obj(cropped_obj)
            if yellow_percentage > 20:
                count_color += 1
            if count_color == 10:
                check_color = 2
        elif check_color == 2:
            red_percentage = red_obj(cropped_obj)
            if red_percentage > 20:
                count_color += 1
            if count_color == 10:
                check_color = 3
        
        # check color = 3 -> 회피        
        # 움직임 제어
        direction = set_direction(slope) 
        speed = set_speed(slope)
        
        if check_color < 15:
            direction = 75
            speed = 1400
            check_color += 1
        
        communicate_with_arduino(py_serial, speed, direction)
    
    count += 1
    
    if count > 500:
        break

device.close()