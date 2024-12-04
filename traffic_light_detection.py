import cv2
from camera_class import Yolo_camera, Depth_camera, check_device
import cv2
import depthai as dai
import numpy as np
from obstacle_detection import detection_and_crop_obj

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

def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
       
# 프레임 표시 함수
# 검출된 객체를 사각형과 레이블로 표시
# 여기는 나중에 구현 파트에 들어가면 되는 거고
def displayFrame_traffic_light(frame, detections):
    color = (255, 0, 0)
    for detection in detections:
        if labelMap[detection.label] == "traffic light":
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cropped_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]  # Y축, X축 순으로 자르기
            return frame, cropped_img
        
    return frame, None

def green_detection(img):
    # BGR을 HSV로 변환
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 초록색 HSV 범위 정의
    lower_green = np.array([35, 100, 100])  # 초록색 범위의 하한값
    upper_green = np.array([85, 255, 255])  # 초록색 범위의 상한값

    # 초록색 마스크 생성
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # 원본 이미지에서 초록색 추출
    result = cv2.bitwise_and(img, img, mask=green_mask)
    
    return result

# 근데 초록말고는 다 색이 애매함. 구별은 하긴 하는데. 오차가 좀 발생함. 뚜렷하지도 않고
# 초록만 사용해서 신호를 판별하도록 합시다
def orange_detection(img):
    # BGR을 HSV로 변환
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 주황색(노란불) HSV 범위 정의
    lower_orange = np.array([12, 150, 150])  # 주황색 하한값
    upper_orange = np.array([20, 255, 255])  # 주황색 상한값

    # 주황색 마스크 생성
    orange_mask = cv2.inRange(hsv_image, lower_orange, upper_orange)

    # 원본 이미지에서 주황색 추출
    result = cv2.bitwise_and(img, img, mask=orange_mask)
    
    return result

def red_detection(img):
    # BGR을 HSV로 변환
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 빨간색 HSV 범위 정의
    lower_red1 = np.array([0, 150, 150])   # 첫 번째 빨강 범위 (하한값 조정)
    upper_red1 = np.array([8, 255, 255])   # 첫 번째 빨강 범위 (상한값 조정)

    lower_red2 = np.array([170, 150, 150]) # 두 번째 빨강 범위 (하한값 조정)
    upper_red2 = np.array([179, 255, 255]) # 두 번째 빨강 범위 (상한값 조정)
    # 빨간색 마스크 생성
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = mask1 | mask2  # 두 마스크를 합침

    # 원본 이미지에서 빨간색 추출
    result = cv2.bitwise_and(img, img, mask=red_mask)
    
    return result

def detect_green_light(img):
    # BGR을 HSV로 변환
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 초록색 HSV 범위 정의
    lower_green = np.array([40, 100, 150])  # 초록색 범위의 하한값
    upper_green = np.array([75, 255, 255])  # 초록색 범위의 상한값

    # 초록색 마스크 생성
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # 초록색 픽셀 개수와 전체 픽셀 개수 계산
    green_pixels = cv2.countNonZero(green_mask)  # 빨간색 픽셀 개수
    total_pixels = img.shape[0] * img.shape[1]  # 전체 픽셀 개수

    # 초록색 비율 계산 (%)
    green_percentage = (green_pixels / total_pixels) * 100

    return green_percentage

# pipeline = dai.Pipeline()

# yolo_camera = Yolo_camera(pipeline=pipeline)
# depth_camera = Depth_camera(pipeline=pipeline)

# yolo_camera.set_camera()
# depth_camera.set_camera()

# device = dai.Device(pipeline)

# yolo_camera.set2_camera(device)
# depth_camera.set2_camera(device)

# check_color = 0
# count = 0

# while check_device(device):
#     yolo_img, line_img, detections = yolo_camera.return_img()
#     # depth_img, depth_color_img = depth_camera.return_img()
    
#     ## detections에서 신호등 부분만 추출
#     detect_traffic_img, traffic_img = displayFrame_traffic_light(line_img, detections)
#     green = green_detection(detect_traffic_img)
#     if traffic_img is not None:
#         cv2.imshow("traffic", traffic_img)
#         green_percentage = detect_green_light(traffic_img)
#         print(green_percentage)
#     orange = orange_detection(detect_traffic_img)
#     red = red_detection(detect_traffic_img)
        
        
#     if yolo_img is None:
#         print("no img")
#     else:
#         cv2.imshow("1", detect_traffic_img)
#         cv2.imshow("green detect", green)
        
#         cv2.imshow("orange detect", orange)
#         cv2.imshow("red detect", red)

        
#     if cv2.waitKey(1) == ord('q'):
#         break

# device.close()
