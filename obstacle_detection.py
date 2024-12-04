import cv2
from camera_class import Yolo_camera, Depth_camera, check_device
import cv2
import depthai as dai
import numpy as np

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
       

# 하나의 물체만 특정해서 뎁스 이미지에 영상을 그리고 
# 물체의 이미지를 따로 분리
def detection_and_crop_obj(img, detections, obj):
    color = (255, 0, 0)
    frame_center_x = img.shape[1] / 2  # 영상 중심 X 좌표
    frame_center_y = img.shape[0] / 2  # 영상 중심 Y 좌표
    shift_factor_x = 0.27  # X축 이동 비율
    shift_factor_y = 0.2  # Y축 이동 비율
    
    cropped_img = None  # 잘라낸 이미지를 저장할 리스트

    for detection in detections:
        if labelMap[detection.label] == obj:
            bbox = frameNorm(img, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            
            # 바운딩 박스 중심 계산
            bbox_center_x = (bbox[0] + bbox[2]) / 2
            bbox_center_y = (bbox[1] + bbox[3]) / 2  # Y축 좌표 수정

            # 중심점 거리 계산
            distance_from_center_x = frame_center_x - bbox_center_x
            distance_from_center_y = frame_center_y - bbox_center_y

            # 이동량 계산
            shift_x = int(distance_from_center_x * shift_factor_x)
            shift_y = int(distance_from_center_y * shift_factor_y)

            # 좌표 이동 (화면 경계 제한)
            bbox[0] = max(0, bbox[0] + shift_x)  # xmin 이동
            bbox[2] = min(img.shape[1], bbox[2] + shift_x)  # xmax 이동
            bbox[1] = max(0, bbox[1] + shift_y)  # ymin 이동
            bbox[3] = min(img.shape[0], bbox[3] + shift_y)  # ymax 이동

            # 사각형 내부 이미지 추출
            cropped_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]  # Y축, X축 순으로 자르기
            
            # 원본 이미지에 텍스트와 바운딩 박스 그리기
            cv2.putText(img, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(img, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    
    return img, cropped_img

# 뎁스 이미지를 색상 별로 구분하는 테스트 코드들
# red, yellow, green detection func
def red_detection(img):
    # BGR을 HSV로 변환
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 빨간색 HSV 범위 정의
    lower_red1 = np.array([0, 100, 100])   # 첫 번째 빨강 범위
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 100, 100]) # 두 번째 빨강 범위
    upper_red2 = np.array([180, 255, 255])

    # 빨간색 마스크 생성
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = mask1 | mask2  # 두 마스크를 합침

    # 원본 이미지에서 빨간색 추출
    result = cv2.bitwise_and(img, img, mask=red_mask)
    
    return result

def yellow_detection(img):
    # BGR을 HSV로 변환
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 노란색 HSV 범위 정의
    lower_yellow = np.array([20, 100, 100])  # 노란색 범위의 하한값
    upper_yellow = np.array([30, 255, 255])  # 노란색 범위의 상한값

    # 노란색 마스크 생성
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # 원본 이미지에서 노란색 추출
    result = cv2.bitwise_and(img, img, mask=yellow_mask)
    
    return result

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

# 추출한 오브젝트를 뎁스카메라로 검사하여 색상 구별로 거리 판단
def green_obj(img):
    # BGR을 HSV로 변환
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 초록색 HSV 범위 정의
    lower_green = np.array([35, 100, 100])  # 초록색 범위의 하한값
    upper_green = np.array([85, 255, 255])  # 초록색 범위의 상한값

    # 초록색 마스크 생성
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # 빨간색 픽셀 개수와 전체 픽셀 개수 계산
    green_pixels = cv2.countNonZero(green_mask)  # 빨간색 픽셀 개수
    total_pixels = img.shape[0] * img.shape[1]  # 전체 픽셀 개수

    # 빨간색 비율 계산 (%)
    green_percentage = (green_pixels / total_pixels) * 100

    return green_percentage

def yellow_obj(img):
    # BGR을 HSV로 변환
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 노란색 HSV 범위 정의
    lower_yellow = np.array([20, 100, 100])  # 노란색 범위의 하한값
    upper_yellow = np.array([30, 255, 255])  # 노란색 범위의 상한값

    # 노란색 마스크 생성
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # 빨간색 픽셀 개수와 전체 픽셀 개수 계산
    yellow_pixels = cv2.countNonZero(yellow_mask)  # 빨간색 픽셀 개수
    total_pixels = img.shape[0] * img.shape[1]  # 전체 픽셀 개수

    # 빨간색 비율 계산 (%)
    yellow_percentage = (yellow_pixels / total_pixels) * 100

    return yellow_percentage

def red_obj(img):
    # BGR을 HSV로 변환
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 빨간색 HSV 범위 정의
    lower_red1 = np.array([0, 100, 100])   # 첫 번째 빨강 범위
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 100, 100]) # 두 번째 빨강 범위
    upper_red2 = np.array([180, 255, 255])

    # 빨간색 마스크 생성
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = mask1 | mask2  # 두 마스크를 합침

    # 빨간색 픽셀 개수와 전체 픽셀 개수 계산
    red_pixels = cv2.countNonZero(red_mask)  # 빨간색 픽셀 개수
    total_pixels = img.shape[0] * img.shape[1]  # 전체 픽셀 개수

    # 빨간색 비율 계산 (%)
    red_percentage = (red_pixels / total_pixels) * 100

    return red_percentage


## test code

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
#     depth_img, depth_color_img = depth_camera.return_img()
    
#     boxed_depth_img, cropped_obj = detection_and_crop_obj(depth_color_img, detections, "keyboard")
    
#     # 테스트용
#     # 빨강 추출
#     red_img = red_detection(boxed_depth_img)
#     yellow_img = yellow_detection(boxed_depth_img)
#     green_img = green_detection(boxed_depth_img)
    
#     # 이제 이걸 가져와서 색상 구별
#     # 그냥 냅다 확인을 하면 안되고 
#     # 일정 퍼센트를 넘기는 것만 카운트해서 n 번 이상 카운트 되면 다음 단계로 넘어가게?
#     # 아니면 셋 다 검사를 하는데 일정 카운트 이상 쌓이면 그 거리에 있다고 인식하기?           
    
#     # 여기 파트를 나중에 떼고
#     if cropped_obj is not None:
#         if check_color == 0:
#             green = green_obj(cropped_obj)
#             if green > 30:
#                 count += 1
#                 if count > 10:
#                     count = 0
#                     check_color = 1
#                     print("detect green")
#         elif check_color == 1:
#             yellow = yellow_obj(cropped_obj)
#             if yellow > 30:
#                 count += 1
#                 if count > 10:
#                     count = 0
#                     check_color = 2
#                     print("detect yellow")
#         elif check_color == 2:
#             red = red_obj(cropped_obj)
#             if red > 30:
#                 count += 1
#                 if count > 10:
#                     count = 0
#                     check_color = 3
#                     print("detect red")
        
        
#     if yolo_img is None:
#         print("no img")
#     else:
#         cv2.imshow("1", red_img)
#         # cv2.imshow("2", line_img)
#         cv2.imshow("detect", yellow_img)
#         cv2.imshow("red", green_img)

        
#     if cv2.waitKey(1) == ord('q'):
#         break

# device.close()
