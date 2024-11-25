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

def depth_img_resizing(depth_img):
    
    # 위아래에서 동일하게 잘라내기
    height, width = depth_img.shape[:2]
    crop_top = 24  # 위쪽에서 잘라낼 픽셀 수
    crop_bottom = 24  # 아래쪽에서 잘라낼 픽셀 수

    # 슬라이싱을 이용해 이미지 자르기
    cropped_image = depth_img[crop_top:height-crop_bottom, :]
    
    return cropped_image

def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
       
# 프레임 표시 함수
# 검출된 객체를 사각형과 레이블로 표시
# 여기는 나중에 구현 파트에 들어가면 되는 거고
# 이거 부담이 많이 가니까. 탐지는 하나만 정해서 진행하고
def displayFrame(frame, detections):
    color = (255, 0, 0)
    frame_center_x = frame.shape[1] / 2  # 영상 중심 X 좌표
    frame_center_y = frame.shape[0] / 2  # 영상 중심 Y 좌표
    shift_factor_x = 0.27  # X축 이동 비율
    shift_factor_y = 0.2  # Y축 이동 비율
    
    for detection in detections:
        bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
        
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
        bbox[2] = min(frame.shape[1], bbox[2] + shift_x)  # xmax 이동
        bbox[1] = max(0, bbox[1] + shift_y)  # ymin 이동
        bbox[3] = min(frame.shape[0], bbox[3] + shift_y)  # ymax 이동
        
        
        cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
    return frame

def distance_measurement(img, detections, obj):
    color = (255, 0, 0)
    frame_center_x = img.shape[1] / 2  # 영상 중심 X 좌표
    frame_center_y = img.shape[0] / 2  # 영상 중심 Y 좌표
    shift_factor_x = 0.27  # X축 이동 비율
    shift_factor_y = 0.2  # Y축 이동 비율
    
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
            
            
            cv2.putText(img, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(img, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
    return img

# 하나의 물체만 특정해서 뎁스 이미지에 영상을 그리고 
# 물체의 이미지를 따로 분리
def crop_obj(img, detections, obj):
    color = (255, 0, 0)
    frame_center_x = img.shape[1] / 2  # 영상 중심 X 좌표
    frame_center_y = img.shape[0] / 2  # 영상 중심 Y 좌표
    shift_factor_x = 0.27  # X축 이동 비율
    shift_factor_y = 0.2  # Y축 이동 비율
    
    cropped_objects = []  # 잘라낸 이미지를 저장할 리스트

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
            cropped_objects.append(cropped_img)  # 리스트에 저장
            
            # 원본 이미지에 텍스트와 바운딩 박스 그리기
            cv2.putText(img, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(img, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    
    return img, cropped_objects


pipeline = dai.Pipeline()

yolo_camera = Yolo_camera(pipeline=pipeline)
depth_camera = Depth_camera(pipeline=pipeline)

yolo_camera.set_camera()
depth_camera.set_camera()


device = dai.Device(pipeline)

yolo_camera.set2_camera(device)
depth_camera.set2_camera(device)

# token = 0

while check_device(device):
    yolo_img, line_img, detections = yolo_camera.return_img()
    depth_img, depth_color_img = depth_camera.return_img()
    
    # 뎁스 이미지 가공
    resized_depth_img = depth_img_resizing(depth_color_img)
    # 모든 라벨 검사
    # boxed_depth_img = displayFrame(resized_depth_img, detections)
    # boxed_depth_img = distance_measurement(resized_depth_img, detections, "keyboard")
    boxed_depth_img, cropped_imgs = crop_obj(resized_depth_img, detections, "keyboard")
    
    if yolo_img is None:
        print("no img")
    else:
        # cv2.imshow("yolo_img", yolo_img)
        # cv2.imshow("depth_img", boxed_depth_img)
        # cv2.imshow("line_img", line_img)
        # 잘라낸 객체가 없을 경우 확인
        if not cropped_imgs:
            print("No objects detected with the specified label.")

        # 잘라낸 이미지 표시
        for i, cropped in enumerate(cropped_imgs):
            cv2.imshow(f"Cropped Object {i+1}", cropped)
        
        
        # for detection in detections:
        #     # 바운딩 박스 좌표 (정규화된 값, 0~1 범위)
        #     xmin, ymin, xmax, ymax = detection.xmin, detection.ymin, detection.xmax, detection.ymax

        #     # 검출된 클래스 ID
        #     label_id = detection.label

        #     # 클래스 이름 (labelMap에서 조회)
        #     label_name = labelMap[label_id]

        #     # 신뢰도 (confidence)
        #     confidence = detection.confidence

        #     print(f"Detected: {label_name} ({confidence*100:.2f}%)")
        #     print(f"Bounding Box: ({xmin}, {ymin}), ({xmax}, {ymax})")

        
    if cv2.waitKey(1) == ord('q'):
        break

device.close()
