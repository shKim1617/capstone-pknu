from camera_class import Yolo_camera, Depth_camera, check_device
import cv2
import depthai as dai

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
    depth_img, _ = depth_camera.return_img()
    
    # 뎁스 이미지 가공
    
    if yolo_img is None:
        print("no img")
    else:
        cv2.imshow("yolo_img", yolo_img)
        cv2.imshow("depth_img", depth_img)
        cv2.imshow("line_img", line_img)
        for detection in detections:
            # 바운딩 박스 좌표 (정규화된 값, 0~1 범위)
            xmin, ymin, xmax, ymax = detection.xmin, detection.ymin, detection.xmax, detection.ymax

            # 검출된 클래스 ID
            label_id = detection.label

            # 클래스 이름 (labelMap에서 조회)
            # label_name = self.labelMap[label_id]

            # 신뢰도 (confidence)
            confidence = detection.confidence

            print(f"Detected: {label_id} ({confidence*100:.2f}%)")
            print(f"Bounding Box: ({xmin}, {ymin}), ({xmax}, {ymax})")
            
            # 이 바운딩 박스는 0부터 1까지의 숫자다.
            # 뎁스 이미지가 더 크니 위아래를 잘라낸 뒤에
            # 범위를 적용시키면 똑같은 박스가 나올 것이다.
            # 해당 박스의 이미지를 정리하면 거리 검출이 가능하다
        
        # 이미지 해상도 확인
        # if img1 is not None and token < 3:
        #     print("Image shape:", img1.shape)  # (Height, Width, Channels)
        #     print("Height:", img1.shape[0])    # 세로 (Height)
        #     print("Width:", img1.shape[1])     # 가로 (Width)
        #     # print("Channels:", img1.shape[2])  # 채널 수 (예: RGB=3, Grayscale=1)
        #     token += 1
        
    if cv2.waitKey(1) == ord('q'):
        break

device.close()
