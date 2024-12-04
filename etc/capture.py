import cv2
from line_tracing import detect_road

def capture_from_camera(camera_index=0, save_image=False):
    # 카메라 연결 (camera_index는 기본값 0)
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다. 연결 상태를 확인하세요.")
        return

    print("카메라가 활성화되었습니다. ESC 키를 눌러 종료하세요.")
    prev_lane_gap = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽는 데 실패했습니다.")
            break

        # 캡처한 이미지 화면에 표시
        cv2.imshow("Camera Capture", frame)

        # 키보드 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC 키를 누르면 종료
            break
        elif key == ord('s') and save_image:  # 's' 키를 누르면 이미지 저장
            cv2.imwrite("captured_image.jpg", frame)
            print("이미지가 저장되었습니다: captured_image.jpg")
        
        img, slope, prev_lane_gap = detect_road(frame, prev_lane_gap)
        if slope is not None:
            print(slope)
        cv2.imshow("result", img)



    # 리소스 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_from_camera(camera_index=1, save_image=True)
