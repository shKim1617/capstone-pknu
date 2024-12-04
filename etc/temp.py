import cv2

file_path = "../images/output_video.avi"  # 동영상 파일 경로
cap = cv2.VideoCapture(file_path)

if not cap.isOpened():
    print("동영상을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:  # 더 이상 프레임을 읽을 수 없는 경우 루프 종료
        print("동영상 재생이 끝났거나 파일을 읽을 수 없습니다.")
        break
    
    cv2.imshow("frame", frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):  # 'q'를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
