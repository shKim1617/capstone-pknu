import cv2
import numpy as np
import matplotlib.pyplot as plt


# 관심 영역 지정 함수
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    return cv2.bitwise_and(img, mask)


# 이상치 제거 함수 (Z-score 기반)
def remove_outliers(x, y, threshold=2):
    x, y = np.array(x), np.array(y)
    x_z = (x - np.mean(x)) / np.std(x)
    y_z = (y - np.mean(y)) / np.std(y)
    valid_indices = (np.abs(x_z) < threshold) & (np.abs(y_z) < threshold)
    return x[valid_indices].tolist(), y[valid_indices].tolist()


# 직선 또는 곡선 피팅 함수
def fit_line(x, y, degree=1):
    if len(x) > degree:
        return np.polyfit(y, x, degree)
    return None


# 피팅된 결과로 x 좌표 계산
def calculate_points(fit, y_values):
    if fit is not None:
        if len(fit) == 2:  # 1차 직선
            slope, intercept = fit
            return slope * y_values + intercept
        elif len(fit) == 3:  # 2차 곡선
            return fit[0] * y_values**2 + fit[1] * y_values + fit[2]
    return None


# 정지선 감지 함수
def detect_stop_line(frame, lower_orange, upper_orange):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # 작은 영역 무시
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            stop_size = [x, y, w, h]
            return stop_size  # 정지선 감지됨
    return None

# 세팅
lower_orange = np.array([5, 150, 150])
upper_orange = np.array([15, 255, 255])
lower_white = np.array([0, 0, 200])
upper_white = np.array([255, 50, 255])


# 차선 검출
def detect_road(frame, prev_lane_gap):
    height, width, _ = frame.shape
    cropped_image = frame[height // 2 :, :]  # 하단 영역 자르기
    blurred = cv2.GaussianBlur(cropped_image, (5, 5), 0)

    # 흰색 차선 검출
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    kernel = np.ones((5, 5), np.uint8)
    mask_eroded = cv2.erode(mask, kernel, iterations=1)

    # 허프 변환을 통한 차선 검출
    lines = cv2.HoughLinesP(mask_eroded, 1, np.pi / 180, 100, minLineLength=40, maxLineGap=25)

    # 정지선 감지
    stop_line = None
    stop_line = detect_stop_line(frame, lower_orange, upper_orange) 
        
         

    # 차선 분류 및 좌표 수집
    left_x, left_y, right_x, right_y = [], [], [], []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 == x1:  # 수직선 제거
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.7:  # 거의 수평인 선 제거
                    continue
                if x1 < width // 2:  # 왼쪽 차선
                    left_x.extend([x1, x2])
                    left_y.extend([y1, y2])
                else:  # 오른쪽 차선
                    right_x.extend([x1, x2])
                    right_y.extend([y1, y2])

    # 이상치 제거
    left_x, left_y = remove_outliers(left_x, left_y)
    right_x, right_y = remove_outliers(right_x, right_y)

    # 직선 피팅
    left_fit = fit_line(left_x, left_y)
    right_fit = fit_line(right_x, right_y)

    # y 값 범위 설정 및 좌표 계산
    y_min, y_max = min(left_y + right_y, default=0), max(left_y + right_y, default=height)
    y_values = np.linspace(y_min, y_max, num=200)
    left_x_values = calculate_points(left_fit, y_values)
    right_x_values = calculate_points(right_fit, y_values)
    
    # 중앙 차선 계산
    center_x_values = None
    if left_x_values is not None and right_x_values is not None:
        # 양쪽 차선이 있는 경우, 평균적인 x값 차이 계산
        lane_gap = np.mean(right_x_values - left_x_values)
        center_x_values = (left_x_values + right_x_values) / 2
        prev_lane_gap = lane_gap  # 평균 차선을 저장
    # elif left_x_values is None and right_x_values is not None:
    #     # 왼쪽 차선이 없는 경우, 이전 프레임의 간격을 기준으로 추정
    #     if prev_lane_gap is not None:
    #         left_x_values = right_x_values - prev_lane_gap
    #         center_x_values = (left_x_values + right_x_values) / 2
    # elif left_x_values is not None and right_x_values is None:
    #     # 오른쪽 차선이 없는 경우, 이전 프레임의 간격을 기준으로 추정
    #     if prev_lane_gap is not None:
    #         right_x_values = left_x_values + prev_lane_gap
    #         center_x_values = (left_x_values + right_x_values) / 2
    else:
        if prev_lane_gap is not None:
            left_x_values = right_x_values - prev_lane_gap
            center_x_values = (left_x_values + right_x_values) / 2
    
    result = cv2.bitwise_and(cropped_image, cropped_image, mask=mask_eroded)
    # Center Line 기울기 계산
    center_slope = None
    if center_x_values is not None:
        # np.polyfit으로 기울기 계산
        fit_center = np.polyfit(y_values, center_x_values, 1)  # 1차 직선 피팅
        center_slope = fit_center[0]  # 기울기
        # print(f"Center Line 기울기: {center_slope}")
        
        
        
    
    
    return result, center_slope, prev_lane_gap, stop_line

# 동영상 넣어서 확인
# 동영상 파일 경로
video_path = "images/line.mp4"

# 동영상 열기
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("동영상을 열 수 없습니다.")
    exit()

print("동영상을 재생합니다. 'q'를 눌러 종료하세요.")

prev_lane_gap = None

while True:
    # 한 프레임 읽기
    ret, frame = cap.read()
    
    if not ret:  # 더 이상 읽을 프레임이 없으면 종료
        print("동영상 재생이 끝났습니다.")
        break
    
    # 프레임 표시
    cv2.imshow("Frame", frame)
    img, slope, prev_lane_gap = detect_road(frame, prev_lane_gap)
    cv2.imshow("img", img)
    
    
    # 'q'를 누르면 종료
    if cv2.waitKey(30) & 0xFF == ord('q'):  # 30ms 딜레이 (FPS 조절 가능)
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()