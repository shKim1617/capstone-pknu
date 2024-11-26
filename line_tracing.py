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
            return True  # 정지선 감지됨
    return False

# 세팅
lower_orange = np.array([5, 150, 150])
upper_orange = np.array([15, 255, 255])
lower_white = np.array([0, 0, 200])
upper_white = np.array([255, 50, 255])


# 차선 검출
def detect_road(frame):
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
    if detect_stop_line(frame, lower_orange, upper_orange):
        print("주황색 정지선 감지")

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
    center_x_values = (left_x_values + right_x_values) / 2 if left_x_values is not None and right_x_values is not None else None
    
    result = cv2.bitwise_and(cropped_image, cropped_image, mask=mask_eroded)
    
    # Center Line 기울기 계산
    if center_x_values is not None:
        # np.polyfit으로 기울기 계산
        fit_center = np.polyfit(y_values, center_x_values, 1)  # 1차 직선 피팅
        center_slope = fit_center[0]  # 기울기
        # print(f"Center Line 기울기: {center_slope}")
    
    return center_slope, result

# 동영상 처리
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video file")
        return

    # 정지선
    lower_orange = np.array([5, 150, 150])
    upper_orange = np.array([15, 255, 255])
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 50, 255])

    plt.ion()  # 실시간 플롯 갱신을 위해 인터랙티브 모드 활성화

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot fetch the frame.")
            break

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
        if detect_stop_line(frame, lower_orange, upper_orange):
            print("주황색 정지선 감지")

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
        center_x_values = (left_x_values + right_x_values) / 2 if left_x_values is not None and right_x_values is not None else None

        # 결과 시각화
        plt.clf()
        plt.imshow(mask_eroded, cmap="gray")
        plt.scatter(left_x, left_y, color="blue", label="Left Lane Points", marker="o")
        plt.scatter(right_x, right_y, color="red", label="Right Lane Points", marker="x")
        if left_x_values is not None:
            plt.plot(left_x_values, y_values, color="blue", linewidth=2, label="Left Lane")
        if right_x_values is not None:
            plt.plot(right_x_values, y_values, color="red", linewidth=2, label="Right Lane")
        if center_x_values is not None:
            plt.plot(center_x_values, y_values, color="green", linestyle="--", linewidth=2, label="Center Line")
        plt.legend(loc="upper center", ncol=3)
        plt.pause(0.001)

        # 원본 프레임에 마스크 적용 및 표시
        result = cv2.bitwise_and(cropped_image, cropped_image, mask=mask_eroded)
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Mask After Erosion", mask_eroded)

        # q 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# 실행
# process_video("images/drive1.mp4")