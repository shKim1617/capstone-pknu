import cv2
import numpy as np
import matplotlib.pyplot as plt

# ROI 지정
# 관심 영역 지정해서 특정 영역만 검사
def region_of_interest(img, vertices):
    mask  = np.zeros_like(img)
    match_mask_color = 255  # Grayscale에서 사용할 마스크 색상
    
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# 이상치 제거 함수
def remove_outliers(x, y, threshold=2):
    """
    Z-점수를 기준으로 이상치 제거
    Args:
        x (list): x 좌표 리스트
        y (list): y 좌표 리스트
        threshold (float): Z-점수 임계값 (기본값: 2)
    Returns:
        filtered_x, filtered_y: 이상치를 제거한 x, y 좌표 리스트
    """
    x, y = np.array(x), np.array(y)

    # Z-score 계산
    x_z_scores = (x - np.mean(x)) / np.std(x)
    y_z_scores = (y - np.mean(y)) / np.std(y)

    # Z-score가 threshold 이하인 값만 유지
    filtered_indices = (np.abs(x_z_scores) < threshold) & (np.abs(y_z_scores) < threshold)

    filtered_x = x[filtered_indices].tolist()
    filtered_y = y[filtered_indices].tolist()

    return filtered_x, filtered_y

# 선 그리기
def fit_line(x, y, degree=1):
    """
    x, y 데이터로부터 선형(1차) 또는 곡선(2차) 피팅을 수행합니다.
    """
    if len(x) > degree:
        return np.polyfit(y, x, degree)  # y를 기준으로 피팅 (세로 축 기준)
    return None

# 그린 선의 기울기 계산
def calculate_points(fit, y_values):
    """
    피팅된 직선 또는 곡선을 기반으로 x 좌표를 계산합니다.
    """
    if fit is not None:
        if len(fit) == 2:  # 1차 직선
            slope, intercept = fit
            return slope * y_values + intercept
        elif len(fit) == 3:  # 2차 곡선
            return abs(fit[0] * y_values**2 + fit[1] * y_values + fit[2])
    return None


# 동영상 파일 경로
video_path = 'images/linevideo.mp4'

# VideoCapture 객체 생성
cap = cv2.VideoCapture(video_path)
prev_lane_gap = None

# 동영상이 열렸는지 확인
if not cap.isOpened():
    print("=======================================")
    print("Error: Cannot open video file")
    print("=======================================")
else:
    while True:
        # 한 프레임씩 읽기
        ret, frame = cap.read()

        # 동영상 끝에 도달했거나 오류 발생 시 중지
        if not ret:
            print("End of video or cannot fetch the frame.")
            break

        height, width, _ = frame.shape
        
        cropped_image = frame[height//2:height, 0:width]

        # 블러링 적용 (GaussianBlur로 노이즈 제거)
        blurred = cv2.GaussianBlur(cropped_image, (5, 5), 0)

        # 프레임을 HSV 색 공간으로 변환
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # 흰색 범위 정의 (HSV 기준)
        lower_white = np.array([0, 0, 200])  # 흰색 하한값
        upper_white = np.array([255, 50, 255])  # 흰색 상한값

        # 흰색 영역 마스크 생성
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # 침식(Erosion) 적용: 작은 잡음 제거
        kernel = np.ones((5, 5), np.uint8)  # 커널 크기 설정
        mask_eroded = cv2.erode(mask, kernel, iterations=1)  # 침식 수행

        # 차선 검출을 할건데..
        lines = cv2.HoughLinesP(mask_eroded, 1, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=25)
        
        # 차선 분류 및 피팅
        left_x, left_y, right_x, right_y = [], [], [], []
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    # 수직선 제거
                    if x2 == x1:
                        continue 
                    slope = (y2 - y1) / (x2 - x1)
                    # 기울기가 0에 가까운 수평선 제거
                    if abs(slope) < 0.7: 
                        continue                        
                    if y1 > y2:
                        if x1 < width//2:
                            left_x.extend([x1, x2])
                            left_y.extend([y1, y2])
                        else:
                            right_x.extend([x1, x2])
                            right_y.extend([y1, y2])
                    else:
                        if x1 < width//2:
                            left_x.extend([x1, x2])
                            left_y.extend([y1, y2])
                        else:
                            right_x.extend([x1, x2])
                            right_y.extend([y1, y2])
                            
        # 이상치 제거
        # if len(left_line_x) > 0 and len(left_line_y) > 0:
        left_x, left_y = remove_outliers(left_x, left_y)
        right_x, right_y = remove_outliers(right_x, right_y)

        # 직선 피팅
        left_fit = fit_line(left_x, left_y, degree=1)
        right_fit = fit_line(right_x, right_y, degree=1)

        # y 값 범위 설정
        y_min, y_max = min(left_y + right_y, default=0), max(left_y + right_y, default=height)
        y_values = np.linspace(y_min, y_max, num=200)

        # 차선 좌표 계산
        left_x_values = calculate_points(left_fit, y_values)
        right_x_values = calculate_points(right_fit, y_values)
        
        
        # 중앙 차선 계산
        center_x_values = None
        if left_x_values is not None and right_x_values is not None:
            # 양쪽 차선이 있는 경우, 평균적인 x값 차이 계산
            lane_gap = np.mean(right_x_values - left_x_values)
            center_x_values = (left_x_values + right_x_values) / 2
            prev_lane_gap = lane_gap  # 평균 차선을 저장
        elif left_x_values is None and right_x_values is not None:
            # 왼쪽 차선이 없는 경우, 이전 프레임의 간격을 기준으로 추정
            if prev_lane_gap is not None:
                left_x_values = right_x_values - prev_lane_gap
                center_x_values = (left_x_values + right_x_values) / 2
        elif left_x_values is not None and right_x_values is None:
            # 오른쪽 차선이 없는 경우, 이전 프레임의 간격을 기준으로 추정
            if prev_lane_gap is not None:
                right_x_values = left_x_values + prev_lane_gap
                center_x_values = (left_x_values + right_x_values) / 2

        # 결과 시각화
        plt.clf()
        # plt.figure(figsize=(10, 5))
        plt.imshow(mask_eroded, cmap='gray')

        plt.scatter(left_x, left_y, color='blue', label="Left Lane Points", marker="o")
        plt.scatter(right_x, right_y, color='red', label="right Lane Points", marker="x")
        # plt.sca

        if left_x_values is not None:
            plt.plot(left_x_values, y_values, color="blue", linewidth=2, label="Left Lane")
        if right_x_values is not None:
            plt.plot(right_x_values, y_values, color="red", linewidth=2, label="Right Lane")
        if center_x_values is not None:
            plt.plot(center_x_values, y_values, color="green", linestyle='--', linewidth=2, label="Center Line")

        plt.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', borderaxespad=0., ncol=3)
        # plt.show()
        plt.pause(0.0001)
        
        # 마스크를 원본 프레임에 적용
        result = cv2.bitwise_and(cropped_image, cropped_image, mask=mask_eroded)

        # 결과 표시
        cv2.imshow('Original Frame', frame)
        # cv2.imshow('Blurred Frame', blurred)
        cv2.imshow('Mask (White Lane)', cropped_image)
        cv2.imshow('Mask After Erosion', mask_eroded)

        # cv2.imshow('Result (White Lane Highlight)', result)

        # q 키를 누르면 종료
        if cv2.waitKey(60) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()#
