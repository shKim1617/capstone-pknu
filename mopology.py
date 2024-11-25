import cv2
import numpy as np
import matplotlib.pyplot as plt

# ROI 지정
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

def fit_line(x, y, degree=1):
    """
    x, y 데이터로부터 선형(1차) 또는 곡선(2차) 피팅을 수행합니다.
    """
    if len(x) > degree:
        return np.polyfit(y, x, degree)  # y를 기준으로 피팅 (세로 축 기준)
    return None

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


def process_image(image_path):
    """
    이미지 처리 및 차선 검출 과정을 수행합니다.
    """
    # 이미지 로드 및 전처리
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape
    region_of_interest_vertices = [
        #왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래
        (0, 600), (width, 600), (width, height), (0, height)
    ]

    # 노이즈 제거와 엣지 검출
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    cropped_image = region_of_interest(edges, np.array([region_of_interest_vertices], np.int32))

    # 팽창-> 침식으로 엣지 보정
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    processed_image = cv2.erode(cv2.dilate(cropped_image, kernel, iterations=2), kernel, iterations=1)

    # 허프 변환으로 직선 검출
    lines = cv2.HoughLinesP(processed_image, 1, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=25)

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
                if abs(slope) < 0.5: 
                    continue
                # 왼쪽 차선
                if slope < 0: 
                    left_x.extend([x1, x2])
                    left_y.extend([y1, y2])
                # 오른쪽 차선
                else:  
                    right_x.extend([x1, x2])
                    right_y.extend([y1, y2])

    # 이상치 제거
    # if len(left_line_x) > 0 and len(left_line_y) > 0:
    left_x, left_y = remove_outliers(left_x, left_y)
    right_x, right_y = remove_outliers(right_x, right_y)

    # 직선 피팅
    left_fit = fit_line(left_x, left_y, degree=2)
    right_fit = fit_line(right_x, right_y, degree=2)

    # y 값 범위 설정
    y_min, y_max = min(left_y + right_y, default=0), max(left_y + right_y, default=height)
    y_values = np.linspace(y_min, y_max, num=200)

    # 차선 좌표 계산
    left_x_values = calculate_points(left_fit, y_values)
    right_x_values = calculate_points(right_fit, y_values)
    
    # 오른쪽 차선 추정 (왼쪽 차선만 검출된 경우)
    if left_fit is not None and right_x_values is None:
        right_fit = np.copy(left_fit)
        right_fit[0] = -right_fit[0]  # 기울기 반전
        right_x_values = calculate_points(right_fit, y_values) + 100



    # 중앙선 계산
    if left_x_values is not None and right_x_values is not None:
        center_x_values = (left_x_values + right_x_values) / 2
    else:
        center_x_values = None

    # 결과 시각화
    plt.figure(figsize=(10, 5))
    plt.imshow(processed_image, cmap='gray')

    plt.scatter(left_x, left_y, color='blue', label="Left Lane Points", marker="o")
    plt.scatter(right_x, right_y, color='red', label="right Lane Points", marker="x")
    plt.sca

    if left_x_values is not None:
        plt.plot(left_x_values, y_values, color="blue", linewidth=2, label="Left Lane")
    if right_x_values is not None:
        plt.plot(right_x_values, y_values, color="red", linewidth=2, label="Right Lane")
    if center_x_values is not None:
        plt.plot(center_x_values, y_values, color="green", linestyle='--', linewidth=2, label="Center Line")

    plt.legend()
    plt.show()

# 실행
process_image('./images/line3.png')





