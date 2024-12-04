import numpy as np

# 중앙선 각도를 이용해서 각도 설정
# 중앙선 각도가 오른쪽이 음수, 왼쪽이 양수
# 서보 모터

# 값 정규화 함수
def rescale(value, in_min=-0.3, in_max=0.3, out_min=40, out_max=140):
    direction = np.interp(value, [in_min, in_max], [out_min, out_max])
    if value < in_min:
        direction = out_min
    elif value > in_max:
        direction = out_max
    return direction

# 오프셋을 정해서 주면 항상 그렇게 실행될 건데
# 함수 내부 오프셋 값을 넣어주자.
# 이렇게 적으면 항상 적어줘야함
def direction_with_offset(direction):
    offset = 15
    return direction + offset

def set_direction(angle):
    direction = rescale(angle)
    direction = direction_with_offset(direction)
    return direction

## 속도 설정
def set_speed(angle, flag=False):
    if abs(angle) < 10:
        speed = 1400
    else:
        speed = 1420
        
    # 정지 상황
    if flag:
        speed = 1500
        
    return speed