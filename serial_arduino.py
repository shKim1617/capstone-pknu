import serial
import time

def open_serial_port(port='/dev/ttyACM0', baudrate=9600, timeout=1):
    """
    시리얼 포트를 열고 초기화합니다.
    """
    try:
        py_serial = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=timeout
        )
        time.sleep(2)  # 연결 안정화 대기
        if py_serial.is_open:
            print(f"Serial port {port} opened successfully.")
        return py_serial
    except serial.SerialException as e:
        print(f"Failed to open serial port {port}: {e}")
        return None

def communicate_with_arduino(py_serial, speed, direction, response_timeout=5, check=False):
    """
    Arduino와 통신합니다.
    """
    # 명령 생성
    command = f"{speed} {direction}\n"
    
    # 명령어 전송
    py_serial.write(command.encode())

    # 응답 대기
    start_time = time.time()
    response = ""
    while time.time() - start_time < response_timeout:
        if py_serial.in_waiting > 0:
            try:
                line = py_serial.readline().decode('utf-8', 'ignore').strip()
                if line == "READY":
                    break
                else:
                    response = line
            except UnicodeDecodeError:
                response = "Decode Error"
                break
    
    if check == True:
        print(f"Sending: {command.strip()}")
        if response:    
            print(f"Received: {response}")
        else:
            print("No response received within timeout.")
        print()

def test():
    # 시리얼 포트 열기
    py_serial = open_serial_port()

    if not py_serial:
        return  # 포트를 열지 못하면 종료

    speed = 1500
    direction = 100
    
    try:
        # Arduino와 통신
        while direction != 90:
            speed, direction = map(int, input("input: ").split())
            communicate_with_arduino(py_serial, speed, direction, check=True)
    finally:
        # 시리얼 포트 닫기
        py_serial.close()
        print("Serial port closed.")
