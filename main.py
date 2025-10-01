import sys
import time
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Локальные импорты
from utils import visualize
from stepper_pca import Stepper28BYJ_PCA
from opencv_multiplot import Plotter

# Зависимости для оборудования
from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory
import Adafruit_PCA9685

# Глобальные переменные для FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()
DETECTION_RESULT = None

# Переменные для ПИД-контроллера
yErrorSum, yErrorPrev = 0, 0
xErrorSum, xErrorPrev = 0, 0
cord = (0, 0)

def run(model: str, min_detection_confidence: float, min_suppression_threshold: float) -> None:
    global yErrorPrev, yErrorSum, xErrorPrev, xErrorSum, cord

    # --- Инициализация камеры (ИЗМЕНЕНО) ---
    # Захват видео с USB-камеры (индекс 0 - обычно камера по умолчанию)
    cap = cv2.VideoCapture(0)
    # Установка желаемого разрешения
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)


    # --- Инициализация оборудования ---
    stepper_x_channels = [0, 1, 2, 3] 
    servo_y_pin = 18

    # Инициализация PCA9685
    pca = Adafruit_PCA9685.PCA9685()
    pca.set_pwm_freq(60)

    # Инициализация шагового двигателя через PCA9685
    stepper = Stepper28BYJ_PCA(pca, stepper_x_channels)

    # Инициализация сервопривода
    pigpio_factory = PiGPIOFactory()
    servo = AngularServo(servo_y_pin, pin_factory=pigpio_factory)
    
    servoDegree = 0
    servo.angle = servoDegree
    time.sleep(2)

    # --- Параметры визуализации ---
    row_size = 50
    left_margin = 24
    text_color = (0, 255, 0)
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    def save_result(result: vision.FaceDetectorResult, unused_output_image: mp.Image, timestamp_ms: int):
        global FPS, COUNTER, START_TIME, DETECTION_RESULT
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()
        DETECTION_RESULT = result
        COUNTER += 1

    # --- Инициализация модели MediaPipe ---
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.FaceDetectorOptions(base_options=base_options,
                                          running_mode=vision.RunningMode.LIVE_STREAM,
                                          min_detection_confidence=min_detection_confidence,
                                          min_suppression_threshold=min_suppression_threshold,
                                          result_callback=save_result)
    detector = vision.FaceDetector.create_from_options(options)

    # --- Инициализация плоттера ---
    plot = Plotter(700, 250, 4)
    plot.setValName(["Y value", "Y setpoint", "X value", "X setpoint"])

    # --- Основной цикл ---
    while True:
        # Захват кадра с USB-камеры (ИЗМЕНЕНО)
        success, image = cap.read()
        if not success:
            sys.stderr.write("Не удалось получить кадр с камеры.")
            continue

        image = cv2.flip(image, 1)
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        current_frame = image
        
        if DETECTION_RESULT and len(DETECTION_RESULT.detections):
            current_frame, cord = visualize(current_frame, DETECTION_RESULT)
            
            yError, ydir = error(current_frame.shape[0], cord[1])
            xError, xdir = error(current_frame.shape[1], cord[0])

            # --- Логика ПИД-регулятора ---
            yKp, yKi, yKd = 10, 0.003, 14
            yP = yKp * ydir * yError
            yErrorSum += yError
            yI = yKi * ydir * yErrorSum
            yD = yKd * ydir * (yError - yErrorPrev)
            yErrorPrev = yError
            yPidOutput = round(yP + yI + yD)
            
            xKp, xKi, xKd = 10, 0.003, 8
            xP = xKp * xError
            xErrorSum += xError
            xI = xKi * xErrorSum
            xD = xKd * (xError - xErrorPrev)
            xErrorPrev = xError
            xPidOutput = round(abs(xP + xI + xD))

            # --- Управление моторами ---
            servoDegree += yPidOutput
            if not -90 < servoDegree < 90:
                servoDegree = 0
            servo.angle = servoDegree

            if xdir == 1:
                stepper.cwStepping(xPidOutput)
            elif xdir == -1:
                stepper.ccwStepping(xPidOutput)
                
            time.sleep(0.001)

        # Отображение FPS
        fps_text = f'FPS={FPS:.1f}'
        cv2.putText(current_frame, fps_text, (left_margin, row_size), cv2.FONT_HERSHEY_DUPLEX,
                    font_size, text_color, font_thickness, cv2.LINE_AA)

        plot.multiplot([cord[1], current_frame.shape[0] // 2, cord[0], current_frame.shape[1] // 2], "PID Plot")
        cv2.imshow('face_detection', current_frame)

        if cv2.waitKey(1) == ord(' '):
            break
    
    # Освобождение ресурсов (ИЗМЕНЕНО)
    detector.close()
    cap.release()
    cv2.destroyAllWindows()


def error(windowMax, x):
    normalised_adjustment = x / windowMax - 0.5
    adjustment_magnitude = abs(round(normalised_adjustment, 1))
    adjustment_direction = -1 if normalised_adjustment > 0 else 1
    return adjustment_magnitude, adjustment_direction

if __name__ == '__main__':
    run("detector.tflite", 0.6, 0.6)