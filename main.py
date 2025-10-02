import cv2
import numpy as np
import time
from adafruit_servokit import ServoKit
import RPi.GPIO as GPIO 

# --- НАСТРОЙКИ ШАГОВОГО ДВИГАТЕЛЯ ---
# Укажите пины GPIO, к которым подключены IN1, IN2, IN3, IN4 драйвера
STEPPER_PINS = [17, 18, 27, 22] 
# Задержка между шагами. Уменьшите для увеличения скорости, увеличьте для большего момента.
STEP_DELAY = 0.001 

# Последовательность для полушагового режима (8 шагов) для более плавного движения
# ЭТО ПОСЛЕДОВАТЕЛЬНОСТЬ ДЛЯ ДВИЖЕНИЯ ВПЕРЕД (НАПРИМЕР, ПО ЧАСОВОЙ СТРЕЛКЕ)
SEQ_FORWARD = [
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 1]
]
# ОБРАТНАЯ ПОСЛЕДОВАТЕЛЬНОСТЬ (Против часовой стрелки) - отзеркаливаем массив
SEQ_BACKWARD = SEQ_FORWARD[::-1] 

# Настройка GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for pin in STEPPER_PINS:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, 0)

# --- ИСХОДНЫЕ НАСТРОЙКИ ---
kit = ServoKit(channels=16, address=0x6F)
TILT_CHANNEL, LASER_CHANNEL = 1, 15 
TILT_ANGLE_MIN, TILT_ANGLE_MAX = 50, 120
tilt_angle = 75
kit.servo[TILT_CHANNEL].angle = tilt_angle
kit._pca.channels[LASER_CHANNEL].duty_cycle = 0

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cols, rows = 640, 480
setpoint = cols // 40

# --- НАСТРОЙКИ PID-РЕГУЛЯТОРОВ ---
# !! ВАЖНО: Увеличен KP для более уверенного движения шагового двигателя
PAN_KP_STEPPER, PAN_KI, PAN_KD = 0.1, 0.0, 0.0 
TILT_KP, TILT_KI, TILT_KD = 0.008, 0.0, 0.0

pan_integral = 0.0
pan_last_time = time.time()
pan_error_prior = 0.0

tilt_integral = 0.0
tilt_last_time = time.time()
tilt_error_prior = 0.0

inTarget = 0 

# --- ФУНКЦИИ ---
# ИЗМЕНЕННАЯ ФУНКЦИЯ: УДАЛЕНА ГЛОБАЛЬНАЯ ПЕРЕМЕННАЯ ИНДЕКСА.
def move_stepper(steps, direction):
    """
    Вращает шаговый двигатель на заданное количество шагов.
    steps: количество шагов (int), должно быть положительным.
    direction: 1 для вращения вперед (SEQ_FORWARD), -1 - назад (SEQ_BACKWARD).
    """
    
    # 1. Выбираем нужную последовательность шагов
    if direction == 1:
        sequence = SEQ_FORWARD
    elif direction == -1:
        sequence = SEQ_BACKWARD
    else:
        return # Если направление не 1 или -1, ничего не делаем

    num_steps = abs(steps)
    seq_len = len(sequence)
    
    # 2. Перемещаемся по выбранной последовательности
    for i in range(num_steps):
        # Используем оператор % для циклического прохода по последовательности
        step_pattern = sequence[i % seq_len]
        
        # Устанавливаем пины в соответствии с текущим шагом
        for pin_index, pin_value in enumerate(step_pattern):
            GPIO.output(STEPPER_PINS[pin_index], pin_value)
            
        time.sleep(STEP_DELAY)

    # 3. После движения ОБЯЗАТЕЛЬНО отключаем пины для экономии энергии и предотвращения перегрева
    for pin in STEPPER_PINS:
        GPIO.output(pin, 0)


def calculate_pid(error, kp, ki, kd, integral, last_time, error_prior):
    current_time = time.time()
    delta_time = current_time - last_time

    proportional = error
    integral += error * delta_time
    derivative = (error - error_prior) / delta_time if delta_time > 0 else 0.0

    output = kp * proportional + ki * integral + kd * derivative
    
    return output, integral, current_time

# --- ОСНОВНОЙ ЦИКЛ ---
cap = cv2.VideoCapture(0)
cap.set(3, cols)
cap.set(4, rows)

pTime = 0.0
fps = 0.0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0] # Берем только первое найденное лицо
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            
            # --- РАСЧЕТ ОШИБКИ И УПРАВЛЕНИЕ ---
            pan_error = face_center_x - cols // 2
            tilt_error = face_center_y - rows // 2

            # ИЗМЕНЕННАЯ ЛОГИКА ВЫЗОВА ДЛЯ ОСИ PAN
            if abs(pan_error) > setpoint:
                pan_output, pan_integral, pan_last_time = calculate_pid(
                    pan_error, PAN_KP_STEPPER, PAN_KI, PAN_KD, pan_integral, pan_last_time, pan_error_prior
                )
                
                steps_to_move = int(pan_output)
                
                # Определяем направление и двигаем мотор
                if steps_to_move > 0:
                    # Положительная ошибка (лицо справа): двигаем мотор в сторону SEQ_BACKWARD (или FORWARD, зависит от проводки)
                    # Выберите 1 или -1, чтобы двигатель преследовал цель.
                    move_stepper(abs(steps_to_move), -1) 
                elif steps_to_move < 0:
                    # Отрицательная ошибка (лицо слева): двигаем мотор в противоположную сторону
                    move_stepper(abs(steps_to_move), 1) 
            
            # Логика для TILT (вертикаль) осталась прежней
            if abs(tilt_error) > setpoint:
                tilt_output, tilt_integral, tilt_last_time = calculate_pid(
                    tilt_error, TILT_KP, TILT_KI, TILT_KD, tilt_integral, tilt_last_time, tilt_error_prior
                )
                tilt_angle = np.clip(tilt_angle + tilt_output, TILT_ANGLE_MIN, TILT_ANGLE_MAX)
                kit.servo[TILT_CHANNEL].angle = tilt_angle

            # Логика лазера
            if abs(pan_error) < setpoint and abs(tilt_error) < setpoint:
                inTarget += 1
                if inTarget > 15:
                    kit._pca.channels[LASER_CHANNEL].duty_cycle = 65535
            else:
                inTarget = 0
                kit._pca.channels[LASER_CHANNEL].duty_cycle = 0

            pan_error_prior = pan_error
            tilt_error_prior = tilt_error
            
            # Отрисовка на экране
            cv2.rectangle(frame, (cols // 2 - setpoint, rows // 2 - setpoint), (cols // 2 + setpoint, rows // 2 + setpoint), (0, 255, 0), 1)
            cv2.line(frame, (face_center_x, 0), (face_center_x, rows), (0, 255, 255), 1)
            cv2.line(frame, (0, face_center_y), (cols, face_center_y), (0, 255, 255), 1)
            cv2.circle(frame, (face_center_x, face_center_y), 50, (0, 255, 255), 1)
        else:
            # Если лиц нет, сбрасываем счетчик цели
            inTarget = 0
            kit._pca.channels[LASER_CHANNEL].duty_cycle = 0

        # Отображение FPS
        cTime = time.time()
        fps = 0.9 * fps + 0.1 * (1 / (cTime - pTime)) if (cTime - pTime) > 0 else fps
        pTime = cTime
        cv2.putText(frame, f"FPS : {int(fps)}", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        cv2.imshow('Face Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == 27: # ESC для выхода
            break
finally:
    # Обязательная очистка GPIO и сброс оборудования при выходе
    print("Завершение работы...")
    GPIO.cleanup()
    kit.servo[TILT_CHANNEL].angle = 90
    kit._pca.channels[LASER_CHANNEL].duty_cycle = 0
    cap.release()
    cv2.destroyAllWindows()
