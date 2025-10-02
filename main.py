import cv2
import numpy as np
import time
from adafruit_servokit import ServoKit
from adafruit_motor import stepper
# Import Stepper class for Stepper motor control
from adafruit_motor.stepper import StepperMotor

# Инициализация ServoKit.
# В этом случае предполагается, что вы используете PCA9685 как "PWM source" для StepperMotor.
# Для 28BYJ-48 потребуется внешний драйвер ULN2003,
# который будет подключен к каналам PCA9685.
kit = ServoKit(channels=16, address=0x6F)

# Каналы для Tilt Servo и Laser
TILT_CHANNEL, LASER_CHANNEL = 4, 15 # TILT изменен на 4, чтобы освободить 0-3 для шаговика

# Каналы для шагового двигателя PAN (28BYJ-48)
# Шаговый двигатель использует 4 канала. Здесь используются каналы 0, 1, 2, 3
COIL_A_1 = kit._pca.channels[0]
COIL_A_2 = kit._pca.channels[1]
COIL_B_1 = kit._pca.channels[2]
COIL_B_2 = kit._pca.channels[3]

# Создание объекта шагового двигателя
pan_motor = StepperMotor(COIL_A_1, COIL_A_2, COIL_B_1, COIL_B_2, microsteps=False)

# Шаговый двигатель 28BYJ-48 (в полношаговом режиме) имеет примерно 2048 шагов на оборот (360 градусов).
# Это дает 360 / 2048 ≈ 0.176 градуса на шаг.
# Нам нужно установить максимальный угол поворота, чтобы ограничить движение.
STEPS_PER_REV = 2048
PAN_DEGREE_PER_STEP = 360 / STEPS_PER_REV
PAN_ANGLE_MIN_DEG, PAN_ANGLE_MAX_DEG = 5, 175 # Углы в градусах
# Переводим углы в шаги для ограничения.
PAN_STEP_MIN = int(PAN_ANGLE_MIN_DEG / PAN_DEGREE_PER_STEP)
PAN_STEP_MAX = int(PAN_ANGLE_MAX_DEG / PAN_DEGREE_PER_STEP)

pan_steps = int(90 / PAN_DEGREE_PER_STEP) # Текущая позиция в шагах (90 градусов)
# pan_motor.release() # Шаговый двигатель не нужно двигать на старте, только Tilt
# Шаговый двигатель 28BYJ-48 сохраняет позицию, когда на него подается ток.

TILT_ANGLE_MIN, TILT_ANGLE_MAX = 50, 120

tilt_angle = 75 # Установить начальный угол для Tilt-серво
kit.servo[TILT_CHANNEL].angle = tilt_angle # Переместить Tilt-серво в начальное положение
kit._pca.channels[LASER_CHANNEL].duty_cycle = 0 # Лазер выключен

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Используемая модель

cols, rows = 640, 480
setpoint = cols // 40 # Окно, в котором считается, что цель достигнута
# PID для PAN (теперь управляет шагами)
PAN_KP, PAN_KI, PAN_KD = 0.0081 * 0.5, 0.0, 0.0 # Меньше, т.к. шаг меньше
# PID для TILT (управляет углом серво)
TILT_KP, TILT_KI, TILT_KD = 0.008, 0.0, 0.0

pan_integral = 0.0
pan_last_time = time.time()
tilt_integral = 0.0
tilt_last_time = time.time()
pan_error_prior = 0.0
tilt_error_prior = 0.0

inTarget = 0 # Счетчик нахождения в цели

def calculate_pid(error, kp, ki, kd, integral, last_time, error_prior): # Функция PID-регулятора
    current_time = time.time()
    # Защита от деления на ноль, если delta_time слишком мало.
    delta_time = max(current_time - last_time, 0.0001)

    proportional = error
    integral += error * delta_time
    # Ограничить integral, чтобы избежать windup
    integral = np.clip(integral, -100, 100)
    derivative = (error - error_prior) / delta_time

    output = kp * proportional + ki * integral + kd * derivative

    return output, integral, current_time

cap = cv2.VideoCapture(0) # Инициализация видеозахвата
cap.set(3, cols)
cap.set(4, rows)

pTime = 0.0 # Отслеживание времени для FPS
cTime = 0.0
fps = 0.0

while True:
    ret, frame = cap.read() # Чтение кадра
    frame = cv2.flip(frame, 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Преобразование в оттенки серого
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5) # Обнаружение лиц

    if not ret:
        print("Не удалось прочитать кадр с камеры")
        break

    if len(faces) > 0:
        (x, y, w, h) = faces[0] # Берем первое обнаруженное лицо
        face_center_x = x + w // 2 # Центр лица
        face_center_y = y + h // 2
        
        # Вычисление ошибки
        pan_error = face_center_x - cols // 2
        tilt_error = face_center_y - rows // 2

        # --- Управление PAN (Шаговый двигатель) ---
        if abs(pan_error) > setpoint:
            pan_output, pan_integral, pan_last_time = calculate_pid(pan_error, PAN_KP, PAN_KI, PAN_KD, pan_integral, pan_last_time, pan_error_prior)
            
            # Конвертируем выход PID (пиксели) в шаги
            # Мы хотим, чтобы output (в пикселях) был преобразован в количество шагов для двигателя.
            # pan_output / (cols/2) * MAX_STEP_CHANGE (максимальное изменение шагов)
            # Упрощенно: преобразуем выход в шаги, где 1 пиксель ошибки = k шагов
            steps_to_move = int(pan_output * 0.005) # Коэффициент 0.005 подобран эмпирически
            
            # Ограничиваем изменение шагов, чтобы не было слишком быстрого движения
            steps_to_move = np.clip(steps_to_move, -50, 50)
            
            pan_steps += steps_to_move
            pan_steps = np.clip(pan_steps, PAN_STEP_MIN, PAN_STEP_MAX)
            
            # Двигаем шаговый двигатель
            if steps_to_move > 0:
                # Вправо: steps.FORWARD или steps.BACKWARD, в зависимости от подключения
                pan_motor.onestep(direction=stepper.FORWARD) 
            elif steps_to_move < 0:
                # Влево
                pan_motor.onestep(direction=stepper.BACKWARD)

        # --- Управление TILT (Сервопривод) ---
        if abs(tilt_error) > setpoint:
            tilt_output, tilt_integral, tilt_last_time = calculate_pid(tilt_error, TILT_KP, TILT_KI, TILT_KD, tilt_integral, tilt_last_time, tilt_error_prior)
            
            tilt_angle = np.clip(tilt_angle + tilt_output, TILT_ANGLE_MIN, TILT_ANGLE_MAX)
            kit.servo[TILT_CHANNEL].angle = tilt_angle
        
        # --- Управление Лазером ---
        if abs(pan_error) < setpoint and abs(tilt_error) < setpoint:
            inTarget += 1
            if inTarget > 15: # В цели в течение определенного периода времени
                kit._pca.channels[LASER_CHANNEL].duty_cycle = 65535 # Включить лазер
        else:
            inTarget = 0 # Сбросить счетчик, если цель покинута
            kit._pca.channels[LASER_CHANNEL].duty_cycle = 0 # Выключить лазер
            
        pan_error_prior = pan_error # Обновление предыдущей ошибки
        tilt_error_prior = tilt_error
        
        # Отрисовка
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2) # Прямоугольник вокруг лица
        cv2.rectangle(frame, (cols // 2 - setpoint, rows // 2 - setpoint), (cols // 2 + setpoint, rows // 2 + setpoint), (0, 255, 0), 1)
        cv2.line(frame, (face_center_x, 0), (face_center_x, rows), (0, 255, 255), 1)
        cv2.line(frame, (0, face_center_y), (cols, face_center_y), (0, 255, 255), 1)
        cv2.circle(frame, (face_center_x, face_center_y), 50, (0, 255, 255), 1)
    else:
        # Если лицо не найдено, выключить лазер и освободить шаговый двигатель
        kit._pca.channels[LASER_CHANNEL].duty_cycle = 0
        pan_motor.release() # Снять напряжение с обмоток шагового двигателя

    # FPS
    cTime = time.time()
    fps = 0.9 * fps + 0.1 * (1 / (cTime - pTime))
    pTime = cTime
    cv2.putText(frame, f"FPS : {int(fps)}", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    # print(int(fps))
    
    cv2.imshow('Face Tracking', frame) # Показать кадр
    
    if cv2.waitKey(1) & 0xFF == 27: # Выход по Esc
        break

# Очистка
kit.servo[TILT_CHANNEL].angle = 90
pan_motor.release() # Освободить шаговый двигатель
kit._pca.channels[LASER_CHANNEL].duty_cycle = 0
cap.release()
cv2.destroyAllWindows()
