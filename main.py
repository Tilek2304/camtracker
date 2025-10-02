import cv2
import numpy as np
import time
# Используем ServoKit, чтобы установить частоту 50 Гц для сервопривода.
from adafruit_servokit import ServoKit 
# Нам нужен низкоуровневый доступ к драйверу PCA9685 для настройки шагового двигателя.
from adafruit_motor import stepper
from adafruit_motor.stepper import StepperMotor
# Нам также нужен объект PWMOut для правильного создания обмоток.
from adafruit_pca9685 import PWMOut 

# Инициализация ServoKit (она также инициализирует PCA9685 и устанавливает частоту ~50 Гц)
kit = ServoKit(channels=16, address=0x6F) 

# Каналы для Tilt Servo и Laser
TILT_CHANNEL, LASER_CHANNEL = 4, 15 

# --- Инициализация шагового двигателя (PAN) ---
# Для шагового двигателя используем каналы 0, 1, 2, 3
# StepperMotor ожидает объекты, которые могут управлять PWM/Logic level.
# В PCA9685, эти объекты могут быть созданы с помощью PWMOut.

# Создаем объекты PWMOut для каждой обмотки шагового двигателя.
# Это позволяет нам "обернуть" низкоуровневые регистры PCA9685.
# NOTE: i2c_device - это объект i2c, который kit уже инициализировал.
# i2c_device в ServoKit находится в kit._pca._i2c_device
i2c_device = kit._pca._i2c_device
pca = kit._pca

# Создаем обмотки (Coils) с помощью PWMOut, указывая нужный канал и PCA
COIL_A_1 = PWMOut(pca.channels[0], duty_cycle=0)
COIL_A_2 = PWMOut(pca.channels[1], duty_cycle=0)
COIL_B_1 = PWMOut(pca.channels[2], duty_cycle=0)
COIL_B_2 = PWMOut(pca.channels[3], duty_cycle=0)

# Создание объекта шагового двигателя
# Важно: здесь мы передаем объекты, которые правильно созданы для PCA9685.
pan_motor = StepperMotor(COIL_A_1, COIL_A_2, COIL_B_1, COIL_B_2, microsteps=False)

# Шаговый двигатель 28BYJ-48
STEPS_PER_REV = 2048 
PAN_DEGREE_PER_STEP = 360 / STEPS_PER_REV
PAN_ANGLE_MIN_DEG, PAN_ANGLE_MAX_DEG = 5, 175 
PAN_STEP_MIN = int(PAN_ANGLE_MIN_DEG / PAN_DEGREE_PER_STEP)
PAN_STEP_MAX = int(PAN_ANGLE_MAX_DEG / PAN_DEGREE_PER_STEP)

pan_steps = int(90 / PAN_DEGREE_PER_STEP) 

TILT_ANGLE_MIN, TILT_ANGLE_MAX = 50, 120
TILT_CHANNEL = 4 # Используем 4 для Tilt, так как 0-3 заняты шаговиком

tilt_angle = 75 
kit.servo[TILT_CHANNEL].angle = tilt_angle 
kit._pca.channels[LASER_CHANNEL].duty_cycle = 0 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

cols, rows = 640, 480 
setpoint = cols // 40
PAN_KP, PAN_KI, PAN_KD = 0.0081 * 0.5, 0.0, 0.0 
TILT_KP, TILT_KI, TILT_KD = 0.008, 0.0, 0.0 

pan_integral = 0.0
pan_last_time = time.time()
tilt_integral = 0.0
tilt_last_time = time.time()
pan_error_prior = 0.0
tilt_error_prior = 0.0

inTarget = 0 

# --- ОСТАЛЬНАЯ ЧАСТЬ КОДА ОСТАЕТСЯ ПРЕЖНЕЙ ---

def calculate_pid(error, kp, ki, kd, integral, last_time, error_prior): # Функция PID-регулятора
    current_time = time.time()
    delta_time = max(current_time - last_time, 0.0001)

    proportional = error
    integral += error * delta_time
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
    ret, frame = cap.read() 
    frame = cv2.flip(frame, 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5) 

    if not ret:
        print("Не удалось прочитать кадр с камеры")
        break

    if len(faces) > 0:
        (x, y, w, h) = faces[0] 
        face_center_x = x + w // 2 
        face_center_y = y + h // 2
        
        pan_error = face_center_x - cols // 2
        tilt_error = face_center_y - rows // 2

        # --- Управление PAN (Шаговый двигатель) ---
        if abs(pan_error) > setpoint:
            pan_output, pan_integral, pan_last_time = calculate_pid(pan_error, PAN_KP, PAN_KI, PAN_KD, pan_integral, pan_last_time, pan_error_prior)
            
            steps_to_move = int(pan_output * 0.005)
            steps_to_move = np.clip(steps_to_move, -50, 50)
            
            pan_steps += steps_to_move
            pan_steps = np.clip(pan_steps, PAN_STEP_MIN, PAN_STEP_MAX)
            
            if steps_to_move > 0:
                pan_motor.onestep(direction=stepper.FORWARD) 
            elif steps_to_move < 0:
                pan_motor.onestep(direction=stepper.BACKWARD)

        # --- Управление TILT (Сервопривод) ---
        if abs(tilt_error) > setpoint:
            tilt_output, tilt_integral, tilt_last_time = calculate_pid(tilt_error, TILT_KP, TILT_KI, TILT_KD, tilt_integral, tilt_last_time, tilt_error_prior)
            
            tilt_angle = np.clip(tilt_angle + tilt_output, TILT_ANGLE_MIN, TILT_ANGLE_MAX)
            kit.servo[TILT_CHANNEL].angle = tilt_angle
        
        # --- Управление Лазером ---
        if abs(pan_error) < setpoint and abs(tilt_error) < setpoint:
            inTarget += 1
            if inTarget > 15: 
                kit._pca.channels[LASER_CHANNEL].duty_cycle = 65535 
        else:
            inTarget = 0 
            kit._pca.channels[LASER_CHANNEL].duty_cycle = 0 
            
        pan_error_prior = pan_error 
        tilt_error_prior = tilt_error
        
        # Отрисовка
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2) 
        cv2.rectangle(frame, (cols // 2 - setpoint, rows // 2 - setpoint), (cols // 2 + setpoint, rows // 2 + setpoint), (0, 255, 0), 1)
        cv2.line(frame, (face_center_x, 0), (face_center_x, rows), (0, 255, 255), 1)
        cv2.line(frame, (0, face_center_y), (cols, face_center_y), (0, 255, 255), 1)
        cv2.circle(frame, (face_center_x, face_center_y), 50, (0, 255, 255), 1)
    else:
        # Если лицо не найдено
        kit._pca.channels[LASER_CHANNEL].duty_cycle = 0
        pan_motor.release() 

    # FPS
    cTime = time.time()
    fps = 0.9 * fps + 0.1 * (1 / (cTime - pTime))
    pTime = cTime
    cv2.putText(frame, f"FPS : {int(fps)}", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    cv2.imshow('Face Tracking', frame) 
    
    if cv2.waitKey(1) & 0xFF == 27: 
        break

# Очистка
kit.servo[TILT_CHANNEL].angle = 90
pan_motor.release() 
kit._pca.channels[LASER_CHANNEL].duty_cycle = 0
cap.release()
cv2.destroyAllWindows()
