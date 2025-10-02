import cv2
import numpy as np
import time
from adafruit_servokit import ServoKit
# from gpiozero import AngularServo # Не используется в финальной версии

# --- КОНСТАНТЫ И НАСТРОЙКИ ---
# Каналы PCA9685 для шагового двигателя X-оси
XMOTOR = [4, 5, 6, 7]
# Каналы для других шаговых двигателей
FHAND = [8, 9, 10, 11]
SHAND = [12, 13, 14, 15]

# Глобальные переменные для отслеживания текущего угла (Используются в основном цикле)
XMCURRENT_MOTOR_ANGLE = 90
FMCURRENT_MOTOR_ANGLE = 90
SMCURRENT_MOTOR_ANGLE = 90

STEPS_PER_REVOLUTION = 4096 # 28BYJ-48 (Half-step)
HALF_STEP_SEQ = [
    [1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0],
    [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [1, 0, 0, 1]
]

# Инициализация ServoKit (Один раз в начале)
kit = ServoKit(channels=16, address=0x40)

# --- ФУНКЦИИ УПРАВЛЕНИЯ ДВИГАТЕЛЕМ ---

def rotateM(MOTOR_PINS, target_degree, current_angle, delay=0.001):
    """
    Вращает шаговый двигатель, подключенный к PCA9685, на заданный абсолютный угол.
    
    :param MOTOR_PINS: Список каналов PCA9685 (напр., [4, 5, 6, 7]).
    :param target_degree: Целевой угол (0-180).
    :param current_angle: Текущий угол двигателя.
    :returns: Обновленный угол двигателя.
    """
    
    # 1. Вычисляем требуемое смещение
    degree_change = target_degree - current_angle
    
    if abs(degree_change) < 1: # Игнорируем минимальные изменения
        return current_angle 
        
    # Определяем направление
    direction = 'forward' if degree_change > 0 else 'backward'
    degrees = abs(degree_change)
        
    # Преобразуем градусы в шаги
    steps_to_take = int((STEPS_PER_REVOLUTION / 360.0) * degrees)
    
    # 2. Определяем последовательность шагов
    step_sequence = HALF_STEP_SEQ
    if direction == 'backward':
        step_sequence = HALF_STEP_SEQ[::-1]
        
    current_step = 0
    
    print(f"Двиг. {MOTOR_PINS[0]}: с {current_angle}° на {target_degree}° ({degrees}°, {direction}, {steps_to_take} шагов)...")
    
    # 3. Выполняем шаги
    for _ in range(steps_to_take):
        pins_on_off = step_sequence[current_step % 8] 
        
        for pin_index in range(4):
            pca_channel = MOTOR_PINS[pin_index]
            # --- КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Управление PCA9685 ---
            # pins_on_off[pin_index] = 1 (HIGH) или 0 (LOW)
            if pins_on_off[pin_index] == 1:
                 # Устанавливаем HIGH (65535)
                kit._pca.channels[pca_channel].duty_cycle = 65535
            else:
                 # Устанавливаем LOW (0)
                kit._pca.channels[pca_channel].duty_cycle = 0
            
        time.sleep(delay)
        current_step += 1
        
    # 4. Выключаем пины (важно для шаговых двигателей)
    for pca_channel in MOTOR_PINS:
        kit._pca.channels[pca_channel].duty_cycle = 0
        
    # 5. Возвращаем новый угол
    return target_degree

# --- ОСТАЛЬНЫЕ ФУНКЦИИ И ИНИЦИАЛИЗАЦИЯ (Опущены для краткости) ---
# ... (kit, TILT_CHANNEL, PAN_ANGLE_MIN, TILT_ANGLE_MAX, face_cascade, cols, rows, PID-коэффициенты) ...

# ... (Определение tilt_angle, pan_angle и т.д.) ...
TILT_CHANNEL, LASER_CHANNEL = 0, 1 # Channels for servo control
# ...
pan_angle = 90
tilt_angle = 90

# ... (calculate_pid, cap.read, pTime/cTime и т.д.) ...

def calculate_pid(error, kp, ki, kd, integral, last_time, error_prior): # PID control function
    current_time = time.time()
    delta_time = current_time - last_time

    proportional = error
    integral += error * delta_time
    derivative = (error - error_prior) / delta_time

    output = kp * proportional + ki * integral + kd * derivative

    return output, integral, current_time

cap = cv2.VideoCapture(0) # Initialize the video capture
cap.set(3, cols)  
cap.set(4, rows)  

pTime = 0.0 #time tracking for FPS
cTime = 0.0
fps=0.0
inTarget = 0 # Необходимо инициализировать

# --- ГЛАВНЫЙ ЦИКЛ ---
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        pan_error = face_center_x - cols // 2
        tilt_error = face_center_y - rows // 2

        # --- УПРАВЛЕНИЕ PAN (Шаговый двигатель) ---
        if abs(pan_error) > setpoint:
            pan_output, pan_integral, pan_last_time = calculate_pid(pan_error, PAN_KP, PAN_KI, PAN_KD, pan_integral, pan_last_time, pan_error_prior)
            # Применяем инверсию PID-выхода, если нужно (изменяем + на -)
            pan_angle = np.clip(pan_angle - pan_output, PAN_ANGLE_MIN, PAN_ANGLE_MAX)
            
            # --- ОБНОВЛЕНИЕ ГЛОБАЛЬНОГО УГЛА ---
            # Важно: сохраняем обновленный угол, который возвращает rotateM
            XMCURRENT_MOTOR_ANGLE = rotateM(XMOTOR, pan_angle, XMCURRENT_MOTOR_ANGLE, delay=0.001)

        # --- УПРАВЛЕНИЕ TILT (Сервопривод) ---
        if abs(tilt_error) > setpoint:
            tilt_output, tilt_integral, tilt_last_time = calculate_pid(tilt_error, TILT_KP, TILT_KI, TILT_KD, tilt_integral, tilt_last_time, tilt_error_prior)
            tilt_angle = np.clip(tilt_angle + tilt_output, TILT_ANGLE_MIN, TILT_ANGLE_MAX)
            kit.servo[TILT_CHANNEL].angle = tilt_angle
            
        # --- ЛАЗЕР/ОПРЕДЕЛЕНИЕ ЦЕЛИ ---
        if abs(pan_error) < setpoint and abs(tilt_error) < setpoint: 
            inTarget += 1
            if inTarget > 15:
                # Включение лазера
                kit._pca.channels[LASER_CHANNEL].duty_cycle = 65535
        else:
            inTarget = 0
            # Выключение лазера
            kit._pca.channels[LASER_CHANNEL].duty_cycle = 0
            
        pan_error_prior = pan_error
        tilt_error_prior = tilt_error
        
        # --- ОТОБРАЖЕНИЕ ---
        cv2.rectangle(frame,(cols//2-setpoint, rows//2-setpoint),(cols//2+setpoint, rows//2+setpoint),(0,255,0),1)
        cv2.line(frame, (face_center_x, 0), (face_center_x, rows), (0, 255, 255), 1)
        cv2.line(frame, (0, face_center_y), (cols, face_center_y), (0, 255, 255), 1)
        cv2.circle(frame, (face_center_x, face_center_y), 50, (0, 255, 255), 1)

        cTime = time.time()
        fps = 0.9*fps+0.1*(1/(cTime-pTime))
        pTime = cTime
        cv2.putText(frame, f"FPS : {int(fps)}", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        print(int(fps))
        
    cv2.imshow('Face Tracking', frame)
    
    if cv2.waitKey(1) & 0xFF == 27: # Break the loop when 'Esc' is pressed
        break

# --- ОЧИСТКА ---
print("Очистка и сброс позиций...")
rotateM(XMOTOR, 90, XMCURRENT_MOTOR_ANGLE) # Сброс шагового двигателя на 90
kit.servo[TILT_CHANNEL].angle = 90
kit._pca.channels[LASER_CHANNEL].duty_cycle = 0
cap.release()
cv2.destroyAllWindows()
