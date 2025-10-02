import cv2
import numpy as np
import time
from adafruit_servokit import ServoKit
from gpiozero import AngularServo
import RPi.GPIO as GPIO
import time
import math # Для функции abs()
# import board
# import busio
# import adafruit_pca9685

# i2c = busio.I2C(board.SCL, board.SDA)
# pca = adafruit_pca9685.PCA9685(i2c)

XMOTOR = [5,6,7,8]
FHAND = [9,10,11,12]
SHAND = [13,14,15,16]
XMCURRENT_MOTOR_ANGLE = 90
FMCURRENT_MOTOR_ANGLE = 90
SMCURRENT_MOTOR_ANGLE = 90
STEPS_PER_REVOLUTION = 4096 # 28BYJ-48 (Half-step)
HALF_STEP_SEQ = [
    [1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0],
    [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [1, 0, 0, 1]
]

# GPIO.setmode(GPIO.BCM)
# for pin in FHAND:
#     GPIO.setup(pin, GPIO.OUT)
#     GPIO.output(pin, 0)
# for pin in SHAND:
#     GPIO.setup(pin, GPIO.OUT)
#     GPIO.output(pin, 0)
# for pin in XMOTOR:
#     GPIO.setup(pin, GPIO.OUT)
#     GPIO.output(pin, 0)

def rotateM(MOTOR_PINS, target_degree, CURRENT_MOTOR_ANGLE, delay=0.001):
    """
    Вращает шаговый двигатель на заданный абсолютный угол (0-180),
    вычисляя направление и смещение относительно текущего положения.

    :param target_degree: Целевой угол (например, 120 или 60).
    :param delay: Задержка между шагами.
    """
    
    # 1. Вычисляем требуемое смещение (разница между целевым и текущим углом)
    degree_change = target_degree - CURRENT_MOTOR_ANGLE
    
    # Если смещение равно 0, ничего не делаем
    if degree_change == 0:
        print(f"Двигатель уже на {target_degree}°.")
        return
        
    # 2. Определяем направление и количество шагов
    
    if degree_change > 0:
        # target_degree > CURRENT_MOTOR_ANGLE (например, с 90 на 120)
        direction = 'forward' # По часовой
        degrees = degree_change
    else:
        # target_degree < CURRENT_MOTOR_ANGLE (например, с 90 на 60)
        direction = 'backward' # Против часовой
        degrees = abs(degree_change) # Всегда используем положительное значение для градусов
        
    # Преобразуем градусы в шаги
    steps_to_take = int((STEPS_PER_REVOLUTION / 360.0) * degrees)
    
    # 3. Выполняем шаги (та же логика, что и ранее)
    step_sequence = HALF_STEP_SEQ
    if direction == 'backward':
        step_sequence = HALF_STEP_SEQ[::-1] # Инвертируем последовательность
        
    current_step = 0
    
    print(f"Перемещение с {CURRENT_MOTOR_ANGLE}° на {target_degree}° ({degrees}°, {direction}, {steps_to_take} шагов)...")
    
    for _ in range(steps_to_take):
        pins_on_off = step_sequence[current_step % 8] 
        for pin_index in range(4):
            if MOTOR_PINS[pin_index] == 0:
                kit._pca.channels[MOTOR_PINS[pin_index]].duty_cycle = 0
            else:
                kit._pca.channels[MOTOR_PINS[pin_index]].duty_cycle = 65535
            # GPIO.output(MOTOR_PINS[pin_index], pins_on_off[pin_index])
            
        time.sleep(delay)
        current_step += 1
        
    # # 4. Выключаем пины и обновляем текущий угол
    # for pin in MOTOR_PINS:
    #     GPIO.output(pin, 0)
        
    CURRENT_MOTOR_ANGLE = target_degree # Обновляем текущий угол
    print(f"Двигатель перемещен. Новый угол: {CURRENT_MOTOR_ANGLE}°.")


kit = ServoKit(channels=16, address=0x40) # Initialize servo controller
servo = AngularServo(18,min_angle=-90, max_angle=90, min_pulse_width=0.0006, max_pulse_width=0.0024)
TILT_CHANNEL, LASER_CHANNEL  =  0, 1 # Channels for servo control
PAN_ANGLE_MIN, PAN_ANGLE_MAX = 5, 175 # Servo angles range of motion
TILT_ANGLE_MIN, TILT_ANGLE_MAX = 5, 175
pan_angle = 90 # Set initial angles for servos
tilt_angle = 90
servo.angle(tilt_angle)
# kit.servo[PAN_CHANNEL].angle = pan_angle # Move servos to initial angle
# kit.servo[TILT_CHANNEL].angle = tilt_angle
# kit._pca.channels[LASER_CHANNEL].duty_cycle = 0 #from 0 to 65535

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #Model used

cols, rows = 640, 480 # 640x480, (0.55 352x288) or( 0.5 320x240)  (0.8 512, 384)(0.7 448, 336)(0.6 384, 288)
setpoint=cols//40
# PAN_KP,PAN_KI,PAN_KD = 0.014, 0.0, 0.0 # for 352x288
# TILT_KP,TILT_KI,TILT_KD = 0.012, 0.0, 0.0 #0.12
PAN_KP,PAN_KI,PAN_KD = 0.0081, 0.0, 0.0 # PID for 640x480
TILT_KP,TILT_KI,TILT_KD = 0.008, 0.0, 0.0 
pan_integral = 0.0 # Initialize the PID controllers for pan and tilt
pan_last_time = time.time()
tilt_integral = 0.0
tilt_last_time = time.time()
pan_error_prior = 0.0 # Initialize the initial error values
tilt_error_prior = 0.0

panTarget, tiltTarget=0, 0 # how many count in target area

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

while True:
    ret, frame = cap.read() # Read the frame from the video capture
    frame = cv2.flip(frame, 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert the frame to grayscalen
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5) # Perform face detection
    for (x, y, w, h) in faces:
        face_center_x = x + w // 2 # Calculate the center of the face
        face_center_y = y + h // 2
        # Calculate error for pan and tilt
        pan_error = face_center_x - cols // 2
        tilt_error = face_center_y - rows // 2
#            print(format(pan_error, ".2f"))

        if abs(pan_error)> setpoint: # Perform PID control to calculate pan and tilt angles
            pan_output, pan_integral, pan_last_time = calculate_pid(pan_error, PAN_KP, PAN_KI, PAN_KD, pan_integral, pan_last_time, pan_error_prior)
            pan_angle = np.clip(pan_angle + pan_output, PAN_ANGLE_MIN, PAN_ANGLE_MAX)
            rotateM(XMOTOR, pan_angle, XMCURRENT_MOTOR_ANGLE)
#            print(pan_error)
        if abs(tilt_error)>setpoint:
            tilt_output, tilt_integral, tilt_last_time = calculate_pid(tilt_error, TILT_KP, TILT_KI, TILT_KD, tilt_integral, tilt_last_time, tilt_error_prior)
            tilt_angle = np.clip(tilt_angle + tilt_output, TILT_ANGLE_MIN, TILT_ANGLE_MAX) # Adjust tilt angles based on PID output
            # kit.servo[1].angle =tilt_angle
            servo.angle(tilt_angle+90)
            
        if abs(pan_error)<setpoint and abs(tilt_error)<setpoint: 
            inTarget=inTarget+1
            if inTarget>15: #in target for a period of time, say 15 counts
                # kit._pca.channels[LASER_CHANNEL].duty_cycle = 65535
                print('something wrong ' + inTarget)
        else:
            inTarget=0 # reset target count if move out
            # kit._pca.channels[LASER_CHANNEL].duty_cycle = 0
            
        pan_error_prior = pan_error # Update the error_prior values for the next iteration
        tilt_error_prior = tilt_error
        
        cv2.rectangle(frame,(cols//2-setpoint, rows//2-setpoint),(cols//2+setpoint, rows//2+setpoint),(0,255,0),1) #draw setpoint box
        cv2.line(frame, (face_center_x, 0), (face_center_x, rows), (0, 255, 255), 1) #draw moving cross
        cv2.line(frame, (0, face_center_y), (cols, face_center_y), (0, 255, 255), 1) 
        cv2.circle(frame, (face_center_x, face_center_y), 50, (0, 255, 255), 1)

        cTime = time.time() # calculate and display FPS
        fps = 0.9*fps+0.1*(1/(cTime-pTime))

        pTime = cTime
        cv2.putText(frame, f"FPS : {int(fps)}", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        print(int(fps))
    cv2.imshow('Face Tracking', frame) # Show the frame
    
    if cv2.waitKey(1) & 0xFF == 27: # Break the loop when 'q' is pressed
        break

rotateM(XMOTOR,90,XMCURRENT_MOTOR_ANGLE) # Reset the servos to their starting positions and turn off laser
servo.angle(0)
# kit.servo[TILT_CHANNEL].angle = 90
kit._pca.channels[LASER_CHANNEL].duty_cycle = 0
cap.release() # Release the video capture and clean up
cv2.destroyAllWindows()
