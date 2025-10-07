import cv2
import numpy as np
import time
import pygame.mixer
from adafruit_servokit import ServoKit
from datetime import datetime
import subprocess
import os
kit = ServoKit(channels=16, address=0x40) # Initialize servo controller

PAN_CHANNEL,TILT_CHANNEL, LASER_CHANNEL  = 0, 1, 15 # Channels for servo control

PAN_ANGLE_MIN, PAN_ANGLE_MAX = 5, 175 # Servo angles range of motion
TILT_ANGLE_MIN, TILT_ANGLE_MAX = 50, 120

pan_angle = 90 # Set initial angles for servos
tilt_angle = 75
kit.servo[PAN_CHANNEL].angle = pan_angle # Move servos to initial angle
kit.servo[TILT_CHANNEL].angle = tilt_angle

for i in range(2,12):
    kit.servo[i].angle = 90

kit._pca.channels[LASER_CHANNEL].duty_cycle = 0 #from 0 to 65535
pygame.mixer.init()
audio = pygame.mixer.Sound('path')
sleep = 30
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #Model used

cols, rows = 320, 240 # 640x480, (0.55 352x288) or( 0.5 320x240)  (0.8 512, 384)(0.7 448, 336)(0.6 384, 288)
setpoint=cols//40
# PAN_KP,PAN_KI,PAN_KD = 0.014, 0.0, 0.0 # for 352x288
# TILT_KP,TILT_KI,TILT_KD = 0.012, 0.0, 0.0 #0.12
PAN_KP,PAN_KI,PAN_KD = 0.05, 0.0, 0.0 # PID for 640x480
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
start_dt = datetime.now()


VLC_PROCESS = None 
MUSIC_FILE = '/путь/к/вашему/файлу.mp3' # <-- Укажите свой путь
def start_alert():
    global VLC_PROCESS
    stop_alert()
    try:
        VLC_PROCESS = subprocess.Popen(['cvlc', '-I', 'dummy', '--play-and-exit', MUSIC_FILE], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Оповещение запущено (PID: {VLC_PROCESS.pid})")
    except FileNotFoundError:
        print("ОШИБКА: Команда 'cvlc' не найдена. Установите VLC.")


def stop_alert():
    global VLC_PROCESS
    
    if VLC_PROCESS is not None:
        if VLC_PROCESS.poll() is None: 
            print(f"Прерывание оповещения (PID: {VLC_PROCESS.pid}).")
            VLC_PROCESS.terminate() # Вежливое завершение
            try:
                VLC_PROCESS.wait(timeout=0.1) 
            except subprocess.TimeoutExpired:
                VLC_PROCESS.kill() # Жесткое завершение, если не сработало
                print("Процесс VLC был убит.")
        VLC_PROCESS = None # Сбрасываем переменную после завершения

# start_alert() 
# time.sleep(0.3) 
# stop_alert() 
# print("Программа продолжает работу.")




def action():
    print('Приветствие')
    kit.servo[2].angle = 90
    kit.servo[4].angle = 10
    kit.servo[7].angle = 90
    for i in range(30,50):
        kit.servo[5].angle = i
        kit.servo[6].angle = i
        time.sleep(0.001)
    time.sleep(1)
    for i in range(50,30,-1):
        kit.servo[6].angle = i
        kit.servo[5].angle = i
        time.sleep(0.001)
    time.sleep(1)




while True:

    
    ret, frame = cap.read() # Read the frame from the video capture
    frame = cv2.flip(frame, 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert the frame to grayscalen
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5) # Perform face detection    
    end_dt = datetime.now()
    if (end_dt -start_dt)>sleep:
        start_alert()
        time.sleep(2)
        start_dt = end_dt
        action()
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
            kit.servo[0].angle =pan_angle
#            print(pan_error)
        if abs(tilt_error)>setpoint:
            tilt_output, tilt_integral, tilt_last_time = calculate_pid(tilt_error, TILT_KP, TILT_KI, TILT_KD, tilt_integral, tilt_last_time, tilt_error_prior)
            tilt_angle = np.clip(tilt_angle + tilt_output, TILT_ANGLE_MIN, TILT_ANGLE_MAX) # Adjust tilt angles based on PID output
            kit.servo[1].angle =tilt_angle
            
        if abs(pan_error)<setpoint and abs(tilt_error)<setpoint: 
            inTarget=inTarget+1
            if inTarget>15: #in target for a period of time, say 15 counts
                kit._pca.channels[LASER_CHANNEL].duty_cycle = 65535
        else:
            inTarget=0 # reset target count if move out
            kit._pca.channels[LASER_CHANNEL].duty_cycle = 0
            
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

kit.servo[PAN_CHANNEL].angle = 90 # Reset the servos to their starting positions and turn off laser
kit.servo[TILT_CHANNEL].angle = 90
kit._pca.channels[LASER_CHANNEL].duty_cycle = 0
cap.release() # Release the video capture and clean up
cv2.destroyAllWindows()
