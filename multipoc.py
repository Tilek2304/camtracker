#!/usr/bin/env python3
import cv2
import numpy as np
import time
import sys
import subprocess
from datetime import datetime, timedelta
from multiprocessing import Process, Event, Value, Array, set_start_method
from multiprocessing import shared_memory
from adafruit_servokit import ServoKit
import ctypes as C
import os
# -----------------------------
# Константы/настройки
# -----------------------------
COLS, ROWS = 640, 480                   # требуемое разрешение
CHANNELS = 3                            # BGR
SETPOINT = COLS // 40                   # ширина «зелёного» окна
PID_HZ = 30.0                           # частота PID
PID_DT = 1.0 / PID_HZ
ALERT_PERIOD = timedelta(seconds=5)    # каждые 30с при видимом объекте
DET_VISIBLE_TIMEOUT = 1               # объект считается видимым 0.5с после последней детекции

# PID коэффициенты (под 640×480)
PAN_KP, PAN_KI, PAN_KD = 0.004, 0.0, 0.0
TILT_KP, TILT_KI, TILT_KD = 0.0015, 0.0, 0.0

# Серво-каналы/диапазоны
PAN_CHANNEL, TILT_CHANNEL, LASER_CHANNEL = 0, 1, 15
PAN_ANGLE_MIN, PAN_ANGLE_MAX = 5, 175
TILT_ANGLE_MIN, TILT_ANGLE_MAX = 50, 120

# Аудио через VLC
MUSIC_FILE = "A.mp3"   # <-- УКАЖИ ПУТЬ К MP3

# -----------------------------
# Вспомогательные
# -----------------------------
class PID:
    def __init__(self, kp, ki, kd, out_min=None, out_max=None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_t = time.monotonic()
        self.out_min, self.out_max = out_min, out_max

    def step(self, error: float):
        t = time.monotonic()
        dt = t - self.prev_t
        if dt <= 0.0:
            dt = 1e-6
        self.integral += error * dt
        deriv = (error - self.prev_error) / dt
        out = self.kp * error + self.ki * self.integral + self.kd * deriv
        if (self.out_min is not None) or (self.out_max is not None):
            lo = -1e9 if self.out_min is None else self.out_min
            hi =  1e9 if self.out_max is None else self.out_max
            out = float(np.clip(out, lo, hi))
        self.prev_error = error
        self.prev_t = t
        return out

def start_alert(vlc_proc_ref):
    # vlc_proc_ref: multiprocessing.Value('l', pid) — храним PID запущенного VLC
    stop_alert(vlc_proc_ref)
    try:
        p = subprocess.Popen(['cvlc', '--play-and-exit', './A.mp3'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        vlc_proc_ref.value = p.pid
        print(f"[VLC] started PID {p.pid}")
        print(MUSIC_FILE)
    except FileNotFoundError:
        print("[VLC] ERROR: 'cvlc' not found. Install: sudo apt install vlc")

def stop_alert(vlc_proc_ref):
    pid = vlc_proc_ref.value
    if pid <= 0:
        return
    try:
        # Вежливо пытаемся завершить
        subprocess.run(['kill', '-TERM', str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=0.2)
        # Если жив — добиваем
        subprocess.run(['kill', '-KILL', str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=0.2)
    except Exception:
        pass
    vlc_proc_ref.value = 0

def action(kit: ServoKit):
    # Короткая анимация — синхронно, <1 сек
    try:
       # os.system('cvlc --play-and-exit A.mp3')
        print('[ACTION] servos animation')
        kit.servo[2].angle = 90
        kit.servo[4].angle = 10
        kit.servo[7].angle = 90
        for i in range(40, 60):
            kit.servo[5].angle = i
            kit.servo[6].angle = i
            time.sleep(0.001)
        time.sleep(0.25)
        for i in range(60, 40, -1):
            kit.servo[6].angle = i
            kit.servo[5].angle = i
            time.sleep(0.001)
        time.sleep(0.25)
    except Exception as e:
        print("[ACTION] servo error:", e)

# -----------------------------
# Процессы
# -----------------------------
def camera_proc(stop_ev: Event, shm_name: str, frame_seq: Value, frame_ts: Value):
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, COLS)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ROWS)
    cap.set(cv2.CAP_PROP_FPS, 60)  # запрашиваем повыше; драйвер отрегулирует
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    if not cap.isOpened():
        print("[CAM] cannot open camera")
        stop_ev.set()
        return

    shm = shared_memory.SharedMemory(name=shm_name)
    frame_np = np.ndarray((ROWS, COLS, CHANNELS), dtype=np.uint8, buffer=shm.buf)

    while not stop_ev.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 0)
        # Копируем в shared memory
        np.copyto(frame_np, frame)
        with frame_seq.get_lock():
            frame_seq.value += 1
        with frame_ts.get_lock():
            frame_ts.value = time.time()

    cap.release()
    shm.close()

def detector_proc(stop_ev: Event, shm_name: str, frame_seq: Value,
                  last_face: Array, face_present: Value,
                  pan_error_v: Value, tilt_error_v: Value, last_det_ts: Value):
    # Загрузка каскада локально в процессе
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    shm = shared_memory.SharedMemory(name=shm_name)
    frame_np = np.ndarray((ROWS, COLS, CHANNELS), dtype=np.uint8, buffer=shm.buf)

    last_seen_seq = 0

    while not stop_ev.is_set():
        # ждём новый кадр
        current_seq = frame_seq.value
        if current_seq == last_seen_seq:
            time.sleep(0.001)
            continue
        last_seen_seq = current_seq

        # Локальная копия кадра для детекции
        frame = frame_np.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) > 0:
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            cx = x + w // 2
            cy = y + h // 2
            pan_err = float(cx - COLS // 2)
            tilt_err = float(cy - ROWS // 2)

            # Пишем результаты
            last_face[0] = x
            last_face[1] = y
            last_face[2] = w
            last_face[3] = h
            face_present.value = 1
            pan_error_v.value = pan_err
            tilt_error_v.value = tilt_err
            last_det_ts.value = time.time()
        else:
            face_present.value = 0
            # ошибки не обнуляем — PID сам «успокоится»
        # небольшой «breath»
        time.sleep(0.01)

    shm.close()

def pid_proc(stop_ev: Event,
             pan_error_v: Value, tilt_error_v: Value, last_det_ts: Value, face_present: Value,
             vlc_pid: Value):
    # В этом процессе один доступ к железу (ServoKit), чтобы избежать гонок
    kit = ServoKit(channels=16, address=0x40)
    # начальные позиции
    pan_angle = 90.0
    tilt_angle = 75.0
    kit.servo[PAN_CHANNEL].angle = pan_angle
    kit.servo[TILT_CHANNEL].angle = tilt_angle
    for i in range(2, 12):
        kit.servo[i].angle = 90
    kit._pca.channels[LASER_CHANNEL].duty_cycle = 0

    pan_pid = PID(PAN_KP, PAN_KI, PAN_KD)
    tilt_pid = PID(TILT_KP, TILT_KI, TILT_KD)

    next_tick = time.monotonic()
    next_action_deadline = datetime.now() + ALERT_PERIOD
    inTarget = 0

    while not stop_ev.is_set():
        now = time.monotonic()
        if now < next_tick:
            time.sleep(max(0.0, next_tick - now))
        next_tick += PID_DT

        # Читаем ошибки/видимость
        pan_err = pan_error_v.value
        tilt_err = tilt_error_v.value
        seen = (face_present.value == 1) and ((time.time() - last_det_ts.value) < DET_VISIBLE_TIMEOUT)

        # PID шаги
        pan_out = pan_pid.step(pan_err)
        tilt_out = tilt_pid.step(tilt_err)

        pan_angle = float(np.clip(pan_angle + pan_out, PAN_ANGLE_MIN, PAN_ANGLE_MAX))
        tilt_angle = float(np.clip(tilt_angle + tilt_out, TILT_ANGLE_MIN, TILT_ANGLE_MAX))

        try:
            kit.servo[PAN_CHANNEL].angle = pan_angle
            kit.servo[TILT_CHANNEL].angle = 180-tilt_angle
        except Exception as e:
            print("[PID] servo set error:", e)

        # Лазер по попаданию в окно
        if abs(pan_err) < SETPOINT and abs(tilt_err) < SETPOINT:
            inTarget += 1
            if inTarget > 15:
                kit._pca.channels[LASER_CHANNEL].duty_cycle = 65535
        else:
            inTarget = 0
            kit._pca.channels[LASER_CHANNEL].duty_cycle = 0

        # Action + mp3 каждые 30с, пока объект виден
        now_dt = datetime.now()
        if seen:
            if now_dt >= next_action_deadline:
                start_alert(vlc_pid)
                action(kit)
                next_action_deadline = now_dt + ALERT_PERIOD
        else:
            # объект не виден — сдвигаем окно
            next_action_deadline = now_dt + ALERT_PERIOD

    # завершение
    stop_alert(vlc_pid)
    try:
        kit.servo[PAN_CHANNEL].angle = 90
        kit.servo[TILT_CHANNEL].angle = 90
        kit._pca.channels[LASER_CHANNEL].duty_cycle = 0
    except Exception:
        pass

# -----------------------------
# Главный процесс (GUI)
# -----------------------------
def main():
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    # shared memory для кадра
    frame_bytes = ROWS * COLS * CHANNELS
    shm = shared_memory.SharedMemory(create=True, size=frame_bytes)
    # метаданные кадра
    frame_seq   = Value('L', 0)         # беззнаковый long
    frame_ts    = Value('d', 0.0)       # timestamp

    # Детектор → результаты
    last_face   = Array('i', 4)         # x,y,w,h
    face_present= Value('i', 0)         # 0/1
    pan_error_v = Value('d', 0.0)
    tilt_error_v= Value('d', 0.0)
    last_det_ts = Value('d', 0.0)

    # VLC PID (для управления внешним процессом)
    vlc_pid     = Value('l', 0)         # PID VLC или 0

    # Сигнал остановки
    stop_ev = Event()

    # Старт процессов
    p_cam = Process(target=camera_proc, args=(stop_ev, shm.name, frame_seq, frame_ts), name="camera")
    p_det = Process(target=detector_proc, args=(stop_ev, shm.name, frame_seq, last_face, face_present,
                                                pan_error_v, tilt_error_v, last_det_ts), name="detector")
    p_pid = Process(target=pid_proc, args=(stop_ev, pan_error_v, tilt_error_v, last_det_ts, face_present, vlc_pid), name="pid")

    p_cam.start()
    p_det.start()
    p_pid.start()

    # Визуализация
    fps, ptime = 0.0, time.time()
    frame_view = np.ndarray((ROWS, COLS, CHANNELS), dtype=np.uint8, buffer=shm.buf)

    try:
        while not stop_ev.is_set():
            # просто отображаем текущий буфер (он всегда «последний»)
            frame = frame_view.copy()

            # рисуем оверлеи
            x, y, w, h = last_face[:]
            seen = (face_present.value == 1) and ((time.time() - last_det_ts.value) < DET_VISIBLE_TIMEOUT)

            # setpoint окно
            cv2.rectangle(frame,
                          (COLS // 2 - SETPOINT, ROWS // 2 - SETPOINT),
                          (COLS // 2 + SETPOINT, ROWS // 2 + SETPOINT),
                          (0, 255, 0), 1)

            if face_present.value == 1:
                cx = x + w // 2
                cy = y + h // 2
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
                cv2.line(frame, (cx, 0), (cx, ROWS), (0, 255, 255), 1)
                cv2.line(frame, (0, cy), (COLS, cy), (0, 255, 255), 1)
                cv2.circle(frame, (cx, cy), 50, (0, 255, 255), 1)

            ctime = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, ctime - ptime))
            ptime = ctime
            cv2.putText(frame, f"FPS:{int(fps)} Seen:{int(seen)}",
                        (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

            cv2.imshow("Face Tracking (MP)", frame)

            if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
                stop_ev.set()
                break
    finally:
        # Чистим
        stop_ev.set()
        for p in (p_cam, p_det, p_pid):
            if p.is_alive():
                p.join(timeout=1.0)
                if p.is_alive():
                    p.terminate()
        cv2.destroyAllWindows()
        shm.close()
        shm.unlink()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted, exiting...")
        sys.exit(0)
