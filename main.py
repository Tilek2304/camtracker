import cv2
import numpy as np
import time
import threading
from datetime import datetime, timedelta
from adafruit_servokit import ServoKit
import subprocess
import sys

# =========================
# ---- Аппаратная часть ----
# =========================
kit = ServoKit(channels=16, address=0x40)

PAN_CHANNEL, TILT_CHANNEL, LASER_CHANNEL = 0, 1, 15
PAN_ANGLE_MIN, PAN_ANGLE_MAX = 5, 175
TILT_ANGLE_MIN, TILT_ANGLE_MAX = 50, 120

pan_angle = 90.0
tilt_angle = 75.0
kit.servo[PAN_CHANNEL].angle = pan_angle
kit.servo[TILT_CHANNEL].angle = tilt_angle
for i in range(2, 12):
    kit.servo[i].angle = 90

kit._pca.channels[LASER_CHANNEL].duty_cycle = 0  # 0..65535

# =========================
# --------- Параметры -----
# =========================
cols, rows = 640, 480                # <-- 640x480 по требованию
setpoint = cols // 40                # зона захвата (зелёный квадрат)
PID_HZ = 30.0                        # частота PID
PID_DT = 1.0 / PID_HZ
ALERT_PERIOD = timedelta(seconds=30) # каждые 30 сек ПРИ видимом объекте
DET_VISIBLE_TIMEOUT = 0.5            # считаем объект видимым, если детекция свежее 0.5с

# PID коэффициенты (под 640x480)
PAN_KP, PAN_KI, PAN_KD = 0.05, 0.0, 0.0
TILT_KP, TILT_KI, TILT_KD = 0.008, 0.0, 0.0

# Каскад
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# =========================
# ---- Общие состояния -----
# =========================
stop_event = threading.Event()

# Последний кадр
frame_lock = threading.Lock()
latest_frame = None
latest_frame_ts = 0.0

# Последняя детекция/ошибка
det_lock = threading.Lock()
last_face = None      # (x, y, w, h) или None
last_error = (0.0, 0.0)
last_det_ts = 0.0

# Счётчик попадания в «окно»
inTarget = 0

# =========================
# ------ VLC (только) ------
# =========================
VLC_PROCESS = None
MUSIC_FILE = '/путь/к/вашему/файлу.mp3'  # <-- УКАЖИТЕ ПУТЬ

def start_alert():
    """Проиграть mp3 через VLC в фоновом процессе и завершить по окончании."""
    global VLC_PROCESS
    stop_alert()
    try:
        VLC_PROCESS = subprocess.Popen(
            ['cvlc', '-I', 'dummy', '--play-and-exit', MUSIC_FILE],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(f"[VLC] Оповещение запущено (PID: {VLC_PROCESS.pid})")
    except FileNotFoundError:
        print("[VLC] ОШИБКА: 'cvlc' не найден. Установите VLC: sudo apt install vlc")

def stop_alert():
    """Остановить проигрывание, если оно идёт."""
    global VLC_PROCESS
    if VLC_PROCESS is not None:
        if VLC_PROCESS.poll() is None:
            VLC_PROCESS.terminate()
            try:
                VLC_PROCESS.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                VLC_PROCESS.kill()
        VLC_PROCESS = None

# =========================
# ------ Анимация серв -----
# =========================
def action():
    """Короткая анимация сервоприводов."""
    print('[ACTION] Запуск анимации сервоприводов')
    try:
        kit.servo[2].angle = 90
        kit.servo[4].angle = 10
        kit.servo[7].angle = 90
        for i in range(30, 50):
            kit.servo[5].angle = i
            kit.servo[6].angle = i
            time.sleep(0.001)
        time.sleep(0.3)
        for i in range(50, 30, -1):
            kit.servo[6].angle = i
            kit.servo[5].angle = i
            time.sleep(0.001)
        time.sleep(0.3)
    except Exception as e:
        print('[ACTION] Ошибка сервопривода:', e)

# =========================
# ---- Граббер кадров ------
# =========================
def camera_thread(dev_index=0):
    global latest_frame, latest_frame_ts
    cap = cv2.VideoCapture(dev_index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cols)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rows)
    cap.set(cv2.CAP_PROP_FPS, 60)  # запросите повыше; драйвер выставит реально возможное
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # если поддерживается — быстрее
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    if not cap.isOpened():
        print("[CAM] Не удалось открыть камеру")
        stop_event.set()
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 0)  # как в исходнике
        with frame_lock:
            latest_frame = frame
            latest_frame_ts = time.time()

    cap.release()

# =========================
# ---- Детектор лиц --------
# =========================
def detector_thread():
    global last_face, last_error, last_det_ts
    while not stop_event.is_set():
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.001)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) > 0:
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            pan_error = face_center_x - cols // 2
            tilt_error = face_center_y - rows // 2
            with det_lock:
                last_face = (x, y, w, h)
                last_error = (float(pan_error), float(tilt_error))
                last_det_ts = time.time()
        else:
            with det_lock:
                last_face = None
                # last_error оставляем как есть

        time.sleep(0.001)

# =========================
# -------- PID петля -------
# =========================
class PID:
    def __init__(self, kp, ki, kd, out_min=None, out_max=None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_t = time.monotonic()
        self.out_min, self.out_max = out_min, out_max

    def step(self, error):
        t = time.monotonic()
        dt = t - self.prev_t
        if dt <= 0.0:
            dt = 1e-6
        self.integral += error * dt
        deriv = (error - self.prev_error) / dt
        out = self.kp * error + self.ki * self.integral + self.kd * deriv
        if self.out_min is not None or self.out_max is not None:
            out = np.clip(out,
                          self.out_min if self.out_min is not None else -1e9,
                          self.out_max if self.out_max is not None else 1e9)
        self.prev_error = error
        self.prev_t = t
        return out

def pid_thread():
    global pan_angle, tilt_angle, inTarget
    pan_pid = PID(PAN_KP, PAN_KI, PAN_KD)
    tilt_pid = PID(TILT_KP, TILT_KI, TILT_KD)

    next_tick = time.monotonic()
    # Таймер «каждые 30 секунд при видимом объекте»
    next_action_deadline = datetime.now() + ALERT_PERIOD

    while not stop_event.is_set():
        now = time.monotonic()
        if now < next_tick:
            time.sleep(max(0.0, next_tick - now))
        next_tick += PID_DT

        # Достаём последнюю ошибку и факт видимости
        with det_lock:
            pan_error, tilt_error = last_error
            face_seen = (last_face is not None) and ((time.time() - last_det_ts) < DET_VISIBLE_TIMEOUT)

        # PID шаг
        pan_out = pan_pid.step(pan_error)
        tilt_out = tilt_pid.step(tilt_error)

        pan_angle = float(np.clip(pan_angle + pan_out, PAN_ANGLE_MIN, PAN_ANGLE_MAX))
        tilt_angle = float(np.clip(tilt_angle + tilt_out, TILT_ANGLE_MIN, TILT_ANGLE_MAX))

        try:
            kit.servo[PAN_CHANNEL].angle = pan_angle
            kit.servo[TILT_CHANNEL].angle = tilt_angle
        except Exception as e:
            print("[PID] Servo error:", e)

        # Попадание в «окно»
        if abs(pan_error) < setpoint and abs(tilt_error) < setpoint:
            inTarget += 1
            if inTarget > 15:
                kit._pca.channels[LASER_CHANNEL].duty_cycle = 65535
        else:
            inTarget = 0
            kit._pca.channels[LASER_CHANNEL].duty_cycle = 0

        # --- ВАЖНО: Action + mp3 каждые 30 сек ТОЛЬКО пока объект виден ---
        now_dt = datetime.now()
        if face_seen:
            if now_dt >= next_action_deadline:
                # Запуск VLC и анимации
                start_alert()
                # Не ждём завершения VLC (он сам завершится), анимацию делаем синхронно/коротко
                action()
                # Следующий дедлайн ещё через 30 секунд
                next_action_deadline = now_dt + ALERT_PERIOD
        else:
            # Объект не виден — сдвигаем окно; 30с пойдут после повторного появления
            next_action_deadline = now_dt + ALERT_PERIOD

# =========================
# ------ Главный поток -----
# =========================
def main():
    th_cam = threading.Thread(target=camera_thread, name="camera", daemon=True)
    th_det = threading.Thread(target=detector_thread, name="detector", daemon=True)
    th_pid = threading.Thread(target=pid_thread, name="pid", daemon=True)

    th_cam.start()
    th_det.start()
    th_pid.start()

    fps = 0.0
    pTime = time.time()

    while not stop_event.is_set():
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is not None:
            with det_lock:
                face = last_face
                pan_error, tilt_error = last_error
                face_seen = (last_face is not None) and ((time.time() - last_det_ts) < DET_VISIBLE_TIMEOUT)

            # Окно setpoint
            cv2.rectangle(frame,
                          (cols // 2 - setpoint, rows // 2 - setpoint),
                          (cols // 2 + setpoint, rows // 2 + setpoint),
                          (0, 255, 0), 1)

            if face is not None:
                (x, y, w, h) = face
                face_center_x = x + w // 2
                face_center_y = y + h // 2
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
                cv2.line(frame, (face_center_x, 0), (face_center_x, rows), (0, 255, 255), 1)
                cv2.line(frame, (0, face_center_y), (cols, face_center_y), (0, 255, 255), 1)
                cv2.circle(frame, (face_center_x, face_center_y), 50, (0, 255, 255), 1)

            # FPS только отображения
            cTime = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, (cTime - pTime)))
            pTime = cTime
            cv2.putText(frame, f"FPS: {int(fps)}  Seen:{int(face_seen)}",
                        (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            cv2.imshow("Face Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            stop_event.set()
            break

    # Завершение
    stop_alert()
    try:
        kit.servo[PAN_CHANNEL].angle = 90
        kit.servo[TILT_CHANNEL].angle = 90
        kit._pca.channels[LASER_CHANNEL].duty_cycle = 0
    except Exception:
        pass
    cv2.destroyAllWindows()
    time.sleep(0.1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        stop_event.set()
        stop_alert()
        try:
            kit.servo[PAN_CHANNEL].angle = 90
            kit.servo[TILT_CHANNEL].angle = 90
            kit._pca.channels[LASER_CHANNEL].duty_cycle = 0
        except Exception:
            pass
        cv2.destroyAllWindows()
        sys.exit(0)
