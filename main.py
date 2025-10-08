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
import face_recognition as fr
import sounddevice as sd
import webrtcvad
from openai import OpenAI



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
# MUSIC_FILE = "A.mp3"   # <-- УКАЖИ ПУТЬ К MP3

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

OPENAI_MODEL_TEXT = "gpt-4.1-mini"
OPENAI_MODEL_STT  = "gpt-4o-mini-transcribe"
OPENAI_MODEL_TTS  = "gpt-4o-mini-tts"
VOICE = "alloy"

def _say_espeak(text, lang):
    code = {"ru":"ru","ky":"ky"}.get(lang,"ru")
    subprocess.run(["espeak-ng", "-v", code, text])

class DialogEngine:
    def __init__(self):
        self.online = False
        try:
            self.client = OpenAI()
            # ключ берётся из OPENAI_API_KEY в окружении
            self.online = True
        except Exception:
            self.online = False

    def stt(self, wav_bytes, lang_hint="ru"):
        if not self.online: return None
        try:
            r = self.client.audio.transcriptions.create(
                model=OPENAI_MODEL_STT,
                file=("speech.wav", wav_bytes),
                language=lang_hint
            )
            return (r.text or "").strip()
        except Exception as e:
            print("[STT]", e); return None

    def chat(self, text_in, lang="ru"):
        if not self.online:
            return "Сейчас я офлайн, повторим позже." if lang=="ru" else "Азыр оффлайнмын."
        sys_prompt = (
            "Ты ассистент гуманоида. Отвечай кратко, дружелюбно, 2-3 предложения, "
            "в конце задай уточняющий вопрос."
        ) if lang=="ru" else (
            "Сен гуманоид жардамчысысың. Кыска, жылуу 2-3 сүйлөм, "
            "акырында тактоочу суроо бер."
        )
        try:
            r = self.client.responses.create(
                model=OPENAI_MODEL_TEXT,
                input=[{"role":"system","content":sys_prompt},
                       {"role":"user","content":text_in}]
            )
            return r.output_text.strip()
        except Exception as e:
            print("[CHAT]", e)
            return "Кечиресиз, ката кетти." if lang=="ky" else "Извини, произошла ошибка."

    def tts(self, text, lang="ru"):
        if not self.online:
            _say_espeak(text, lang); return
        try:
            audio = self.client.audio.speech.create(
                model=OPENAI_MODEL_TTS, voice=VOICE, input=text
            )
            data = audio.read()
            out = "/tmp/reply.wav"
            with open(out,"wb") as f: f.write(data)
            subprocess.run(["aplay","-q",out])
        except Exception as e:
            print("[TTS]", e); _say_espeak(text, lang)

def record_utterance(max_sec=6, sample_rate=16000):
    import io, wave
    vad = webrtcvad.Vad(2)
    block_ms = 30
    block_len = int(sample_rate*block_ms/1000)
    buf = io.BytesIO()
    stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16")
    with stream:
        started = False; silence = 0; t0 = time.time()
        while time.time()-t0 < max_sec:
            chunk = stream.read(block_len)[0].tobytes()
            speech = vad.is_speech(chunk, sample_rate)
            if speech:
                started = True; silence = 0; buf.write(chunk)
            else:
                if started:
                    silence += 1; buf.write(chunk)
                    if silence*block_ms > 700: break
    raw = buf.getvalue()
    if len(raw) < sample_rate*0.3*2: return None
    out = io.BytesIO()
    with wave.open(out,"wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sample_rate)
        w.writeframes(raw)
    return out.getvalue()





# def start_alert(vlc_proc_ref):
#     # vlc_proc_ref: multiprocessing.Value('l', pid) — храним PID запущенного VLC
#     stop_alert(vlc_proc_ref)
#     try:
#         p = subprocess.Popen(['cvlc', '--play-and-exit', './A.mp3'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#         vlc_proc_ref.value = p.pid
#         print(f"[VLC] started PID {p.pid}")
#         print(MUSIC_FILE)
#     except FileNotFoundError:
#         print("[VLC] ERROR: 'cvlc' not found. Install: sudo apt install vlc")

# def stop_alert(vlc_proc_ref):
#     pid = vlc_proc_ref.value
#     if pid <= 0:
#         return
#     try:
#         # Вежливо пытаемся завершить
#         subprocess.run(['kill', '-TERM', str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=0.2)
#         # Если жив — добиваем
#         subprocess.run(['kill', '-KILL', str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=0.2)
#     except Exception:
#         pass
#     vlc_proc_ref.value = 0

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
                  pan_error_v: Value, tilt_error_v: Value, last_det_ts: Value,
                  recognized_id: Value, recognized_conf: Value,
                  known_encs, known_names):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    shm = shared_memory.SharedMemory(name=shm_name)
    frame_np = np.ndarray((ROWS, COLS, CHANNELS), dtype=np.uint8, buffer=shm.buf)
    last_seen_seq = 0
    known_encs = np.array(known_encs)

    while not stop_ev.is_set():
        current_seq = frame_seq.value
        if current_seq == last_seen_seq:
            time.sleep(0.001); continue
        last_seen_seq = current_seq

        frame = frame_np.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            (x,y,w,h) = max(faces, key=lambda f: f[2]*f[3])
            cx, cy = x + w//2, y + h//2
            pan_error_v.value  = float(cx - COLS//2)
            tilt_error_v.value = float(cy - ROWS//2)

            # FaceID
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encs = fr.face_encodings(rgb, known_face_locations=[(y, x+w, y+h, x)])
            ridx, rconf = -1, 0.0
            if encs:
                enc = encs[0]
                dists = fr.face_distance(known_encs, enc)
                idx = int(np.argmin(dists)); dist = float(dists[idx])
                if dist < 0.55:
                    ridx = idx
                    rconf = float(max(0.0, 1.0 - (dist/0.6)))

            last_face[0], last_face[1], last_face[2], last_face[3] = x,y,w,h
            face_present.value = 1
            recognized_id.value = ridx
            recognized_conf.value = rconf
            last_det_ts.value = time.time()
        else:
            face_present.value = 0
        time.sleep(0.01)

    shm.close()

def pid_proc(stop_ev: Event,
             pan_error_v: Value, tilt_error_v: Value, last_det_ts: Value, face_present: Value,
             vlc_pid: Value,
             recognized_id: Value=None, recognized_conf: Value=None,
             dialog_request: Value=None, last_dialog_ts: Value=None):

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

        #/////////////////////////////////new
        try:
            ridx = recognized_id.value if recognized_id is not None else -1
            rcf  = recognized_conf.value if recognized_conf is not None else 0.0
            seen_good = ((face_present.value == 1) and ((time.time()-last_det_ts.value) < DET_VISIBLE_TIMEOUT)
                        and ridx >= 0 and rcf > 0.5)
            on_target = (abs(pan_err) < SETPOINT and abs(tilt_err) < SETPOINT)
            if seen_good and on_target:
                cooldown_ok = (time.time() - (last_dialog_ts.value if last_dialog_ts else 0.0)) > 20.0
                if cooldown_ok and dialog_request is not None and dialog_request.value == 0:
                    dialog_request.value = 1
        except Exception:
            pass
        #///////////////////////////////////newend
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
                # start_alert(vlc_pid)
                action(kit)
                next_action_deadline = now_dt + ALERT_PERIOD
        else:
            # объект не виден — сдвигаем окно
            next_action_deadline = now_dt + ALERT_PERIOD

    # завершение
    # stop_alert(vlc_pid)
    try:
        kit.servo[PAN_CHANNEL].angle = 90
        kit.servo[TILT_CHANNEL].angle = 90
        kit._pca.channels[LASER_CHANNEL].duty_cycle = 0
    except Exception:
        pass
    

def dialog_proc(stop_ev: Event, dialog_request: Value, lang_pref: Value, last_dialog_ts: Value, names_arr):
    engine = DialogEngine()
    print("[DIALOG] online =", engine.online)
    while not stop_ev.is_set():
        if dialog_request.value == 1:
            dialog_request.value = 0
            last_dialog_ts.value = time.time()
            lang = "ru" if (lang_pref.value == 0) else "ky"
            greet = "Привет! Рад тебя видеть. Чем помочь?" if lang=="ru" else "Салам! Кандай жардам берем?"
            engine.tts(greet, lang)
            wav = record_utterance(max_sec=7)
            if wav is None:
                engine.tts("Не расслышал, повтори." if lang=="ru" else "Укпай калдым, кайра айтчы.", lang)
                continue
            text = engine.stt(wav, lang_hint=lang) or ""
            if not text.strip():
                engine.tts("Не понял, ещё раз." if lang=="ru" else "Түшүнбөдүм, дагы бир айтчы.", lang)
                continue
            reply = engine.chat(text, lang=lang)
            engine.tts(reply, lang=lang)
        else:
            time.sleep(0.05)


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

    recognized_id   = Value('i', -1)
    recognized_conf = Value('d', 0.0)
    dialog_request  = Value('i', 0)   # 0/1
    lang_pref       = Value('i', 0)   # 0=ru, 1=ky
    last_dialog_ts  = Value('d', 0.0)



    # Сигнал остановки
    stop_ev = Event()

    def _auto_load_faces_db():
        DB = "faces_db.npz"
        if os.path.exists(DB):
            z = np.load(DB)
            return z["names"], z["encs"]
        # Автосборка из папки people/
        PEOPLE_DIR = "people"
        names, encs = [], []
        if os.path.isdir(PEOPLE_DIR):
            for person in sorted(os.listdir(PEOPLE_DIR)):
                pdir = os.path.join(PEOPLE_DIR, person)
                if not os.path.isdir(pdir): continue
                for fn in os.listdir(pdir):
                    if not fn.lower().endswith((".jpg",".jpeg",".png")): continue
                    img = fr.load_image_file(os.path.join(pdir, fn))
                    locs = fr.face_locations(img, model="hog")
                    if not locs: continue
                    e = fr.face_encodings(img, known_face_locations=locs)
                    if e:
                        encs.append(e[0]); names.append(person)
            if encs:
                np.savez(DB, names=np.array(names), encs=np.array(encs))
                return np.array(names), np.array(encs)
        # пустая база
        return np.array([]), np.empty((0,128))

    names, encs = _auto_load_faces_db()



    # Старт процессов
    # p_cam = Process(target=camera_proc, args=(stop_ev, shm.name, frame_seq, frame_ts), name="camera")
    # p_det = Process(target=detector_proc, args=(stop_ev, shm.name, frame_seq, last_face, face_present,
    #                                             pan_error_v, tilt_error_v, last_det_ts), name="detector")
    # p_pid = Process(target=pid_proc, args=(stop_ev, pan_error_v, tilt_error_v, last_det_ts, face_present, vlc_pid), name="pid")

    # p_cam.start()
    # p_det.start()
    # p_pid.start()

    # Старт процессов
    p_cam = Process(
        target=camera_proc,
        args=(stop_ev, shm.name, frame_seq, frame_ts),
        name="camera"
    )

    p_det = Process(
        target=detector_proc,
        args=(stop_ev, shm.name, frame_seq, last_face, face_present,
            pan_error_v, tilt_error_v, last_det_ts,
            recognized_id, recognized_conf, encs, names),
        name="detector"
    )

    p_pid = Process(
        target=pid_proc,
        args=(stop_ev, pan_error_v, tilt_error_v, last_det_ts, face_present, vlc_pid,
            recognized_id, recognized_conf, dialog_request, last_dialog_ts),
        name="pid"
    )

    p_dlg = Process(
        target=dialog_proc,
        args=(stop_ev, dialog_request, lang_pref, last_dialog_ts, names),
        name="dialog"
    )

    p_cam.start(); p_det.start(); p_pid.start(); p_dlg.start()

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
            cv2.putText(frame, f"FPS:{int(fps)} Seen:{int(seen)}", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
            ridx = recognized_id.value
            rcf  = recognized_conf.value
            label = "Unknown"
            if ridx >= 0:
                try: label = f"{names[ridx]} ({rcf:.2f})"
                except: pass
            cv2.putText(frame, f"ID: {label}", (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

            cv2.imshow("Face Tracking (MP)", frame)

            if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
                stop_ev.set()
                break
    finally:
        # Чистим
        stop_ev.set()
        for p in (p_cam, p_det, p_pid, p_dlg):
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
