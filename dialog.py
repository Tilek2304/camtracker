import os, io, time, queue, subprocess
import numpy as np
import sounddevice as sd
import webrtcvad
from openai import OpenAI

OPENAI_MODEL_TEXT = "gpt-4.1-mini"             # дешево/шустро
OPENAI_MODEL_STT  = "gpt-4o-mini-transcribe"   # STT
OPENAI_MODEL_TTS  = "gpt-4o-mini-tts"          # TTS
VOICE = "alloy"                                # выбери голос из доступных

# язык по умолчанию; можно переключать извне
LANG = "ru"   # "ru" или "ky"

SYS_PROMPT_RU = (
    "Ты ассистент гуманоида. Отвечай кратко, дружелюбно, без обсуждения политики, "
    "не более 2-3 предложений. Поддерживай разговор, задавай уточняющий вопрос."
)
SYS_PROMPT_KY = (
    "Сен гуманоид жардамчысысың. Кыска, жылуу, сылык жооп бер. "
    "Сүйлөшүүнү колдо, 2-3 сүйлөмдөн ашпасын, акырында бир тактоочу суроо кош."
)

def _say_espeak(text, lang):
    # Фолбэк озвучки без интернета
    code = {"ru":"ru","ky":"ky"}.get(lang,"ru")
    subprocess.run(["espeak-ng", "-v", code, text])

class DialogEngine:
    def __init__(self):
        self.client = None
        self.online = False
        key = os.getenv("OPENAI_API_KEY")
        if key:
            try:
                self.client = OpenAI()
                self.online = True
            except Exception:
                self.online = False

    def stt(self, wav_bytes, lang_hint="ru"):
        if not self.online:
            return None
        # батч STT
        try:
            resp = self.client.audio.transcriptions.create(
                model=OPENAI_MODEL_STT,
                file=("speech.wav", wav_bytes),
                # можно подсказать язык
                language=lang_hint
            )
            return resp.text.strip()
        except Exception as e:
            print("[STT]", e)
            return None

    def chat(self, text_in, lang="ru"):
        if not self.online:
            # оффлайн ответа нет — просто эхо
            return "Извините, сейчас я офлайн. Повторите позже."
        sys_prompt = SYS_PROMPT_RU if lang == "ru" else SYS_PROMPT_KY
        try:
            resp = self.client.responses.create(
                model=OPENAI_MODEL_TEXT,
                input=[{"role":"system","content":sys_prompt},
                       {"role":"user","content":text_in}],
            )
            out = resp.output_text.strip()
            return out
        except Exception as e:
            print("[CHAT]", e); 
            return "Кечиресиз, ката кетти."

    def tts(self, text, lang="ru"):
        if not self.online:
            _say_espeak(text, lang)
            return
        try:
            audio = self.client.audio.speech.create(
                model=OPENAI_MODEL_TTS,
                voice=VOICE,
                input=text,
                # язык определится по тексту; при необходимости можно добавить style/lang hints
            )
            pcm = audio.read()  # bytes of WAV/MP3 в зависимости от SDK
            # Проигрываем через aplay/sox или напрямую через sounddevice, если PCM
            # Здесь безопасно сохраним как WAV и воспроизведём:
            out = "/tmp/reply.wav"
            with open(out, "wb") as f:
                f.write(pcm)
            subprocess.run(["aplay", "-q", out])
        except Exception as e:
            print("[TTS]", e)
            _say_espeak(text, lang)

    def record_utterance(self, max_sec=6, sample_rate=16000):
        """ Простая VAD-запись: слушаем до паузы или max_sec """
        vad = webrtcvad.Vad(2)
        block_ms = 30
        block_len = int(sample_rate * block_ms / 1000)
        buf = io.BytesIO()
        stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16")
        with stream:
            started = False
            silence = 0
            start_t = time.time()
            while time.time() - start_t < max_sec:
                audio = stream.read(block_len)[0].tobytes()
                is_speech = vad.is_speech(audio, sample_rate)
                if is_speech:
                    started = True
                    silence = 0
                    buf.write(audio)
                else:
                    if started:
                        silence += 1
                        buf.write(audio)
                        if silence * block_ms > 700:  # ~0.7s тишины → стоп
                            break
        data = buf.getvalue()
        if len(data) < sample_rate * 0.3 * 2:
            return None
        # оборачиваем в WAV
        import wave
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(data)
        return out.getvalue()
