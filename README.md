sudo usermod -aG audio tilek
sudo -u tilek bash -lc 'echo export OPENAI_API_KEY= >> ~/.bashrc'
sudo -u tilek python3 main.py
Зависимости один раз:
sudo apt-get install -y espeak-ng sox portaudio19-dev python3-pyaudio
pip install openai==1.* sounddevice==0.* webrtcvad face_recognition
(если dlib не ставится из pip — поставь sudo apt install python3-dlib)

sudo apt-get update
sudo apt-get install -y espeak-ng sox portaudio19-dev python3-pyaudio
pip install openai==1.* sounddevice==0.* numpy opencv-python face_recognition webrtcvad
# face_recognition требует dlib; на rpi3b+ удобнее поставить готовый whl или sudo apt install python3-dlib


полезные заметки для RPi 3B+

Если dlib из pip на твоей системе не ставится (ARMv7 колёса не всегда доступны), поставь системный пакет:

sudo apt install python3-dlib


и удали строку dlib==19.24.4 из requirements.txt (остальное оставить).

Аналогично, на Raspberry Pi часто проще ставить OpenCV через apt:

sudo apt install python3-opencv


и тогда удали строку с opencv-python (или opencv-python-headless) из requirements.txt.

Для работы adafruit-blinka убедись, что настроен I²C и установлены базовые пакеты:

sudo raspi-config  # включить I2C
sudo apt install python3-rpi.gpio i2c-tools