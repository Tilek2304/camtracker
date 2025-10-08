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
