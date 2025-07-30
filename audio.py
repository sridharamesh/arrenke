import pyaudio
import wave
from pydub import AudioSegment
import os

def play(file, speed=1.25):
    CHUNK = 1024

    # Convert MP3 to WAV if necessary
    if file.lower().endswith(".mp3"):
        temp_wav = "temp_output.wav"
        sound = AudioSegment.from_mp3(file)
        sound.export(temp_wav, format="wav")
        file = temp_wav

    wf = wave.open(file, 'rb')
    p = pyaudio.PyAudio()

    # Adjust rate for playback speed
    original_rate = wf.getframerate()
    new_rate = int(original_rate * speed)

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=new_rate,
                    output=True)

    data = wf.readframes(CHUNK)
    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(CHUNK)

    stream.stop_stream()
    stream.close()
    p.terminate()
    
    wf.close()

    if file == "temp_output.wav":
        os.remove(file)
