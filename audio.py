import os
import platform
import pyaudio
import wave
from pydub import AudioSegment

def is_audio_output_available():
    return platform.system() == "Windows" or "DISPLAY" in os.environ

def play(file, speed=1.25):
    if not is_audio_output_available():
        print("Audio playback skipped: No output device available.")
        return

    CHUNK = 1024
    temp_file_created = False

    try:
        # Convert MP3 to WAV if needed
        if file.lower().endswith(".mp3"):
            temp_wav = "temp_output.wav"
            sound = AudioSegment.from_mp3(file)
            sound.export(temp_wav, format="wav")
            file = temp_wav
            temp_file_created = True

        wf = wave.open(file, 'rb')
        p = pyaudio.PyAudio()

        original_rate = wf.getframerate()
        new_rate = int(original_rate * speed)

        try:
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=new_rate,
                            output=True)
        except OSError as e:
            print(f"Playback failed (audio device error): {e}")
            return

        data = wf.readframes(CHUNK)
        while data:
            stream.write(data)
            data = wf.readframes(CHUNK)

        stream.stop_stream()
        stream.close()
        p.terminate()
        wf.close()

    except Exception as e:
        print(f"Playback failed: {e}")

    finally:
        if temp_file_created and os.path.exists(file):
            os.remove(file)
