import matplotlib.pyplot as plt
import os

from pydub import AudioSegment
from pydub.utils import make_chunks
from simple_diarizer.diarizer import Diarizer
from simple_diarizer.utils import combined_waveplot
import torchaudio
import whisper
import tkinter as tk

if __name__ == "__main__":
    FRAME_RATE = 16000
    DIR = "raw"
    filenames = os.listdir(DIR)
    PATH = "raw/OMA #5_ Abduh dan Semua Opininya.mp3"
    sound = AudioSegment.from_file(PATH)  # Test
    sound = sound.set_frame_rate(FRAME_RATE)
    sound = sound.split_to_mono()[0]
    WAV_FILE = "wav/OMA #5_ Abduh dan Semua Opininya.wav"
    sound.export(out_f=WAV_FILE, format="wav")
    signal, fs = torchaudio.load(WAV_FILE)
    NUM_SPEAKERS = 3
    diar = Diarizer(
                      embed_model='xvec',  # 'xvec' and 'ecapa' supported
                      cluster_method='sc'  # 'ahc' and 'sc' supported
                   )

    segments = diar.diarize(WAV_FILE, num_speakers=NUM_SPEAKERS, outfile="segments.txt", silence_tolerance=2)

    model = whisper.load_model("small")

    speeches = []
    for i, segment in enumerate(segments):
        speech = signal[0, segment['start_sample']:segment['end_sample']]
        result = model.transcribe(speech)
        speeches.append(f"speaker_{segment['label']}: {result['text']} \n \n")
        print(f"{(i+1)/len(segments)*100}%")

    with open("podcast_2.txt", "w", encoding="utf-8") as tf:
        tf.writelines(speeches)
