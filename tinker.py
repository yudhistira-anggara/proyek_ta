import tkinter as tk
import os
import torchaudio
import whisper

from simple_diarizer.diarizer import Diarizer
from tkinter import filedialog
from pydub import AudioSegment

root = tk.Tk()
from_path = tk.StringVar()
to_path = tk.StringVar()
speaker_count = tk.IntVar()


def upload_action(event=None):
    filename = filedialog.askopenfilename()
    from_path.set(filename)
    print('Selected:', filename)


def save_as_action(event=None):
    filename = filedialog.asksaveasfilename()
    to_path.set(filename)
    print(f"Save as: {filename}")


def transcribe(event=None):
    filename = from_path.get()
    split_ext = os.path.splitext(filename)
    sound = AudioSegment.from_file(filename)
    tempname = f"{split_ext[0]}.wav"
    if split_ext[1] != "wav" and sound.frame_rate != 16000:
        sound = sound.set_frame_rate(16000)
        sound = sound.split_to_mono()[0]
        sound.export(out_f=tempname, format="wav")

    signal, fs = torchaudio.load(tempname)

    diar = Diarizer(
                      embed_model='xvec',  # 'xvec' and 'ecapa' supported
                      cluster_method='sc'  # 'ahc' and 'sc' supported
                   )

    segments = diar.diarize(tempname, num_speakers=speaker_count.get(), outfile="segments.txt", silence_tolerance=2)

    model = whisper.load_model("small", download_root="model")

    speeches = []
    for segment in segments:
        speech = signal[0, segment['start_sample']:segment['end_sample']]
        result = model.transcribe(speech)
        speeches.append(f"speaker_{segment['label']}: {result['text']} \n\n")

    with open(to_path.get(), "w", encoding="utf-8") as tf:
        tf.writelines(speeches)


if __name__ == "__main__":
    entry_from = tk.Entry(root, textvariable=from_path, font=('calibre', 10, 'normal'))
    upload_btn = tk.Button(root, text='Upload', command=upload_action)

    entry_to = tk.Entry(root, textvariable=to_path, font=('calibre', 10, 'normal'))
    save_as_btn = tk.Button(root, text='Save As', command=save_as_action)

    spinbox = tk.Spinbox(root, from_=2, to=8, width=10, relief="sunken", repeatdelay=500, repeatinterval=100,
                         font=("Arial", 12), textvariable=speaker_count)
    transcribe_btn = tk.Button(root, text='Transcribe', command=transcribe)

    entry_from.grid(row=0, column=0)
    upload_btn.grid(row=0, column=1)
    entry_to.grid(row=1, column=0)
    save_as_btn.grid(row=1, column=1)
    spinbox.grid(row=2, column=0)
    transcribe_btn.grid(row=2, column=1)

    root.mainloop()