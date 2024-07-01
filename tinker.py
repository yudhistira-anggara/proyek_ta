import json
import threading
import time

from simple_diarizer.diarizer import Diarizer
from tkinter import filedialog, scrolledtext
from tkinter import ttk
from tkinter import messagebox
from tkinter import *
from transformers import WhisperForConditionalGeneration, WhisperTokenizer
from transformers import AutomaticSpeechRecognitionPipeline as pipeline
from transformers import WhisperFeatureExtractor

import tkinter as tk
import torch
import torchaudio
import whisper
import os


def read_audio(path: str, sampling_rate: int = 16000):
    sox_backends = {'sox', 'sox_io'}
    audio_backends = torchaudio.list_audio_backends()

    if len(sox_backends.intersection(audio_backends)) > 0:
        effects = [
            ['channels', '1'],
            ['rate', str(sampling_rate)]
        ]

        wav, sr = torchaudio.sox_effects.apply_effects_file(path, effects=effects)
    else:
        wav, sr = torchaudio.load(path)

        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != sampling_rate:
            transform = torchaudio.transforms.Resample(orig_freq=sr,
                                                       new_freq=sampling_rate)
            wav = transform(wav)
            sr = sampling_rate

    assert sr == sampling_rate
    return wav.squeeze(0)


def save_audio(path: str, tensor: torch.Tensor, sampling_rate: int = 16000):
    torchaudio.save(path, tensor.unsqueeze(0), sampling_rate, bits_per_sample=16)


def get_model(model_name):
    if model_name != "custom":
        return whisper.load_model("small", download_root="model")

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Indonesian", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path="whisper-small-ina/"
    )
    return pipeline(model=model, feature_extractor=feature_extractor, tokenizer=tokenizer)


class TranscriptionApp:
    def __init__(self, root: Tk):
        self.upload_path = tk.StringVar()
        self.saveAs_path = tk.StringVar()
        self.speaker_count = tk.IntVar()
        self.model_type = tk.StringVar()
        self.embedding = tk.StringVar()
        self.clustering = tk.StringVar()
        self.ttsProgress = IntVar()
        self.HOME_DIR = os.path.expanduser("~")
        self.model_type.set("whisper")
        self.speeches = list()
        self.mainframe = ttk.Frame(root, padding='8 8 16 16')
        self.mainframe.grid(column=0, row=0, sticky='nwes')
        self.uploadEntry = ttk.Entry(self.mainframe, textvariable=self.upload_path, state="disabled")
        self.uploadEntry.grid(column=0, row=0, sticky='we')
        ttk.Button(self.mainframe, text="Pilih file", command=self.upload).grid(column=1, row=0, sticky='we')
        ttk.Label(self.mainframe, text="Jumlah pembicara: ").grid(column=0, row=1, sticky='we')
        tk.Spinbox(self.mainframe, from_=2, to=8, width=10, relief="sunken", repeatdelay=500, repeatinterval=100,
                   font=("Arial", 12), textvariable=self.speaker_count).grid(column=1, row=1, sticky='we')
        ttk.Radiobutton(self.mainframe, text="Whisper Model", variable=self.model_type,
                        value="whisper").grid(row=2, column=0)

        ttk.Radiobutton(self.mainframe, text="Custom Model", variable=self.model_type,
                        value="custom").grid(row=2, column=1)
        ttk.Label(self.mainframe, text="Embedding").grid(row=3, column=0)
        embedCombobox = ttk.Combobox(self.mainframe, textvariable=self.embedding)
        embedCombobox['value'] = ('xvec', 'ecapa')
        embedCombobox.current(0)
        embedCombobox.grid(row=3, column=1)

        ttk.Label(self.mainframe, text="Clustering").grid(row=4, column=0)
        clusterCombobox = ttk.Combobox(self.mainframe, textvariable=self.clustering)
        clusterCombobox['value'] = ('ahc', 'sc')
        clusterCombobox.current(0)
        clusterCombobox.grid(row=4, column=1)

        ttk.Button(self.mainframe, text='Unggah', command=self.main).grid(row=6, column=0)

    def upload(self):
        music_dir = os.path.join(self.HOME_DIR, "Music")
        files = [('MPEG Audio Layer 3', '*.mp3'),
                 ('Waveform Audio File Format', '*.wav'),
                 ('MPEG-4 Audio', '*.m4a'),
                 ('SubRip Subtitle Format', '*.srt'),
                 ('All file format', '*.*')]
        filename = filedialog.askopenfilename(initialdir=music_dir, filetypes=files)
        self.upload_path.set(filename)
        self.uploadEntry.xview_moveto(1)

    def saveAs(self):
        files = [('SubRip Subtitle Format', '*.srt')]
        no_extension = os.path.splitext(os.path.basename(self.upload_path.get()))[0]
        initialDirectory = os.path.join(self.HOME_DIR, "Documents")
        filename = filedialog.asksaveasfilename(initialfile=f"{no_extension}.srt",
                                                initialdir=initialDirectory,
                                                filetypes=files)
        if filename == "":
            return

        with open(filename, "w", encoding="utf-8") as file:
            file.write(self.transcriptText.get("1.0", END))

        messagebox.showinfo("", "transkripsi berhasil disimpan.")
        self.popupEdit.destroy()
        self.popupEdit.update()

    def segmentation(self, filename):
        basename = os.path.splitext(os.path.basename(filename))[0]
        jsonSegments = f"{basename}_segments.json"
        if os.path.exists(jsonSegments):
            with open(jsonSegments, "r", encoding='utf=8') as jsonFile:
                jsonReader = json.load(jsonFile)
                self.segments = jsonReader['segments']
            self.popupEdit.event_generate("<<TranscribeDone>>")
            return

        diar = Diarizer(
            embed_model=self.embedding.get(),  # 'xvec' and 'ecapa' supported
            cluster_method=self.clustering.get()  # 'ahc' and 'sc' supported
        )

        self.segments = diar.diarize(filename, num_speakers=self.speaker_count.get(), silence_tolerance=5)

        for i, segment in enumerate(self.segments):
            self.segments[i]["label"] = int(segment["label"])
            self.segments[i]["start_sample"] = int(segment["start_sample"])
            self.segments[i]["end_sample"] = int(segment["end_sample"])
        with open(jsonSegments, "w", encoding='utf-8') as file:
            json.dump({"segments": self.segments}, file)

        self.popupEdit.event_generate("<<TranscribeDone>>")
        time.sleep(1)

    def transcribe(self, audio):
        model = get_model(self.model_type.get())

        progress = 0
        progress_step = float(100.0 / len(self.segments))
        for i, segment in enumerate(self.segments):
            root.update()
            speech = audio[segment['start_sample']:segment['end_sample']]

            if self.model_type.get() != "custom":
                result = model.transcribe(speech)
            else:
                speechArray = speech.numpy()
                result = model(inputs=speechArray)

            self.speeches.append({
                "start": segment["start"],
                "end": segment["end"],
                "speaker": f"speaker {segment['label']}",
                "text": result['text']
            })

            progress += progress_step
            self.ttsProgress.set(progress)

    def display(self):
        speechText = ""
        for i, speech in enumerate(self.speeches):
            hhStart = round(speech['end'] / 3600)
            mmStart = round(speech['end'] / 60)
            ssStart = round(speech['end'] % 60)
            msStart = int(round(speech['end'] % 1, 3) * 1000)
            start = f"{hhStart:02d}:{mmStart:02d}:{ssStart:02d},{msStart}"

            hhEnd = round(speech['end'] / 3600)
            mmEnd = round(speech['end'] / 60)
            ssEnd = round(speech['end'] % 60)
            msEnd = int(round(speech['end'] % 1, 3) * 1000)
            end = f"{hhEnd:02d}:{mmEnd:02d}:{ssEnd:02d},{msEnd}"
            content = f"[{speech['speaker']}] {speech['text']}"
            speechText += f"{i + 1}\n{start} --> {end}\n{content}\n\n"

        self.transcriptText.insert(0.0, speechText)

    def mainThread(self, filename, audio):
        self.segmentation(filename=filename)
        self.transcribe(audio=audio)
        self.display()

    def main(self):
        self.popupEdit = tk.Toplevel()
        self.popupEdit.bind("<<TranscribeDone>>", self.diarizeDone)
        headerFrame = Frame(self.popupEdit, padx=16, pady=8)
        headerFrame.pack(side='top', expand=True, fill='x')

        headerFrame.columnconfigure(0, weight=1)
        headerFrame.columnconfigure(1, weight=4)
        headerFrame.rowconfigure(0, weight=1)
        headerFrame.rowconfigure(1, weight=1)

        contentFrame = Frame(self.popupEdit, padx=16, pady=8)
        contentFrame.pack(side='top', expand=True, fill='both')

        contentFrame.columnconfigure(0, weight=1)
        contentFrame.rowconfigure(0, weight=1)
        contentFrame.rowconfigure(1, weight=1)

        Label(headerFrame, text="Progres Diarisasi:", anchor='w').grid(row=0, column=0, sticky='w', pady=4)
        self.diarizeProgressbar = ttk.Progressbar(headerFrame, mode='indeterminate')
        self.diarizeProgressbar .grid(row=0, column=1, sticky='we', pady=4)
        Label(headerFrame, text="Progres Transkripsi:", anchor='w').grid(row=1, column=0, sticky='w', pady=4)
        self.ttsProgressbar = ttk.Progressbar(headerFrame, variable=self.ttsProgress, maximum=100.0)
        self.ttsProgressbar.grid(row=1, column=1, sticky='we', pady=4)

        self.transcriptText = scrolledtext.ScrolledText(contentFrame, wrap="word")
        self.transcriptText.grid(row=0, column=0, pady=4)

        saveAsButton = Button(contentFrame, text="Save", command=self.saveAs)
        saveAsButton.grid(row=1, column=0, pady=4)

        root.pack_slaves()

        filename = self.upload_path.get()
        split_ext = os.path.splitext(filename)

        if split_ext[1] == ".srt":
            self.diarizeProgressbar.configure(mode="determinate")
            with open(filename, "r", encoding='utf-8') as file:
                self.transcriptText.insert(0.0, file.read())
            return

        basename = os.path.basename(split_ext[0])
        audio = read_audio(filename, 16000)
        tempname = os.path.join(self.HOME_DIR, "Music", f"{basename}.wav")
        if not os.path.exists(tempname):
            save_audio(tempname, audio, sampling_rate=16000)

        jsonSegment = f"{basename}_segment.json"

        processThread = threading.Thread(target=self.mainThread, args=[tempname, audio])
        processThread.start()
        self.diarizeProgressbar.start(5)

    def diarizeDone(self, event=None):
        self.diarizeProgressbar.stop()
        self.diarizeProgressbar.configure(mode='determinate')
        self.diarizeProgressbar['value'] = 100


if __name__ == "__main__":
    root = Tk()
    TranscriptionApp(root=root)
    root.mainloop()
