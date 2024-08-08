from math import floor
from typing import List, Dict

import whisper
from silero_vad import load_silero_vad, read_audio, collect_chunks, get_speech_timestamps
from tkinter import ttk, filedialog, messagebox, scrolledtext, END

from silero_vad.utils_vad import OnnxWrapper
from torch import Tensor
from whisper import Whisper

from speaker_diarization import SpeakerDiarizer
from tkinter import Tk
import tkinter as tk
import os

SAMPLING_RATE = 16000


class TranscriptionApp(tk.Tk):
    def __init__(
            self, 
            title: str = "Title", 
            size=None, 
            vad_model: OnnxWrapper = load_silero_vad(), 
            stt_model: Whisper = whisper.load_model("small", download_root="model"),
            sd_model: SpeakerDiarizer = SpeakerDiarizer()
    ):

        # Main Setup
        super().__init__()
        if size is None:
            size = [800, 650]
        self.title(title)
        self.geometry(f"{size[0]}x{size[1]}")
        self.minsize(width=size[0], height=size[1])

        self.transcript_text = ""
        self.transcripts = list()
        self.read_path = tk.StringVar()
        self.num_speaker = tk.IntVar()
        self.BASE_DIR = os.path.expanduser("~")
        self.vad_model = vad_model
        self.stt_model = stt_model
        self.sd_model = sd_model

        self.columnconfigure(index=0, weight=1)
        self.rowconfigure(index=0, weight=2)
        self.rowconfigure(index=1, weight=1)
        self.rowconfigure(index=2, weight=4)

        self.menu = Menu(self)
        self.editor = Editor(self)

        self.mainloop()

    def execute(self):
        is_scrolledtext_filled = self.editor.editor_scrolledtext.get("1.0", END) != ""
        is_new_file = self.editor.save_path != self.menu.read_path
        have_save = self.editor.have_save
        self.read_path = self.menu.read_path
        if self.read_path.get() == "":
            messagebox.showwarning(title='Warning!', message='Please input an audio or a subtitle file first!')
            return
        elif is_scrolledtext_filled and is_new_file:
            title = 'Warning!'
            message = "Transcription result hasn't been saved yet, save it first?"
            ask_save = messagebox.askyesno(title=title,message=message)
            if ask_save is True:
                self.editor.saver()
        elif os.path.splitext(self.read_path.get())[1] == ".srt":
            self.editor.editor_progressbar['value'] = 100
            self.editor.editor_scrolledtext.configure(state='normal')
            with open(self.read_path.get(), "r", encoding="utf-8") as file:
                self.transcript_text = file.read()
            self.editor.editor_scrolledtext.insert(index=0.0, chars=self.transcript_text)
            return
        self.num_speaker = self.menu.num_speaker
        self.audio_cleaner(threshold=0.9)
        self.editor.editor_progressbar.start()
        self.transcribe()
        self.segmentation()
        self.editor.save_path = self.read_path
        self.editor.editor_progressbar.stop()
        self.editor.editor_progressbar['value'] = 100
        self.editor.editor_scrolledtext.configure(state='normal')
        self.editor.editor_scrolledtext.insert(index=0.0, chars=self.transcript_text)
        messagebox.showinfo(title="Information", message="Audio have been successfully transcribe!")

    def audio_cleaner(self, threshold: float = 0.9):
        audio = read_audio(path=self.read_path.get(), sampling_rate=SAMPLING_RATE)
        speech_timestamps = get_speech_timestamps(audio=audio, model=self.vad_model, threshold=threshold)
        self.audio = collect_chunks(tss=speech_timestamps, wav=audio)

    def segmentation(self):
        timestamps = [{
            'start': int(segment['start'] * SAMPLING_RATE),
            'end': int(segment['end'] * SAMPLING_RATE)
        } for segment in self.transcripts]
        cluster_label = self.sd_model.get_speaker_segment(
            audio=self.audio, timestamps=timestamps, num_speaker=self.num_speaker.get()
        )

        transcript_text = ""
        for i, segment in enumerate(self.transcripts):
            hhStart = floor(segment['start'] / 3600)
            mmStart = floor(segment['start'] / 60)
            ssStart = floor(segment['start'] % 60)
            msStart = int(round(segment['start'] % 1, 3) * 1000)
            start = f"{hhStart:02d}:{mmStart:02d}:{ssStart:02d},{msStart}"

            hhEnd = floor(segment['end'] / 3600)
            mmEnd = floor(segment['end'] / 60)
            ssEnd = floor(segment['end'] % 60)
            msEnd = int(round(segment['end'] % 1, 3) * 1000)
            end = f"{hhEnd:02d}:{mmEnd:02d}:{ssEnd:02d},{msEnd}"
            content = f"[speaker_{cluster_label[i]}] {segment['text']}"
            transcript_text += f"{i + 1}\n{start} --> {end}\n{content}\n\n"
            print(f"{i + 1}\n{start} --> {end}\n{content}\n\n")

        self.transcript_text = transcript_text

    def transcribe(self):
        result = self.stt_model.transcribe(audio=self.audio)
        self.transcripts = result['segments']


class Menu(ttk.Frame):
    def __init__(self, parent):
        # Main setup
        super().__init__(parent)
        # ttk.Label(self, background='red').pack(expand=True, fill='both')
        self.grid(row=0, sticky='nsew', padx=(16, 16), pady=(8, 8))
        self.parent = parent
        self.read_path = tk.StringVar()
        self.num_speaker = tk.IntVar()
        self.BASE_DIR = os.path.expanduser("~")
        self.warning = "Work in progress"

        self.create_widgets()
        self.create_layout()

    def reader(self):
        music_dir = os.path.join(self.BASE_DIR, "Music")
        files = [('MPEG Audio Layer 3', '*.mp3'),
                 ('Waveform Audio File Format', '*.wav'),
                 ('MPEG-4 Audio', '*.m4a'),
                 ('SubRip Subtitle Format', '*.srt'),
                 ('All file format', '*.*')]
        filename = filedialog.askopenfilename(initialdir=music_dir, filetypes=files)
        self.read_path.set(filename)
        self.reader_entry.xview_moveto(1)

    def create_widgets(self):
        self.upper_frame = ttk.Frame(self)
        self.reader_label = ttk.Label(self.upper_frame, text="Input path: ")
        self.reader_entry = ttk.Entry(self.upper_frame, textvariable=self.read_path, state="disabled")
        self.reader_button = ttk.Button(self.upper_frame, text="Select File", command=self.reader)


        self.lower_frame = ttk.Frame(self)
        self.num_speaker_label = ttk.Label(self.lower_frame, text="Num Speaker: ")
        self.num_speaker_spinbox = tk.Spinbox(self.lower_frame, from_=2, to=8, textvariable=self.num_speaker)
        self.transcribe_button = ttk.Button(self.lower_frame, text='Transcribe', command=self.parent.execute)

    def create_layout(self):
        self.upper_frame.grid(row=0, column=0, sticky='we', padx=(16, 16))
        self.reader_label.pack(side='left', padx=(2, 2))
        self.reader_entry.pack(side='left', expand=True, fill='x', padx=(2, 2))
        self.reader_button.pack(side='left', padx=(2, 2))

        self.lower_frame.grid(row=1, column=0, sticky='we', padx=(16, 16))
        self.num_speaker_label.pack(side='left', padx=(2, 2))
        self.num_speaker_spinbox.pack(side='left', expand=True, fill='x', padx=(2, 2))
        self.transcribe_button.pack(side='left', padx=(2, 2))


class Editor(ttk.Frame):
    def __init__(self, parent):
        # Main setup
        super().__init__(parent)
        self.grid(row=1, sticky='nsew', padx=(16, 16), pady=(8, 8))

        self.save_path = tk.StringVar()
        self.num_speaker = tk.IntVar()
        self.BASE_DIR = os.path.expanduser("~")
        self.have_save = False

        self.create_widgets()
        self.create_layout()

    def create_widgets(self):
        self.editor_progressbar = ttk.Progressbar(self)
        self.editor_scrolledtext = scrolledtext.ScrolledText(self, wrap="word", state='disabled')
        self.editor_button = ttk.Button(self, text='Save', command=self.saver)

    def saver(self):
        files = [('SubRip Subtitle Format', '*.srt')]
        no_extension = os.path.splitext(os.path.basename(self.save_path.get()))[0]
        initial_directory = os.path.join(self.BASE_DIR, "Documents")
        filename = filedialog.asksaveasfilename(initialfile=f"{no_extension}.srt",
                                                initialdir=initial_directory,
                                                filetypes=files)
        if filename == "":
            return

        with open(filename, "w", encoding="utf-8") as file:
            file.write(self.editor_scrolledtext.get("1.0", END))

        self.have_save = True
        messagebox.showinfo("Info", "Transcription successfully saved.")

    def create_layout(self):
        self.editor_progressbar.pack(side='top', expand=True, fill='x')
        self.editor_scrolledtext.pack(side='top', expand=True, fill='x', pady=(1, 1))
        self.editor_button.pack(side='top')


if __name__ == "__main__":
    title = "ADT: Automatic Dialogue Transcriptor"
    vad_model = load_silero_vad()
    stt_model = whisper.load_model("small", download_root="model")
    sd_model = SpeakerDiarizer(embed_model='xvec', cluster_method='ahc')
    TranscriptionApp(title= title, vad_model=vad_model, stt_model=stt_model, sd_model=sd_model)
