from silero_vad import load_silero_vad
from silero_vad import get_speech_timestamps
from silero_vad import read_audio, save_audio, collect_chunks
from matplotlib import pyplot as plt
from pydub.utils import make_chunks
from torch import Tensor
from typing import List
import numpy as np
import time
import os


def vad_test(audio: Tensor, name: str = "", thresholds: List = None, duration: int = 60) -> None:
    model = load_silero_vad()
    if thresholds is None:
        thresholds = [.5, .75, .9]
    result_dir = "vad_segment"
    for threshold in thresholds:
        vad_start = time.time()
        speech_timestamps = get_speech_timestamps(
            audio=audio, model=model, threshold=threshold
        )
        vad_time = time.time() - vad_start
        print(f"kecepatan threshold({threshold}) di skenario ({name}): {vad_time}")
        save_dir = os.path.join(result_dir, f"{name}_threshold({threshold})")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        basepath = os.path.join(save_dir, "base.wav")
        result_path = os.path.join(save_dir, "result.wav")
        clean_audio = collect_chunks(tss=speech_timestamps, wav=audio)
        save_audio(path=basepath, tensor=audio, sampling_rate=SAMPLE_RATE)
        save_audio(path=result_path, tensor=clean_audio, sampling_rate=SAMPLE_RATE)
        vad_result = [0 for _ in range(len(audio))]
        neg_start = 0
        for i, speech_ts in enumerate(speech_timestamps):
            pos_start = speech_ts['start']
            pos_end = speech_ts['end']
            vad_result[pos_start:pos_end] = [1 for _ in range(pos_end - pos_start)]
            positive_path = os.path.join(save_dir, f"positive_{i}.wav")
            negative_path = os.path.join(save_dir, f"negative_{i}.wav")
            save_audio(path=positive_path, tensor=audio[pos_start:pos_end], sampling_rate=SAMPLE_RATE)
            save_audio(path=negative_path, tensor=audio[neg_start:pos_start], sampling_rate=SAMPLE_RATE)
            neg_start = pos_end

        if speech_ts['end'] < len(audio):
            negative_path = os.path.join(save_dir, f"negative_{i + 1}.wav")
            save_audio(path=negative_path, tensor=audio[pos_end:len(audio)], sampling_rate=SAMPLE_RATE)

        x = np.linspace(0, duration, duration * SAMPLE_RATE)
        fig, axs = plt.subplots(2, 1, layout='constrained')
        axs[0].set_title(f"({threshold}) threshold")
        axs[0].set_ylabel("Sample Audio")
        axs[1].set_ylabel("Deteksi suara")
        axs[0].plot(x, audio)
        axs[1].plot(x, vad_result)
        axs[1].set_xlabel('Waktu (detik)')
        save_path = os.path.join(RESULT_DIR, f"{name}_threshold({threshold}).png")
        plt.savefig(save_path)


SAMPLE_RATE = 16000


if __name__ == "__main__":
    DIR = "raw"

    RESULT_DIR = "vad_result"
    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    thresholds = [.5, .6, .7, .8, .9]

    filenames = os.listdir(DIR)
    filenames = {"skenario_1": filenames[0], "skenario_2": filenames[5]}
    chunk_size = 30 * SAMPLE_RATE  # 1 minutes * 16000 frames per second = 480000 frames
    vad_model = load_silero_vad()
    _1m_segment_dir = "1m_dataset"
    if not os.path.exists(_1m_segment_dir):
        os.mkdir(_1m_segment_dir)

    for key in filenames:
        read_path = os.path.join(DIR, filenames[key])
        audio = read_audio(read_path)

        chunk = make_chunks(audio, chunk_size)[0]
        vad_test(name=key, audio=chunk, thresholds=thresholds, duration=chunk_size)
