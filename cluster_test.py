
from simple_diarizer.diarizer import Diarizer
from silero_vad import read_audio, get_speech_timestamps, load_silero_vad, collect_chunks
from sklearn.metrics import silhouette_score
import json
import time
import os

import warnings
warnings.filterwarnings('ignore')

SAMPLING_RATE = 16000


def diarize(signal, speech_ts, em="xvec", cm="sc", num_speakers=2):
    diarizer = Diarizer(embed_model=em, cluster_method=cm)

    embeds, segments = recording_embeds(diarizer, signal=signal, fs=SAMPLING_RATE, speech_ts=speech_ts)

    return diarizer.cluster(embeds, n_clusters=num_speakers), embeds


def recording_embeds(diarizer, signal, fs, speech_ts):
    all_embeds = []
    all_segments = []
    for utt in speech_ts:
        start = utt["start"]
        end = utt["end"]

        utt_signal = signal[start:end]
        utt_embeds, utt_segments = diarizer.windowed_embeds(
            utt_signal, fs, diarizer.window, diarizer.period
        )
        average_embed = [sum(x)/len(x) for x in zip(*utt_embeds)]
        all_embeds.append(average_embed)
        all_segments.append([start, end])

    return all_embeds, all_segments


if __name__ == "__main__":
    DIR = "raw"
    filenames = os.listdir(DIR)
    scenarios = [filenames[0], filenames[5]]
    audios = {
        f"scenario_{i+1}": read_audio(os.path.join(DIR, path), sampling_rate=SAMPLING_RATE)
        for i, path in enumerate(scenarios)
    }
    vad_model = load_silero_vad()

    for key in audios:
        tss = get_speech_timestamps(
            audio=audios[key], model=vad_model, threshold=.9, sampling_rate=SAMPLING_RATE
        )
        clean_audio = collect_chunks(tss=tss, wav=audios[key])
        with open(f"{key}.json", "r", encoding='utf-8') as file:
            result = json.load(file)

        asr_segments = result['segments']
        speech_timestamps = [{
            'start': int(segment['start'] * SAMPLING_RATE),
            'end': int(segment['end'] * SAMPLING_RATE)
        } for segment in asr_segments]
        num_speaker = 2
        if key == "scenario_2":
            num_speaker = 3
        for em in ["xvec", "ecapa"]:
            for cm in ["sc", "ahc"]:
                cluster_start = time.time()
                cluster_label, embeds = diarize(
                    signal=clean_audio,
                    speech_ts=speech_timestamps,
                    em=em,
                    cm=cm,
                    num_speakers=num_speaker,
                )
                cluster_time = time.time() - cluster_start

                ss = silhouette_score(X=embeds, labels=cluster_label)
                print(f"combination for {key} of {em} and {cm} silhouette score: {ss} within: {cluster_time}s")
