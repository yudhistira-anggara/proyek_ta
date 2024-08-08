from silero_vad import read_audio, get_speech_timestamps, load_silero_vad, collect_chunks
from evaluate import load
import whisper
import json
import os


def transcribe_process(data_directories, result_dir):
    vad_model = load_silero_vad()
    stt_model = whisper.load_model("small", download_root="model")
    for directory in data_directories:
        data_dir = os.path.join(MAIN_DIR, directory)
        filenames = os.listdir(data_dir)
        target_dir = os.path.join(result_dir, directory)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        for filename in filenames:
            split_extension = os.path.splitext(filename)[0]
            read_path = os.path.join(data_dir, filename)
            audio = read_audio(read_path, sampling_rate=SAMPLE_RATE)
            speech_ts = get_speech_timestamps(audio=audio, model=vad_model, threshold=0.9)
            clean_audio = collect_chunks(tss=speech_ts, wav=audio)
            result = stt_model.transcribe(audio=clean_audio)
            pure_path = os.path.join(target_dir, "pure", f"{split_extension}.json")
            editorial_path = os.path.join(target_dir, "editorial", f"{split_extension}.json")
            with open(pure_path, "w", encoding='utf-8') as file:
                json.dump(result, file)
            with open(editorial_path, "w", encoding='utf-8') as file:
                json.dump(result, file)


SAMPLE_RATE= 16000


if __name__ == "__main__":
    wer = load("wer")
    MAIN_DIR = "dataset"
    RESULT_DIR = "target"
    data_directories = os.listdir(MAIN_DIR)
    transcribe_process(data_directories=data_directories, result_dir=RESULT_DIR)

    evaluate()
    