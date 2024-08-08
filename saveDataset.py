from silero_vad import get_speech_timestamps, load_silero_vad
from simple_diarizer.diarizer import Diarizer
import torchaudio
import whisper
import torch
import time
import json
import os


def save_as_jsonl(data, filename):
    with open(filename, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    inputDirectory = "wav"
    textDirectory = "target"
    if not os.path.exists(textDirectory):
        os.mkdir(textDirectory)
    SAMPLING_RATE = 16000
    filenames = os.listdir(inputDirectory)
    torch.set_num_threads(1)

    stt_model = whisper.load_model("small", download_root="model")
    diarizer = Diarizer(embed_model="ecapa", cluster_method="ahc")
    vad_model = load_silero_vad()
    dataset_path = list()
    for filename in filenames[:1]:
        audio_path = os.path.join(inputDirectory, filename)
        no_ext = os.path.splitext(filename)[0]
        text_path = os.path.join(textDirectory, f"{no_ext}.txt")
        dataset_path.append({"audio": audio_path, "text": text_path})
        # if os.path.exists(text_path):
        #     print(f"target for {no_ext} already exist")
        #     continue
        wav, fs = torchaudio.load(audio_path)

        start_time = time.time()
        result = stt_model.transcribe(wav[0, :])
        segments = result['segments']
        for segment in segments:
            wav_segment = wav[:, segment['start']:segment['end']]
            speech_timestamps = get_speech_timestamps(audio=wav_segment, model=vad_model, threshold=.9)
            embeds, segments = diarizer.recording_embeds(wav_segment, fs, speech_timestamps)
            cluster_labels = diarizer.cluster(
                embeds,
                n_clusters=2,
            )

        execution_time = time.time() - start_time
        print("--- %s seconds ---" % (execution_time))
        print(result)


    # audios = [os.path.join(inputDirectory, f"{i}-{j}.wav") for i in range(10) for j in range(10)]
    # texts = [os.path.join(textDirectory, f"{i}-{j}.txt") for i in range(10) for j in range(10)]
    # dataset_path = [{"audio": audios[i], "text": texts[i]} for i in range(100)]

    # TRAIN_SPLIT, TEST_SPLIT = 0.8, 0.2
    # splitter = int(len(dataset_path) * TRAIN_SPLIT)
    # train_dataset = dataset_path[:splitter]
    # test_dataset = dataset_path[splitter:]
    #
    # if not os.path.exists("json_directory"):
    #     os.makedirs("json_directory")
    #
    # save_as_jsonl(train_dataset, "json_directory/train.jsonl")
    # save_as_jsonl(test_dataset, "json_directory/test.jsonl")
    # save_as_jsonl(dataset_path, "json_directory/dataset.jsonl")
    print("done")
    