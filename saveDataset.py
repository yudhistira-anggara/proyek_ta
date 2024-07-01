import torchaudio
import whisper
import json
import os
import torch


def save_as_jsonl(data, filename):
    with open(filename, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    inputDirectory = "dataset"
    textDirectory = "target"
    if not os.path.exists(textDirectory):
        os.mkdir(textDirectory)
    SAMPLING_RATE = 16000
    filenames = os.listdir(inputDirectory)
    torch.set_num_threads(1)

    model = whisper.load_model("small", download_root="model")
    dataset_path = list()
    for filename in filenames:
        audio_path = os.path.join(inputDirectory, filename)
        no_ext = os.path.splitext(filename)[0]
        text_path = os.path.join(textDirectory, f"{no_ext}.txt")
        if os.path.exists(text_path):
            continue
        wav, fs = torchaudio.load(audio_path)

        result = model.transcribe(wav[0, :])
        with open(text_path, "w", encoding="utf-8") as f:
            f.writelines(result['text'])
        dataset_path.append({"audio": audio_path, "text": text_path})
        print(f"done packing {no_ext}.")

    TRAIN_SPLIT, TEST_SPLIT = 0.8, 0.2

    train_dataset = dataset_path[:int(len(dataset_path) * TRAIN_SPLIT)]
    test_dataset = dataset_path[:int(len(dataset_path) * TEST_SPLIT)]

    if not os.path.exists("json_directory"):
        os.makedirs("json_directory")

    save_as_jsonl(train_dataset, "json_directory/train.jsonl")
    save_as_jsonl(test_dataset, "json_directory/test.jsonl")

    print("done")
    