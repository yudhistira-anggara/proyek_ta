from pydub.utils import make_chunks
import os
import torch

if __name__ == "__main__":
    inputDirectory = "raw"
    outputDirectory = "dataset"
    tempDirectory = "wav"
    if not os.path.exists(outputDirectory):
        os.mkdir(outputDirectory)

    SAMPLING_RATE = 16000
    torch.set_num_threads(1)

    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True, onnx=False)

    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils

    filenames = os.listdir(inputDirectory)
    chunk_length_ms = 15 * 16000  # unit segmentasi dalam milidetik
    for i, filename in enumerate(filenames):
        file_path = os.path.join(inputDirectory, filename)
        sound = read_audio(file_path, sampling_rate=SAMPLING_RATE)
        chunks = make_chunks(sound, chunk_length_ms)  # fungsi untuk membagi dataset menjadi audio berdurasi 30 detik
        for j, chunk in enumerate(chunks[: 100]):
            speech_timestamps = get_speech_timestamps(chunk, model, sampling_rate=SAMPLING_RATE)
            if len(speech_timestamps) < 1:
                print(f"Chunk has no speech, skipping chunk {i}-{j}.")
                continue
            chunk_name = os.path.join(outputDirectory, f"{i}-{j}.wav")
            print('exporting ', chunk_name)
            save_audio(chunk_name, collect_chunks(speech_timestamps, chunk), sampling_rate=SAMPLING_RATE)

    print("export done")
