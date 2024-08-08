from pydub.utils import make_chunks
from silero_vad import read_audio, save_audio
import os
import torch

if __name__ == "__main__":
    audio_dir = "raw"
    main_dir = "dataset"
    if not os.path.exists(main_dir):
        os.mkdir(main_dir)

    SAMPLING_RATE = 16000
    torch.set_num_threads(1)

    filenames = os.listdir(audio_dir)
    chunk_length = 60 * 16000  # unit segmentasi dalam detik
    for i, filename in enumerate(filenames):
        file_path = os.path.join(audio_dir, filename)
        split_extension = os.path.splitext(filename)[0]
        wav_path = os.path.join("wav", f"{split_extension}.wav")
        save_dir = os.path.join(main_dir, f"{split_extension}")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        sound = read_audio(file_path, sampling_rate=SAMPLING_RATE)
        save_audio(wav_path, sound, sampling_rate=SAMPLING_RATE)
        
        audio_chunks = make_chunks(sound, chunk_length=chunk_length)

        for j, chunk in enumerate(audio_chunks[:10]):
            save_path = os.path.join(save_dir, f"{j}.wav")
            if os.path.exists(save_path):
                continue
            save_audio(path=save_path, tensor=chunk, sampling_rate=SAMPLING_RATE)

    print("synthesizing done")
