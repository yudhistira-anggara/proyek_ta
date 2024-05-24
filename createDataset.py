from pydub import AudioSegment
from pydub.utils import make_chunks
import os

if __name__ == "__main__":
    inputDirectory = "raw"
    outputDirectory = "dataset"
    tempDirectory = "wav"
    if not os.path.exists(outputDirectory):
        os.mkdir(outputDirectory)

    filenames = os.listdir(inputDirectory)
    chunk_length_ms = 30000  # unit segmentasi dalam milidetik
    for i, filename in enumerate(filenames):
        file_path = os.path.join(inputDirectory, filename)
        sound = AudioSegment.from_mp3(file_path)
        sound.set_frame_rate(16000)  # fungsi untuk mengubah sample rate menjadi 16kHz
        sound_name = os.path.join(tempDirectory, f"{i}.wav")
        sound = sound.split_to_mono()[0]
        sound.export(sound_name, format='wav')
        chunks = make_chunks(sound, chunk_length_ms)  # fungsi untuk membagi dataset menjadi audio berdurasi 30 detik
        for j, chunk in enumerate(chunks):
            chunk_name = os.path.join(outputDirectory, f"{i}-{j}.wav")
            print('exporting ', chunk_name)
            chunk.export(chunk_name, format='wav')
    print("export done")
