from silero_vad import read_audio
from speechbrain.inference.speaker import EncoderClassifier
from simple_diarizer.cluster import cluster_AHC, cluster_SC
from simple_diarizer.diarizer import Diarizer
from speechbrain.inference import Pretrained
from typing import Tuple, List
from torch import Tensor
from math import ceil
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
import torch
import os

SAMPLING_RATE = 16000


def get_embed(embed_model: str = "xvec") -> Pretrained:
    run_opts = ({"device": "cuda:0"} if torch.cuda.is_available() else {"device": "cpu"})

    source = "speechbrain/spkrec-xvect-voxceleb"
    savedir = "pretrained_models/spkrec-xvect-voxceleb"

    if embed_model.lower() == "ecapa":
        source = "speechbrain/spkrec-ecapa-voxceleb"
        savedir = "pretrained_models/spkrec-ecapa-voxceleb"

    return EncoderClassifier.from_hparams(source=source, savedir=savedir, run_opts=run_opts)


def get_cluster(cluster_method: str = "sc"):
    if cluster_method.lower() == "ahc":
        return cluster_AHC

    return cluster_SC


def audio_segment(audio: Tuple[Tensor, int], window: float = 2.0, step: float = 1.0) -> List[tuple]:
    chunk_size = int(window * SAMPLING_RATE)
    offset = int(step * SAMPLING_RATE)
    number_of_chunks = int(ceil(len(audio) / float(chunk_size)) * 2)

    return [audio[i * offset:(i + offset) * chunk_size] for i in range(int(number_of_chunks))]


def audio_embed(embed_model, audio_chunks: List[tuple]) -> List:
    chunks_embedding = list()

    with torch.no_grad():
        for chunk in audio_chunks:
            chunk_embed = embed_model.encode_batch(chunk)
            chunks_embedding.append(chunk_embed.squeeze(0).squeeze(0).cpu().numpy())

    return chunks_embedding


class SpeakerDiarization:
    embed_model: Pretrained

    def __init__(self, embed_model: str = "xvec", cluster_method: str = "sc", window: float = 2.0, step: float = 1.0):
        assert embed_model in [
            "xvec",
            "ecapa",
        ], "Only xvec and ecapa are supported options"
        assert cluster_method in [
            "ahc",
            "sc",
        ], "Only ahc and sc in the supported clustering options"

        self.cluster = get_cluster(cluster_method=cluster_method)
        self.embed_model = get_embed(embed_model=embed_model)
        self.window = window
        self.step = step

    def diarizer(self, audio: str | Tuple[Tensor, int], speaker_count: int = 2):
        if type(audio) is str:
            audio = read_audio(audio, sampling_rate=16000)

        if speaker_count < 2:
            raise Exception("Number of speaker must at least be 2 or more")

        audio_chunks = audio_segment(audio=audio, window=self.window, step=self.step)
        assert len(audio_chunks) >= 1, "Couldn't find any speech during VAD"

        chunks_embedding = audio_embed(embed_model=self.embed_model, audio_chunks=audio_chunks)


if __name__ == "__main__":
    DIR = "raw"
    filenames = os.listdir(DIR)
    path = os.path.join(DIR, filenames[0])
    audio = read_audio(path, sampling_rate=16000)
    # audio = audio[:30 * 16000]

    diarizer = Diarizer(embed_model="ecapa", cluster_method="ahc")
    
    timestamps = diarizer.vad(audio)

    embed_list, _ = diarizer.recording_embeds(signal=audio, fs=SAMPLING_RATE, speech_ts=timestamps)
    cluster_labels = diarizer.cluster(
        embed_list,
        n_clusters=2,
    )
    
    _3d_embed = PCA(n_components=3).fit_transform(embed_list)
    _3d_shape = _3d_embed.shape
    _3d_embed = _3d_embed.reshape(_3d_shape[1], _3d_shape[0])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(_3d_embed[0, :], _3d_embed[1, :], _3d_embed[2, :])
    ax.set_title("Hasil potongan segmen berdasarkan ECAPA Embedding")
    plt.show()
    pickle.dump(fig, open('ECAPA_Embedding_Scatterplot_3D.fig.pickle', 'wb'))
