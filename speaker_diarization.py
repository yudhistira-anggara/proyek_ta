from speechbrain.inference.speaker import EncoderClassifier
from simple_diarizer.diarizer import Diarizer
from typing import List, Dict
from torch import Tensor
import numpy as np
import torch


class SpeakerDiarizer(Diarizer):
    def __init__(self, embed_model: str = "xvec", cluster_method: str = "sc"):
        super().__init__(cluster_method=cluster_method)
        self.sampling_rate = 16000

        if embed_model == "xvec":
            self.embed_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-xvect-voxceleb",
                savedir="pretrained_models/spkrec-xvect-voxceleb",
                run_opts=self.run_opts,
            )
        elif embed_model == "ecapa":
            self.embed_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts=self.run_opts,
            )

    def segment_embedding(self, audio: Tensor, timestamps: List[Dict[str, int]]):
        embeddings = []

        for utt in timestamps:
            start = utt["start"]
            end = utt["end"]

            utt_signal = audio[start:end]
            utt_embeds, utt_segments = self.windowed_embeds(
                utt_signal, self.sampling_rate, self.window, self.period
            )
            average_embed = [sum(x) / len(x) for x in zip(*utt_embeds)]
            embeddings.append(average_embed)

        return embeddings

    def windowed_embeds(self, signal, fs, window=1.5, period=0.75):
        """
        Calculates embeddings for windows across the signal

        window: length of the window, in seconds
        period: jump of the window, in seconds

        returns: embeddings, segment info
        """
        len_window = int(window * fs)
        len_period = int(period * fs)
        len_signal = signal.shape[0]

        # Get the windowed segments
        segments = []
        start = 0
        while start + len_window < len_signal:
            segments.append([start, start + len_window])
            start += len_period

        segments.append([start, len_signal - 1])
        embeds = []

        with torch.no_grad():
            for i, j in segments:
                signal_seg = signal[i:j]
                seg_embed = self.embed_model.encode_batch(signal_seg)
                embeds.append(seg_embed.squeeze(0).squeeze(0).cpu().numpy())

        embeds = np.array(embeds)
        return embeds, np.array(segments)

    def get_speaker_segment(self, audio: Tensor, timestamps: List[Dict[str, int]], num_speaker: int = 2):
        speaker_embeddings = self.segment_embedding(audio=audio, timestamps=timestamps)

        cluster_labels = self.cluster(speaker_embeddings, n_clusters=num_speaker)

        return cluster_labels
        