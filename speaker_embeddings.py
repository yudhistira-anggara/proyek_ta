from speechbrain.inference import Pretrained
from speechbrain.inference.speaker import EncoderClassifier


class SpeakerEmbedding(Pretrained):
    def __init__(self, source=):
        super().__init__()
