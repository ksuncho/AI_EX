from typing import Tuple, List
import torch
import torch.nn as nn

from model.deepspeech2 import DeepSpeech2Model
from model.ctc import CTCLossWrapper, CTCDecoder


class DeepSpeech2Wrapper(nn.Module):

    def __init__(self,
                 feature_dim: int,
                 num_layers: int,
                 hidden_dim: int,
                 vocab_size: int,
                 blank_idx: int = 0) -> None:
        super().__init__()

        self.model = DeepSpeech2Model(feature_dim, num_layers, hidden_dim, vocab_size)
        self.loss = CTCLossWrapper(blank_idx, reduction="mean")
        self.decoder = CTCDecoder(blank_idx, vocab_size=vocab_size)

    def forward(self,
                features: torch.Tensor,
                labels: torch.Tensor,
                feature_lengths: torch.Tensor,
                label_lengths: torch.Tensor) -> torch.Tensor:
        # this function returns CTC loss, so it requires target (=labels)
        feature, feature_lengths = self.model(features, feature_lengths)
        loss = self.loss(feature, labels, feature_lengths, label_lengths)
        return loss

    def decode(self,
               features: torch.Tensor,
               feature_lengths: torch.Tensor,
               beam_width: int = 1) -> Tuple[List[List[int]], List[float]]:
        # this function returns decoded result, so it does not require target (=labels)
        feature, feature_lengths = self.model(features, feature_lengths)
        prediction, logp_score = self.decoder(feature, feature_lengths, beam_width)
        return prediction, logp_score
