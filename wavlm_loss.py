import torch
import torch.nn as nn
from transformers import AutoModel


class WavLMFeatureMatchingLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        wavlm = AutoModel.from_pretrained("microsoft/wavlm-large")
        self.wavlm_encoder = wavlm.feature_extractor
        self.wavlm_encoder._freeze_parameters()
        self.wavlm_encoder.to(device)

    def forward(self, prediction, target):
        # squeeze channels dim, since HF model will add it back
        prediction = prediction.squeeze(1)
        target = target.squeeze(1)
        prediction_features = self.wavlm_encoder(prediction)
        target_features = self.wavlm_encoder(target)

        loss = torch.nn.functional.mse_loss(prediction_features, target_features)

        return loss
        