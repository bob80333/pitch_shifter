import dac
import torch
import torch.nn as nn

class DACFeatureMatchingLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        dac_path = dac.utils.download()
        self.dac_encoder = dac.DAC.load(dac_path).to(device).encoder.block[:-2] # remove last two layers, not actually features but for the quantization / compression
        self.dac_encoder.eval()

    def forward(self, x, y):
        loss = 0
        prev_x = x
        prev_y = y
        for i in range(len(self.dac_encoder)):
            prev_x = self.dac_encoder[i](prev_x)
            prev_y = self.dac_encoder[i](prev_y)
            if i == 0:
                continue # first layer is single conv, not useful for feature matching
            
            # l1 loss between features at each layer
            loss += torch.nn.functional.l1_loss(prev_x, prev_y)

        return loss

