import dac
import torch
import torch.nn as nn

class DACFeatureMatchingLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        dac_path = dac.utils.download()
        self.dac_encoder = dac.DAC.load(dac_path).to(device).encoder.block[:2]
        print(len(self.dac_encoder))
        # print # model params
        print(sum(p.numel() for p in self.dac_encoder.parameters()))
        self.dac_encoder.eval()

    def forward(self, x, y):
        loss = 0
        prev_x = x
        prev_y = y
        for i in range(len(self.dac_encoder)):
            layer = self.dac_encoder[i]
            # is layer EncoderBlock?
            if hasattr(layer, 'block'):
                for block in layer.block:
                    prev_x = block(prev_x)
                    prev_y = block(prev_y)

                    # is block a ResidualUnit?
                    if hasattr(block, 'block'):
                        # l1 loss between features at each layer
                        loss += torch.nn.functional.l1_loss(prev_x, prev_y)

                    # don't do feature matching losses at other layers (activations, downsampling convs, etc)
            else:
                prev_x = layer(prev_x)
                prev_y = layer(prev_y)
            if i == 0:
                continue # first layer is single conv, not useful for feature matching
            
            # l1 loss between features at each layer
            

        return loss

