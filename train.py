from muon import Muon
import torch
import torch.optim as optim
from model import AudioUNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    model = AudioUNet()
    model.to(device)

    # Find ≥2D parameters in the body of the network -- these will be optimized by Muon
    muon_params = [p for p in model.parameters() if p.ndim >= 2]
    # Find everything else -- these will be optimized by AdamW
    adamw_params = [p for p in model.parameters() if p.ndim < 2]
    # Create the optimizer
    optimizer = Muon(muon_params, lr=0.02, momentum=0.95,
                    adamw_params=adamw_params, adamw_lr=3e-4, adamw_betas=(0.90, 0.95), adamw_wd=0.01)
    
    