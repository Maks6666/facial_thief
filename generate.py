from generator_model import model
import torch
import cv2
import numpy as np

device = "mps" if torch.backends.mps.is_available() else "cpu"


def generate(tensor, amount, path_to_save):
    mu, logvar = model.encoder(tensor)

    for i in range(amount):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        out = model.decoder(z)
        out = out.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        out = (out * 255)
        cv2.imwrite(f"{path_to_save}/generated_{i}.jpg", out)