from generator_model import model
import torch
import cv2
from torchvision import transforms
from PIL import Image

device = "mps" if torch.backends.mps.is_available() else "cpu"

transformer = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.CenterCrop((128, 128)),
    transforms.ToTensor()

])


def generate(tensor, amount, path_to_save):
    img = cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    tensor = transformer(img)

    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)

    model.to(device)
    mu, logvar = model.encoder(tensor)

    for i in range(amount):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        out = model.decoder(z)
        out = out.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        out = (out * 255).astype("uint8")
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

        cv2.imwrite(f"{path_to_save}/generated_{i}.jpg", out)