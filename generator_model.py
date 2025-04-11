import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        # 3, 128, 128
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.bnorm1 = nn.BatchNorm2d(32)
        # 32, 64, 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bnorm2 = nn.BatchNorm2d(64)
        # 64, 32, 32
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bnorm3 = nn.BatchNorm2d(128)
        # 128, 16, 16
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bnorm4 = nn.BatchNorm2d(256)
        # 256, 8, 8
        self.conv5 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bnorm5 = nn.BatchNorm2d(512)
        # 512, 4, 4

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(512 * 4 * 4, 1024)

        self.linear2_mu = nn.Linear(1024, latent_dim)

        self.linear2_logvar = nn.Linear(1024, latent_dim)

    def forward(self, x):
        out = F.leaky_relu(self.bnorm1(self.conv1(x)))
        # print(x.shape)
        out = F.leaky_relu(self.bnorm2(self.conv2(out)))
        out = F.leaky_relu(self.bnorm3(self.conv3(out)))
        out = F.leaky_relu(self.bnorm4(self.conv4(out)))
        out = F.leaky_relu(self.bnorm5(self.conv5(out)))

        out = self.flatten(out)

        h = F.leaky_relu(self.linear1(out))

        mu = self.linear2_mu(h)
        logvar = self.linear2_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()

        self.linear1 = nn.Linear(latent_dim, 1024)
        self.linear2 = nn.Linear(1024, 512 * 4 * 4)
        # x.view(256, 7, 7)
        self.unflatten = nn.Unflatten(1, (512, 4, 4))

        # 512, 4, 4
        self.t_conv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bnorm1 = nn.BatchNorm2d(256)
        # 256, 8, 8
        self.t_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bnorm2 = nn.BatchNorm2d(128)
        # 128, 16, 16
        self.t_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bnorm3 = nn.BatchNorm2d(64)
        # 64, 32, 32
        self.t_conv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bnorm4 = nn.BatchNorm2d(32)
        # 32, 64, 64
        self.t_conv5 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        # 3, 128, 128

        # self.upsmaple1 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        # print(x.shape)
        out = F.leaky_relu(self.linear1(x))
        # print(out.shape)
        out = F.leaky_relu(self.linear2(out))

        out = self.unflatten(out)
        out = F.leaky_relu(self.bnorm1(self.t_conv1(out)))
        # print(out.shape)
        out = F.leaky_relu(self.bnorm2(self.t_conv2(out)))
        # print(out.shape)
        out = F.leaky_relu(self.bnorm3(self.t_conv3(out)))
        # print(out.shape)
        out = F.leaky_relu(self.bnorm4(self.t_conv4(out)))
        rec = F.sigmoid(self.t_conv5(out))
        return rec


class VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        # z = μ+σ*ε
        z = mu + std * eps
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        rec = self.decoder(z)
        return rec, mu, logvar

    def predict(self, x):
        self.eval()

        with torch.no_grad():
            if len(x.shape) == 4:
                out, _, _ = self.forward(x)
            elif len(x.shape) < 4:
                x = x.unsqueeze(0)
                out, _, _ = self.forward(x)


            out = out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

        return out


def model():
    model = VAE()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    model.load_state_dict(torch.load("/Users/maxkucher/opencv/facial_thief/generator_weights/face_generator_VAE_02.pt", map_location=device))
    print(f"Model loaded!")
    return model

model = model()