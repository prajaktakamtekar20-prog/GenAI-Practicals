import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Pixel Normalization
# ============================================================

class PixelNorm(nn.Module):
    def forward(self, x):
        return x / torch.sqrt(
            torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8
        )


# ============================================================
# Mapping Network
# Z -> W
# ============================================================

class MappingNetwork(nn.Module):

    def __init__(self, z_dim=256, w_dim=256, n_layers=6):
        super().__init__()

        layers = [PixelNorm()]

        in_dim = z_dim

        for _ in range(n_layers):
            layers.append(
                nn.Linear(in_dim, w_dim)
            )
            layers.append(
                nn.LeakyReLU(0.2, inplace=True)
            )

            in_dim = w_dim

        self.mapping = nn.Sequential(*layers)

    def forward(self, z):
        return self.mapping(z)


# ============================================================
# AdaIN
# ============================================================

class AdaIN(nn.Module):

    def __init__(self, channels, w_dim):
        super().__init__()

        self.instance_norm = nn.InstanceNorm2d(
            channels,
            affine=False
        )

        self.style_scale = nn.Linear(
            w_dim,
            channels
        )

        self.style_bias = nn.Linear(
            w_dim,
            channels
        )

        nn.init.ones_(
            self.style_scale.bias
        )

        nn.init.zeros_(
            self.style_bias.bias
        )

    def forward(self, x, w):

        x = self.instance_norm(x)

        scale = self.style_scale(w)
        scale = scale.unsqueeze(2).unsqueeze(3)

        bias = self.style_bias(w)
        bias = bias.unsqueeze(2).unsqueeze(3)

        return scale * x + bias


# ============================================================
# Noise Injection
# ============================================================

class NoiseInjection(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.weight = nn.Parameter(
            torch.zeros(1, channels, 1, 1)
        )

    def forward(self, x, noise=None):

        if noise is None:

            noise = torch.randn(
                x.size(0),
                1,
                x.size(2),
                x.size(3),
                device=x.device
            )

        return x + self.weight * noise


# ============================================================
# Synthesis Block
# ============================================================

class SynthesisBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        w_dim,
        upsample=True
    ):

        super().__init__()

        self.upsample = upsample

        self.conv1 = nn.Conv2d(
            in_ch,
            out_ch,
            3,
            padding=1
        )

        self.noise1 = NoiseInjection(
            out_ch
        )

        self.adain1 = AdaIN(
            out_ch,
            w_dim
        )

        self.conv2 = nn.Conv2d(
            out_ch,
            out_ch,
            3,
            padding=1
        )

        self.noise2 = NoiseInjection(
            out_ch
        )

        self.adain2 = AdaIN(
            out_ch,
            w_dim
        )

        self.act = nn.LeakyReLU(
            0.2,
            inplace=True
        )

    def forward(self, x, w):

        if self.upsample:

            x = F.interpolate(
                x,
                scale_factor=2,
                mode="bilinear",
                align_corners=False
            )

        x = self.act(
            self.conv1(x)
        )

        x = self.noise1(x)

        x = self.adain1(
            x,
            w
        )

        x = self.act(
            self.conv2(x)
        )

        x = self.noise2(x)

        x = self.adain2(
            x,
            w
        )

        return x


# ============================================================
# StyleGAN Generator
# ============================================================

class Generator(nn.Module):

    def __init__(
        self,
        w_dim=256,
        base_channels=256,
        img_channels=3
    ):

        super().__init__()

        self.channels = {
            4: 256,
            8: 256,
            16: 128,
            32: 64,
            64: 32
        }

        self.resolutions = [
            4,
            8,
            16,
            32,
            64
        ]

        # Learned constant
        self.starting_constant = nn.Parameter(
            torch.randn(
                1,
                self.channels[4],
                4,
                4
            )
        )

        # Initial layers
        self.initial_noise = NoiseInjection(
            self.channels[4]
        )

        self.initial_adain1 = AdaIN(
            self.channels[4],
            w_dim
        )

        self.initial_conv = nn.Conv2d(
            self.channels[4],
            self.channels[4],
            3,
            padding=1
        )

        self.initial_noise2 = NoiseInjection(
            self.channels[4]
        )

        self.initial_adain2 = AdaIN(
            self.channels[4],
            w_dim
        )

        self.act = nn.LeakyReLU(
            0.2,
            inplace=True
        )

        # Synthesis blocks
        self.blocks = nn.ModuleList()

        for i in range(
            1,
            len(self.resolutions)
        ):

            in_ch = self.channels[
                self.resolutions[i - 1]
            ]

            out_ch = self.channels[
                self.resolutions[i]
            ]

            self.blocks.append(
                SynthesisBlock(
                    in_ch,
                    out_ch,
                    w_dim,
                    upsample=True
                )
            )

        # RGB output
        self.to_rgb = nn.Conv2d(
            self.channels[64],
            img_channels,
            kernel_size=1
        )

        self.num_style_inputs = (
            1 + len(self.blocks)
        )

    def forward(self, ws):

        if isinstance(
            ws,
            torch.Tensor
        ):

            ws = [
                ws
            ] * self.num_style_inputs

        assert len(ws) == self.num_style_inputs

        batch = ws[0].shape[0]

        x = self.starting_constant.expand(
            batch,
            -1,
            -1,
            -1
        )

        x = self.initial_noise(x)

        x = self.initial_adain1(
            x,
            ws[0]
        )

        x = self.act(
            self.initial_conv(x)
        )

        x = self.initial_noise2(x)

        x = self.initial_adain2(
            x,
            ws[0]
        )

        for i, block in enumerate(
            self.blocks
        ):

            x = block(
                x,
                ws[i + 1]
            )

        return torch.tanh(
            self.to_rgb(x)
        )